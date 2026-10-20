"""Geometry, dispatch validation and bounded error context; no game mechanics.

No new persistence, model inference, automatic retry or action dispatch.
"""
import ast as _sc_ast
import json as _sc_json


def component_geometry(cells):
    r0 = min(p[0] for p in cells)
    c0 = min(p[1] for p in cells)
    r1 = max(p[0] for p in cells)
    c1 = max(p[1] for p in cells)
    return {'bbox_rc': [r0, c0, r1, c1],
            'bbox_center2_rc': [r0 + r1, c0 + c1]}


def validate_mouse_coordinates(actions, shape):
    """Validate the whole normalized batch BEFORE its first dispatch; no coercion."""
    for i, item in enumerate(actions, 1):
        if item.get('action', '').upper() != 'MOUSE':
            continue
        if 'row' not in item or 'col' not in item:
            raise ValueError(f'MOUSE action {i:d} requires both integer row and col.')
        r, c = item['row'], item['col']
        if type(r) is not int or type(c) is not int:
            raise TypeError(f'MOUSE action {i:d}: row and col must each be an integer, not nested coordinates, floats, strings or booleans. Use one explicit (row, col) pair.')
        if len(shape) != 2 or not (0 <= r < shape[0] and 0 <= c < shape[1]):
            raise ValueError(f'MOUSE action {i:d} coordinate is outside current_frame.shape.')


def _value_shape(v, depth=0):
    # Never call repr/str/len/iteration hooks on a user-defined object.
    t = type(v)
    if v is None:
        return 'None'
    for cls, label in ((bool, 'bool'), (int, 'int'), (float, 'float')):
        if t is cls:
            return label
    if t is str:
        return f'str(length={len(v):d})'
    if t is dict:
        keys = []
        for i, k in enumerate(v):
            if i >= 8:
                break
            if type(k) is str:
                keys.append(k[:24])
        return 'dict(keys={}{})'.format(','.join(keys), ',...' if len(v) > 8 else '')
    if t in (list, tuple):
        name = 'list' if t is list else 'tuple'
        if depth >= 2:
            return f'{name}(length={len(v):d})'
        return '{}(length={:d},first={})'.format(name, len(v),
                   ','.join(_value_shape(x, depth+1) for x in v[:3]))
    if t in (set, frozenset):
        return f'set(length={len(v):d})'
    return 'object'


def exception_shape_context(exc, code):
    """Type/shape facts at the failing generated frame, never candidate contents."""
    tb = exc.__traceback__
    selected = None
    while tb is not None:
        if tb.tb_frame.f_code.co_filename in ('<python_tool>', '<retained>'):
            selected = tb
        tb = tb.tb_next
    if selected is None:
        return ''
    f = selected.tb_frame
    names = []
    if f.f_code.co_filename == '<python_tool>':
        try:
            tree = _sc_ast.parse(code)
            for node in _sc_ast.walk(tree):
                if (isinstance(node, _sc_ast.Name) and node.lineno == selected.tb_lineno
                        and node.id not in names):
                        names.append(node.id)
        except (SyntaxError, ValueError):
            pass
    # Include local arguments of retained helpers and lambdas, with a firm cap.
    for name in f.f_code.co_varnames:
        if name not in names:
            names.append(name)
    shapes = {}
    for name in names[:10]:
        if name.startswith('__'):
            continue
        if name in f.f_locals:
            value = f.f_locals[name]
        elif name in f.f_globals:
            value = f.f_globals[name]
        else:
            shapes[name[:40]] = 'MISSING'
            continue
        shapes[name[:40]] = _value_shape(value)[:180]
    if not shapes:
        return ''
    omitted = 0
    text = _sc_json.dumps({'values': shapes, 'omitted': omitted}, separators=(',', ':'), ensure_ascii=True)
    while len(text) > 1800 and shapes:
        shapes.popitem(); omitted += 1
        text = _sc_json.dumps({'values': shapes, 'omitted': omitted}, separators=(',', ':'), ensure_ascii=True)
    return '\n[value_shapes] ' + text + '\nShapes only; no data restored or repaired. Pass snippet-local data explicitly into retained functions.'


def execution_receipt(results, frame, uncertain=False):
    """Host-owned facts: never infer exact counts from an ambiguous batch reply."""
    count = 0
    complete = True
    calls = 0
    for r in results:
        executed = r.get('executed')
        if executed:
            calls += 1
        n = r.get('executed_count')
        if type(n) is int and n >= 0:
            count += n
        elif type(executed) is int and executed >= 0:
            count += executed
        elif executed is True:
            names = r.get('executed_actions')
            if type(names) is list:
                count += len(names)
            else:
                complete = False
        elif executed not in (False, None):
            complete = False
    details = {'action_calls_confirmed': calls,
               'confirmed_actions': count if complete else None,
               'execution_uncertain': bool(uncertain)}
    for name in ('level', 'step'):
        value = frame.get(name) if type(frame) is dict else None
        if type(value) is int:
            details['last_confirmed_' + name] = value
    return ('\n[execution_receipt] ' + _sc_json.dumps(details, separators=(',', ':')) +
            '\nNo automatic retry. Re-read current_frame/history before repairing the code; never replay an already executed prefix. The last confirmed frame may be stale if execution is uncertain.')
