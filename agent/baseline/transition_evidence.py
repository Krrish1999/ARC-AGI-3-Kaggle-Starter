"""Bounded visual/terminal evidence, not a resource decoder or death classifier.
Standard library only; source is injected into the isolated Python tool.
"""
import itertools


def _te_grid(frame):
    g = frame.get('grid') if type(frame) is dict else None
    if type(g) not in (list, tuple) or not g or len(g) > 256:
        return None
    if type(g[0]) not in (list, tuple) or not g[0] or len(g[0]) > 256 or len(g)*len(g[0]) > 8192:
        return None
    w = len(g[0])
    if any(type(row) not in (list, tuple) or len(row) != w or any(type(x) is not int or not 0 <= x < 16 for x in row) for row in g):
        return None
    return tuple(tuple(row) for row in g)


def _te_contiguous(a, b):
    if type(a) is not dict or type(b) is not dict:
        return False
    if any(type(f.get(k)) is not int for f in (a,b) for k in ('step','level')):
        return False
    action = b.get('action')
    result = b.get('result')
    if type(action) is not str or not action or action.strip().upper().startswith('RESET') or type(result) is not dict:
        return False
    if any(result.get(k) is True for k in ('level_completed','run_complete','execution_uncertain')):
        return False
    if 'executed' in result and result['executed'] in (False,0,None):
        return False
    # A death frame itself is evidence; do not discard the terminal transition.
    return a['level'] == b['level'] and b['step'] == a['step']+1


def _te_stripes(frames, colors):
    if len(frames) < 3 or not all(_te_contiguous(a,b) for a,b in zip(frames[-3:],frames[-2:])):
        return []
    grids = [_te_grid(f) for f in frames[-3:]]
    if any(g is None for g in grids):
        return []
    grids = [g for g in grids if g is not None]
    if any(g is None for g in grids) or len({(len(g),len(g[0])) for g in grids}) != 1:
        return []
    results = []
    for axis in ('row','col'):
        lines = grids if axis == 'row' else [tuple(zip(*g)) for g in grids]
        for index in range(len(lines[0])):
            stripe = [g[index] for g in lines]
            # A contiguous two-color fill is only a CANDIDATE indicator. No edge,
            # palette role, game identifier or pixel coordinate is assumed.
            palette = set().union(*(set(line) for line in stripe))
            if len(palette) != 2 or any(sum(a != b for a,b in itertools.pairwise(line)) > 1 for line in stripe):
                continue
            for color in sorted(palette):
                counts = [line.count(color) for line in stripe]
                if counts[0] > counts[1] > counts[2]:
                    results.append({'axis':axis,'index':index,'color':colors[color],
                        'counts':counts,'steps':[f['step'] for f in frames[-3:]],
                        'length':len(stripe[0]),'empty_now':counts[-1] == 0,
                        'interpretation':'possible resource/counter or ordinary geometry; unresolved'})
    results.sort(key=lambda x:(not x['empty_now'],-(x['counts'][0]-x['counts'][-1]),x['axis'],x['index']))
    return results


def transition_evidence(frames, colors):
    """Frames: at most 64 recent host snapshots including action/result metadata.

Raw grids remain unchanged. No physical IDs, hazard rules, resource units,
remaining-action estimates or terminal causes are asserted.
"""
    out = {'status':'ok','stripe_candidates':[],'same_scene_comparisons':[],
           'recent_death':None,'cause':'unknown','candidates_truncated':False,
           'limits':{'history':64,'grid_cells':8192,'dimension':256,'stripes':4,'comparisons':2},
           'coverage':'narrow visual evidence, not complete resource/hazard state'}
    if type(frames) not in (list,tuple) or not frames or any(type(f) is not dict for f in frames[-64:]):
        out.update(status='unavailable',reason='invalid_or_missing_history');return out
    frames = frames[-64:]
    current = frames[-1]
    grid = _te_grid(current)
    if grid is None:
        out.update(status='unavailable',reason='invalid_or_oversized_grid');return out
    candidates = _te_stripes(frames,colors)
    out['candidates_truncated'] = len(candidates)>4
    out['stripe_candidates'] = candidates[:4]
    # Equal visible scene EXCEPT an explicitly identified candidate stripe does
    # not imply equivalent state. Both the differing counts and terminal flags
    # are retained, including comparisons across attempts in the same level.
    for stripe in candidates[:4]:
        axis,index = stripe['axis'],stripe['index']
        rows = grid if axis == 'row' else tuple(zip(*grid))
        key = rows[:index]+rows[index+1:]
        for old in reversed(frames[:-1]):
            if old.get('level') != current.get('level') or old.get('step') == current.get('step'):
                continue
            previous = _te_grid(old)
            if previous is None or (len(previous),len(previous[0])) != (len(grid),len(grid[0])):
                continue
            other = previous if axis == 'row' else tuple(zip(*previous))
            if other[index] == rows[index] or other[:index]+other[index+1:] != key:
                continue
            color = colors.index(stripe['color'])
            result_old = old.get('result') if type(old.get('result')) is dict else {}
            result_now = current.get('result') if type(current.get('result')) is dict else {}
            out['same_scene_comparisons'].append({'axis':axis,'index':index,'color':stripe['color'],
                'steps':[old.get('step'),current.get('step')],
                'counts':[other[index].count(color),rows[index].count(color)],
                'game_over':[result_old.get('game_over') is True,result_now.get('game_over') is True],
                'outside_stripe_equal':True,'state_equivalence':'not established'})
            break
        if len(out['same_scene_comparisons']) >= 2:
            break
    # Include a just-ended attempt even when the harness has appended RESET.
    # Do not leak a previous level's death or repeatedly warn many actions later.
    for i in range(len(frames)-1,max(-1,len(frames)-5),-1):
        end = frames[i]
        result = end.get('result')
        if end.get('level') != current.get('level') or type(result) is not dict or result.get('game_over') is not True:
            continue
        death = {'step':end.get('step'),'cause':'unknown','stripe_candidates':[],
                 'colors_lost_completely':[],'interpretation':'terminal evidence, not a collision/exhaustion verdict'}
        death['stripe_candidates'] = _te_stripes(frames[max(0,i-2):i+1],colors)[:4]
        if i and _te_contiguous(frames[i-1],end):
            before,after = _te_grid(frames[i-1]),_te_grid(end)
            if before and after and (len(before),len(before[0])) == (len(after),len(after[0])):
                ca,cb = [0]*16,[0]*16
                for row in before:
                    for x in row:ca[x]+=1
                for row in after:
                    for x in row:cb[x]+=1
                death['colors_lost_completely'] = [{'color':colors[c],'before_pixels':ca[c],'after_pixels':0} for c in range(16) if ca[c] and not cb[c]][:4]
        out['recent_death'] = death
        break
    return out


def transition_evidence_note(evidence):
    """Only internally generated plain data is formatted; never a user callback."""
    if evidence['status'] != 'ok':
        return ''
    parts = []
    death = evidence['recent_death']
    if death:
        parts.append('recent game_over step='+str(death['step'])+' cause=unknown')
        for stripe in death['stripe_candidates'][:1]:
            parts.append('terminal stripe '+stripe['axis']+'='+str(stripe['index'])+' '+stripe['color']+' counts='+str(stripe['counts'])+' empty='+str(stripe['empty_now']))
        if death['colors_lost_completely']:
            parts.append('colors absent after death='+str([x['color'] for x in death['colors_lost_completely']]))
    for stripe in evidence['stripe_candidates'][:1]:
        parts.append('candidate stripe '+stripe['axis']+'='+str(stripe['index'])+' '+stripe['color']+' counts='+str(stripe['counts']))
    for pair in evidence['same_scene_comparisons'][:1]:
        parts.append('same pixels outside stripe at steps='+str(pair['steps'])+' but counts='+str(pair['counts'])+' game_over='+str(pair['game_over']))
    if not parts:
        return ''
    return ('[transition_evidence] '+'; '.join(parts)+
        '; resource meaning unresolved. Do not equate these states or infer new hazard rules from death alone. Details: current_frame.transition_evidence.')[:850]


def frozen_frame_defaults(callback, frame_type):
    """Detect direct frame defaults on plain functions only; no arbitrary hooks.
    Not a universal stale-closure detector. Do not mutate or invoke the callback.
    """
    if type(callback) is not type(lambda:None):
        return False
    defaults = callback.__defaults__
    kw = callback.__kwdefaults__
    return (type(defaults) is tuple and any(type(x) is frame_type for x in defaults[:64])) or (type(kw) is dict and any(type(x) is frame_type for x in list(kw.values())[:64]))
