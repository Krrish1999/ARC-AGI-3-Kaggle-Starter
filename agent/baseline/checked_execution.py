"""Generic checked execution. Environment access is supplied by the sandbox adapter.

No game identifiers, decoders, transition rules, retries, persistence or search.
The existing outer Python-tool deadline is the hard timeout.
"""
from copy import deepcopy as _exec_copy


def checked_execute(actions, observe, predict, send_one, *, expected_start,
                    max_steps=32):
    """Run a finite concrete plan, checking every nonterminal transition.

    observe() returns a deepcopy-able model state with ordinary equality.
    predict(state, action) returns its predicted successor. Neither may act.
    send_one(action) sends exactly one action via the existing guarded interface.
    The adapter owns callback action blocking and syntax validation of actions.

    A completed plan is NOT a won game. Reports retain confirmed execution even
    on ordinary callback errors. BaseException guards and hard timeouts propagate
    unchanged; the host's action records remain authoritative in those cases.
    """
    if type(actions) not in (list, tuple) or len(actions) > 128:
        raise ValueError('actions must be a finite list/tuple of at most 128 single actions')
    if type(max_steps) is not int or not 1 <= max_steps <= 128:
        raise ValueError('max_steps must be an integer in [1, 128]')
    if not all(callable(f) for f in (observe, predict, send_one)):
        raise TypeError('observe, predict and send_one must be callable')
    report = {'status': None, 'phase': 'prepare', 'attempted': 0,
              'confirmed_executed': 0, 'executed_actions': [],
              'execution_uncertain': False, 'last_action_result': None,
              'expected': None, 'observed': None, 'error': None}

    def finish(status):
        report['status'] = status
        return report

    def failure(status, exc):
        try:
            message = str(exc)[:600]
        except Exception:  # noqa: BLE001 - archived callback/retention failure boundary
            message = '<exception message unavailable>'
        report['error'] = type(exc).__name__ + ': ' + message
        return finish(status)

    try:
        plan = _exec_copy(list(actions))
        expected = _exec_copy(expected_start)
        report['expected'] = _exec_copy(expected)
        if not plan:
            return finish('plan_complete')
        report['phase'] = 'observe_start'
        state = _exec_copy(observe())
        report['observed'] = _exec_copy(state)
        if state != expected:
            return finish('start_mismatch')
    except Exception as exc:  # noqa: BLE001 - archived callback/retention failure boundary
        return failure('callback_error', exc)

    for index, item in enumerate(plan):
        if index >= max_steps:
            report['phase'] = 'step_limit'
            return finish('limit')
        try:
            report['phase'] = 'predict'
            predicted = _exec_copy(predict(_exec_copy(state), _exec_copy(item)))
            report['expected'] = _exec_copy(predicted)
            # Save the exact request before dispatch; copies cannot trigger a
            # post-action bookkeeping failure before execution is recorded.
            sent = _exec_copy(item)
        except Exception as exc:  # noqa: BLE001 - archived callback/retention failure boundary
            return failure('callback_error', exc)
        report['phase'] = 'action'
        report['attempted'] += 1
        report['observed'] = None
        report['last_action_result'] = None
        try:
            reply = send_one(sent)
        except Exception as exc:  # noqa: BLE001 - archived callback/retention failure boundary
            # No retry: transport/refresh errors can happen after a real action.
            report['execution_uncertain'] = True
            return failure('action_error', exc)
        if not isinstance(reply, dict):
            report['execution_uncertain'] = True
            return failure('action_error', TypeError('action result must be a dict'))
        report['last_action_result'] = reply
        count = reply.get('executed')
        confirmed = count is True or (type(count) is int and count == 1)
        if confirmed:
            report['confirmed_executed'] += 1
            report['executed_actions'].append(item)
        elif count is not False and not (type(count) is int and count == 0):
            report['execution_uncertain'] = True
        # Terminal flags MUST precede old-level decoding and comparison.
        report['phase'] = 'action_result'
        if reply.get('run_complete') or reply.get('done'):
            return finish('run_complete')
        if reply.get('game_over'):
            return finish('game_over')
        if reply.get('level_completed'):
            return finish('level_completed')
        if report['execution_uncertain']:
            return finish('action_error')
        if not confirmed:
            return finish('not_executed')
        if reply.get('no_op') or reply.get('gameplay_changed') is False:
            return finish('no_op')
        if reply.get('stop_reason'):
            return finish('stopped')
        try:
            report['phase'] = 'observe_after'
            observed = _exec_copy(observe())
            report['observed'] = _exec_copy(observed)
            if observed != predicted:
                return finish('mismatch')
            state = observed
        except Exception as exc:  # noqa: BLE001 - archived callback/retention failure boundary
            return failure('callback_error', exc)
    report['phase'] = 'complete'
    return finish('plan_complete')
