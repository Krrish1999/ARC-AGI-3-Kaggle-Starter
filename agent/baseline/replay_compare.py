"""Bounded, optional named-state replay and advisory progress text; stdlib only.
No game rules, actions, identity inference, global budgets or authorization.
"""
import json as _rc_json
import math as _rc_math
import time as _rc_time


def _rc_snapshot(value, root=True, depth=0, budget=None):
    if budget is None:
        budget = [4096]
    budget[0] -= 1
    if budget[0] < 0 or depth > 8:
        raise ValueError('named state exceeds 4096 values or depth 8')
    t = type(value)
    if root and (t is not dict or not value):
        raise TypeError('observe/predict must return a nonempty dict of named fields, not a positional tuple')
    if t is dict:
        if len(value) > 64 or any(type(k) is not str or len(k) > 80 for k in value):
            raise ValueError('state dict needs at most 64 string keys, each at most 80 characters')
        return {k: _rc_snapshot(v, False, depth+1, budget) for k, v in value.items()}
    if t in (list, tuple):
        if len(value) > 1024:
            raise ValueError('state sequence exceeds 1024 items')
        items = [_rc_snapshot(v, False, depth+1, budget) for v in value]
        return tuple(items) if t is tuple else items
    if t is str:
        if len(value) > 256:
            raise ValueError('state strings must be at most 256 characters')
        return value
    if t is int:
        if value.bit_length() > 256:
            raise ValueError('state integer exceeds 256 bits')
        return value
    if t is float:
        if not _rc_math.isfinite(value):
            raise ValueError('state numbers must be finite')
        return value
    if value is None or t is bool:
        return value
    raise TypeError('state values must be plain dict/list/tuple and finite scalar data')


def _rc_preview(value):
    text = _rc_json.dumps(value, ensure_ascii=True, separators=(',', ':'))
    return {'type': type(value).__name__, 'value_preview': text[:160], 'truncated': len(text) > 160}


def _rc_differences(predicted, observed):
    out = []; count = 0
    def add(path, kind, p=None, o=None, missing=None):
        nonlocal count
        count += 1
        if len(out) < 8:
            d = {'path': list(path), 'kind': kind}
            if missing != 'predicted': d['predicted'] = _rc_preview(p)
            if missing != 'observed': d['observed'] = _rc_preview(o)
            out.append(d)
    def walk(p, o, path):
        if type(p) is not type(o):
            add(path, 'type', p, o)
        elif type(p) is dict:
            for key in sorted(p.keys() | o.keys()):
                if key not in p: add(path+(key,), 'missing_predicted_field', o=o[key], missing='predicted')
                elif key not in o: add(path+(key,), 'extra_predicted_field', p=p[key], missing='observed')
                else: walk(p[key], o[key], path+(key,))
        elif type(p) in (list, tuple):
            if len(p) != len(o): add(path, 'length', len(p), len(o))
            for i in range(min(len(p),len(o))): walk(p[i], o[i], path+(i,))
        elif p != o:
            add(path, 'value', p, o)
    walk(predicted, observed, ())
    return out, count


def _rc_skip(record):
    before = getattr(record, 'before_frame', None)
    after = getattr(record, 'after_frame', None)
    action_name = getattr(record, 'action', None)
    result = getattr(record, 'result', None)
    if type(action_name) is not str or not action_name.strip() or len(action_name) > 256:
        return 'invalid_action_label'
    if action_name.strip().upper() == 'RESET': return 'reset'
    if type(result) is not dict: return 'missing_result'
    if result.get('execution_uncertain'): return 'execution_uncertain'
    if any(result.get(k) for k in ('game_over','level_completed','run_complete','done')): return 'terminal_boundary'
    if 'executed' in result and not result['executed']: return 'not_executed'
    if result.get('stop_reason'): return 'stopped'
    if before is None or after is None: return 'missing_frame'
    if getattr(before,'level',None) != getattr(after,'level',None): return 'level_boundary'
    bs = getattr(before,'step',None); es = getattr(after,'step',None)
    if type(bs) is not int or type(es) is not int or es != bs+1: return 'nonconsecutive_steps'
    return None


def replay_named_transitions(records, observe, predict, *, max_cases=16, timeout_s=2.0):
    """Each prediction starts at an independently decoded observed before-state.
    Only endpoint fields supplied by the caller are compared. A match is not
    safety, completeness, physical update-order proof or a plan execution ticket.
    """
    if type(records) not in (list,tuple): raise TypeError('records must be a finite list/tuple of TransitionView records')
    if not callable(observe) or not callable(predict): raise TypeError('observe and predict must be callable')
    if type(max_cases) is not int or max_cases < 1: raise ValueError('max_cases must be a positive integer')
    if type(timeout_s) not in (int,float) or not _rc_math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError('timeout_s must be finite and positive')
    effective_cases = min(max_cases,64); effective_time = min(timeout_s,5.0)
    start = _rc_time.monotonic(); expired = lambda: _rc_time.monotonic()-start >= effective_time
    report = {'status':'empty','scope':'selected observed nonterminal transitions and supplied named fields only; not safety or model certification',
        'input_count':len(records),'scanned_count':0,'input_truncated':len(records)>128,
        'eligible_count':0,'selected_count':0,'selection_truncated':False,
        'counts':{'matched':0,'mismatched':0,'errors':0,'skipped':0},'skip_reasons':{},'cases':[],
        'effective_limits':{'max_cases':effective_cases,'timeout_s':effective_time,'scan_records':128,'diffs_per_case':8},
        'clamped_limits':[k for k,v in (('max_cases',max_cases>64),('timeout_s',timeout_s>5.0)) if v]}
    eligible=[]
    for i in range(max(0,len(records)-128),len(records)):
        if expired(): report['status']='limit';report['reason']='time';break
        record=records[i];report['scanned_count']+=1;why=_rc_skip(record)
        if why:
            report['counts']['skipped']+=1
            report['skip_reasons'][why]=report['skip_reasons'].get(why,0)+1
        else: eligible.append((i,record))
    report['eligible_count']=len(eligible)
    chosen=eligible[-effective_cases:];report['selected_count']=len(chosen)
    report['selection_truncated']=len(chosen)<len(eligible)
    if report['status']!='limit':
        for index,record in chosen:
            if expired(): report['status']='limit';report['reason']='time';break
            entry={'index':index,'before_step':record.before_frame.step,'after_step':record.after_frame.step,'level':record.before_frame.level,'action':record.action}
            phase='observe_before'
            try:
                before=_rc_snapshot(observe(record.before_frame))
                if expired(): report['status']='limit';report['reason']='time';break
                phase='observe_after';observed=_rc_snapshot(observe(record.after_frame))
                if expired(): report['status']='limit';report['reason']='time';break
                phase='predict';predicted=_rc_snapshot(predict(_rc_snapshot(before),record.action))
                if expired(): report['status']='limit';report['reason']='time';break
                diffs,total=_rc_differences(predicted,observed)
                entry.update(status='mismatch' if total else 'match',fields=sorted(observed) if isinstance(observed, dict) else [],differences=diffs,difference_count=total,differences_truncated=total>len(diffs))
                report['counts']['mismatched' if total else 'matched']+=1
            except Exception as exc:  # noqa: BLE001 - archived callback/retention failure boundary
                entry.update(status='callback_error',phase=phase,error_type=type(exc).__name__)
                # No repr/str on callback-owned exceptions or arbitrary objects.
                report['counts']['errors']+=1
                report['cases'].append(entry);report['status']='callback_error';break
            report['cases'].append(entry)
        if report['status'] not in ('limit','callback_error'):
            report['status']='mismatch' if report['counts']['mismatched'] else ('matches_observed' if report['counts']['matched'] else 'empty')
    report['elapsed_s']=round(_rc_time.monotonic()-start,6)
    return report


def replay_summary(report):
    c=report['counts'];text='[replay_transitions] status={} matched={:d} mismatched={:d} errors={:d} skipped={:d} selected={:d}; supplied fields only, not safety.'.format(report['status'],c['matched'],c['mismatched'],c['errors'],c['skipped'],report['selected_count'])
    first=next((x for x in report['cases'] if x['status']!='match'),None)
    if first:
        text+=' First issue: step {} {}'.format(first['after_step'],first['status'])
        if first.get('differences'): text+=' path='+_rc_json.dumps(first['differences'][0]['path'],ensure_ascii=True)[:120]
        if first.get('phase'):text+=' phase='+first['phase']+' type='+first['error_type']
    if report['input_truncated'] or report['selection_truncated']:text+=' Selected tail only; earlier evidence not checked.'
    return text[:480]


def planning_progress_note(previous, key, now, timing, executed=None):
    """Readonly session timing snapshot; no synthesized/extended game deadline.
    executed is None for opener, False for a call without confirmed action,
    True for confirmed actions. No-action count is not lack-of-learning proof.
    """
    if type(previous) is not dict or previous.get('key')!=key or executed is True:
        previous={'key':key,'since':now,'calls':0}
    current=dict(previous)
    if executed is False: current['calls']=min(999999,current.get('calls',0)+1)
    elapsed=max(0.0,now-current['since'])
    remaining=timing.get('time_remaining_seconds') if type(timing) is dict else None
    if not isinstance(remaining, (int, float)) or isinstance(remaining, bool) or not _rc_math.isfinite(remaining) or remaining<0:
        remaining=None
    clock='unknown' if remaining is None else f'{remaining:.1f}s'
    text='[planning_progress] game_remaining_at_note={}; no_action_calls={:d}; interval={:.1f}s. Clock snapshot ages during generation.'.format(clock,current['calls'],elapsed)
    if current['calls']>=3 or (remaining is not None and remaining<=120):
        text+=' Prefer a short evidence-checked execution or a probe that distinguishes predictions over another rewrite. A printed plan executes nothing; raw actions remain available. Never blindly execute an unvalidated route.'
    return current,text[:480]
