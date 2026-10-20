"""Bounded local OpenAI-compatible inference; generated code stays in the worker."""
from __future__ import annotations

import http.client
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

PROTOCOL = '''Return exactly one JSON object, without markdown.
Use only the observed state, objects, actions and replay evidence in this conversation.
Return {"actions": [{"action": "ACTION1", "expect": [{"x": 0, "y": 0, "value": 0}],
"expect_level_up": false}]} for a concrete plan; coordinates x/y are required for a
complex action. Every expect must predict non-HUD cells; never invent observations.
Alternatively return {"hypotheses": [{"source": "def step(state, action): ...",
"probe": {"action": "ACTION1", "expect": [{"x": 0, "y": 0, "value": 0}],
"expect_level_up": false}, "alternative_expect": [{"x": 0, "y": 0, "value": 1}],
"scope": "level"}]}. A simulator request returns {"simulator_source": "def step(state, action): ..."}.
Synthesized step receives a JSON state with grid, level and state, and an action
dictionary. It must return that complete state; unknown mechanics stay unknown.
The example coordinates/colors above illustrate the schema only, not a game rule.
Your proposals undergo full historical replay and forward-prediction checks.
'''


class LocalBackend:
    def __init__(self, directory: Path, limits: Callable[[], tuple[int, float]],
                 context: Callable[[], str], *, model: str = 'flashnext') -> None:
        self.directory, self.limits, self.context, self.model = directory, limits, context, model

    def __call__(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        tokens, timeout = self.limits()
        if tokens <= 0 or timeout <= 0:
            raise TimeoutError('reasoning budget exhausted')
        # Send the immutable complete history verbatim. No timestamps, injected
        # early messages, pruning, or mutation of prior assistant turns here.
        payload = {'model': self.model, 'messages': messages, 'max_tokens': tokens,
                   'temperature': 0.7, 'top_p': 0.95,
                   'response_format': {'type': 'json_object'}}
        encoded = json.dumps(payload, separators=(',', ':')).encode()
        began = time.monotonic()
        connection = http.client.HTTPConnection('127.0.0.1', 8001, timeout=min(30.0, timeout))
        record: dict[str, Any] = {'context': self.context(), 'model': self.model,
                                  'messages': len(messages), 'cache_hit_rate': None}
        try:
            connection.request('POST', '/v1/chat/completions', body=encoded,
                               headers={'Content-Type': 'application/json'})
            result = connection.getresponse()
            raw = result.read(4_194_305)
            if result.status != 200 or len(raw) > 4_194_304:
                raise RuntimeError(f'local inference HTTP {result.status} or oversized response')
            data = json.loads(raw)
            content = data['choices'][0]['message']['content']
            response = json.loads(content)
            if not isinstance(response, dict) or not any(response.get(k) for k in (
                    'actions', 'hypotheses', 'simulator_source')):
                raise ValueError('local backend returned no structured proposal')
            usage = data.get('usage', {})
            prompt, completion = usage.get('prompt_tokens'), usage.get('completion_tokens')
            details = usage.get('prompt_tokens_details') or {}
            cached = details.get('cached_tokens')
            record.update(prompt_tokens=prompt, completion_tokens=completion, cached_tokens=cached,
                          status='response')
            if type(prompt) is int and prompt > 0 and type(cached) is int and 0 <= cached <= prompt:
                record['cache_hit_rate'] = cached / prompt
            return response
        except Exception as error:
            record.update(status='error', error_type=type(error).__name__, error=str(error))
            raise
        finally:
            connection.close()
            record['elapsed_seconds'] = time.monotonic() - began
            completion = record.get('completion_tokens')
            if type(completion) is int and record['elapsed_seconds'] > 0:
                record['completion_tokens_per_second'] = completion / record['elapsed_seconds']
            self.directory.mkdir(parents=True, exist_ok=True)
            with (self.directory / 'llm_requests.jsonl').open('a') as stream:
                stream.write(json.dumps(record, sort_keys=True) + '\n')
