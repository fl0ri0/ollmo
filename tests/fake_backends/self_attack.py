"""Self-attack adapter over the existing isolated fake-provider E2E harness."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import hashlib
import struct
from unittest.mock import patch

from tests.fake_backends import FakeBackendHarness
from ollmo_g.semantic_role_profile import build_semantic_role_profile
from ollmo_g.request_meta import extract_request_meta
from scripts.run_graph_rebase_shadow_corpus import HttpResult
from scripts.run_graph_rebase_shadow_corpus import atomic_write_json, utc_now


class SelfAttackBackend(FakeBackendHarness):
    """Fake model decisions, real response/graph/closure/frame and Late Fill owners.

    No remote ports, model lifecycle, or policy learning. Late Fill is drained
    synchronously through its normal owner so profile isolation is well defined.
    """

    def __enter__(self):
        super().__enter__()
        import ollmo_webserver as server
        self._stack.enter_context(patch.object(server, '_resolve_late_fill_route', self.resolve_late_fill))
        self._stack.enter_context(patch.object(server, '_schedule_response_late_fill', self.schedule))
        self._stack.enter_context(patch.object(server._LATE_FILL_RUNTIME, 'schedule_response_late_fill', self.schedule))
        self.scheduled = []
        self.raw_responses = []
        return self

    def checkpoint(self, stage, **details):
        atomic_write_json(self.root.parent / 'execution-progress.json',
                          dict(stage=stage, observed_at=utc_now(), **details))
        atomic_write_json(self.root.parent / 'backend_calls.json',
                          {'calls': dict(self.calls), 'records': self.call_records})

    def _execute_chat_backend_request(self, **kwargs):
        result = super()._execute_chat_backend_request(**kwargs)
        if getattr(self, 'source_output', None):
            result = self.source_output
        self.call_records[-1]['result'] = result
        self.checkpoint('chat_provider_returned')
        return result

    def _invoke_internal_api_json_route(self, *args, **kwargs):
        payload = kwargs.get('payload') or {}
        input_path = Path(str(payload.get('file_path') or ''))
        input_sha = hashlib.sha256(input_path.read_bytes()).hexdigest() if input_path.is_file() else None
        result = super()._invoke_internal_api_json_route(*args, **kwargs)
        body, status = result
        if status == 200 and body.get('mode') == 'text_to_speech':
            # A deterministic fake codec: STT reads the source from the exact
            # saved WAV bytes, never from the requested/expected transcript.
            text = str(payload.get('content_payload') or payload.get('prompt') or '')
            source = text.encode('utf-8')
            original = Path(body['saved_audio_path']).read_bytes()
            chunk = b'oltx' + struct.pack('<I', len(source)) + source + (b'\0' if len(source) % 2 else b'')
            wav = original[:4] + struct.pack('<I', len(original) + len(chunk) - 8) + original[8:] + chunk
            path = self.audio_dir / f'{hashlib.sha256(wav).hexdigest()}.wav'
            path.write_bytes(wav)
            body['saved_audio_path'] = str(path)
        elif status == 200 and body.get('mode') == 'speech_to_text':
            path = Path(str(payload.get('file_path') or ''))
            transcript = None
            if path.is_file():
                wav = path.read_bytes()
                offset = 12
                while offset + 8 <= len(wav):
                    name, size = wav[offset:offset + 4], struct.unpack('<I', wav[offset + 4:offset + 8])[0]
                    if name == b'oltx':
                        transcript = wav[offset + 8:offset + 8 + size].decode('utf-8')
                        break
                    offset += 8 + size + size % 2
            if transcript is None:
                result = ({'error': 'Fake STT has no decodable source in this exact input artifact.'}, 422)
            else:
                body['content'] = transcript
                body['result']['transcript'] = transcript
        self.call_records[-1]['input_sha256'] = input_sha
        output_path = Path(str(body.get('saved_audio_path') or ''))
        if output_path.is_file():
            self.call_records[-1]['output_sha256'] = hashlib.sha256(output_path.read_bytes()).hexdigest()
        self.checkpoint('capability_provider_returned')
        return result

    def _resolve_ghost_auto_route(self, data, *args, **kwargs):
        route, error = super()._resolve_ghost_auto_route(data, *args, **kwargs)
        route['route_runtime']['semantic_role_profile'] = build_semantic_role_profile(
            {'prompt': data.get('prompt', ''), 'runtime': route['route_runtime']},
            request_meta=extract_request_meta(data))
        return route, error

    def resolve_late_fill(self, request_payload, *, expected_capability, **kwargs):
        instance = self.instances[expected_capability]
        effective = dict(request_payload, capability=expected_capability)
        return effective, self._route_info(expected_capability, instance), None

    def schedule(self, **kwargs):
        kwargs.pop('complete_response_late_fill', None)
        self.scheduled.append(deepcopy(kwargs))
        return True

    def post(self, path, payload, *, timeout):
        import ollmo_webserver as server
        if path != '/api/responses':
            raise ValueError('Self-attack may only execute canonical Responses.')
        self.source_output = payload.get('content_payload')
        response, status = self.post_response(dict(payload))
        self.raw_responses.append(deepcopy(response))
        self.checkpoint('initial_response_returned', response_id=payload.get('response_id'),
                        lifecycle_state=response.get('lifecycle_state'))
        # Keep the runtime's own repair budgets. The enclosing worker process
        # has the explicit profile wall-time budget; no hidden continuation cap.
        while self.scheduled:
            work = self.scheduled.pop(0)
            self.checkpoint('late_fill_owner_running', response_id=payload.get('response_id'))
            server._complete_response_late_fill(**work)
            self.checkpoint('late_fill_owner_returned', response_id=payload.get('response_id'))
        return HttpResult(status, response)

    def get(self, path, *, timeout):
        if path == '/api/graph_rebase/readiness':
            # Do not read the production trusted readiness registry in a fake run.
            return HttpResult(200, {'kind': 'isolated_readiness_not_evaluated', 'runtime_effect': 'none'})
        response = self.client.get(path)
        return HttpResult(response.status_code, response.get_json(), len(response.data))
