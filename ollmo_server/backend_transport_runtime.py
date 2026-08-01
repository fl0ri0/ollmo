"""Provider transport and media adapter owner for Ollmo."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ollmo_services.transports import (
    extract_generate_content,
    extract_generate_seed,
    extract_image_data_url_from_generate_output,
    extract_saved_image_path_from_generate_output,
    extract_text_payload,
    locate_saved_image_file_from_generate_output,
    mlx_audio_speech,
    mlx_chat_completions,
    ollama_chat_with_options,
    ollama_generate,
    ollama_openai_image_generation,
    persist_audio_bytes_locally,
    persist_image_data_url_locally,
    whisper_transcribe,
)


@dataclass
class BackendTransportRuntimeOwner:
    hooks: dict[str, Any]
    capability_chat: str
    request_timeout_error: type[Exception]
    request_connection_error: type[Exception]
    request_exception_error: type[Exception]

    def _hook(self, name: str) -> Any:
        return self.hooks[name]

    def execute_chat_backend_request(
        self,
        *,
        target_port: int,
        model_name: str,
        backend: str,
        capability: str,
        messages: list[dict],
        request_model_override: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout_override_sec: Optional[int] = None,
    ) -> str:
        chat_timeout_seconds = self._hook('chat_timeout_seconds')
        normalize_chat_messages_for_backend = self._hook('normalize_chat_messages_for_backend')
        requests_post = self._hook('requests_post')

        chat_timeout_sec = int(timeout_override_sec or chat_timeout_seconds(model_name, backend, capability))
        prepared_messages = normalize_chat_messages_for_backend(messages, backend=backend)

        if backend in {'mlx', 'llama_cpp'}:
            openai_url = f'http://127.0.0.1:{target_port}/v1/chat/completions'
            openai_payload = {
                'model': request_model_override or model_name or 'default_model',
                'messages': prepared_messages,
                'stream': False,
            }
            if backend == 'mlx':
                openai_payload['enable_thinking'] = False
            if temperature is not None:
                openai_payload['temperature'] = temperature
            if top_p is not None:
                openai_payload['top_p'] = top_p
            if max_tokens is not None:
                openai_payload['max_tokens'] = max_tokens
            logging.info(
                'Forwarding request to %s: %s for model %s',
                backend,
                openai_url,
                openai_payload['model'],
            )
            response = requests_post(openai_url, json=openai_payload, timeout=chat_timeout_sec)
            response.raise_for_status()
            data = response.json()
            choices = data.get('choices') or []
            if not choices:
                raise ValueError(f"{backend} response missing 'choices'.")
            openai_message = choices[0].get('message', {}) if isinstance(choices[0], dict) else {}
            assistant_message = extract_text_payload(openai_message.get('content'))
            if not assistant_message:
                assistant_message = extract_text_payload(openai_message.get('reasoning'))
            return assistant_message

        ollama_url = f'http://localhost:{target_port}/api/chat'
        ollama_payload = {'model': model_name, 'messages': prepared_messages, 'stream': False}
        if temperature is not None or top_p is not None:
            ollama_payload['options'] = {}
            if temperature is not None:
                ollama_payload['options']['temperature'] = temperature
            if top_p is not None:
                ollama_payload['options']['top_p'] = top_p
        if max_tokens is not None:
            ollama_payload.setdefault('options', {})['num_predict'] = max_tokens
        logging.info('Forwarding request to Ollama: %s for model %s', ollama_url, model_name)
        response = requests_post(ollama_url, json=ollama_payload, timeout=chat_timeout_sec)
        response.raise_for_status()
        ollama_response_data = response.json()
        return ollama_response_data.get('message', {}).get('content', '')

    def extract_stream_delta_payload(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts: list[str] = []
            for item in value:
                chunk = self.extract_stream_delta_payload(item)
                if chunk:
                    parts.append(chunk)
            return ''.join(parts)
        if isinstance(value, dict):
            for key in ('text', 'content', 'response', 'output'):
                if key in value:
                    chunk = self.extract_stream_delta_payload(value.get(key))
                    if chunk:
                        return chunk
        return ''

    def normalize_stream_line(self, raw_line: Any) -> str:
        if isinstance(raw_line, bytes):
            return raw_line.decode('utf-8', errors='replace').strip()
        return str(raw_line or '').strip()

    def open_ollama_chat_stream(
        self,
        *,
        target_port: int,
        model_name: str,
        messages: list[dict],
        timeout_sec: int,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> tuple[Any, int]:
        normalize_chat_messages_for_backend = self._hook('normalize_chat_messages_for_backend')
        requests_post = self._hook('requests_post')

        payload: dict[str, Any] = {
            'model': model_name,
            'messages': normalize_chat_messages_for_backend(messages),
            'stream': True,
        }
        model_name_lower = str(model_name or '').lower()
        if 'deepseek-ocr' in model_name_lower:
            payload['options'] = {'num_ctx': 4096, 'num_keep': 0}
        if temperature is not None:
            payload.setdefault('options', {})['temperature'] = temperature
        if top_p is not None:
            payload.setdefault('options', {})['top_p'] = top_p
        if max_tokens is not None:
            payload.setdefault('options', {})['num_predict'] = max_tokens

        candidate_ports = [int(target_port)]
        if int(target_port) != 11434:
            candidate_ports.append(11434)

        last_exc: Optional[Exception] = None
        for port in candidate_ports:
            try:
                response = requests_post(
                    f'http://127.0.0.1:{port}/api/chat',
                    json=payload,
                    timeout=max(30, int(timeout_sec)),
                    stream=True,
                )
                response.raise_for_status()
                return response, port
            except self.request_timeout_error as exc:
                last_exc = exc
                continue
            except self.request_connection_error as exc:
                last_exc = exc
                continue
            except self.request_exception_error as exc:
                last_exc = exc
                raise exc
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                break

        if last_exc:
            raise last_exc
        raise RuntimeError('Ollama stream failed without a captured exception.')

    def iter_ollama_stream_deltas(self, response) -> Any:
        try:
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                line = self.normalize_stream_line(raw_line)
                if not line:
                    continue
                if line.startswith('event:'):
                    continue
                if line.startswith('data:'):
                    line = line[5:].strip()
                if not line or line == '[DONE]':
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    logging.warning('Skipping non-JSON Ollama stream chunk: %s', line[:200])
                    continue
                if not isinstance(data, dict):
                    continue
                message = data.get('message') if isinstance(data.get('message'), dict) else None
                delta = self.extract_stream_delta_payload(message.get('content') if message else None)
                if not delta:
                    delta = self.extract_stream_delta_payload(data.get('response'))
                if delta:
                    yield delta
                if data.get('done'):
                    break
        finally:
            try:
                response.close()
            except Exception:  # noqa: BLE001
                pass

    def open_openai_chat_stream(
        self,
        *,
        backend: str,
        target_port: int,
        request_model_override: Optional[str],
        model_name: str,
        messages: list[dict],
        timeout_sec: int,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        normalize_chat_messages_for_backend = self._hook('normalize_chat_messages_for_backend')
        requests_post = self._hook('requests_post')

        payload = {
            'model': request_model_override or model_name or 'default_model',
            'messages': normalize_chat_messages_for_backend(messages, backend=backend),
            'stream': True,
        }
        if backend == 'mlx':
            payload['enable_thinking'] = False
        if temperature is not None:
            payload['temperature'] = temperature
        if top_p is not None:
            payload['top_p'] = top_p
        if max_tokens is not None:
            payload['max_tokens'] = max_tokens
        response = requests_post(
            f'http://127.0.0.1:{target_port}/v1/chat/completions',
            json=payload,
            timeout=max(30, int(timeout_sec)),
            stream=True,
        )
        response.raise_for_status()
        return response

    def request_exception_details(self, exc: Exception) -> str:
        details = str(exc)
        response = getattr(exc, 'response', None)
        if response is None:
            return details
        try:
            payload = response.json()
            if isinstance(payload, dict):
                details = payload.get('error') or payload.get('message') or details
        except Exception:  # noqa: BLE001
            details = getattr(response, 'text', details)[:300]
        return details

    def iter_openai_stream_deltas(self, response) -> Any:
        try:
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                line = self.normalize_stream_line(raw_line)
                if not line:
                    continue
                if line.startswith('event:'):
                    continue
                if line.startswith('data:'):
                    line = line[5:].strip()
                if not line or line == '[DONE]':
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    logging.warning('Skipping non-JSON MLX stream chunk: %s', line[:200])
                    continue
                if not isinstance(data, dict):
                    continue
                choices = data.get('choices') or []
                if not isinstance(choices, list) or not choices:
                    continue
                first_choice = choices[0] if isinstance(choices[0], dict) else {}
                delta_payload = first_choice.get('delta') if isinstance(first_choice.get('delta'), dict) else {}
                message_payload = first_choice.get('message') if isinstance(first_choice.get('message'), dict) else {}
                delta = self.extract_stream_delta_payload(delta_payload.get('content'))
                if not delta:
                    delta = self.extract_stream_delta_payload(message_payload.get('content'))
                if delta:
                    yield delta
        finally:
            try:
                response.close()
            except Exception:  # noqa: BLE001
                pass

    def iter_mlx_stream_deltas(self, response) -> Any:
        yield from self.iter_openai_stream_deltas(response)

    def ollama_generate(
        self,
        port: int,
        model_name: str,
        prompt: str,
        images: Optional[list[str]] = None,
        timeout_sec: int = 900,
        options: Optional[dict[str, Any]] = None,
        *,
        max_retries: int = 3,
        allow_port_fallback: bool = True,
    ) -> dict:
        requests_module = self._hook('requests_module')
        return ollama_generate(
            port,
            model_name,
            prompt,
            requests_module,
            self.request_timeout_error,
            self.request_connection_error,
            self.request_exception_error,
            images=images,
            timeout_sec=timeout_sec,
            options=options,
            max_retries=max_retries,
            allow_port_fallback=allow_port_fallback,
        )

    def extract_generate_content(self, data: dict) -> str:
        return extract_generate_content(data)

    def locate_saved_image_file_from_generate_output(self, data: dict) -> Optional[Path]:
        to_base64_func = self._hook('to_base64')
        return locate_saved_image_file_from_generate_output(data, to_base64_func=to_base64_func)

    def extract_image_data_url_from_generate_output(self, data: dict) -> Optional[str]:
        to_base64_func = self._hook('to_base64')
        return extract_image_data_url_from_generate_output(data, to_base64_func=to_base64_func)

    def extract_saved_image_path_from_generate_output(self, data: dict) -> Optional[str]:
        to_base64_func = self._hook('to_base64')
        return extract_saved_image_path_from_generate_output(data, to_base64_func=to_base64_func)

    def extract_generate_seed(self, data: dict) -> Optional[int]:
        return extract_generate_seed(data)

    def persist_image_data_url_locally(self, image_data_url: Optional[str], model_name: str) -> Optional[str]:
        return persist_image_data_url_locally(image_data_url, model_name)

    def ollama_openai_image_generation(
        self,
        port: int,
        model_name: str,
        prompt: str,
        *,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> Optional[str]:
        requests_module = self._hook('requests_module')
        return ollama_openai_image_generation(
            port,
            model_name,
            prompt,
            requests_module,
            width=width,
            height=height,
        )

    def persist_audio_bytes_locally(
        self,
        audio_bytes: Optional[bytes],
        model_name: str,
        *,
        response_format: Optional[str] = None,
        content_type: Optional[str] = None,
    ) -> Optional[str]:
        generated_audio_dir = self._hook('generated_audio_dir')
        return persist_audio_bytes_locally(
            audio_bytes,
            model_name=model_name,
            output_dir=generated_audio_dir,
            response_format=response_format,
            content_type=content_type,
        )

    def ollama_chat(self, port: int, model_name: str, messages: list[dict]) -> dict:
        return self.ollama_chat_with_options(
            port=port,
            model_name=model_name,
            messages=messages,
            timeout_sec=180,
            allow_port_fallback=False,
        )

    def extract_text_payload(self, value: Any) -> str:
        return extract_text_payload(value)

    def ollama_chat_with_options(
        self,
        *,
        port: int,
        model_name: str,
        messages: list[dict],
        timeout_sec: int,
        allow_port_fallback: bool,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> dict:
        requests_module = self._hook('requests_module')
        return ollama_chat_with_options(
            port=port,
            model_name=model_name,
            messages=messages,
            requests_module=requests_module,
            request_timeout_error=self.request_timeout_error,
            request_connection_error=self.request_connection_error,
            request_exception_error=self.request_exception_error,
            timeout_sec=timeout_sec,
            allow_port_fallback=allow_port_fallback,
            temperature=temperature,
            top_p=top_p,
        )

    def whisper_transcribe(
        self,
        port: int,
        audio_path: Path,
        task: str = 'transcribe',
        language: Optional[str] = None,
    ) -> dict:
        requests_module = self._hook('requests_module')
        return whisper_transcribe(port, audio_path, requests_module, task=task, language=language)

    def mlx_audio_speech(
        self,
        port: int,
        model_name: str,
        prompt: str,
        *,
        instruct: Optional[str] = None,
        voice: Optional[str] = None,
        response_format: Optional[str] = None,
        speed: float = 1.0,
        pitch: float = 1.0,
        lang_code: Optional[str] = None,
        timeout_sec: int = 600,
    ) -> dict:
        requests_module = self._hook('requests_module')
        return mlx_audio_speech(
            port,
            model_name,
            prompt,
            requests_module,
            instruct=instruct,
            voice=voice,
            response_format=response_format,
            speed=speed,
            pitch=pitch,
            lang_code=lang_code,
            timeout_sec=timeout_sec,
        )

    def mlx_chat_completions(
        self,
        port: int,
        model_name: str,
        messages: list[dict],
        timeout_sec: int = 600,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> dict:
        requests_module = self._hook('requests_module')
        return mlx_chat_completions(
            port,
            model_name,
            messages,
            requests_module,
            timeout_sec=timeout_sec,
            temperature=temperature,
            top_p=top_p,
        )

    def openai_chat_completions(
        self,
        backend: str,
        port: int,
        model_name: str,
        messages: list[dict],
        timeout_sec: int = 600,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> dict:
        if backend == 'mlx':
            return self.mlx_chat_completions(
                port,
                model_name,
                messages,
                timeout_sec=timeout_sec,
                temperature=temperature,
                top_p=top_p,
            )
        content = self.execute_chat_backend_request(
            target_port=port,
            model_name=model_name,
            backend=backend,
            capability=self.capability_chat,
            messages=messages,
            request_model_override=model_name,
            temperature=temperature,
            top_p=top_p,
            timeout_override_sec=timeout_sec,
        )
        return {'content': content, 'result': {'choices': [{'message': {'content': content}}]}}
