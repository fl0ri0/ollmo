import json
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _run_node(script: str) -> dict:
    result = subprocess.run(
        ['node', '-e', textwrap.dedent(script)],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(result.stdout)


@pytest.mark.skipif(shutil.which('node') is None, reason='node is required for frontend VM tests')
def test_explicit_external_target_submits_all_selected_inputs_in_one_turn():
    result = _run_node(
        r"""
        const fs = require('fs');
        const vm = require('vm');
        const context = {
          console,
          state: { arena: { enabled: false } },
          elements: { userInput: { value: 'Inspect these files together.' } },
          setTimeout: (callback) => { callback(); return 1; },
          clearTimeout() {},
          setInterval,
          clearInterval,
          TextDecoder,
          Uint8Array,
        };
        context.globalThis = context;
        vm.createContext(context);
        vm.runInContext(fs.readFileSync('static/ui/request-lifecycle.js', 'utf8'), context);
        vm.runInContext(`
          const imageFile = { name: 'diagram.png' };
          const notesFile = { name: 'notes.txt' };
          let pendingItems = [
            { source: 'upload', file: imageFile },
            { source: 'upload', file: notesFile },
            { source: 'path', localPath: '/work/context.pdf' },
            { source: 'path', localPath: '/work/data.csv' },
          ];
          let activeTarget = {
            instance_id: 'external:codex',
            target_kind: 'external',
            capability: 'chat',
          };
          let clearCount = 0;
          let consentCount = 0;
          const calls = [];

          getPendingInputItems = () => pendingItems;
          getActiveConversationId = () => 'conversation-test';
          isResponsesWorkbenchActive = () => false;
          isResponsesWorkbenchAutoTarget = () => false;
          getActivePromptTargetInstance = () => activeTarget;
          getSelectedReferenceArtifacts = () => [];
          ensureCodexFileConsent = async () => { consentCount += 1; return true; };
          canSendPrompt = () => true;
          basenameFromPath = (value) => String(value || '').split('/').pop() || '';
          clearPendingAttachment = () => { clearCount += 1; };
          updateGlobalModelStatus = () => {};
          sendSingleMessage = async (...args) => { calls.push(args); };

          globalThis.runTest = async () => {
            await sendMessage();
            const externalCalls = calls.splice(0);

            activeTarget = {
              instance_id: 'local-chat',
              target_kind: 'local',
              capability: 'chat',
            };
            pendingItems = [
              { source: 'upload', file: imageFile },
              { source: 'path', localPath: '/work/context.pdf' },
            ];
            await sendMessage();

            return {
              externalCalls,
              localCalls: calls,
              clearCount,
              consentCount,
            };
          };
        `, context);
        context.runTest().then((value) => {
          process.stdout.write(JSON.stringify(value));
        }).catch((error) => {
          console.error(error);
          process.exit(1);
        });
        """
    )

    assert result['consentCount'] == 1
    assert result['clearCount'] == 2

    external_calls = result['externalCalls']
    assert len(external_calls) == 1
    external_call = external_calls[0]
    assert external_call[0] == 'external:codex'
    assert external_call[2]['name'] == 'diagram.png'
    assert external_call[4] == ''
    assert [item['name'] for item in external_call[5]['externalAttachments']] == [
        'diagram.png',
        'notes.txt',
    ]
    assert external_call[5]['externalLocalPaths'] == [
        '/work/context.pdf',
        '/work/data.csv',
    ]

    local_calls = result['localCalls']
    assert len(local_calls) == 2
    assert local_calls[0][0] == 'local-chat'
    assert local_calls[0][2]['name'] == 'diagram.png'
    assert local_calls[0][4] == ''
    assert local_calls[1][2] is None
    assert local_calls[1][4] == '/work/context.pdf'
    assert 'externalAttachments' not in local_calls[0][5]
    assert 'externalLocalPaths' not in local_calls[1][5]


@pytest.mark.skipif(shutil.which('node') is None, reason='node is required for frontend VM tests')
def test_external_transport_uses_plural_multipart_without_changing_local_fields():
    result = _run_node(
        r"""
        const fs = require('fs');
        const vm = require('vm');

        class CaptureFormData {
          constructor() { this.items = []; }
          append(key, value, filename) {
            this.items.push({
              key,
              value: value && typeof value === 'object'
                ? { name: value.name || null }
                : value,
              filename: filename || null,
            });
          }
        }

        const posts = [];
        const context = {
          console,
          state: {
            flaskServerUrl: 'http://127.0.0.1:5001',
            inference: { inferTimeoutMs: 60000 },
          },
          FormData: CaptureFormData,
          posts,
          axios: {
            post: async (url, body, options) => {
              posts.push({
                url,
                isFormData: body instanceof CaptureFormData,
                body: body instanceof CaptureFormData ? body.items : body,
                timeout: options.timeout,
              });
              return { data: { id: `response-${posts.length}` } };
            },
          },
        };
        context.globalThis = context;
        vm.createContext(context);
        vm.runInContext(fs.readFileSync('static/ui/request-transport.js', 'utf8'), context);
        vm.runInContext(`
          getRequestExecutionInstance = (instance) => instance;
          normalizeCapability = (value) => String(value || '').trim().toLowerCase();
          normalizeBackend = (value) => String(value || '').trim().toLowerCase();
          inferFileKindFromName = () => 'binary';
          parseExplicitBatchPrompts = () => [];
          buildGhostRoutingConversationSnapshot = () => [];
          buildGhostExecutionPreviewPayload = () => null;
          getResponsesGhostPreferencesPayload = () => null;
          getResponsesGhostRequestMetaPayload = () => null;
          buildSelectedReferenceArtifactPayload = () => null;
          buildSessionControlRequestFields = () => ({});
          sendViaResponsesStream = async () => { throw new Error('file turns must not stream'); };

          const externalTarget = {
            instance_id: 'external:codex',
            target_kind: 'external',
            model: 'codex:auto',
            backend: 'codex_cli',
            capability: 'chat',
          };
          const localTarget = {
            instance_id: 'local-chat',
            target_kind: 'local',
            model: 'local-model',
            backend: 'ollama',
            capability: 'chat',
          };
          const ghostAutoTarget = {
            instance_id: '__responses_ghost_auto__',
            ghostAuto: true,
          };
          const imageFile = { name: 'diagram.png' };
          const notesFile = { name: 'notes.txt' };

          globalThis.runTest = async () => {
            await sendViaResponsesTransport(
              externalTarget,
              externalTarget.instance_id,
              'external-conversation',
              'Inspect these together.',
              imageFile,
              '',
              '',
              {
                requestInstance: externalTarget,
                requestControlFields: {},
                externalAttachments: [imageFile, notesFile],
                externalLocalPaths: ['/work/context.pdf', '/work/data.csv'],
              },
              'response-external'
            );
            await sendViaResponsesTransport(
              localTarget,
              localTarget.instance_id,
              'local-conversation',
              'Inspect this.',
              imageFile,
              '/work/context.pdf',
              '',
              { requestInstance: localTarget, requestControlFields: {} },
              'response-local-upload'
            );
            await sendViaResponsesTransport(
              localTarget,
              localTarget.instance_id,
              'local-conversation',
              'Inspect this path.',
              null,
              '/work/context.pdf',
              '',
              { requestInstance: localTarget, requestControlFields: {} },
              'response-local-path'
            );
            await sendViaResponsesTransport(
              externalTarget,
              externalTarget.instance_id,
              'external-conversation',
              'Inspect these paths.',
              null,
              '/work/context.pdf',
              '',
              {
                requestInstance: externalTarget,
                requestControlFields: {},
                externalAttachments: [],
                externalLocalPaths: ['/work/context.pdf', '/work/data.csv'],
              },
              'response-external-paths'
            );
            await sendViaResponsesTransport(
              ghostAutoTarget,
              ghostAutoTarget.instance_id,
              'ghost-conversation',
              'Inspect this through Ghost.',
              imageFile,
              '',
              '',
              {
                requestInstance: externalTarget,
                requestControlFields: {},
                externalAttachments: [imageFile, notesFile],
              },
              'response-ghost'
            );
            return posts;
          };
        `, context);
        context.runTest().then((value) => {
          process.stdout.write(JSON.stringify(value));
        }).catch((error) => {
          console.error(error);
          process.exit(1);
        });
        """
    )

    external_upload = result[0]
    assert external_upload['isFormData'] is True
    external_upload_keys = [item['key'] for item in external_upload['body']]
    assert external_upload_keys.count('files') == 2
    assert external_upload_keys.count('file_paths_json') == 1
    assert 'file' not in external_upload_keys
    assert 'file_path' not in external_upload_keys
    uploaded_names = [
        item['filename']
        for item in external_upload['body']
        if item['key'] == 'files'
    ]
    assert uploaded_names == ['diagram.png', 'notes.txt']
    path_item = next(item for item in external_upload['body'] if item['key'] == 'file_paths_json')
    assert json.loads(path_item['value']) == ['/work/context.pdf', '/work/data.csv']

    local_upload = result[1]
    assert local_upload['isFormData'] is True
    local_upload_keys = [item['key'] for item in local_upload['body']]
    assert local_upload_keys.count('file') == 1
    assert local_upload_keys.count('file_path') == 1
    assert 'files' not in local_upload_keys
    assert 'file_paths_json' not in local_upload_keys

    local_path = result[2]
    assert local_path['isFormData'] is False
    assert local_path['body']['file_path'] == '/work/context.pdf'
    assert 'files' not in local_path['body']
    assert 'file_paths_json' not in local_path['body']

    external_paths = result[3]
    assert external_paths['isFormData'] is True
    external_path_keys = [item['key'] for item in external_paths['body']]
    assert 'files' not in external_path_keys
    assert external_path_keys.count('file_paths_json') == 1
    external_path_item = next(item for item in external_paths['body'] if item['key'] == 'file_paths_json')
    assert json.loads(external_path_item['value']) == ['/work/context.pdf', '/work/data.csv']

    ghost_upload = result[4]
    assert ghost_upload['isFormData'] is True
    ghost_upload_keys = [item['key'] for item in ghost_upload['body']]
    assert ghost_upload_keys.count('file') == 1
    assert ghost_upload_keys.count('ghost_messages_json') == 1
    assert 'files' not in ghost_upload_keys
    assert 'file_paths_json' not in ghost_upload_keys
