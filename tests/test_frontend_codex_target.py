import json
import re
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
WEB_UI_PATH = ROOT / "ollmo_webUI.html"
MODELS_PATH = ROOT / "static" / "ui" / "models.js"
CONVERSATIONS_PATH = ROOT / "static" / "ui" / "conversations.js"
REQUEST_LIFECYCLE_PATH = ROOT / "static" / "ui" / "request-lifecycle.js"
REQUEST_TRANSPORT_PATH = ROOT / "static" / "ui" / "request-transport.js"
MESSAGE_STATE_PATH = ROOT / "static" / "ui" / "message-state.js"
UI_CSS_PATH = ROOT / "static" / "ui" / "ollmo.css"
LANDING_PATH = ROOT / "site" / "index.html"
LANDING_CSS_PATH = ROOT / "site" / "landing.css"


def _run_node(script: str) -> dict:
    result = subprocess.run(
        ["node", "-e", textwrap.dedent(script)],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(result.stdout)


def _css_rule(source: str, selector: str) -> str:
    match = re.search(rf"{re.escape(selector)}\s*\{{(?P<body>[^}}]*)\}}", source)
    assert match, f"missing CSS rule for {selector}"
    return match.group("body")


def test_codex_connection_is_a_compact_visible_external_card():
    html = WEB_UI_PATH.read_text()
    css = UI_CSS_PATH.read_text()

    opening_tag = re.search(
        r'<div id="codex-connection" class="external-target-card"[^>]*>',
        html,
    )
    assert opening_tag
    assert "hidden" not in opening_tag.group(0)
    assert "Optional cloud provider" in html
    assert '<span>ChatGPT</span>' in html
    assert "exact model variant is not exposed to Ollmo" in html
    assert 'class="badge-soft badge-muted badge-inline">External</span>' in html
    assert html.index('id="models-list"') < html.index('id="codex-connection"')

    card_rule = _css_rule(css, ".external-target-card")
    assert "display: grid" in card_rule
    assert "grid-template-columns: minmax(0, 1fr)" in card_rule
    assert "padding: 0.58rem 0.7rem" in card_rule
    assert "min-height" not in card_rule
    assert "font-size: 0.56rem" in _css_rule(css, ".external-target-card__detail")


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required for frontend VM tests")
def test_selectable_codex_projects_into_direct_targets_and_tabs_not_running_instances():
    result = _run_node(
        r"""
        const fs = require('fs');
        const vm = require('vm');

        function makeNode(tag = 'div') {
          const listeners = {};
          const node = {
            tagName: String(tag).toUpperCase(),
            children: [],
            dataset: {},
            style: {},
            hidden: false,
            disabled: false,
            className: '',
            title: '',
            value: '',
            textContent: '',
            classList: { add() {}, remove() {}, toggle() {} },
            setAttribute() {},
            removeAttribute() {},
            addEventListener(type, handler) { listeners[type] = handler; },
            appendChild(child) { this.children.push(child); return child; },
            focus() {},
            listeners,
          };
          let html = '';
          Object.defineProperty(node, 'innerHTML', {
            get() { return html; },
            set(value) { html = String(value); this.children = []; },
          });
          return node;
        }

        const nodes = new Map();
        const getNode = (id) => {
          if (!nodes.has(id)) nodes.set(id, makeNode());
          return nodes.get(id);
        };
        const context = {
          console,
          window: {
            location: { protocol: 'http:', hostname: '127.0.0.1', port: '5001' },
            matchMedia: () => ({
              matches: false,
              addEventListener() {},
              addListener() {},
            }),
          },
          document: {
            getElementById: getNode,
            querySelectorAll: () => [],
            querySelector: () => makeNode(),
            addEventListener() {},
            createElement: makeNode,
          },
          setTimeout,
          clearTimeout,
          setInterval,
          clearInterval,
          requestAnimationFrame: (callback) => callback(),
          localStorage: { getItem() { return null; }, setItem() {} },
          sessionStorage: { getItem() { return null; }, setItem() {} },
        };
        context.globalThis = context;
        vm.createContext(context);
        vm.runInContext(fs.readFileSync('static/ui/models.js', 'utf8'), context);
        vm.runInContext(fs.readFileSync('static/ui/conversations.js', 'utf8'), context);

        const page = fs.readFileSync('ollmo_webUI.html', 'utf8');
        const scripts = [...page.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)]
          .map((match) => match[1]);
        const inline = scripts.find((source) => source.includes('const settingsDefaults'));
        vm.runInContext(inline, context);

        vm.runInContext(`
          state.runningInstances = [];
          state.responsesWorkbench.codexStatus = {
            id: 'external:codex',
            status: 'available',
            enabled: true,
            selectable: true,
            source: 'path',
          };
          state.activeWorkspace = 'responses';
          renderResponsesWorkbenchTargetOptions = () => {};
          updateActiveTabHighlight = () => {};
          updateChatControls = () => {};
          renderModelTabs();
          globalThis.result = {
            runningInstanceIds: state.runningInstances.map((item) => item.instance_id),
            directTargets: getDirectConversationTargets(),
            tabs: elements.modelTabs.children.map((node) => ({
              className: node.className,
              dataset: { ...node.dataset },
              html: node.innerHTML,
            })),
            instanceMeta: getInstanceMeta('external:codex'),
            sourceLabels: {
              app: formatCodexSourceLabel({ source: 'chatgpt_app_system' }),
              cli: formatCodexSourceLabel({ source: 'path' }),
              generic: formatCodexSourceLabel({ source: 'future_source' }),
            },
            displayNames: {
              model: formatModelDisplayName('codex:auto'),
              instance: formatModelDisplayName('external:codex'),
            },
          };
        `, context);

        process.stdout.write(JSON.stringify(context.result));
        """
    )

    assert result["runningInstanceIds"] == []
    assert len(result["directTargets"]) == 1
    target = result["directTargets"][0]
    assert target["instance_id"] == "external:codex"
    assert target["target_kind"] == "external"
    assert target["lifecycle_managed"] is False
    assert target["text_only"] is False
    assert target["inputs"] == ["text", "image", "file"]
    assert target["outputs"] == ["text"]
    assert target["label"] == "ChatGPT"
    assert target["model"] == "codex:auto"
    assert result["instanceMeta"]["instance_id"] == "external:codex"

    assert len(result["tabs"]) == 2
    assert "Ollmo" in result["tabs"][0]["html"]
    assert result["tabs"][1]["dataset"] == {
        "instanceId": "external:codex",
        "targetKind": "external",
    }
    assert "ChatGPT" in result["tabs"][1]["html"]
    assert "Cloud" in result["tabs"][1]["html"]
    assert result["displayNames"] == {"model": "ChatGPT", "instance": "ChatGPT"}

    assert result["sourceLabels"] == {
        "app": "the ChatGPT app",
        "cli": "a separate Codex CLI",
        "generic": "the detected Codex connection",
    }

    models_source = MODELS_PATH.read_text()
    stop_all_source = models_source[
        models_source.index("async function stopAllModels()"):
        models_source.index("async function stopCurrentModel()")
    ]
    assert "state.runningInstances.map" in stop_all_source
    assert "getSelectableExternalTargets" not in stop_all_source
    assert "ChatGPT has no local process to stop" in models_source


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required for frontend VM tests")
def test_ollmo_auto_and_chatgpt_remain_sendable_with_explicit_context():
    result = _run_node(
        r"""
        const fs = require('fs');
        const vm = require('vm');
        const context = {
          console,
          state: {
            arena: { enabled: false, modelA: null, modelB: null },
            responsesWorkbench: { targetInstanceId: '__responses_ghost_auto__' },
            runningInstances: [],
            inference: {},
          },
          elements: { userInput: { value: 'hello from Ollmo' } },
          setTimeout,
          clearTimeout,
          setInterval,
          clearInterval,
          TextDecoder,
          Uint8Array,
        };
        context.globalThis = context;
        vm.createContext(context);
        vm.runInContext(fs.readFileSync('static/ui/request-lifecycle.js', 'utf8'), context);
        vm.runInContext(`
          let pendingItems = [];
          let selectedRefs = [];
          let activeTarget = null;
          getPendingInputItems = () => pendingItems;
          isResponsesWorkbenchActive = () => true;
          isResponsesWorkbenchAutoTarget = () => true;
          getResponsesWorkbenchConversationId = () => 'ollmo-auto';
          hasPendingConversationPreview = () => false;
          hasPendingConversationChatRequest = () => false;
          arenaSelectionsAreChatCapable = () => false;
          isInstanceRequestPending = () => false;
          globalThis.autoAllowedWithNoModels = canSendPrompt();

          isResponsesWorkbenchActive = () => false;
          isResponsesWorkbenchAutoTarget = () => false;
          activeTarget = {
            instance_id: 'external:codex',
            target_kind: 'external',
            text_only: false,
            inputs: ['text', 'image', 'file'],
            capability: 'chat',
          };
          getActivePromptTargetInstance = () => activeTarget;
          getActiveConversationId = () => 'codex-conversation';
          isExternalConversationTarget = (value) => value?.target_kind === 'external';
          normalizeCapability = (value) => String(value || '').toLowerCase();
          getSelectedReferenceArtifacts = () => selectedRefs;

          globalThis.externalTextAllowed = canSendPrompt();
          pendingItems = [{ source: 'upload', file: { name: 'private.png' } }];
          globalThis.externalAttachmentAllowed = canSendPrompt();
          pendingItems = [];
          selectedRefs = [{ artifact_ref: 'artifact:private' }];
          globalThis.externalReferenceAllowed = canSendPrompt();
        `, context);
        process.stdout.write(JSON.stringify({
          autoAllowedWithNoModels: context.autoAllowedWithNoModels,
          externalTextAllowed: context.externalTextAllowed,
          externalAttachmentAllowed: context.externalAttachmentAllowed,
          externalReferenceAllowed: context.externalReferenceAllowed,
        }));
        """
    )

    assert result == {
        "autoAllowedWithNoModels": True,
        "externalTextAllowed": True,
        "externalAttachmentAllowed": True,
        "externalReferenceAllowed": True,
    }


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required for frontend VM tests")
def test_external_codex_stream_payload_preserves_explicit_artifact_references():
    lifecycle_source = REQUEST_LIFECYCLE_PATH.read_text()
    transport_source = REQUEST_TRANSPORT_PATH.read_text()
    assert "buildSelectedReferenceArtifactPayload(conversationId)" in lifecycle_source
    assert "buildSelectedReferenceArtifactPayload(conversationId)" in transport_source

    result = _run_node(
        r"""
        const fs = require('fs');
        const vm = require('vm');
        const context = {
          console,
          state: { flaskServerUrl: 'http://127.0.0.1:5001' },
          elements: {},
          TextDecoder,
          Uint8Array,
          setTimeout,
          clearTimeout,
          setInterval,
          clearInterval,
        };
        context.globalThis = context;
        vm.createContext(context);
        vm.runInContext(fs.readFileSync('static/ui/request-lifecycle.js', 'utf8'), context);
        vm.runInContext(`
          getRequestExecutionInstance = (instance) => instance;
          buildResponsesInputForHistory = () => [{ type: 'message', role: 'user' }];
          buildGhostRoutingConversationSnapshot = () => [];
          getResponsesGhostPreferencesPayload = () => null;
          getResponsesGhostRequestMetaPayload = () => null;
          buildSessionControlRequestFields = () => ({});
          isExternalConversationTarget = (instance) => instance?.target_kind === 'external';
          buildSelectedReferenceArtifactPayload = () => ({
            type: 'image',
            path: '/tmp/explicit-reference.png',
            artifact_ref: 'artifact:explicit-reference',
          });
          buildCanonicalResponseId = () => 'resp-codex-test';
          markActiveResponseSseStream = () => {};
          fetch = async (_url, options) => {
            globalThis.capturedPayload = JSON.parse(options.body);
            return {
              ok: true,
              body: {
                getReader() {
                  return { read: async () => ({ done: true, value: undefined }) };
                },
              },
            };
          };
        `, context);
        vm.runInContext(`
          sendViaResponsesStream(
            {
              instance_id: 'external:codex',
              target_kind: 'external',
              text_only: false,
              inputs: ['text', 'image', 'file'],
            },
            'external:codex',
            'conv-codex',
            '',
            null,
            'hello'
          )
        `, context).then(() => {
          process.stdout.write(JSON.stringify(context.capturedPayload));
        }).catch((error) => {
          console.error(error);
          process.exit(1);
        });
        """
    )

    assert result["instance_id"] == "external:codex"
    assert result["stream"] is True
    assert result["reference_artifacts"]["artifact_ref"] == "artifact:explicit-reference"
    assert "file_path" not in result


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required for frontend VM tests")
def test_external_codex_stream_error_preserves_canonical_response_truth():
    result = _run_node(
        r"""
        const fs = require('fs');
        const vm = require('vm');
        const canonicalFailure = {
          id: 'resp-codex-failed',
          lifecycle_state: 'failed',
          outputs: [{ type: 'text', status: 'blocked' }],
          error_ref: { code: 'CODEX_AUTH_REQUIRED', stage: 'external_execution' },
          recovery_hint: 'Sign in through Codex, then try again.',
          response_frame: { lifecycle_state: 'failed' },
          error: "Codex execution ended with status 'auth_required'.",
        };
        const context = {
          console,
          canonicalFailure,
          state: { flaskServerUrl: 'http://127.0.0.1:5001' },
          elements: {},
          TextDecoder,
          Uint8Array,
          setTimeout,
          clearTimeout,
          setInterval,
          clearInterval,
        };
        context.globalThis = context;
        vm.createContext(context);
        vm.runInContext(fs.readFileSync('static/ui/request-lifecycle.js', 'utf8'), context);
        vm.runInContext(`
          getRequestExecutionInstance = (instance) => instance;
          buildResponsesInputForHistory = () => [{ type: 'message', role: 'user', content: 'hello' }];
          buildGhostRoutingConversationSnapshot = () => [];
          getResponsesGhostPreferencesPayload = () => null;
          getResponsesGhostRequestMetaPayload = () => null;
          buildSessionControlRequestFields = () => ({});
          buildSelectedReferenceArtifactPayload = () => null;
          buildCanonicalResponseId = () => 'resp-codex-failed';
          fetch = async () => ({
            ok: false,
            status: 401,
            text: async () => JSON.stringify(canonicalFailure),
          });
        `, context);
        vm.runInContext(`
          sendViaResponsesStream(
            { instance_id: 'external:codex', target_kind: 'external', text_only: true },
            'external:codex',
            'conv-codex',
            '',
            null,
            'hello'
          )
        `, context).then(() => {
          process.stderr.write('expected request to fail');
          process.exit(1);
        }).catch((error) => {
          process.stdout.write(JSON.stringify({
            message: error.message,
            status: error.response && error.response.status,
            data: error.response && error.response.data,
          }));
        });
        """
    )

    assert result["status"] == 401
    assert result["data"]["lifecycle_state"] == "failed"
    assert result["data"]["outputs"][0]["status"] == "blocked"
    assert result["data"]["error_ref"]["code"] == "CODEX_AUTH_REQUIRED"
    assert result["data"]["response_frame"]["lifecycle_state"] == "failed"
    assert result["data"]["recovery_hint"]


def test_external_codex_copy_discloses_independent_turns_and_observer_preserves_tab():
    html = WEB_UI_PATH.read_text()
    models_source = MODELS_PATH.read_text()
    fetch_running_source = models_source[
        models_source.index("async function fetchRunningInstances()"):
        models_source.index("async function fetchAvailableModels()")
    ]

    assert "Each turn is independent" in html
    assert "Send a prompt and any explicitly selected files to ChatGPT." in html
    assert "exact model variant is not exposed to Ollmo" in html
    assert "processed by OpenAI" in html
    assert "each turn is independent" in models_source
    assert "const activeExternalTarget" in fetch_running_source
    assert "if (activeExternalTarget)" in fetch_running_source
    assert "ChatGPT remains ready through Codex." in fetch_running_source


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required for frontend VM tests")
def test_external_codex_provenance_uses_chatgpt_label_without_inventing_model_identity():
    result = _run_node(
        r"""
        const fs = require('fs');
        const vm = require('vm');
        const context = {
          console,
          state: {},
          elements: {},
          normalizeBackend: (value) => String(value || '').trim().toLowerCase(),
          normalizeCapability: (value) => String(value || '').trim().toLowerCase(),
          formatBackendLabel: (value) => String(value || ''),
          formatModelDisplayName: (value) => String(value || ''),
          isResponsesWorkbenchConversationId: () => true,
          getConversationInstanceMeta: () => null,
          getInstanceMeta: () => null,
        };
        context.globalThis = context;
        vm.createContext(context);
        vm.runInContext(fs.readFileSync('static/ui/message-state.js', 'utf8'), context);
        const message = {
          role: 'assistant',
          content: 'I am GPT-5.6 Sol.',
          responseModel: 'codex:auto',
          responseBackend: 'codex_cli',
          responseInstanceId: 'external:codex',
          routeSource: 'ghost_carried',
          routeRouterModel: 'codex:auto',
        };
        const legacyMessage = {
          role: 'assistant',
          content: 'Cloud answer from an older conversation.',
          responseBackend: 'codex_cli',
          responseInstanceId: 'external:codex',
        };
        process.stdout.write(JSON.stringify({
          provenance: context.formatAssistantProvenanceText(message, 'chatgpt-test'),
          legacyProvenance: context.formatAssistantProvenanceText(legacyMessage, 'chatgpt-test'),
        }));
        """
    )

    assert result["provenance"] == (
        "Answered by ChatGPT through Ollmo (via Codex) · automatic model"
    )
    assert result["legacyProvenance"] == result["provenance"]
    assert "codex:auto" not in result["provenance"]
    assert "GPT-5.6" not in result["provenance"]


def test_landing_keeps_third_party_provider_and_skill_as_a_demoted_runtime_note():
    html = LANDING_PATH.read_text()
    css = LANDING_CSS_PATH.read_text()

    about_start = html.index('<section id="about"')
    setup_start = html.index('<section id="setup"')
    about = html[about_start:setup_start]
    setup = html[setup_start:]

    assert 'class="codex-skill"' not in html
    assert 'class="codex-skill-note"' not in html
    assert 'class="provider-note"' in about
    assert "<strong>Third-party providers.</strong>" in about
    assert "(currently ChatGPT)" in about
    assert "Ollmo also offers an optional companion skill" in about
    assert "execute requests through Ollmo" in about
    assert 'class="provider-note"' not in setup

    skill_note_rule = _css_rule(css, ".today > .provider-note")
    assert "font-size: 0.84rem" in skill_note_rule
    assert "width: 100%" in skill_note_rule
    assert "max-width: none" in skill_note_rule
    assert "display: grid" not in skill_note_rule
    assert ".codex-skill h2" not in css
    assert ".codex-skill-note" not in css
    assert ".install {" not in css
