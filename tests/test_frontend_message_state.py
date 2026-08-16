import shutil
import subprocess
import textwrap
import unittest
from pathlib import Path


class FrontendMessageStateTests(unittest.TestCase):
    def test_default_response_ui_hides_destructive_and_debug_actions(self):
        messages_source = Path("static/ui/messages.js").read_text()

        self.assertNotIn('<i class="fas fa-trash"></i> Trash', messages_source)
        self.assertNotIn('Use As Reference', messages_source)
        self.assertNotIn('Use Reply As Reference', messages_source)
        self.assertNotIn('formatAssistantImageDebugText(message, conversationId)', messages_source)
        self.assertIn('Reference Reply', messages_source)
        self.assertIn('Reference Image', messages_source)
        self.assertIn('chat-message-work-compact', messages_source)
        self.assertIn('chat-work-indicator', messages_source)
        self.assertIn('createCompactLateFillBranchControlsForMessage', messages_source)
        self.assertIn('chat-work-branch-control', messages_source)
        self.assertIn('shouldPrependAssistantWorkStatusMeta', messages_source)
        self.assertIn('assistantWorkStatusMeta', messages_source)
        self.assertIn('bubble.appendChild(body);', messages_source)
        self.assertIn('filterUserVisibleResponseWorkItems(outputs)', messages_source)
        self.assertIn('artifacts.forEach((artifact) => {', messages_source)
        self.assertLess(
            messages_source.index('bubble.appendChild(body);'),
            messages_source.index('if (assistantWorkStatusMeta)'),
        )
        self.assertNotIn(
            'const isAssistantMessage = !isUser\n        && !message.isLoading',
            messages_source,
        )

    def test_compact_work_indicator_styles_are_available(self):
        css_source = Path("static/ui/ollmo.css").read_text()

        self.assertIn('.chat-message-work-compact', css_source)
        self.assertIn('.chat-message-work-compact + .chat-message__body', css_source)
        self.assertIn('.chat-work-indicator', css_source)
        self.assertIn('.chat-work-branch-control', css_source)
        self.assertIn('.chat-work-branch-popover', css_source)
        self.assertIn('@keyframes chat-work-pulse', css_source)

    def test_hydrated_open_responses_resume_late_fill_polling(self):
        lifecycle_source = Path("static/ui/request-lifecycle.js").read_text()
        history_source = Path("static/ui/settings-history.js").read_text()

        self.assertIn('function resumeHydratedLateFillResponses', lifecycle_source)
        self.assertIn('historyMessageHasOpenResponseWork', lifecycle_source)
        self.assertIn('historyMessageNeedsResponseTruthHydration', lifecycle_source)
        self.assertIn('historyMessageNeedsResponseLookup', lifecycle_source)
        self.assertIn('scheduleLateFillResponsePoll(', lifecycle_source)
        self.assertIn('resumeHydratedLateFillResponses(instanceId)', history_source)

    def test_stream_push_truth_updates_loading_response_by_id(self):
        lifecycle_source = Path("static/ui/request-lifecycle.js").read_text()
        message_state_source = Path("static/ui/message-state.js").read_text()

        self.assertIn("response.state.updated", lifecycle_source)
        self.assertIn("response.late_fill.updated", lifecycle_source)
        self.assertIn("response.route.resolved", lifecycle_source)
        self.assertIn("response.backend.started", lifecycle_source)
        self.assertIn("response.late_fill.branch.updated", lifecycle_source)
        self.assertIn("response.requires_action", lifecycle_source)
        self.assertIn("buildResponseSseLiveStatusLines", lifecycle_source)
        self.assertIn("applySseLiveStatus", lifecycle_source)
        self.assertIn("applyPushedResponsePayload", lifecycle_source)
        self.assertIn("updateAssistantResponseByResponseId(", lifecycle_source)
        self.assertIn("updateLoadingAssistantMessageFromResponsePayload(", lifecycle_source)
        self.assertIn("ensureLoadingResponseBinding(createdResponseId)", lifecycle_source)
        self.assertIn("finalResponseFinalized", lifecycle_source)
        self.assertIn("!extras.forceRender", message_state_source)
        self.assertIn("function updateLoadingAssistantMessageFromResponsePayload", message_state_source)

    @unittest.skipIf(shutil.which("node") is None, "node is required for frontend message-state tests")
    def test_public_artifact_ledger_excludes_dossiers_and_repair_outputs(self):
        script = r"""
            const assert = require('assert');
            const fs = require('fs');
            const vm = require('vm');

            const context = {
              console,
              window: {},
              state: {},
              elements: {},
              normalizeCapability: (value) => String(value || '').trim().toLowerCase() || null,
              normalizeBackend: (value) => String(value || '').trim().toLowerCase() || null,
              basenameFromPath: (value) => String(value || '').split(/[\\/]/).filter(Boolean).pop() || '',
            };
            vm.createContext(context);
            vm.runInContext(fs.readFileSync('static/ui/message-state.js', 'utf8'), context);

            const frameArtifacts = context.sanitizeResponseArtifacts({
              output: [
                {
                  type: 'text',
                  path: '/tmp/final/index.html',
                  artifact_ref: 'artifact:index',
                },
              ],
              dossiers: {
                'artifact:text_generated_image_bad': {
                  artifact: {
                    type: 'text',
                    path: '/tmp/intermediate/generated-image-index.html',
                    artifact_ref: 'artifact:text_generated_image_bad',
                  },
                },
              },
            });
            assert.strictEqual(
              JSON.stringify(frameArtifacts.map((artifact) => artifact.artifact_ref)),
              JSON.stringify(['artifact:index']),
            );

            const message = {
              role: 'assistant',
              artifacts: [
                {
                  type: 'text',
                  path: '/tmp/final/index.html',
                  artifact_ref: 'artifact:index',
                },
                {
                  type: 'text',
                  path: '/tmp/intermediate/generated-image-index.html',
                  artifact_ref: 'artifact:text_generated_image_bad',
                },
              ],
              outputSlots: [
                {
                  slot_id: 'output-html',
                  type: 'text',
                  status: 'fulfilled',
                  artifact_ref: 'artifact:index',
                },
                {
                  slot_id: 'output-repair',
                  type: 'text',
                  status: 'fulfilled',
                  artifact_ref: 'artifact:text_generated_image_bad',
                },
              ],
              outputs: [
                {
                  slot_id: 'output-html',
                  type: 'text',
                  status: 'fulfilled',
                  artifact_ref: 'artifact:index',
                },
                {
                  slot_id: 'output-repair',
                  type: 'text',
                  status: 'fulfilled',
                  artifact_ref: 'artifact:text_generated_image_bad',
                },
              ],
            };
            context.syncMessageArtifactLedger(message);
            assert.strictEqual(
              JSON.stringify(message.artifacts.map((artifact) => artifact.artifact_ref)),
              JSON.stringify(['artifact:index']),
            );
        """
        subprocess.run(
            ["node", "-e", textwrap.dedent(script)],
            cwd=".",
            check=True,
            text=True,
            capture_output=True,
        )

    @unittest.skipIf(shutil.which("node") is None, "node is required for frontend message-state tests")
    def test_same_caption_images_keep_distinct_artifact_refs(self):
        script = r"""
            const assert = require('assert');
            const fs = require('fs');
            const vm = require('vm');

            const context = {
              console,
              window: {},
              state: {},
              elements: {},
              normalizeCapability: (value) => String(value || '').trim().toLowerCase() || null,
              normalizeBackend: (value) => String(value || '').trim().toLowerCase() || null,
              basenameFromPath: (value) => String(value || '').split(/[\\/]/).filter(Boolean).pop() || '',
            };
            vm.createContext(context);
            vm.runInContext(fs.readFileSync('static/ui/message-state.js', 'utf8'), context);

            const artifacts = context.sanitizeResponseArtifacts([
              {
                type: 'image',
                name: 'Generated pet portrait',
                prompt: 'same frontend caption',
                artifact_ref: 'artifact:image_one',
              },
              {
                type: 'image',
                name: 'Generated pet portrait',
                prompt: 'same frontend caption',
                artifact_ref: 'artifact:image_two',
              },
            ]);
            assert.strictEqual(
              JSON.stringify(artifacts.map((artifact) => artifact.artifact_ref)),
              JSON.stringify(['artifact:image_one', 'artifact:image_two']),
            );

            const message = {
              role: 'assistant',
              artifacts,
              outputSlots: [
                { slot_id: 'output-image-1', type: 'image', status: 'fulfilled', artifact_ref: 'artifact:image_one' },
                { slot_id: 'output-image-2', type: 'image', status: 'fulfilled', artifact_ref: 'artifact:image_two' },
              ],
              outputs: [
                { slot_id: 'output-image-1', type: 'image', status: 'fulfilled', artifact_ref: 'artifact:image_one' },
                { slot_id: 'output-image-2', type: 'image', status: 'fulfilled', artifact_ref: 'artifact:image_two' },
              ],
            };
            context.syncMessageArtifactLedger(message);
            assert.strictEqual(
              JSON.stringify(message.artifacts.map((artifact) => artifact.artifact_ref)),
              JSON.stringify(['artifact:image_one', 'artifact:image_two']),
            );
        """
        subprocess.run(
            ["node", "-e", textwrap.dedent(script)],
            cwd=".",
            check=True,
            text=True,
            capture_output=True,
        )

    @unittest.skipIf(shutil.which("node") is None, "node is required for frontend request-lifecycle tests")
    def test_stale_loading_cleanup_preserves_active_push_owned_messages(self):
        script = r"""
            const assert = require('assert');
            const fs = require('fs');
            const vm = require('vm');

            const oldTimestamp = new Date(Date.now() - 10 * 60 * 1000).toISOString();
            const context = {
              console,
              state: {
                inference: {
                  staleLoadingMs: 10000,
                  pendingRequests: {
                    req_pending: {
                      phase: 'running',
                      conversationId: 'conv-live',
                      loadingMessageId: 'client-pending',
                    },
                  },
                },
                conversations: {
                  'conv-live': [
                    { role: 'assistant', isLoading: true, clientMessageId: 'client-sse', responseId: 'resp-sse', timestamp: oldTimestamp },
                    { role: 'assistant', isLoading: true, clientMessageId: 'client-pending', timestamp: oldTimestamp },
                    { role: 'assistant', isLoading: true, clientMessageId: 'client-stale', timestamp: oldTimestamp },
                    { role: 'user', content: 'Keep the live frame.' },
                  ],
                },
                arena: { enabled: false },
              },
              elements: {},
              window: { setTimeout: () => 0, clearTimeout: () => {} },
              sessionStorage: { getItem: () => null, setItem: () => {}, removeItem: () => {} },
              normalizeCapability: (value) => String(value || '').trim().toLowerCase() || null,
              isConversationVisible: () => false,
              renderConversation: () => {},
              getActiveConversationId: () => 'conv-live',
              updateSendButtonState: () => {},
            };
            vm.createContext(context);
            vm.runInContext(fs.readFileSync('static/ui/request-lifecycle.js', 'utf8'), context);

            context.markActiveResponseSseStream('resp-sse', true);
            context.clearStaleLoadingMessages('conv-live');
            assert.deepStrictEqual(
              context.state.conversations['conv-live'].map((message) => message.clientMessageId || message.role),
              ['client-sse', 'client-pending', 'user'],
            );

            context.clearStaleLoadingMessages('conv-live', true);
            assert.deepStrictEqual(
              context.state.conversations['conv-live'].map((message) => message.role),
              ['user'],
            );
        """
        subprocess.run(
            ["node", "-e", textwrap.dedent(script)],
            cwd=".",
            check=True,
            text=True,
            capture_output=True,
        )

    @unittest.skipIf(shutil.which("node") is None, "node is required for frontend history hydration tests")
    def test_history_hydration_preserves_active_live_loading_message(self):
        script = r"""
            const assert = require('assert');
            const fs = require('fs');
            const vm = require('vm');

            let savedSlots = 0;
            let renderedHistory = 0;
            let resumed = 0;
            const liveMessage = {
              role: 'assistant',
              isLoading: true,
              clientMessageId: 'client-live',
              responseId: 'resp-live',
              content: 'HTML saved; image still running...',
              artifacts: [{ type: 'text', path: '/tmp/artifacts/index.html' }],
              outputSlots: [{ slot_id: 'output-image', type: 'image', status: 'pending' }],
              timestamp: new Date(Date.now() - 10 * 60 * 1000).toISOString(),
            };
            const context = {
              console,
              window: {},
              state: {
                conversations: {
                  'conv-live': [
                    { role: 'user', content: 'Build the page.', clientMessageId: 'user-1' },
                    liveMessage,
                  ],
                },
                chatHistoryLoaded: {},
              },
              elements: {},
              normalizeCapability: (value) => String(value || '').trim().toLowerCase() || null,
              normalizeBackend: (value) => String(value || '').trim().toLowerCase() || null,
              basenameFromPath: (value) => String(value || '').split(/[\\/]/).filter(Boolean).pop() || '',
              buildConversationLedgerMetadata: () => ({}),
              registerConversationMetadata: () => null,
              appendConversationToSlotHistory: () => {},
              setSlotHistoryConversationIds: () => {},
              saveConversationSlots: () => { savedSlots += 1; },
              renderConversationHistoryList: () => { renderedHistory += 1; },
              resumeHydratedLateFillResponses: () => { resumed += 1; },
              responseHasActiveSseStream: (responseId) => responseId === 'resp-live',
              listPendingRequests: () => [],
            };
            vm.createContext(context);
            vm.runInContext(fs.readFileSync('static/ui/message-state.js', 'utf8'), context);
            vm.runInContext(fs.readFileSync('static/ui/settings-history.js', 'utf8'), context);

            context.hydrateConversationFromHistoryPayload('conv-live', {
              messages: [{ role: 'user', content: 'Build the page.', message_id: 'user-1' }],
            });
            assert.strictEqual(context.state.conversations['conv-live'].length, 2);
            assert.strictEqual(context.state.conversations['conv-live'][1].isLoading, true);
            assert.strictEqual(context.state.conversations['conv-live'][1].responseId, 'resp-live');
            assert.strictEqual(context.state.conversations['conv-live'][1].artifacts[0].path, '/tmp/artifacts/index.html');
            assert.strictEqual(savedSlots, 1);
            assert.strictEqual(renderedHistory, 1);
            assert.strictEqual(resumed, 1);

            context.state.conversations['conv-live'] = [
              { role: 'user', content: 'Build the page.', clientMessageId: 'user-1' },
              liveMessage,
            ];
            context.hydrateConversationFromHistoryPayload('conv-live', {
              messages: [
                { role: 'user', content: 'Build the page.', message_id: 'user-1' },
                {
                  role: 'assistant',
                  content: 'Done.',
                  response_id: 'resp-live',
                  lifecycle_state: 'completed',
                  status_semantics: { is_terminal: true, has_open_continuation: false },
                },
              ],
            });
            assert.strictEqual(context.state.conversations['conv-live'].length, 2);
            assert.strictEqual(context.state.conversations['conv-live'][1].isLoading, false);
            assert.strictEqual(context.state.conversations['conv-live'][1].content, 'Done.');
        """
        subprocess.run(
            ["node", "-e", textwrap.dedent(script)],
            cwd=".",
            check=True,
            text=True,
            capture_output=True,
        )

    @unittest.skipIf(shutil.which("node") is None, "node is required for frontend request-lifecycle tests")
    def test_sse_event_status_reducer_uses_event_intent(self):
        script = r"""
            const assert = require('assert');
            const fs = require('fs');
            const vm = require('vm');

            const context = {
              console,
              state: {},
              elements: {},
              window: { setTimeout: () => 0, clearTimeout: () => {} },
              sessionStorage: { getItem: () => null, setItem: () => {}, removeItem: () => {} },
            };
            vm.createContext(context);
            vm.runInContext(fs.readFileSync('static/ui/request-lifecycle.js', 'utf8'), context);

            assert.strictEqual(
              JSON.stringify(context.buildResponseSseLiveStatusLines('response.backend.started', {}, {})),
              JSON.stringify(['Backend running...'])
            );
            assert.strictEqual(
              JSON.stringify(context.buildResponseSseLiveStatusLines(
                'response.late_fill.branch.updated',
                {
                  branch: { branch_id: 'branch-image-1', capability: 'image_generation', status: 'running' },
                  late_fill_status: 'running',
                },
                {}
              )),
              JSON.stringify(['Image running...'])
            );
            assert.strictEqual(
              JSON.stringify(context.buildResponseSseLiveStatusLines(
                'response.late_fill.branch.updated',
                {
                  branch: { branch_id: 'branch-image-1', capability: 'image_generation', status: 'completed' },
                  late_fill_status: 'running',
                },
                {}
              )),
              JSON.stringify(['Image completed; remaining work pending...'])
            );
            assert.strictEqual(
              JSON.stringify(context.buildResponseSseLiveStatusLines('response.completed', {}, {})),
              JSON.stringify([])
            );
        """
        subprocess.run(
            ["node", "-e", textwrap.dedent(script)],
            cwd=".",
            check=True,
            text=True,
            capture_output=True,
        )

    def test_history_serialization_preserves_response_truth_fields(self):
        history_source = Path("static/ui/settings-history.js").read_text()

        self.assertIn('const lateFill = sanitizeMessageLateFill(source.late_fill || source.lateFill)', history_source)
        self.assertIn('lifecycle_state: normalized.lifecycle_state || null', history_source)
        self.assertIn('status_semantics: normalized.status_semantics || null', history_source)
        self.assertIn('late_fill: normalized.late_fill || null', history_source)
        self.assertIn('surface_state: normalized.surface_state || null', history_source)
        self.assertIn('lifecycleState: normalized.lifecycle_state || null', history_source)
        self.assertIn('statusSemantics: normalized.status_semantics || null', history_source)
        self.assertIn('lateFill: normalized.late_fill || null', history_source)
        self.assertIn('surfaceState: normalized.surface_state || null', history_source)
        self.assertIn('response_frame_sequence: Number.isFinite(Number(normalized.response_frame_sequence))', history_source)
        self.assertIn('responseFrameSequence: Number.isFinite(Number(normalized.response_frame_sequence))', history_source)

    def test_backend_history_hydration_copies_response_truth_fields(self):
        webserver_source = Path("ollmo_webserver.py").read_text()

        self.assertIn("hydrated['status_semantics'] = dict(status_semantics)", webserver_source)
        self.assertIn("hydrated['surface_state'] = dict(surface_state)", webserver_source)
        self.assertIn("for key in ('lifecycle_state', 'state_version', 'canonical_status_field', 'status_compatibility'):", webserver_source)
        self.assertIn("hydrated['response_frame_sequence'] = frame_sequence", webserver_source)
        self.assertIn("hydrated['response_frame_id'] = frame_id", webserver_source)

    def test_response_observer_polling_uses_compact_status_and_history_dedupe(self):
        lifecycle_source = Path("static/ui/request-lifecycle.js").read_text()
        history_source = Path("static/ui/settings-history.js").read_text()

        self.assertIn('fetchResponseLookupStatusPayload', lifecycle_source)
        self.assertIn("view: 'status'", lifecycle_source)
        self.assertIn('getResponseLookupStateVersion', lifecycle_source)
        self.assertIn('LATE_FILL_UNCHANGED_POLL_DELAYS_MS', lifecycle_source)
        self.assertIn('rememberLateFillPollVersion', lifecycle_source)
        self.assertIn('responsePayloadHasOpenPublicArtifactProjectionGap', lifecycle_source)
        self.assertIn('state.chatHistoryPersistedSnapshotsById', history_source)
        self.assertIn('state.chatHistoryPersistRequestsInFlight', history_source)
        self.assertIn('serializedPayload', history_source)

    def test_long_html_css_code_previews_are_collapsible(self):
        messages_source = Path("static/ui/messages.js").read_text()
        css_source = Path("static/ui/ollmo.css").read_text()

        self.assertIn('COLLAPSIBLE_CODE_PREVIEW_LINE_LIMIT = 15', messages_source)
        self.assertIn('isHtmlCssCodePreview', messages_source)
        self.assertIn('data-toggle-code-preview', messages_source)
        self.assertIn('bindMarkdownCodePreviewToggles', messages_source)
        self.assertIn('chat-markdown__code-block--collapsible', messages_source)
        self.assertIn('.chat-markdown__code-block--collapsible.is-collapsed pre', css_source)
        self.assertIn('.chat-markdown__code-toggle', css_source)
        self.assertIn('Show all', messages_source)

    def test_streaming_message_patch_keeps_in_progress_artifact_controls_live(self):
        messages_source = Path("static/ui/messages.js").read_text()
        state_source = Path("static/ui/message-state.js").read_text()

        self.assertIn('function messageHasStructuredRenderSurface', messages_source)
        self.assertIn('streamPatchWouldDropStructuredControls', state_source)
        self.assertIn('!streamPatchWouldDropStructuredControls', state_source)
        self.assertIn('bindMarkdownCodeCopyButtons(body)', state_source)
        self.assertIn('bindMarkdownCodePreviewToggles(body)', state_source)
        self.assertIn('bindMarkdownBlockquoteCopyButtons(body)', state_source)

    def test_saved_text_artifact_view_and_copy_actions_do_not_wait_for_bundle(self):
        messages_source = Path("static/ui/messages.js").read_text()

        self.assertIn('function artifactIsTextLike', messages_source)
        self.assertIn('function copySavedArtifactContent', messages_source)
        self.assertIn("'<i class=\"fas fa-up-right-from-square\"></i> View'", messages_source)
        self.assertIn("'<i class=\"fas fa-copy\"></i> Copy'", messages_source)
        self.assertIn('const canUseInteractivePreview = isHtmlArtifact && (htmlPreviewPath || responseId)', messages_source)
        self.assertIn('htmlPreviewPath || savedArtifactPath', messages_source)
        self.assertIn("htmlPreviewPath ? '' : responseId", messages_source)
        self.assertIn(': buildSavedArtifactViewUrl(savedArtifactPath)', messages_source)
        self.assertIn("window.open(targetUrl, '_blank', 'noopener,noreferrer')", messages_source)
        self.assertIn('copySavedArtifactContent(savedArtifactPath, copyArtifactButton)', messages_source)

    @unittest.skipIf(shutil.which("node") is None, "node is required for frontend message-state tests")
    def test_html_preview_mapping_prefers_exact_source_path_across_bundle_history(self):
        script = r"""
            const assert = require('assert');
            const fs = require('fs');
            const vm = require('vm');

            const context = {
              console,
              window: {},
              state: { flaskServerUrl: 'http://127.0.0.1:5001' },
              elements: {},
              normalizeCapability: (value) => String(value || '').trim().toLowerCase() || null,
              normalizeBackend: (value) => String(value || '').trim().toLowerCase() || null,
              basenameFromPath: (value) => String(value || '').split(/[\\/]/).filter(Boolean).pop() || '',
            };
            vm.createContext(context);
            vm.runInContext(fs.readFileSync('static/ui/messages.js', 'utf8'), context);

            const artifact = { artifact_ref: 'artifact:index', path: '/artifacts/documents/index.html' };
            const message = {
              artifactBundles: [
                {
                  bundle_id: 'older-exact',
                  copied_artifacts: [
                    {
                      artifact_ref: 'artifact:index',
                      source_path: '/artifacts/documents/index.html',
                      path: '/artifacts/bundles/older/index.html',
                    },
                  ],
                },
                {
                  bundle_id: 'newer-ref-alias',
                  copied_artifacts: [
                    {
                      artifact_ref: 'artifact:index',
                      source_path: '/artifacts/documents/other.html',
                      path: '/artifacts/bundles/newer/other.html',
                    },
                  ],
                },
              ],
            };

            assert.strictEqual(
              context.resolveArtifactHtmlPreviewPath(artifact, message, artifact.path),
              '/artifacts/bundles/older/index.html'
            );
            assert.strictEqual(
              context.buildSavedArtifactPreviewUrl(artifact.path, 'resp test'),
              'http://127.0.0.1:5001/api/preview_saved_artifact?path=%2Fartifacts%2Fdocuments%2Findex.html&response_id=resp%20test'
            );
        """
        subprocess.run(
            ["node", "-e", textwrap.dedent(script)],
            cwd=".",
            check=True,
            text=True,
            capture_output=True,
        )

    def test_ordered_artifact_rendering_preserves_streamed_preview_text(self):
        messages_source = Path("static/ui/messages.js").read_text()
        state_source = Path("static/ui/message-state.js").read_text()

        self.assertIn('function getPreservedAssistantPreviewText', messages_source)
        self.assertIn('assistantPreviewTextIsWorthPreserving', messages_source)
        self.assertIn('chat-message__output-text--preview', messages_source)
        self.assertIn('const preservedPreviewText = getPreservedAssistantPreviewText', messages_source)
        self.assertIn('if (preservedPreviewText)', messages_source)
        self.assertIn('displayTextOutputCount: displayTextOutputs.length', messages_source)
        self.assertIn('assistantPreviewTextIsGenericSummary', messages_source)
        self.assertIn('preferredPreviewText', state_source)
        self.assertIn('assistantPreviewTextIsWorthPreserving(preferredPreviewText)', state_source)

    def test_resolved_compact_work_indicator_uses_success_tone(self):
        messages_source = Path("static/ui/messages.js").read_text()
        css_source = Path("static/ui/ollmo.css").read_text()

        self.assertIn("token === 'active' || token === 'resolved'", messages_source)
        self.assertIn("return 'resolved';", messages_source)
        self.assertIn('TRANSIENT_RESOLVED_WORK_STATUS_VISIBLE_MS = 5000', messages_source)
        self.assertIn('scheduleTransientResolvedWorkStatusDismissal', messages_source)
        self.assertIn('markTransientResolvedWorkStatusDismissed', messages_source)
        self.assertIn('options.transientResolved === true', messages_source)
        self.assertIn('transientResolvedWorkStatusIsDismissed', messages_source)
        self.assertIn('.chat-message-work-compact--resolved', css_source)
        self.assertIn('.chat-message-work-compact--resolved .chat-work-indicator', css_source)
        self.assertIn('.chat-message-work-compact--transient.is-expiring', css_source)

    def test_response_artifact_bundle_ui_action_and_card_are_available(self):
        messages_source = Path("static/ui/messages.js").read_text()
        css_source = Path("static/ui/ollmo.css").read_text()
        history_source = Path("static/ui/settings-history.js").read_text()
        state_source = Path("static/ui/message-state.js").read_text()

        self.assertIn('canBundleAssistantMessage', messages_source)
        self.assertIn('responseArtifactBundleHasActiveWork', messages_source)
        self.assertIn('getLatestOpenableArtifactBundle', messages_source)
        self.assertIn('getResponseArtifactBundleActionState', messages_source)
        self.assertIn("label: existingBundle ? 'Open Bundle' : 'Bundle'", messages_source)
        self.assertIn("mode = existingBundle ? 'open' : 'create'", messages_source)
        self.assertIn('bundleButton.disabled = !bundleAction.enabled', messages_source)
        self.assertIn('/bundle_artifacts', messages_source)
        self.assertIn('Bundle', messages_source)
        self.assertIn('appendResponseArtifactBundleCards', messages_source)
        self.assertIn('openSavedArtifactEntry', messages_source)
        self.assertIn('buildSavedArtifactPreviewUrl', messages_source)
        self.assertIn("buildSavedArtifactUrl(savedPath, 'preview_saved_artifact')", messages_source)
        self.assertIn('artifactIsHtmlLike', messages_source)
        self.assertIn('resolveArtifactHtmlPreviewPath', messages_source)
        self.assertIn('bundle.copied_artifacts', messages_source)
        self.assertIn("buildSavedArtifactViewUrl(savedArtifactPath)", messages_source)
        self.assertNotIn('open_file: true', messages_source)
        self.assertIn('Open Entry', messages_source)
        self.assertIn('chat-artifact-bundle-card', css_source)
        self.assertIn('width: fit-content', css_source)
        self.assertIn('chat-artifact-bundle-card--failed', css_source)
        self.assertIn('artifact_bundles', history_source)
        self.assertIn('status_semantics', history_source)
        self.assertIn('lifecycle_state', history_source)
        self.assertIn('artifactBundles', state_source)

    @unittest.skipIf(shutil.which("node") is None, "node is required for frontend message-state tests")
    def test_response_frame_artifacts_and_outputs_survive_normalization(self):
        script = r"""
            const assert = require('assert');
            const fs = require('fs');
            const vm = require('vm');

            const context = {
              console,
              window: {},
              state: {},
              elements: {},
              normalizeCapability: (value) => String(value || '').trim().toLowerCase() || null,
              normalizeBackend: (value) => String(value || '').trim().toLowerCase() || null,
              basenameFromPath: (value) => String(value || '').split(/[\\/]/).filter(Boolean).pop() || '',
            };
            vm.createContext(context);
            vm.runInContext(fs.readFileSync('static/ui/message-state.js', 'utf8'), context);

            const frame = {
              artifacts: {
                dossiers: {
                  'artifact:image_one': {
                    artifact: {
                      type: 'image',
                      kind: 'image',
                      path: '/tmp/artifacts/images/one.png',
                      artifact_ref: 'artifact:image_one',
                      artifact_id: 'image_one',
                      origin: 'assistant_output',
                    },
                    metadata: { availability: 'available' },
                  },
                  'artifact:image_two': {
                    artifact: {
                      type: 'image',
                      kind: 'image',
                      path: '/tmp/artifacts/images/two.png',
                      artifact_ref: 'artifact:image_two',
                      artifact_id: 'image_two',
                      origin: 'assistant_output',
                    },
                    metadata: { availability: 'available' },
                  },
                },
                output: [
                  {
                    type: 'image',
                    path: '/tmp/artifacts/images/one.png',
                    artifact_ref: 'artifact:image_one',
                  },
                  {
                    type: 'image',
                    path: '/tmp/artifacts/images/two.png',
                    artifact_ref: 'artifact:image_two',
                  },
                ],
              },
              planning: {
                artifact_flow: {
                  output_slots: [
                    { slot_id: 'output-text', type: 'text', status: 'fulfilled' },
                    { slot_id: 'output-image-1', type: 'image', status: 'fulfilled', artifact_ref: 'artifact:image_one' },
                    { slot_id: 'output-image-2', type: 'image', status: 'fulfilled', artifact_ref: 'artifact:image_two' },
                  ],
                },
              },
              output: {
                outputs: [
                  { slot_id: 'output-text', type: 'text', status: 'fulfilled', value: 'Two image prompts.' },
                  {
                    slot_id: 'output-image-1',
                    type: 'image',
                    status: 'fulfilled',
                    artifact_ref: 'artifact:image_one',
                    artifacts: [{ type: 'image', path: '/tmp/artifacts/images/one.png', artifact_ref: 'artifact:image_one' }],
                  },
                  {
                    slot_id: 'output-image-2',
                    type: 'image',
                    status: 'fulfilled',
                    artifact_ref: 'artifact:image_two',
                    artifacts: [{ type: 'image', path: '/tmp/artifacts/images/two.png', artifact_ref: 'artifact:image_two' }],
                  },
                ],
              },
            };

            const expectPaths = (actual, expected) => {
              assert.strictEqual(JSON.stringify(Array.from(actual).sort()), JSON.stringify([...expected].sort()));
            };

            const artifacts = context.sanitizeResponseArtifacts(frame.artifacts);
            expectPaths(Array.from(artifacts, (artifact) => artifact.path), [
              '/tmp/artifacts/images/one.png',
              '/tmp/artifacts/images/two.png',
            ]);

            const outputSlots = context.extractResponseOutputSlots(frame);
            assert.strictEqual(outputSlots.length, 3);

            const outputs = context.extractResponseOutputs(frame, { artifacts, outputSlots });
            assert.strictEqual(outputs.length, 3);
            expectPaths(
              Array.from(outputs.flatMap((output) => output.artifacts || []), (artifact) => artifact.path),
              ['/tmp/artifacts/images/one.png', '/tmp/artifacts/images/two.png'],
            );

            const message = { role: 'assistant', artifacts: [], outputs };
            context.syncMessageArtifactLedger(message);
            expectPaths(Array.from(message.artifacts, (artifact) => artifact.path), [
              '/tmp/artifacts/images/one.png',
              '/tmp/artifacts/images/two.png',
            ]);
        """
        subprocess.run(
            ["node", "-e", textwrap.dedent(script)],
            cwd=".",
            check=True,
            text=True,
            capture_output=True,
        )

    @unittest.skipIf(shutil.which("node") is None, "node is required for frontend message-state tests")
    def test_response_truth_reconciler_rejects_older_frame_sequence(self):
        script = r"""
            const assert = require('assert');
            const fs = require('fs');
            const vm = require('vm');

            let persisted = 0;
            const context = {
              console,
              window: {},
              state: {
                arena: { enabled: false },
                conversations: {
                  'conv-1': [
                    {
                      role: 'assistant',
                      content: 'Done.',
                      responseId: 'resp-truth',
                      responseFrameSequence: 4,
                      responseFrameId: 'resp-truth:frame-4',
                      lifecycleState: 'completed',
                      statusSemantics: { canonicalLifecycleState: 'completed', hasOpenContinuation: false },
                      lateFill: { status: 'completed' },
                      artifacts: [],
                      outputs: [],
                      outputSlots: [],
                      outputBranches: [],
                    },
                  ],
                },
              },
              elements: {},
              normalizeCapability: (value) => String(value || '').trim().toLowerCase() || null,
              normalizeBackend: (value) => String(value || '').trim().toLowerCase() || null,
              basenameFromPath: (value) => String(value || '').split(/[\\/]/).filter(Boolean).pop() || '',
              getConversationInstanceMeta: () => null,
              getRequestExecutionInstance: (value) => value,
              getInstanceMeta: () => null,
              mergeRequestSnapshotInputArtifacts: (snapshot) => snapshot || null,
              formatMessageHtml: (value) => value,
              renderConversationHistoryList: () => {},
              renderArenaConversations: () => {},
              isConversationVisible: () => false,
              renderConversation: () => {},
              persistChatHistory: () => { persisted += 1; },
            };
            vm.createContext(context);
            vm.runInContext(fs.readFileSync('static/ui/message-state.js', 'utf8'), context);

            const staleApplied = context.updateAssistantResponseByResponseId(
              'conv-1',
              'resp-truth',
              {
                id: 'resp-truth',
                status: 'completed',
                lifecycle_state: 'late_fill_running',
                status_semantics: {
                  canonical_lifecycle_state: 'late_fill_running',
                  has_open_continuation: true,
                },
                late_fill: { status: 'running' },
                response_frame: { frame_sequence: 2, frame_id: 'resp-truth:frame-2' },
                output_text: 'Old running projection.',
              },
              null
            );

            assert.strictEqual(staleApplied, false);
            assert.strictEqual(context.state.conversations['conv-1'][0].responseFrameSequence, 4);
            assert.strictEqual(context.state.conversations['conv-1'][0].lifecycleState, 'completed');
            assert.strictEqual(context.state.conversations['conv-1'][0].lateFill.status, 'completed');
            assert.strictEqual(persisted, 0);

            const newerApplied = context.updateAssistantResponseByResponseId(
              'conv-1',
              'resp-truth',
              {
                id: 'resp-truth',
                status: 'completed',
                lifecycle_state: 'completed',
                status_semantics: {
                  canonical_lifecycle_state: 'completed',
                  has_open_continuation: false,
                  is_terminal: true,
                },
                late_fill: { status: 'completed' },
                response_frame: { frame_sequence: 5, frame_id: 'resp-truth:frame-5' },
                outputs: [{ type: 'text', status: 'fulfilled', value: 'Final truth.' }],
                output_text: 'Final truth.',
              },
              null
            );

            assert.strictEqual(newerApplied, true);
            assert.strictEqual(context.state.conversations['conv-1'][0].responseFrameSequence, 5);
            assert.strictEqual(context.state.conversations['conv-1'][0].responseFrameId, 'resp-truth:frame-5');
            assert.strictEqual(context.state.conversations['conv-1'][0].content, 'Final truth.');
            assert.strictEqual(context.state.conversations['conv-1'][0].lifecycleState, 'completed');
            assert.strictEqual(persisted, 1);

            context.state.conversations['conv-2'] = [
              {
                role: 'assistant',
                content: 'Text blocked.',
                responseId: 'resp-petsie-like',
                artifacts: [{ type: 'text', path: '/tmp/index.html', artifact_ref: 'artifact:index' }],
                outputSlots: [
                  { slot_id: 'output-phase-6', branch_id: 'branch-chat-1', phase_id: 'phase-6', type: 'text', status: 'fulfilled' },
                  { slot_id: 'output-phase-7', branch_id: 'branch-text_artifact-1', phase_id: 'phase-7', type: 'text', status: 'blocked' },
                  { slot_id: 'output-repair-chat', branch_id: 'repair-chat', phase_id: 'repair-chat', type: 'text', status: 'pending' },
                ],
                outputs: [
                  { slot_id: 'output-phase-7', branch_id: 'branch-text_artifact-1', phase_id: 'phase-7', type: 'text', status: 'blocked' },
                  { slot_id: 'output-repair-chat', branch_id: 'repair-chat', phase_id: 'repair-chat', type: 'text', status: 'pending' },
                ],
              },
            ];
            const terminalApplied = context.updateAssistantResponseByResponseId(
              'conv-2',
              'resp-petsie-like',
              {
                id: 'resp-petsie-like',
                status: 'completed',
                lifecycle_state: 'completed',
                status_semantics: {
                  canonical_lifecycle_state: 'completed',
                  compatibility_status: 'completed',
                  has_open_continuation: false,
                  has_actionable_repair: false,
                  is_terminal: true,
                },
                late_fill: { status: 'completed' },
                status_lookup: { state_version: 'version-terminal', frame_sequence: 4, frame_id: 'resp-petsie-like:frame-4' },
                output_slots: [
                  { slot_id: 'output-phase-6', branch_id: 'branch-chat-1', phase_id: 'phase-6', type: 'text', status: 'fulfilled' },
                  { slot_id: 'output-phase-7', branch_id: 'branch-text_artifact-1', phase_id: 'phase-7', type: 'text', status: 'fulfilled' },
                  { slot_id: 'output-repair-chat', branch_id: 'repair-chat', phase_id: 'repair-chat', type: 'text', status: 'fulfilled' },
                ],
                outputs: [
                  { slot_id: 'output-phase-6', branch_id: 'branch-chat-1', phase_id: 'phase-6', type: 'text', status: 'fulfilled', value: 'Final page.' },
                  { slot_id: 'output-phase-7', branch_id: 'branch-text_artifact-1', phase_id: 'phase-7', type: 'text', status: 'fulfilled' },
                  { slot_id: 'output-repair-chat', branch_id: 'repair-chat', phase_id: 'repair-chat', type: 'text', status: 'fulfilled' },
                ],
                output_text: 'Final page.',
              },
              null
            );

            assert.strictEqual(terminalApplied, true);
            assert.strictEqual(context.state.conversations['conv-2'][0].responseStateVersion, 'version-terminal');
            assert.strictEqual(
              JSON.stringify(context.state.conversations['conv-2'][0].outputSlots.map((slot) => slot.status)),
              JSON.stringify(['fulfilled', 'fulfilled', 'fulfilled'])
            );
            assert.strictEqual(
              JSON.stringify(context.formatAssistantCurrentWorkStatusLines(context.state.conversations['conv-2'][0])),
              JSON.stringify([])
            );
        """
        subprocess.run(
            ["node", "-e", textwrap.dedent(script)],
            cwd=".",
            check=True,
            text=True,
            capture_output=True,
        )

    @unittest.skipIf(shutil.which("node") is None, "node is required for frontend message-state tests")
    def test_pushed_response_truth_hydrates_loading_message_status(self):
        script = r"""
            const assert = require('assert');
            const fs = require('fs');
            const vm = require('vm');

            let renderCount = 0;
            const context = {
              console,
              window: {},
              setTimeout: () => 'timer',
              clearTimeout: () => {},
              state: {
                arena: { enabled: false },
                streamDomPatches: {},
                conversations: {
                  'conv-live': [
                    {
                      role: 'assistant',
                      isLoading: true,
                      clientMessageId: 'client-live',
                      responseId: 'resp-live',
                      content: '<span class="loading-dots">Thinking</span>',
                      trustedHtml: true,
                    },
                  ],
                },
              },
              elements: {},
              normalizeCapability: (value) => String(value || '').trim().toLowerCase() || null,
              normalizeBackend: (value) => String(value || '').trim().toLowerCase() || null,
              basenameFromPath: (value) => String(value || '').split(/[\\/]/).filter(Boolean).pop() || '',
              getConversationInstanceMeta: () => null,
              getRequestExecutionInstance: (value) => value,
              getInstanceMeta: () => null,
              mergeRequestSnapshotInputArtifacts: (snapshot) => snapshot || null,
              formatMessageHtml: (value) => value,
              renderConversationHistoryList: () => {},
              renderArenaConversations: () => {},
              isConversationVisible: () => true,
              scrollConversationToBottom: () => {},
              renderConversation: () => { renderCount += 1; },
              persistChatHistory: () => {},
            };
            vm.createContext(context);
            vm.runInContext(fs.readFileSync('static/ui/message-state.js', 'utf8'), context);

            const payload = {
              id: 'resp-live',
              status: 'completed',
              lifecycle_state: 'late_fill_pending',
              status_semantics: {
                canonical_lifecycle_state: 'late_fill_pending',
                compatibility_status: 'completed',
                has_open_continuation: true,
                has_actionable_repair: false,
                is_terminal: false,
              },
              late_fill: {
                status: 'pending',
                expected_capability: 'image_generation',
                missing_artifact_type: 'image',
                pending_branches: [
                  { branch_id: 'branch-image-1', phase_id: 'phase-2', capability: 'image_generation', output_type: 'image', status: 'pending' },
                ],
              },
              output_slots: [
                { slot_id: 'output-phase-1', type: 'document', status: 'fulfilled', lifecycle: 'materialized_output' },
                { slot_id: 'output-phase-2', branch_id: 'branch-image-1', phase_id: 'phase-2', type: 'image', status: 'pending', lifecycle: 'deferred_output' },
              ],
              output_text: 'Generated landing page text.',
            };

            assert.strictEqual(
              context.updateLoadingAssistantMessageFromResponsePayload(
                'conv-live',
                'resp-live',
                payload,
                null,
                {
                  clientMessageId: 'client-live',
                  content: 'Generated landing page text.',
                  forceRender: true,
                }
              ),
              true
            );

            const message = context.state.conversations['conv-live'][0];
            assert.strictEqual(message.isLoading, true);
            assert.strictEqual(message.content, 'Generated landing page text.');
            assert.strictEqual(message.lateFill.status, 'pending');
            assert.strictEqual(message.outputSlots[1].status, 'pending');
            assert.strictEqual(
              JSON.stringify(context.formatAssistantCurrentWorkStatusLines(message)),
              JSON.stringify(['Image queued for late fill...'])
            );
            assert.strictEqual(renderCount, 1);

            context.updateLoadingAssistantMessage(
              'conv-live',
              'Generated landing page text with another delta.',
              { clientMessageId: 'client-live' }
            );
            assert.strictEqual(
              JSON.stringify(context.formatAssistantCurrentWorkStatusLines(context.state.conversations['conv-live'][0])),
              JSON.stringify(['Image queued for late fill...'])
            );
        """
        subprocess.run(
            ["node", "-e", textwrap.dedent(script)],
            cwd=".",
            check=True,
            text=True,
            capture_output=True,
        )

    @unittest.skipIf(shutil.which("node") is None, "node is required for frontend message-state tests")
    def test_ordered_artifact_output_text_projection_hides_internal_surfaces(self):
        script = r"""
            const assert = require('assert');
            const fs = require('fs');
            const vm = require('vm');

            const context = {
              console,
              window: {},
              state: {},
              elements: {},
              normalizeCapability: (value) => String(value || '').trim().toLowerCase() || null,
              normalizeBackend: (value) => String(value || '').trim().toLowerCase() || null,
              basenameFromPath: (value) => String(value || '').split(/[\\/]/).filter(Boolean).pop() || '',
            };
            vm.createContext(context);
            vm.runInContext(fs.readFileSync('static/ui/message-state.js', 'utf8'), context);

            const imageOutput = {
              type: 'image',
              status: 'fulfilled',
              artifact_ref: 'artifact:image_one',
              artifacts: [
                { type: 'image', path: '/tmp/artifacts/images/one.png', artifact_ref: 'artifact:image_one' },
              ],
            };
            const prepOutput = {
              type: 'document',
              status: 'fulfilled',
              value: 'Image Generation Prompts\n1. A luminous harbor at sunset.',
            };
            const textArtifactOutput = {
              type: 'text',
              status: 'fulfilled',
              artifact_ref: 'artifact:index_html',
              value: '<!DOCTYPE html>\n<html><body>Saved page.</body></html>',
              artifacts: [
                { type: 'text', path: '/tmp/artifacts/documents/index.html', artifact_ref: 'artifact:index_html' },
              ],
            };
            const repairOutput = {
              type: 'text',
              status: 'fulfilled',
              value: 'Target text artifact: artifacts/documents/index.html\nDeterministic syntax sanity issues:\n- placeholder image link remains',
            };
            const repairChatOutput = {
              slot_id: 'output-repair-chat',
              branch_id: 'repair-chat',
              phase_id: 'repair-chat',
              type: 'text',
              status: 'fulfilled',
              value: 'Internal repaired HTML content.',
            };
            const finalOutput = {
              type: 'text',
              status: 'fulfilled',
              value: 'Done. The saved page and image artifacts are ready.',
            };

            assert.strictEqual(
              context.shouldRenderAssistantOutputText(prepOutput, [prepOutput, imageOutput], null),
              false
            );
            assert.strictEqual(
              context.shouldRenderAssistantOutputText(textArtifactOutput, [textArtifactOutput, imageOutput], null),
              false
            );
            assert.strictEqual(
              context.shouldRenderAssistantOutputText(repairOutput, [repairOutput, imageOutput], null),
              false
            );
            assert.strictEqual(context.responseWorkItemIsInternalProjection(repairChatOutput), true);
            assert.strictEqual(
              JSON.stringify(context.filterUserVisibleResponseWorkItems([textArtifactOutput, repairChatOutput]).map((item) => item.artifact_ref || '')),
              JSON.stringify(['artifact:index_html'])
            );
            assert.strictEqual(
              context.shouldRenderAssistantOutputText(finalOutput, [finalOutput, imageOutput], null),
              true
            );
            assert.strictEqual(
              context.shouldRenderAssistantOutputText(prepOutput, [prepOutput], null),
              true
            );
        """
        subprocess.run(
            ["node", "-e", textwrap.dedent(script)],
            cwd=".",
            check=True,
            text=True,
            capture_output=True,
        )

    @unittest.skipIf(shutil.which("node") is None, "node is required for frontend message-state tests")
    def test_same_path_audio_artifacts_dedupe_with_partial_mime(self):
        script = r"""
            const assert = require('assert');
            const fs = require('fs');
            const vm = require('vm');

            const context = {
              console,
              window: {},
              state: {},
              elements: {},
              normalizeCapability: (value) => String(value || '').trim().toLowerCase() || null,
              normalizeBackend: (value) => String(value || '').trim().toLowerCase() || null,
              basenameFromPath: (value) => String(value || '').split(/[\\/]/).filter(Boolean).pop() || '',
            };
            vm.createContext(context);
            vm.runInContext(fs.readFileSync('static/ui/message-state.js', 'utf8'), context);

            const artifacts = context.sanitizeResponseArtifacts([
              {
                type: 'audio',
                path: '/tmp/artifacts/audio/out.wav',
                artifact_ref: 'artifact:audio_first',
                mime_type: 'audio/wav',
              },
              {
                type: 'audio',
                path: '/tmp/artifacts/audio/out.wav',
                artifact_ref: 'artifact:audio_second',
              },
            ]);

            assert.strictEqual(artifacts.length, 1);
            assert.strictEqual(artifacts[0].path, '/tmp/artifacts/audio/out.wav');
            assert.strictEqual(artifacts[0].mime_type, 'audio/wav');
        """
        subprocess.run(
            ["node", "-e", textwrap.dedent(script)],
            cwd=".",
            check=True,
            text=True,
            capture_output=True,
        )

    @unittest.skipIf(shutil.which("node") is None, "node is required for frontend message-state tests")
    def test_current_work_status_lists_multiple_active_slots(self):
        script = r"""
            const assert = require('assert');
            const fs = require('fs');
            const vm = require('vm');

            const context = {
              console,
              window: {},
              state: {},
              elements: {},
              normalizeCapability: (value) => String(value || '').trim().toLowerCase() || null,
              normalizeBackend: (value) => String(value || '').trim().toLowerCase() || null,
              basenameFromPath: (value) => String(value || '').split(/[\\/]/).filter(Boolean).pop() || '',
            };
            vm.createContext(context);
            vm.runInContext(fs.readFileSync('static/ui/message-state.js', 'utf8'), context);

            const message = {
              lateFill: { status: 'pending', missingArtifactType: 'image' },
              outputSlots: [
                { slot_id: 'image-1', type: 'image', status: 'pending' },
                { slot_id: 'audio-1', type: 'audio', status: 'running' },
              ],
            };

            assert.strictEqual(
              JSON.stringify(context.formatAssistantCurrentWorkStatusLines(message)),
              JSON.stringify([
                'Image queued for late fill...',
                'Audio still generating...',
              ])
            );
            assert.strictEqual(
              context.formatAssistantLateFillStatusText(message),
              'Image queued for late fill... · Audio still generating...'
            );

            const displayContent = context.getAssistantDisplayContent({
              late_fill: { status: 'pending', missing_artifact_type: 'image' },
              output_slots: [
                { slot_id: 'image-1', type: 'image', status: 'pending' },
                { slot_id: 'audio-1', type: 'audio', status: 'pending' },
              ],
            });
            assert.strictEqual(
              displayContent,
              'Image queued for late fill... Audio queued for late fill...'
            );

            const staleMessage = {
              outputSlots: [
                { slot_id: 'text-1', type: 'text', status: 'pending' },
              ],
            };
            assert.strictEqual(
              JSON.stringify(context.formatAssistantCurrentWorkStatusLines(staleMessage)),
              JSON.stringify(['Text output still open.'])
            );
            assert.strictEqual(
              context.getAssistantDisplayContent({
                output_slots: [
                  { slot_id: 'text-1', type: 'text', status: 'pending' },
                ],
              }),
              'Text output still open.'
            );

            const completedLateFillMessage = {
              lateFill: {
                status: 'completed',
                completedBranches: [
                  { branch_id: 'branch-image-1', phase_id: 'phase-2', type: 'image', status: 'fulfilled' },
                ],
              },
              outputSlots: [
                { slot_id: 'output-phase-2', branch_id: 'branch-image-1', phase_id: 'phase-2', type: 'image', status: 'pending' },
              ],
            };
            assert.strictEqual(
              JSON.stringify(context.formatAssistantCurrentWorkStatusLines(completedLateFillMessage)),
              JSON.stringify([])
            );

            const completedTerminalTruthWithStalePendingMessage = {
              lifecycleState: 'completed',
              statusSemantics: {
                canonical_lifecycle_state: 'completed',
                has_open_continuation: false,
                has_actionable_repair: false,
                is_terminal: true,
              },
              lateFill: {
                status: 'completed',
                completedBranches: [
                  { branch_id: 'branch-image_generation-5', phase_id: 'phase-6', capability: 'image_generation', type: 'image', status: 'fulfilled' },
                ],
              },
              surfaceState: {
                status: 'pending',
                category_counts: {
                  open: 2,
                  late_fill_pending: 2,
                },
              },
              outputSlots: [
                { slot_id: 'output-phase-2', branch_id: 'branch-image_generation-1', phase_id: 'phase-2', type: 'image', status: 'pending' },
                { slot_id: 'output-phase-4', branch_id: 'branch-chat-1', phase_id: 'phase-4', type: 'text', status: 'running' },
              ],
              outputs: [
                { id: 'output-phase-2', branch_id: 'branch-image_generation-1', phase_id: 'phase-2', type: 'image', status: 'pending' },
              ],
            };
            assert.strictEqual(
              JSON.stringify(context.formatAssistantCurrentWorkStatusLines(completedTerminalTruthWithStalePendingMessage)),
              JSON.stringify([])
            );
            assert.ok(
              !context.getAssistantDisplayContent(completedTerminalTruthWithStalePendingMessage).includes('in progress')
            );

            const completedTerminalTruthWithStaleBlockedHistoryMessage = {
              lifecycleState: 'completed',
              statusSemantics: {
                canonical_lifecycle_state: 'completed',
                compatibility_status: 'completed',
                has_open_continuation: false,
                has_actionable_repair: false,
                is_terminal: true,
              },
              lateFill: {
                status: 'completed',
                completedBranches: [
                  { branch_id: 'branch-chat-1', phase_id: 'phase-6', type: 'text', status: 'fulfilled' },
                  { branch_id: 'repair-chat', phase_id: 'repair-chat', type: 'text', status: 'fulfilled' },
                ],
              },
              outputSlots: [
                { slot_id: 'output-phase-6', branch_id: 'branch-chat-1', phase_id: 'phase-6', type: 'text', status: 'fulfilled' },
                { slot_id: 'output-phase-7', branch_id: 'branch-text_artifact-1', phase_id: 'phase-7', type: 'text', status: 'blocked', lifecycle: 'blocked_output' },
                { slot_id: 'output-repair-chat', branch_id: 'repair-chat', phase_id: 'repair-chat', type: 'text', status: 'pending', lifecycle: 'deferred_output' },
              ],
              outputs: [
                { slot_id: 'output-phase-7', branch_id: 'branch-text_artifact-1', phase_id: 'phase-7', type: 'text', status: 'blocked', lifecycle: 'blocked_output' },
                { slot_id: 'output-repair-chat', branch_id: 'repair-chat', phase_id: 'repair-chat', type: 'text', status: 'pending', lifecycle: 'deferred_output' },
              ],
            };
            assert.strictEqual(
              JSON.stringify(context.formatAssistantCurrentWorkStatusLines(completedTerminalTruthWithStaleBlockedHistoryMessage)),
              JSON.stringify([])
            );

            const fulfilledOutputsWithStaleActiveLateFillMessage = {
              lateFill: {
                status: 'running',
                activeBranches: [
                  { branch_id: 'branch-text_artifact-1', phase_id: 'phase-5', type: 'text', status: 'running' },
                ],
                materialization_contract_open_check_count: 4,
              },
              outputSlots: [
                { slot_id: 'output-phase-1', branch_id: 'phase-1', phase_id: 'phase-1', type: 'document', status: 'fulfilled' },
                { slot_id: 'output-phase-2', branch_id: 'branch-image_generation-1', phase_id: 'phase-2', type: 'image', status: 'fulfilled' },
                { slot_id: 'output-phase-3', branch_id: 'branch-image_generation-2', phase_id: 'phase-3', type: 'image', status: 'fulfilled' },
                { slot_id: 'output-phase-4', branch_id: 'branch-image_generation-3', phase_id: 'phase-4', type: 'image', status: 'fulfilled' },
                { slot_id: 'output-phase-5', branch_id: 'branch-text_artifact-1', phase_id: 'phase-5', type: 'text', status: 'fulfilled' },
                { slot_id: 'output-phase-6', branch_id: 'branch-text_artifact-2', phase_id: 'phase-6', type: 'text', status: 'fulfilled' },
                { slot_id: 'output-repair-chat', branch_id: 'repair-chat', phase_id: 'repair-chat', type: 'text', status: 'fulfilled' },
              ],
              outputs: [
                { slot_id: 'output-phase-1', branch_id: 'phase-1', phase_id: 'phase-1', type: 'document', status: 'fulfilled' },
                { slot_id: 'output-phase-5', branch_id: 'branch-text_artifact-1', phase_id: 'phase-5', type: 'text', status: 'fulfilled' },
                { slot_id: 'output-repair-chat', branch_id: 'repair-chat', phase_id: 'repair-chat', type: 'text', status: 'fulfilled' },
              ],
            };
            assert.strictEqual(
              JSON.stringify(context.formatAssistantCurrentWorkStatusLines(fulfilledOutputsWithStaleActiveLateFillMessage)),
              JSON.stringify([])
            );
            assert.strictEqual(
              context.responseMessageOutputContractIndicatesCleanResolution(fulfilledOutputsWithStaleActiveLateFillMessage),
              true
            );

            const terminalRepairAttentionMessage = {
              lifecycleState: 'completed',
              statusSemantics: {
                canonical_lifecycle_state: 'completed',
                has_open_continuation: false,
                has_actionable_repair: true,
                is_terminal: true,
              },
              lateFill: {
                status: 'completed',
                final_materialization_contract_status: 'unmet',
                materialization_contract_open_check_count: 2,
              },
              surfaceState: {
                status: 'blocked',
                category_counts: {
                  repair_pending: 1,
                },
              },
              outputSlots: [
                { slot_id: 'output-phase-2', branch_id: 'branch-image_generation-1', phase_id: 'phase-2', type: 'image', status: 'pending' },
              ],
            };
            assert.strictEqual(
              JSON.stringify(context.formatAssistantCurrentWorkStatusLines(terminalRepairAttentionMessage)),
              JSON.stringify(['Image output still open.', 'Repair needed: 2 open checks.', 'Repair pending: 1.'])
            );

            const staleSurfaceCompletedMessage = {
              lateFill: {
                status: 'completed',
                completedBranches: [
                  { branch_id: 'branch-audio-1', phase_id: 'phase-2', type: 'audio', status: 'fulfilled' },
                ],
                surfaceState: {
                  status: 'pending',
                  category_counts: {
                    open: 1,
                    late_fill_pending: 1,
                    controlled_attention_pending: 56,
                    aspiration_pending: 6,
                    commitment_pending: 11,
                  },
                },
              },
              outputSlots: [
                { slot_id: 'output-phase-1', branch_id: 'phase-1', phase_id: 'phase-1', type: 'text', status: 'fulfilled' },
                { slot_id: 'output-phase-2', branch_id: 'branch-audio-1', phase_id: 'phase-2', type: 'audio', status: 'fulfilled' },
              ],
            };
            assert.strictEqual(
              JSON.stringify(context.formatAssistantCurrentWorkStatusLines(staleSurfaceCompletedMessage)),
              JSON.stringify([])
            );

            const staleActiveCompletedMessage = {
              lateFill: {
                status: 'completed',
                activeBranches: [
                  { branch_id: 'branch-image-1', phase_id: 'phase-2', type: 'image', status: 'planned' },
                ],
                completedBranches: [
                  { branch_id: 'branch-image-1', phase_id: 'phase-2', type: 'image', status: 'planned' },
                ],
              },
              outputSlots: [
                { slot_id: 'output-phase-2', branch_id: 'branch-image-1', phase_id: 'phase-2', type: 'image', status: 'fulfilled' },
              ],
            };
            assert.strictEqual(
              JSON.stringify(context.formatAssistantCurrentWorkStatusLines(staleActiveCompletedMessage)),
              JSON.stringify([])
            );

            const terminalPlaceholderMessage = {
              lateFill: { status: 'completed' },
              outputSlots: [
                {
                  slot_id: 'output-phase-2',
                  branch_id: 'branch-audio-1',
                  phase_id: 'phase-2',
                  type: 'audio',
                  status: 'fulfilled',
                  lifecycle: 'materialized_output',
                  placeholder_ref: 'pending-output-branch-audio-1',
                },
              ],
              outputBranches: [
                {
                  slot_id: 'output-phase-2',
                  branch_id: 'branch-audio-1',
                  phase_id: 'phase-2',
                  type: 'audio',
                  status: 'fulfilled',
                  lifecycle: 'materialized_output',
                  placeholder_ref: 'pending-output-branch-audio-1',
                },
              ],
            };
            assert.strictEqual(
              context.sanitizeResponseOutputSlots(terminalPlaceholderMessage.outputSlots)[0].placeholder_ref,
              null
            );
            assert.strictEqual(
              context.sanitizeResponseOutputBranches(terminalPlaceholderMessage.outputBranches)[0].placeholder_ref,
              null
            );
            assert.strictEqual(
              JSON.stringify(context.formatAssistantCurrentWorkStatusLines(terminalPlaceholderMessage)),
              JSON.stringify([])
            );

            const blockedLateFillMessage = {
              lateFill: {
                status: 'failed',
                failedBranches: [
                  {
                    branch_id: 'branch-audio-1',
                    phase_id: 'phase-3',
                    type: 'audio',
                    status: 'blocked',
                    blocked_reason: 'missing dependency audio',
                  },
                ],
              },
              outputSlots: [
                { slot_id: 'output-phase-3', branch_id: 'branch-audio-1', phase_id: 'phase-3', type: 'audio', status: 'pending' },
              ],
            };
            assert.strictEqual(
              JSON.stringify(context.formatAssistantCurrentWorkStatusLines(blockedLateFillMessage)),
              JSON.stringify(['Audio blocked: missing dependency audio'])
            );

            const failedLateFillMessage = {
              lateFill: {
                status: 'failed',
                failedBranches: [
                  {
                    branch_id: 'branch-image-1',
                    phase_id: 'phase-2',
                    type: 'image',
                    status: 'failed',
                    error_ref: { code: 'DEPENDENCY_CHAIN_REPAIR_REQUIRED' },
                  },
                ],
              },
              outputSlots: [
                { slot_id: 'output-phase-2', branch_id: 'branch-image-1', phase_id: 'phase-2', type: 'image', status: 'pending' },
              ],
            };
            const failedLines = context.formatAssistantCurrentWorkStatusLines(failedLateFillMessage);
            assert.ok(failedLines.some((line) => line.includes('late fill failed')));
            assert.ok(!failedLines.join(' ').includes('queued'));
            assert.ok(!failedLines.join(' ').includes('still generating'));

            const waivedSupersededMessage = {
              lateFill: {
                status: 'completed',
                completedBranches: [
                  {
                    branch_id: 'branch-image-waived',
                    phase_id: 'phase-waived',
                    type: 'image',
                    status: 'waived',
                    waiver_reason: 'user accepted text only',
                  },
                  {
                    branch_id: 'branch-audio-superseded',
                    phase_id: 'phase-superseded',
                    type: 'audio',
                    status: 'superseded',
                    supersession_reason: 'newer branch fulfilled the obligation',
                  },
                ],
              },
            };
            assert.strictEqual(
              JSON.stringify(context.formatAssistantCurrentWorkStatusLines(waivedSupersededMessage)),
              JSON.stringify([
                'Image waived: user accepted text only',
                'Audio superseded: newer branch fulfilled the obligation',
              ])
            );

            const repairNeededMessage = {
              surfaceState: {
                status: 'blocked',
                category_counts: {
                  repair_pending: 1,
                },
              },
            };
            assert.strictEqual(
              JSON.stringify(context.formatAssistantCurrentWorkStatusLines(repairNeededMessage)),
              JSON.stringify(['Repair pending: 1.'])
            );

            const controlledAttentionMessage = {
              surfaceState: {
                status: 'pending',
                category_counts: {
                  controlled_attention_advisory: 2,
                  aspiration_advisory: 1,
                  commitment_advisory: 1,
                },
              },
            };
            assert.strictEqual(
              JSON.stringify(context.formatAssistantCurrentWorkStatusLines(controlledAttentionMessage)),
              JSON.stringify([])
            );
        """
        subprocess.run(
            ["node", "-e", textwrap.dedent(script)],
            cwd=".",
            check=True,
            text=True,
            capture_output=True,
        )

    @unittest.skipIf(shutil.which("node") is None, "node is required for frontend message-state tests")
    def test_current_work_status_lists_cancelled_late_fill_branch(self):
        script = r"""
            const assert = require('assert');
            const fs = require('fs');
            const vm = require('vm');

            const context = {
              console,
              window: {},
              state: {},
              elements: {},
              normalizeCapability: (value) => String(value || '').trim().toLowerCase() || null,
              normalizeBackend: (value) => String(value || '').trim().toLowerCase() || null,
              basenameFromPath: (value) => String(value || '').split(/[\\/]/).filter(Boolean).pop() || '',
            };
            vm.createContext(context);
            vm.runInContext(fs.readFileSync('static/ui/message-state.js', 'utf8'), context);

            const message = {
              lateFill: {
                status: 'cancelled',
                cancelledBranches: [
                  {
                    branch_id: 'branch-image-1',
                    phase_id: 'phase-2',
                    type: 'image',
                    status: 'cancelled',
                    cancel_reason: 'not needed',
                  },
                ],
              },
            };

            assert.strictEqual(
              JSON.stringify(context.formatAssistantCurrentWorkStatusLines(message)),
              JSON.stringify(['Image cancelled: not needed'])
            );
            assert.strictEqual(context.lateFillStatusIsActive(message.lateFill), false);
        """
        subprocess.run(
            ["node", "-e", textwrap.dedent(script)],
            cwd=".",
            check=True,
            text=True,
            capture_output=True,
        )

    @unittest.skipIf(shutil.which("node") is None, "node is required for frontend message-state tests")
    def test_resolved_linked_artifact_rebind_is_compact_and_workspace_relative(self):
        script = r"""
            const assert = require('assert');
            const fs = require('fs');
            const vm = require('vm');

            const context = {
              console,
              window: {},
              state: {},
              elements: {},
              normalizeCapability: (value) => String(value || '').trim().toLowerCase() || null,
              normalizeBackend: (value) => String(value || '').trim().toLowerCase() || null,
              basenameFromPath: (value) => String(value || '').split(/[\\/]/).filter(Boolean).pop() || '',
            };
            vm.createContext(context);
            vm.runInContext(fs.readFileSync('static/ui/message-state.js', 'utf8'), context);

            const lateFill = context.sanitizeMessageLateFill({
              status: 'completed',
              linked_artifact_rebind_status: 'applied',
              linked_artifact_rebinds: [
                {
                  status: 'applied',
                  target_path: '/Users/example/Projects/ollmo/artifacts/documents/index.html',
                  change_count: 2,
                  changes: [
                    {
                      kind: 'attribute_link',
                      from: 'abyss7_station.png',
                      to: '../images/aethelgard.png',
                      linked_path: '/Users/example/Projects/ollmo/artifacts/images/aethelgard.png',
                    },
                  ],
                },
              ],
            });

            assert.strictEqual(
              context.formatAssistantCurrentWorkStatusLines({ lateFill }).join(' | '),
              'Unresolved bindings resolved: 2 links.'
            );
            assert.strictEqual(lateFill.linkedArtifactRebinds[0].targetPath, 'artifacts/documents/index.html');
            assert.strictEqual(lateFill.linkedArtifactRebinds[0].changes[0].linkedPath, 'artifacts/images/aethelgard.png');
            assert.strictEqual(
              context.isResolvedLinkedArtifactBindingOutput(
                {
                  value: 'Target text artifact: artifacts/documents/index.html\nUnresolved linked artifact binding:\n- text artifact still contains placeholder\nResolved runtime artifacts:\n- image: artifacts/images/aethelgard.png\nUpdate only the target text artifact.',
                },
                lateFill
              ),
              true
            );
        """
        subprocess.run(
            ["node", "-e", textwrap.dedent(script)],
            cwd=".",
            check=True,
            text=True,
            capture_output=True,
        )

    @unittest.skipIf(shutil.which("node") is None, "node is required for frontend message-state tests")
    def test_late_fill_cancel_control_describes_exact_branch_target(self):
        script = r"""
            const assert = require('assert');
            const fs = require('fs');
            const vm = require('vm');

            const context = {
              console,
              window: {},
              state: {},
              elements: {},
              normalizeCapability: (value) => String(value || '').trim().toLowerCase() || null,
              normalizeBackend: (value) => String(value || '').trim().toLowerCase() || null,
              basenameFromPath: (value) => String(value || '').split(/[\\/]/).filter(Boolean).pop() || '',
            };
            vm.createContext(context);
            vm.runInContext(fs.readFileSync('static/ui/message-state.js', 'utf8'), context);

            const branch = context.sanitizeLateFillBranchList([
              {
                branch_id: 'branch-image_generation-3',
                phase_id: 'phase-4',
                type: 'image',
                capability: 'image_generation',
                status: 'running',
                depends_on: ['phase-1', 'phase-2'],
                content_payload_source: 'late_fill_results:phase-1',
                phase_summary: 'generate the Art Deco underwater jazz poster variant',
              },
            ], 'running')[0];

            const target = context.describeLateFillBranchControlTarget(branch, { actionLabel: 'Cancel' });

            assert.strictEqual(target.label, 'Cancel Image #3');
            assert.ok(target.title.startsWith('Cancel image generation #3: generate the Art Deco underwater jazz poster variant'));
            assert.ok(target.title.includes('Waits for: phase-1, phase-2'));
            assert.ok(target.title.includes('Branch: branch-image_generation-3'));
            assert.ok(target.title.includes('Phase: phase-4'));
            assert.ok(target.title.includes('Capability: image generation'));
            assert.ok(target.title.includes('Source: late fill results:phase-1'));
            assert.ok(!target.title.startsWith('Cancel image #3; branch'));
            assert.strictEqual(target.ariaLabel, target.title);
        """
        subprocess.run(
            ["node", "-e", textwrap.dedent(script)],
            cwd=".",
            check=True,
            text=True,
            capture_output=True,
        )

    @unittest.skipIf(shutil.which("node") is None, "node is required for frontend message-state tests")
    def test_late_fill_branch_progress_filters_completed_active_branches(self):
        script = r"""
            const assert = require('assert');
            const fs = require('fs');
            const vm = require('vm');

            const context = {
              console,
              window: {},
              state: {},
              elements: {},
              normalizeCapability: (value) => String(value || '').trim().toLowerCase() || null,
              normalizeBackend: (value) => String(value || '').trim().toLowerCase() || null,
              basenameFromPath: (value) => String(value || '').split(/[\\/]/).filter(Boolean).pop() || '',
            };
            vm.createContext(context);
            vm.runInContext(fs.readFileSync('static/ui/message-state.js', 'utf8'), context);

            const lateFill = {
              status: 'running',
              active_branches: [
                { branch_id: 'branch-image-1', phase_id: 'phase-image-1', capability: 'image_generation', status: 'running' },
                { branch_id: 'branch-image-2', phase_id: 'phase-image-2', capability: 'image_generation', status: 'running' },
                { branch_id: 'branch-image-3', phase_id: 'phase-image-3', capability: 'image_generation', status: 'running' },
              ],
              branch_progress: [
                { branch_id: 'branch-image-1', phase_id: 'phase-image-1', capability: 'image_generation', status: 'completed' },
              ],
            };
            const sanitized = context.sanitizeMessageLateFill(lateFill);
            assert.deepStrictEqual(
              sanitized.activeBranches.map((branch) => branch.branch_id),
              ['branch-image-2', 'branch-image-3']
            );
            assert.deepStrictEqual(
              sanitized.branchProgress.map((branch) => branch.branch_id),
              ['branch-image-1']
            );
            const lines = context.formatAssistantCurrentWorkStatusLines({ role: 'assistant', lateFill });
            assert.strictEqual(lines.length, 1);
            assert.strictEqual(lines[0], '2 images still generating...');
        """
        subprocess.run(
            ["node", "-e", textwrap.dedent(script)],
            cwd=".",
            check=True,
            text=True,
            capture_output=True,
        )

    @unittest.skipIf(shutil.which("node") is None, "node is required for frontend message-state tests")
    def test_late_fill_failed_branch_progress_stays_visible(self):
        script = r"""
            const assert = require('assert');
            const fs = require('fs');
            const vm = require('vm');

            const context = {
              console,
              window: {},
              state: {},
              elements: {},
              normalizeCapability: (value) => String(value || '').trim().toLowerCase() || null,
              normalizeBackend: (value) => String(value || '').trim().toLowerCase() || null,
              basenameFromPath: (value) => String(value || '').split(/[\\/]/).filter(Boolean).pop() || '',
            };
            vm.createContext(context);
            vm.runInContext(fs.readFileSync('static/ui/message-state.js', 'utf8'), context);

            const lateFill = {
              status: 'running',
              active_branches: [
                { branch_id: 'branch-image-1', phase_id: 'phase-image-1', capability: 'image_generation', status: 'running' },
                { branch_id: 'branch-image-2', phase_id: 'phase-image-2', capability: 'image_generation', status: 'running' },
              ],
              branch_progress: [
                { branch_id: 'branch-image-1', phase_id: 'phase-image-1', capability: 'image_generation', status: 'failed' },
              ],
            };
            const sanitized = context.sanitizeMessageLateFill(lateFill);
            assert.deepStrictEqual(
              sanitized.activeBranches.map((branch) => branch.branch_id),
              ['branch-image-2']
            );
            const lines = context.formatAssistantCurrentWorkStatusLines({ role: 'assistant', lateFill });
            assert.ok(lines.includes('Image late fill failed.'));
            assert.ok(lines.includes('Image still generating...'));
        """
        subprocess.run(
            ["node", "-e", textwrap.dedent(script)],
            cwd=".",
            check=True,
            text=True,
            capture_output=True,
        )

    @unittest.skipIf(shutil.which("node") is None, "node is required for frontend message-state tests")
    def test_active_work_primary_label_overrides_stale_surface_attention(self):
        script = r"""
            const assert = require('assert');
            const fs = require('fs');
            const vm = require('vm');

            const context = {
              console,
              window: {},
              state: {},
              elements: {},
              normalizeCapability: (value) => String(value || '').trim().toLowerCase() || null,
              normalizeBackend: (value) => String(value || '').trim().toLowerCase() || null,
              basenameFromPath: (value) => String(value || '').split(/[\\/]/).filter(Boolean).pop() || '',
            };
            vm.createContext(context);
            vm.runInContext(fs.readFileSync('static/ui/message-state.js', 'utf8'), context);
            vm.runInContext(fs.readFileSync('static/ui/messages.js', 'utf8'), context);

            const activeMessage = {
              lifecycleState: 'late_fill_running',
              statusSemantics: {
                canonical_lifecycle_state: 'late_fill_running',
                has_open_continuation: true,
                has_actionable_repair: false,
                is_terminal: false,
              },
              lateFill: {
                status: 'running',
                active_branches: [
                  { branch_id: 'branch-text-1', phase_id: 'phase-5', capability: 'chat', type: 'text', status: 'planned' },
                  { branch_id: 'branch-text-2', phase_id: 'phase-6', capability: 'chat', type: 'text', status: 'planned' },
                ],
              },
              surfaceState: {
                status: 'pending',
                category_counts: {
                  repair_pending: 6,
                  open: 3,
                  late_fill_pending: 3,
                },
              },
            };

            const activeLines = context.formatAssistantCurrentWorkStatusLines(activeMessage);
            assert.strictEqual(
              JSON.stringify(activeLines),
              JSON.stringify(['2 text outputs still generating...'])
            );
            assert.strictEqual(context.formatCompactWorkStatusLabel(activeLines), 'Working');
            assert.strictEqual(context.formatCompactWorkStatusLabel(['Backend running...']), 'Working');
            assert.strictEqual(context.formatCompactWorkStatusLabel(['Image running...']), 'Working');
            assert.strictEqual(context.inferMessageWorkGlowTone(activeMessage, activeLines), 'active');

            const repairMessage = {
              ...activeMessage,
              statusSemantics: {
                canonical_lifecycle_state: 'late_fill_running',
                has_open_continuation: true,
                has_actionable_repair: true,
                is_terminal: false,
              },
            };
            const repairLines = context.formatAssistantCurrentWorkStatusLines(repairMessage);
            assert.ok(repairLines.includes('Repairs pending: 6.'));
            assert.strictEqual(context.formatCompactWorkStatusLabel(repairLines), 'Needs Attention');

            const autoExecutableRepairMessage = {
              lifecycleState: 'late_fill_pending',
              statusSemantics: {
                canonical_lifecycle_state: 'late_fill_pending',
                has_open_continuation: true,
                has_actionable_repair: true,
                is_terminal: false,
              },
              lateFill: {
                status: 'pending',
                repair_loop: {
                  status: 'promoted',
                  auto_execute: true,
                  repair_work_available: true,
                },
                repair_rebuild_contracts: [
                  {
                    branch_id: 'repair-chat',
                    phase_id: 'repair-chat',
                    capability: 'chat',
                    output_type: 'text',
                    status: 'promoted',
                    repair_action: 'retry_same_branch',
                  },
                ],
                pending_branches: [
                  {
                    branch_id: 'branch-image_generation-1',
                    phase_id: 'phase-image-1',
                    capability: 'image_generation',
                    type: 'image',
                    status: 'pending',
                  },
                  {
                    branch_id: 'branch-text_artifact-1',
                    phase_id: 'phase-text-1',
                    capability: 'chat',
                    type: 'text',
                    status: 'repair_needed',
                    repair_action: 'retry_same_branch',
                  },
                  {
                    branch_id: 'repair-chat',
                    phase_id: 'repair-chat',
                    capability: 'chat',
                    type: 'text',
                    status: 'repair_needed',
                    repair_action: 'retry_same_branch',
                  },
                ],
              },
              surfaceState: {
                status: 'pending',
                category_counts: {
                  repair_pending: 2,
                  late_fill_pending: 3,
                },
              },
            };
            const autoRepairLines = context.formatAssistantCurrentWorkStatusLines(autoExecutableRepairMessage);
            assert.strictEqual(
              JSON.stringify(autoRepairLines),
              JSON.stringify(['Image queued for late fill...', '2 text outputs queued for late fill...'])
            );
            assert.strictEqual(context.formatCompactWorkStatusLabel(autoRepairLines), 'Working');
            assert.strictEqual(
              context.lateFillNeedsRepairAttention(autoExecutableRepairMessage.lateFill),
              false
            );
            assert.strictEqual(
              context.getAssistantDisplayContent(autoExecutableRepairMessage),
              'Image queued for late fill... 2 text outputs queued for late fill...'
            );
        """
        subprocess.run(
            ["node", "-e", textwrap.dedent(script)],
            cwd=".",
            check=True,
            text=True,
            capture_output=True,
        )

    @unittest.skipIf(shutil.which("node") is None, "node is required for frontend request-lifecycle tests")
    def test_response_lifecycle_state_drives_late_fill_polling(self):
        script = r"""
            const assert = require('assert');
            const fs = require('fs');
            const vm = require('vm');

            let timeoutCalls = 0;
            const context = {
              console,
              window: {
                setTimeout: (_fn, _delay) => `timer-${++timeoutCalls}`,
                clearTimeout: (_timer) => {},
              },
              state: { inference: { lateFillPollers: {}, pendingRequests: {}, resumePollers: {} } },
              elements: {},
              sessionStorage: { getItem: () => null, setItem: () => {}, removeItem: () => {} },
              normalizeCapability: (value) => String(value || '').trim().toLowerCase() || null,
              sanitizeRequestSnapshot: (value) => value || null,
            };
            vm.createContext(context);
            vm.runInContext(fs.readFileSync('static/ui/request-lifecycle.js', 'utf8'), context);

            assert.strictEqual(
              context.responseHasPendingLateFill({
                status: 'completed',
                lifecycle_state: 'late_fill_running',
                late_fill: { status: 'running' },
              }),
              true
            );
            assert.strictEqual(
              context.responseHasPendingLateFill({
                status: 'completed',
                lifecycle_state: 'late_fill_pending',
                late_fill: { status: 'pending' },
              }),
              true
            );
            assert.strictEqual(
              context.responseHasPendingLateFill({
                status: 'completed',
                lifecycle_state: 'late_fill_failed',
                late_fill: { status: 'running' },
              }),
              false
            );
            assert.strictEqual(
              context.responseHasPendingLateFill({
                status: 'completed',
                status_semantics: {
                  compatibility_status: 'completed',
                  canonical_lifecycle_state: 'late_fill_failed',
                  has_open_continuation: false,
                },
                late_fill: { status: 'running' },
              }),
              false
            );
            assert.strictEqual(
              context.responseHasPendingLateFill({
                status: 'completed',
                lifecycle_state: 'late_fill_completed',
                late_fill: { status: 'completed' },
              }),
              false
            );
            assert.strictEqual(
              context.responseHasPendingLateFill({
                status: 'completed',
                lifecycle_state: 'repair_needed',
                status_semantics: {
                  compatibility_status: 'completed',
                  canonical_lifecycle_state: 'repair_needed',
                  has_open_continuation: false,
                  has_actionable_repair: true,
                },
                late_fill: { status: 'failed' },
              }),
              false
            );
            assert.strictEqual(
              context.responseHasPendingLateFill({
                late_fill: { status: 'running' },
              }),
              true
            );
            assert.strictEqual(
              context.responseHasPendingLateFill({
                status: 'completed',
                lifecycle_state: 'completed',
                late_fill: { status: 'completed' },
              }),
              false
            );
            assert.strictEqual(
              context.responseHasPendingLateFill({
                status: 'completed',
                lifecycle_state: 'completed',
                status_semantics: {
                  canonical_lifecycle_state: 'completed',
                  has_open_continuation: false,
                  is_terminal: true,
                },
                late_fill: { status: 'running' },
              }),
              false
            );
            assert.strictEqual(
              context.responseHasPendingLateFill({
                status: 'completed',
                status_semantics: {
                  compatibility_status: 'completed',
                  canonical_lifecycle_state: 'late_fill_running',
                  has_open_continuation: true,
                },
                late_fill: { status: 'running' },
              }),
              true
            );
            context.maybeWatchLateFillResponse(
              'conv-1',
              {
                id: 'resp-lifecycle-active',
                status: 'completed',
                lifecycle_state: 'late_fill_running',
                late_fill: { status: 'running' },
              },
              null
            );
            assert.strictEqual(context.state.inference.lateFillPollers['resp-lifecycle-active'], 'timer-1');
            context.markActiveResponseSseStream('resp-lifecycle-streaming', true);
            context.maybeWatchLateFillResponse(
              'conv-1',
              {
                id: 'resp-lifecycle-streaming',
                status: 'completed',
                lifecycle_state: 'late_fill_running',
                late_fill: { status: 'running' },
              },
              null
            );
            assert.strictEqual(context.state.inference.lateFillPollers['resp-lifecycle-streaming'], undefined);
            context.markActiveResponseSseStream('resp-lifecycle-streaming', false);
            context.maybeWatchLateFillResponse(
              'conv-1',
              {
                id: 'resp-lifecycle-completed-needs-truth',
                status: 'completed',
                output_slots: [
                  { slot_id: 'output-phase-7', type: 'text', status: 'blocked' },
                  { slot_id: 'output-repair-chat', type: 'text', status: 'pending' },
                ],
              },
              null
            );
            assert.strictEqual(context.state.inference.lateFillPollers['resp-lifecycle-completed-needs-truth'], 'timer-2');
            context.state.inference.lateFillPollers['resp-lifecycle-failed'] = 'timer-old';
            context.maybeWatchLateFillResponse(
              'conv-1',
              {
                id: 'resp-lifecycle-failed',
                status: 'completed',
                lifecycle_state: 'late_fill_failed',
                late_fill: { status: 'failed' },
              },
              null
            );
            assert.strictEqual(context.state.inference.lateFillPollers['resp-lifecycle-failed'], undefined);
            const terminalProjectionGap = {
              id: 'resp-terminal-gap',
              status: 'completed',
              lifecycle_state: 'completed',
              artifacts: [{ type: 'image', path: '/tmp/one.png' }],
              outputs: [
                { type: 'image', path: '/tmp/one.png' },
                { type: 'image', path: '/tmp/two.png' },
              ],
            };
            assert.strictEqual(context.responsePayloadHasPublicArtifactProjectionGap(terminalProjectionGap), true);
            assert.strictEqual(context.responsePayloadHasOpenPublicArtifactProjectionGap(terminalProjectionGap), false);
            context.maybeWatchLateFillResponse('conv-1', terminalProjectionGap, null);
            assert.strictEqual(context.state.inference.lateFillPollers['resp-terminal-gap'], 'timer-3');
            assert.strictEqual(
              context.responsePayloadHasOpenPublicArtifactProjectionGap({
                ...terminalProjectionGap,
                id: 'resp-active-gap',
                lifecycle_state: 'late_fill_running',
                late_fill: { status: 'running' },
              }),
              true
            );
            assert.strictEqual(context.getResponseLookupStateVersion({ state_version: 'version-a' }), 'version-a');
            assert.strictEqual(
              context.getResponseLookupStateVersion({ response_frame: { frame_sequence: 4 } }),
              'frame:4'
            );
            assert.strictEqual(
              context.getResponseLookupStateVersion({ status_lookup: { latest_frame_id: 'resp-1:frame-5' } }),
              'frame:resp-1:frame-5'
            );
            context.rememberLateFillPollVersion('resp-backoff', { state_version: 'version-a' }, { changed: true });
            assert.strictEqual(context.getLateFillPollDelayMs('resp-backoff', { changed: true }), 650);
            context.rememberLateFillPollVersion('resp-backoff', { state_version: 'version-a' }, { changed: false });
            assert.strictEqual(context.getLateFillPollDelayMs('resp-backoff'), 900);
            context.rememberLateFillPollVersion('resp-backoff', { state_version: 'version-a' }, { changed: false });
            assert.strictEqual(context.getLateFillPollDelayMs('resp-backoff'), 1200);
            context.rememberLateFillPollVersion('resp-backoff', { state_version: 'version-a' }, { changed: false });
            assert.strictEqual(context.getLateFillPollDelayMs('resp-backoff'), 1500);
            for (let i = 0; i < 10; i += 1) {
                context.rememberLateFillPollVersion('resp-backoff', { state_version: 'version-a' }, { changed: false });
            }
            assert.strictEqual(context.getLateFillPollDelayMs('resp-backoff'), 2000);
            context.rememberLateFillPollVersion('resp-backoff', {}, { error: true });
            assert.strictEqual(context.getLateFillPollDelayMs('resp-backoff', { error: true }), 1500);
        """
        subprocess.run(
            ["node", "-e", textwrap.dedent(script)],
            cwd=".",
            check=True,
            text=True,
            capture_output=True,
        )

    @unittest.skipIf(shutil.which("node") is None, "node is required for frontend request-lifecycle tests")
    def test_response_lookup_fetch_defaults_to_ui_view(self):
        script = r"""
            const assert = require('assert');
            const fs = require('fs');
            const vm = require('vm');

            const calls = [];
            const context = {
              console,
              window: { setTimeout: () => 'timer', clearTimeout: () => {} },
              state: {
                flaskServerUrl: 'http://localhost:5001',
                inference: { lateFillPollers: {}, pendingRequests: {}, resumePollers: {} },
              },
              elements: {},
              sessionStorage: { getItem: () => null, setItem: () => {}, removeItem: () => {} },
              sanitizeRequestSnapshot: (value) => value || null,
              axios: {
                get: async (url, options) => {
                  calls.push({ url, options });
                  return { data: { id: 'resp-ui' } };
                },
              },
            };
            vm.createContext(context);
            vm.runInContext(fs.readFileSync('static/ui/request-lifecycle.js', 'utf8'), context);

            (async () => {
              await context.fetchResponseLookupPayload('resp-ui');
              await context.fetchResponseLookupStatusPayload('resp-ui');

              assert.strictEqual(calls[0].options.params.view, 'ui');
              assert.strictEqual(calls[1].options.params.view, 'status');
            })().catch((error) => {
              console.error(error);
              process.exit(1);
            });
        """
        subprocess.run(
            ["node", "-e", textwrap.dedent(script)],
            cwd=".",
            check=True,
            text=True,
            capture_output=True,
        )

    @unittest.skipIf(shutil.which("node") is None, "node is required for frontend request-lifecycle tests")
    def test_reload_recovery_uses_status_view_for_active_late_fill(self):
        script = r"""
            const assert = require('assert');
            const fs = require('fs');
            const vm = require('vm');

            let timeoutCalls = 0;
            const clearedTimers = [];
            const lookupCalls = [];
            const loadingUpdates = [];
            const statusPayload = {
              id: 'resp-active',
              response_id: 'resp-active',
              status: 'completed',
              lifecycle_state: 'late_fill_running',
              status_semantics: {
                compatibility_status: 'completed',
                canonical_lifecycle_state: 'late_fill_running',
                has_open_continuation: true,
                terminal: false,
              },
              late_fill: {
                status: 'running',
                active_count: 12,
                pending_count: 36,
                completed_count: 15,
                failed_count: 1,
              },
            };
            const context = {
              console,
              PENDING_REQUEST_SNAPSHOT_STORAGE_KEY: 'pending-requests-test',
              window: {
                setTimeout: (_fn, delay) => `timer-${++timeoutCalls}-${delay}`,
                clearTimeout: (timer) => { clearedTimers.push(timer); },
              },
              clearTimeout: (timer) => { clearedTimers.push(timer); },
              state: {
                flaskServerUrl: 'http://localhost:5001',
                conversations: { 'conv-1': [] },
                arena: { enabled: false },
                inference: {
                  lateFillPollers: {},
                  lateFillPollStates: {},
                  pendingByInstance: {},
                  pendingRequests: {
                    'req-1': {
                      requestId: 'req-1',
                      conversationId: 'conv-1',
                      responseId: 'resp-active',
                      phase: 'recovering',
                      loadingMessageId: 'loading-1',
                      requestSnapshot: { requestId: 'snapshot-1' },
                    },
                  },
                  resumePollers: { 'req-1': 'resume-old' },
                  interruptedRequestsByConversation: {},
                },
              },
              elements: {},
              sessionStorage: { getItem: () => null, setItem: () => {}, removeItem: () => {} },
              normalizeCapability: (value) => String(value || '').trim().toLowerCase() || null,
              sanitizeRequestSnapshot: (value) => value || null,
              addLoadingAssistantMessage: (conversationId, requestSnapshot, clientMessageId) => {
                context.state.conversations[conversationId].push({
                  role: 'assistant',
                  isLoading: true,
                  clientMessageId,
                  content: '<span class="loading-dots">Thinking</span>',
                  trustedHtml: true,
                  requestSnapshot,
                });
                return clientMessageId;
              },
              updateLoadingAssistantMessage: (conversationId, content, extras = {}) => {
                loadingUpdates.push({ conversationId, content, extras });
                const message = context.state.conversations[conversationId].find((item) => (
                  item.clientMessageId === extras.clientMessageId
                ));
                if (message) {
                  message.content = content;
                  message.responseId = extras.responseId || null;
                  message.lifecycleState = extras.lifecycleState || null;
                  message.statusSemantics = extras.statusSemantics || null;
                  message.lateFill = extras.lateFill || null;
                  message.liveStatusLines = extras.liveStatusLines || [];
                }
              },
              axios: {
                get: async (_url, options) => {
                  lookupCalls.push(options?.params?.view || '');
                  if (options?.params?.view === 'status') {
                    return { data: statusPayload };
                  }
                  throw new Error('full UI lookup should not run for active late-fill recovery');
                },
              },
            };
            vm.createContext(context);
            vm.runInContext(fs.readFileSync('static/ui/request-lifecycle.js', 'utf8'), context);

            (async () => {
              await context.pollPendingResponseAfterReload('req-1');

              assert.deepStrictEqual(lookupCalls, ['status']);
              assert.strictEqual(context.state.inference.pendingRequests['req-1'], undefined);
              assert.strictEqual(clearedTimers.includes('resume-old'), true);
              assert.ok(context.state.inference.lateFillPollers['resp-active']);
              assert.strictEqual(loadingUpdates.length, 1);
              assert.strictEqual(loadingUpdates[0].content, '<span class="loading-dots">Working</span>');
              assert.strictEqual(
                JSON.stringify(loadingUpdates[0].extras.liveStatusLines),
                JSON.stringify(['12 branches running...'])
              );
              assert.strictEqual(
                context.state.conversations['conv-1'][0].lateFill.status,
                'running'
              );
            })().catch((error) => {
              console.error(error);
              process.exit(1);
            });
        """
        subprocess.run(
            ["node", "-e", textwrap.dedent(script)],
            cwd=".",
            check=True,
            text=True,
            capture_output=True,
        )

    @unittest.skipIf(shutil.which("node") is None, "node is required for frontend request-lifecycle tests")
    def test_sse_completed_rearms_terminal_truth_hydration_after_stream_closes(self):
        script = r"""
            const assert = require('assert');
            const fs = require('fs');
            const vm = require('vm');

            let timeoutCalls = 0;
            const lookupCalls = [];
            const appliedPayloads = [];
            const completedEvent = [
              'event: response.completed',
              'data: ' + JSON.stringify({
                type: 'response.completed',
                response: {
                  id: 'resp-stream-final',
                  status: 'completed',
                  lifecycle_state: 'completed',
                  status_semantics: {
                    canonical_lifecycle_state: 'completed',
                    has_open_continuation: false,
                    is_terminal: true,
                  },
                  output_counts: { artifact_count: 3, output_count: 4, output_slot_count: 4 },
                },
              }),
              '',
              '',
            ].join('\n');
            let chunkSent = false;
            const context = {
              console,
              TextDecoder,
              window: {
                setTimeout: (_fn, _delay) => `timer-${++timeoutCalls}`,
                clearTimeout: (_timer) => {},
              },
              state: {
                flaskServerUrl: 'http://localhost:5001',
                conversations: { 'conv-1': [] },
                inference: {
                  lateFillPollers: {},
                  pendingRequests: {},
                  resumePollers: {},
                  activeResponseStreams: {},
                  lateFillPollStates: {},
                },
              },
              elements: {},
              sessionStorage: { getItem: () => null, setItem: () => {}, removeItem: () => {} },
              axios: {
                get: async (url, options) => {
                  lookupCalls.push({ url, options });
                  return {
                    data: {
                      id: 'resp-stream-final',
                      status: 'completed',
                      lifecycle_state: 'completed',
                      status_semantics: {
                        canonical_lifecycle_state: 'completed',
                        has_open_continuation: false,
                        is_terminal: true,
                      },
                      output_counts: { artifact_count: 3, output_count: 4, output_slot_count: 4 },
                      artifacts: [
                        { type: 'image', artifact_ref: 'artifact:image-1', path: '/tmp/image-1.png' },
                        { type: 'audio', artifact_ref: 'artifact:audio-1', path: '/tmp/audio-1.wav' },
                        { type: 'document', artifact_ref: 'artifact:doc-1', path: '/tmp/report.md' },
                      ],
                      outputs: [
                        { type: 'text', status: 'fulfilled', value: 'Artifacts generated.' },
                        { type: 'image', status: 'fulfilled', artifact_ref: 'artifact:image-1' },
                        { type: 'audio', status: 'fulfilled', artifact_ref: 'artifact:audio-1' },
                        { type: 'document', status: 'fulfilled', artifact_ref: 'artifact:doc-1' },
                      ],
                      output_slots: [
                        { type: 'text', status: 'fulfilled', slot_id: 'slot-text-1' },
                        { type: 'image', status: 'fulfilled', slot_id: 'slot-image-1', artifact_ref: 'artifact:image-1' },
                        { type: 'audio', status: 'fulfilled', slot_id: 'slot-audio-1', artifact_ref: 'artifact:audio-1' },
                        { type: 'document', status: 'fulfilled', slot_id: 'slot-doc-1', artifact_ref: 'artifact:doc-1' },
                      ],
                      state_version: 'frame:2',
                    },
                  };
                },
              },
              fetch: async () => ({
                ok: true,
                body: {
                  getReader: () => ({
                    read: async () => {
                      if (chunkSent) return { done: true, value: new Uint8Array() };
                      chunkSent = true;
                      return { done: false, value: Buffer.from(completedEvent, 'utf8') };
                    },
                  }),
                },
              }),
              normalizeCapability: (value) => String(value || '').trim().toLowerCase() || null,
              sanitizeRequestSnapshot: (value) => value || null,
              mergeRequestSnapshotInputArtifacts: (requestSnapshot) => requestSnapshot || null,
              getRequestExecutionInstance: (instance) => instance || {},
              buildResponsesInputForHistory: () => [],
              buildGhostRoutingConversationSnapshot: () => [],
              getResponsesGhostPreferencesPayload: () => null,
              getResponsesGhostRequestMetaPayload: () => null,
              buildSessionControlRequestFields: () => ({}),
              buildSelectedReferenceArtifactPayload: () => null,
              buildCanonicalResponseId: () => 'resp-fallback',
              updateLoadingAssistantMessage: () => {},
              getAssistantDisplayContent: () => 'Generated artifacts.',
              updateLoadingAssistantMessageFromResponsePayload: () => true,
              updateAssistantResponseByResponseId: (_conversationId, _responseId, payload) => {
                appliedPayloads.push(payload);
                return true;
              },
              finalizeLoadingAssistantResponse: () => {},
            };
            vm.createContext(context);
            vm.runInContext(fs.readFileSync('static/ui/request-lifecycle.js', 'utf8'), context);

            (async () => {
              await context.sendViaResponsesStream(
                { instance_id: 'chat-1', capability: 'chat' },
                'chat-1',
                'conv-1',
                'client-1',
                { requestInstance: { capability: 'chat' }, requestSnapshot: { request_id: 'req-1' } },
                'create four images',
                'resp-stream-final',
                'request-1',
              );

              assert.strictEqual(context.state.inference.activeResponseStreams['resp-stream-final'], undefined);
              assert.strictEqual(timeoutCalls, 0);
              assert.strictEqual(context.state.inference.lateFillPollers['resp-stream-final'], undefined);
              assert.strictEqual(lookupCalls.length, 1);
              assert.strictEqual(lookupCalls[0].url, 'http://localhost:5001/api/responses/resp-stream-final');
              assert.strictEqual(lookupCalls[0].options.params.view, 'ui');
              assert.ok(appliedPayloads.length >= 1);
              const fullPayload = appliedPayloads.find((payload) => (
                Array.isArray(payload.artifacts) && payload.artifacts.length === 3
              ));
              assert.ok(fullPayload);
              assert.deepStrictEqual(
                fullPayload.artifacts.map((artifact) => artifact.type).sort(),
                ['audio', 'document', 'image']
              );
              assert.strictEqual(fullPayload.outputs.length, 4);
              assert.strictEqual(fullPayload.output_slots.length, 4);
            })().catch((error) => {
              console.error(error);
              process.exit(1);
            });
        """
        subprocess.run(
            ["node", "-e", textwrap.dedent(script)],
            cwd=".",
            check=True,
            text=True,
            capture_output=True,
        )

    @unittest.skipIf(shutil.which("node") is None, "node is required for frontend request-lifecycle tests")
    def test_hydrated_response_without_truth_schedules_lookup_once(self):
        script = r"""
            const assert = require('assert');
            const fs = require('fs');
            const vm = require('vm');

            const timers = [];
            const context = {
              console,
              window: {
                setTimeout: (_fn, delay) => {
                  const timer = `timer-${timers.length + 1}`;
                  timers.push({ timer, delay });
                  return timer;
                },
                clearTimeout: (_timer) => {},
              },
              state: {
                conversations: {
                  'conv-1': [
                    { role: 'assistant', responseId: 'resp-missing-truth', content: 'artifact card' },
                    {
                      role: 'assistant',
                      responseId: 'resp-terminal',
                      content: 'done',
                      lifecycleState: 'completed',
                      statusSemantics: { canonicalLifecycleState: 'completed', hasOpenContinuation: false, isTerminal: true },
                      lateFill: { status: 'completed' },
                      lines: ['Image queued for late fill...'],
                    },
                    {
                      role: 'assistant',
                      responseId: 'resp-open',
                      content: 'running',
                      lateFill: { status: 'running' },
                    },
                  ],
                },
                inference: { lateFillPollers: {}, pendingRequests: {}, resumePollers: {} },
              },
              elements: {},
              sessionStorage: { getItem: () => null, setItem: () => {}, removeItem: () => {} },
              normalizeCapability: (value) => String(value || '').trim().toLowerCase() || null,
              sanitizeRequestSnapshot: (value) => value || null,
              sanitizeMessageLateFill: (value) => value || null,
              lateFillStatusIsActive: (lateFill) => ['pending', 'queued', 'running', 'scheduled', 'accepted'].includes(String(lateFill?.status || '').toLowerCase()),
              formatAssistantCurrentWorkStatusLines: (message) => message.lines || [],
            };
            vm.createContext(context);
            vm.runInContext(fs.readFileSync('static/ui/request-lifecycle.js', 'utf8'), context);

            assert.strictEqual(
              context.historyMessageNeedsResponseTruthHydration(context.state.conversations['conv-1'][0]),
              true
            );
            assert.strictEqual(
              context.historyMessageNeedsResponseTruthHydration(context.state.conversations['conv-1'][1]),
              false
            );

            context.resumeHydratedLateFillResponses('conv-1');

            assert.deepStrictEqual(
              Object.keys(context.state.inference.lateFillPollers).sort(),
              ['resp-missing-truth', 'resp-open']
            );
            assert.strictEqual(context.state.inference.lateFillPollers['resp-terminal'], undefined);
            assert.strictEqual(timers.length, 2);
            assert.strictEqual(timers[0].delay, 250);
            assert.strictEqual(timers[1].delay, 250);
        """
        subprocess.run(
            ["node", "-e", textwrap.dedent(script)],
            cwd=".",
            check=True,
            text=True,
            capture_output=True,
        )

    @unittest.skipIf(shutil.which("node") is None, "node is required for frontend request-lifecycle tests")
    def test_late_fill_terminal_poll_fetches_full_frame_when_status_version_unchanged(self):
        script = r"""
            const assert = require('assert');
            const fs = require('fs');
            const vm = require('vm');

            let timeoutCalls = 0;
            let clearedTimer = null;
            let statusLookups = 0;
            let fullLookups = 0;
            const applied = [];
            const context = {
              console,
              window: {
                setTimeout: (_fn, _delay) => `timer-${++timeoutCalls}`,
                clearTimeout: (timer) => { clearedTimer = timer; },
              },
              state: {
                inference: {
                  lateFillPollers: { 'resp-terminal': 'timer-old' },
                  lateFillPollStates: {
                    'resp-terminal': { stateVersion: 'version-same', unchangedCount: 3, errorCount: 0 },
                  },
                  pendingRequests: {},
                  resumePollers: {},
                },
              },
              elements: {},
              sessionStorage: { getItem: () => null, setItem: () => {}, removeItem: () => {} },
              normalizeCapability: (value) => String(value || '').trim().toLowerCase() || null,
              sanitizeRequestSnapshot: (value) => value || null,
              mergeRequestSnapshotInputArtifacts: (snapshot, inputArtifacts) => ({
                ...(snapshot || {}),
                inputArtifacts: inputArtifacts || [],
              }),
              updateAssistantResponseByResponseId: (conversationId, responseId, payload, requestSnapshot) => {
                applied.push({ conversationId, responseId, payload, requestSnapshot });
              },
              axios: {
                get: async (_url, options) => {
                  if (options?.params?.view === 'status') {
                    statusLookups += 1;
                    return {
                      data: {
                        id: 'resp-terminal',
                        state_version: 'version-same',
                        lifecycle_state: 'completed',
                        status: 'completed',
                        late_fill: { status: 'completed' },
                      },
                    };
                  }
                  fullLookups += 1;
                  return {
                    data: {
                      id: 'resp-terminal',
                      state_version: 'version-same',
                      lifecycle_state: 'completed',
                      status: 'completed',
                      late_fill: { status: 'completed', fill_results: [{ branch_id: 'branch-text' }] },
                      artifacts: [{ type: 'image', path: '/tmp/one.png' }],
                      outputs: [
                        { type: 'image', path: '/tmp/one.png' },
                        { type: 'image', path: '/tmp/two.png' },
                      ],
                      input_artifacts: [{ path: 'artifacts/documents/index.html' }],
                    },
                  };
                },
              },
            };
            vm.createContext(context);
            vm.runInContext(fs.readFileSync('static/ui/request-lifecycle.js', 'utf8'), context);

            (async () => {
              await context.pollLateFillResponse('conv-1', 'resp-terminal', { requestId: 'req-1' });

              assert.strictEqual(statusLookups, 1);
              assert.strictEqual(fullLookups, 1);
              assert.strictEqual(applied.length, 1);
              assert.strictEqual(applied[0].conversationId, 'conv-1');
              assert.strictEqual(applied[0].responseId, 'resp-terminal');
              assert.strictEqual(applied[0].payload.late_fill.fill_results[0].branch_id, 'branch-text');
              assert.strictEqual(applied[0].requestSnapshot.inputArtifacts[0].path, 'artifacts/documents/index.html');
              assert.strictEqual(clearedTimer, 'timer-old');
              assert.strictEqual(context.state.inference.lateFillPollers['resp-terminal'], undefined);
              assert.strictEqual(context.state.inference.lateFillPollStates['resp-terminal'], undefined);
            })().catch((error) => {
              console.error(error);
              process.exit(1);
            });
        """
        subprocess.run(
            ["node", "-e", textwrap.dedent(script)],
            cwd=".",
            check=True,
            text=True,
            capture_output=True,
        )


if __name__ == "__main__":
    unittest.main()
