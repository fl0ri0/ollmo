const TRANSIENT_RESOLVED_WORK_STATUS_VISIBLE_MS = 5000;
const TRANSIENT_RESOLVED_WORK_STATUS_FADE_MS = 220;

function showConversationLoadingPlaceholder(title, message = 'Loading chat history…') {
    if (!elements.chatArea) return;
    elements.chatArea.innerHTML = `
        <div class="chat-placeholder h-full">
            <div class="chat-placeholder__icon"><span class="spinner"></span></div>
            <h3>${escapeHtml(title)}</h3>
            <p>${escapeHtml(message)}</p>
        </div>
    `;
}

function resolveRenderableConversationId(instanceId = '') {
    return String(instanceId || '').trim();
}

function queueConversationRerender(instanceId = '') {
    const targetId = String(instanceId || '').trim();
    if (!targetId) return;
    state.pendingConversationRerenders = state.pendingConversationRerenders || {};
    state.pendingConversationRerenders[targetId] = true;
}

function conversationNeedsHistoryHydration(conversationId = '') {
    const key = String(conversationId || '').trim();
    if (!key) return false;
    if (state.chatHistoryLoaded?.[key]) return false;
    if (getPersistedConversationMessageCount(key) > 0) return false;
    return !state.chatHistoryRequestsInFlight?.[key];
}

async function hydrateConversationLineageForRender(instanceId, lineageIds = []) {
    if (!Array.isArray(lineageIds) || lineageIds.length === 0) {
        return [];
    }
    const hasUnhydratedSegments = lineageIds.some((conversationId) => conversationNeedsHistoryHydration(conversationId));
    if (!hasUnhydratedSegments) {
        return lineageIds;
    }
    if (state.chatHistoryRequestsInFlight?.[instanceId]) {
        return lineageIds;
    }
    await ensureConversationLineageLoaded(instanceId);
    return getConversationLineageIds(instanceId);
}

async function renderConversation(instanceId) {
    instanceId = resolveRenderableConversationId(instanceId);
    state.renderConversationInFlight = state.renderConversationInFlight || {};
    if (state.renderConversationInFlight[instanceId]) {
        queueConversationRerender(instanceId);
        return state.renderConversationInFlight[instanceId];
    }
    const renderPromise = (async () => {
        if (state.arena.enabled) {
            renderArenaConversations();
            return;
        }
        if (!elements.chatArea) return;
        elements.chatArea.classList.remove('chat-area--arena');
        if (!instanceId) {
            renderNoModelSelected();
            return;
        }
        if (isResponsesWorkbenchConversationId(instanceId)) {
            return renderResponsesWorkbenchConversation(instanceId);
        }
        appendInterruptedPendingRequestNotices(instanceId);
        const conversation = Array.isArray(state.conversations[instanceId]) ? state.conversations[instanceId] : [];
        const historyLoaded = Boolean(state.chatHistoryLoaded[instanceId]);
        if (!historyLoaded && conversation.length === 0) {
            showConversationLoadingPlaceholder(getConversationDisplayLabel(instanceId));
            return;
        }
        if (conversation.length === 0) {
            elements.chatArea.innerHTML = `
                <div class="chat-placeholder h-full">
                    <div class="chat-placeholder__icon"><i class="fas fa-comments"></i></div>
                    <h3>${getConversationDisplayLabel(instanceId)}</h3>
                    <p>This draft is empty. Say hello to kick off a new chat.</p>
                </div>
            `;
            return;
        }

        elements.chatArea.innerHTML = '';
        conversation.forEach((message) => {
            elements.chatArea.appendChild(createChatMessageElement(message, instanceId));
        });

        scrollConversationToBottom(instanceId);
    })();
    state.renderConversationInFlight[instanceId] = renderPromise;
    try {
        return await renderPromise;
    } finally {
        if (state.renderConversationInFlight?.[instanceId] === renderPromise) {
            delete state.renderConversationInFlight[instanceId];
        }
        const pendingRerender = Boolean(state.pendingConversationRerenders?.[instanceId]);
        if (pendingRerender) {
            delete state.pendingConversationRerenders[instanceId];
            if (isConversationVisible(instanceId) || String(getActiveConversationId() || '').trim() === instanceId) {
                queueMicrotask(() => {
                    renderConversation(instanceId);
                });
            }
        }
    }
}

async function renderResponsesWorkbenchConversation(conversationId = getResponsesWorkbenchConversationId()) {
    appendInterruptedPendingRequestNotices(conversationId);
    const target = getResponsesWorkbenchTargetInstance();
    const conversation = Array.isArray(state.conversations[conversationId]) ? state.conversations[conversationId] : [];
    const historyLoaded = Boolean(state.chatHistoryLoaded[conversationId]);
    if (!historyLoaded && conversation.length === 0) {
        showConversationLoadingPlaceholder('Ollmo');
        return;
    }
    if (!target && !isResponsesWorkbenchAutoTarget() && conversation.length === 0) {
        elements.chatArea.innerHTML = `
            <div class="chat-placeholder h-full">
                <div class="chat-placeholder__icon"><i class="fas fa-arrows-turn-right"></i></div>
                <h3>Ollmo</h3>
                <p>Start a local model or explicitly enable an external provider, then choose where this draft should run.</p>
            </div>
        `;
        return;
    }
    if (conversation.length === 0) {
        const isAuto = isResponsesWorkbenchAutoTarget();
        const resolvedAutoTarget = isAuto ? getGhostResolvedTargetInstance() : null;
        if (isAuto && !resolvedAutoTarget) {
            elements.chatArea.innerHTML = `
                <div class="chat-placeholder h-full">
                    <div class="chat-placeholder__icon"><i class="fas fa-route"></i></div>
                    <h3>Ollmo</h3>
                    <p>This draft is empty. Ollmo will choose an available local model or an explicitly enabled external provider.</p>
                </div>
            `;
            return;
        }
        const label = isAuto
            ? formatResponsesAutoLabel()
            : (
                isExternalConversationTarget(target)
                    ? 'ChatGPT'
                    : (target ? formatModelDisplayName(target.model || target.instance_id) : 'target model')
            );
        const backendLabel = isAuto
            ? 'resolved target'
            : (
                isExternalConversationTarget(target)
                    ? 'external provider'
                    : (target ? formatBackendLabel(target.backend) : 'runtime')
            );
        elements.chatArea.innerHTML = `
            <div class="chat-placeholder h-full">
                <div class="chat-placeholder__icon"><i class="fas fa-route"></i></div>
                <h3>Ollmo</h3>
                <p>This draft is empty. Everything sent here goes through Ollmo to ${label} (${backendLabel}).</p>
            </div>
        `;
        return;
    }
    elements.chatArea.innerHTML = '';
    conversation.forEach((message) => {
        elements.chatArea.appendChild(createChatMessageElement(message, conversationId));
    });
    scrollConversationToBottom(conversationId);
}

function renderArenaConversations() {
    if (!elements.chatArea) return;
    elements.chatArea.classList.add('chat-area--arena');
    if (!arenaHasDistinctSelections()) {
        elements.chatArea.innerHTML = `
            <div class="chat-placeholder h-full">
                <div class="chat-placeholder__icon"><i class="fas fa-columns"></i></div>
                <h3>Configure Arena</h3>
                <p>Select two different running models to compare their responses.</p>
            </div>
        `;
        return;
    }
    const columns = [
        { label: 'Model A', instanceId: state.arena.modelA },
        { label: 'Model B', instanceId: state.arena.modelB }
    ];
    elements.chatArea.innerHTML = '';
    columns.forEach(({ label, instanceId }) => {
        const column = document.createElement('div');
        column.className = 'arena-column';
        const header = document.createElement('div');
        header.className = 'arena-column__header';
        const meta = getInstanceMeta(instanceId);
        const title = formatModelDisplayName(meta?.model || instanceId);
        const backendLabel = formatBackendLabel(meta?.backend);
        header.innerHTML = `<span>${label}</span><span>${title} • ${backendLabel}</span>`;
        column.appendChild(header);
        const body = document.createElement('div');
        body.className = 'arena-column__body';
        const conversationId = getArenaConversationId(instanceId);
        const conversation = conversationId ? (state.conversations[conversationId] || []) : [];
        if (!conversationId || conversation.length === 0) {
            body.innerHTML = '<div class="arena-empty">Awaiting first exchange.</div>';
        } else {
            conversation.forEach(message => {
                body.appendChild(createChatMessageElement(message, conversationId));
            });
        }
        column.appendChild(body);
        elements.chatArea.appendChild(column);
    });
}

function buildRenderableMessageArtifacts(message = {}) {
    return buildCanonicalMessageArtifacts(message);
}

function isRedundantUserAttachmentMarkerLine(line = '', artifactNames = new Set()) {
    const normalizedLine = String(line || '').trim();
    if (!normalizedLine || !artifactNames.size) {
        return false;
    }
    const prefixes = [
        '[Attachment: ',
        '[Local file: ',
        '[Audio: ',
        '[Reference image: ',
        '[Image: ',
        '[Text file: ',
        '[PDF: ',
    ];
    for (const name of artifactNames) {
        if (!name) continue;
        for (const prefix of prefixes) {
            if (normalizedLine === `${prefix}${name}]`) {
                return true;
            }
        }
    }
    return false;
}

function getRenderableMessageContent(message = {}, artifacts = []) {
    const content = String(message.content || '');
    if (String(message.role || '').trim().toLowerCase() !== 'user' || !artifacts.length) {
        return content;
    }
    const requestSnapshot = sanitizeRequestSnapshot(message.requestSnapshot || message.request_snapshot);
    const artifactNames = new Set(
        [
            requestSnapshot?.attachment?.name,
            ...artifacts.map((artifact) => (
                artifact.name
                || basenameFromPath(artifact.source_path || artifact.path || '')
                || null
            )),
        ]
            .map((value) => String(value || '').trim())
            .filter(Boolean)
    );
    if (!artifactNames.size) {
        return content;
    }
    const lines = content.split('\n');
    while (lines.length) {
        const lastLine = String(lines[lines.length - 1] || '').trim();
        if (!isRedundantUserAttachmentMarkerLine(lastLine, artifactNames)) {
            break;
        }
        lines.pop();
        while (lines.length && !String(lines[lines.length - 1] || '').trim()) {
            lines.pop();
        }
    }
    return lines.join('\n');
}

function isArtifactUnavailable(artifact = {}) {
    const availability = String(artifact.availability || '').trim().toLowerCase();
    return availability === 'missing' || availability === 'purged';
}

function formatArtifactAvailabilityText(artifact = {}) {
    const availability = String(artifact.availability || '').trim().toLowerCase();
    if (availability === 'purged') {
        const purgedAt = String(artifact.purged_at || '').trim();
        return purgedAt
            ? `Unavailable: purged from disk (${purgedAt}).`
            : 'Unavailable: purged from disk.';
    }
    if (availability === 'missing') {
        const checkedAt = String(artifact.availability_checked_at || '').trim();
        return checkedAt
            ? `Unavailable: file missing on disk (${checkedAt}).`
            : 'Unavailable: file missing on disk.';
    }
    return '';
}

function canUseArtifactAsReference(artifact = {}) {
    const type = String(artifact.type || '').trim().toLowerCase();
    const path = String(artifact.path || '').trim();
    return Boolean(path) && !isArtifactUnavailable(artifact) && ['image', 'audio', 'text', 'document'].includes(type);
}

function isLocalBundleableArtifact(artifact = {}) {
    const path = String(artifact.path || artifact.source_path || '').trim();
    if (!path || isArtifactUnavailable(artifact) || /^(?:https?:|data:|blob:)/i.test(path)) {
        return false;
    }
    return true;
}

function artifactLooksLikeBundleEntrypoint(artifact = {}) {
    const type = String(artifact.type || artifact.kind || '').trim().toLowerCase();
    const path = String(artifact.path || artifact.source_path || '').trim().toLowerCase();
    const mimeType = String(artifact.mime_type || artifact.mimeType || '').trim().toLowerCase();
    return ['text', 'document'].includes(type)
        || path.endsWith('.html')
        || path.endsWith('.htm')
        || path.endsWith('.css')
        || path.endsWith('.js')
        || mimeType.startsWith('text/')
        || mimeType.includes('html');
}

function canBundleAssistantMessage(message = {}, artifacts = []) {
    if (!message || String(message.role || '').trim().toLowerCase() !== 'assistant') return false;
    if (message.ephemeralUiNotice) return false;
    if (!String(message.responseId || message.response_id || '').trim()) return false;
    const localArtifacts = sanitizeResponseArtifacts(artifacts).filter(isLocalBundleableArtifact);
    if (!localArtifacts.length) return false;
    return localArtifacts.some(artifactLooksLikeBundleEntrypoint) || localArtifacts.length >= 2;
}

function responseArtifactBundleHasActiveWork(message = {}) {
    if (message.isLoading || message.ephemeralUiNotice) return true;
    const statusSemantics = message.statusSemantics || message.status_semantics || {};
    if (
        typeof responseMessageOutputContractIndicatesCleanResolution === 'function'
        && responseMessageOutputContractIndicatesCleanResolution(message)
        && statusSemantics.hasOpenContinuation !== true
        && statusSemantics.has_open_continuation !== true
    ) {
        return false;
    }
    if (statusSemantics.hasOpenContinuation === true || statusSemantics.has_open_continuation === true) {
        return true;
    }
    const lifecycleState = String(
        message.lifecycleState
        || message.lifecycle_state
        || statusSemantics.canonicalLifecycleState
        || statusSemantics.canonical_lifecycle_state
        || ''
    ).trim().toLowerCase();
    if (['in_progress', 'queued', 'running', 'pending', 'late_fill_running', 'late_fill_pending'].includes(lifecycleState)) {
        return true;
    }
    const lateFillStatus = String(
        message.lateFill?.status
        || message.late_fill?.status
        || ''
    ).trim().toLowerCase();
    if (['running', 'pending', 'queued', 'in_progress'].includes(lateFillStatus)) {
        return true;
    }
    return false;
}

function getLatestOpenableArtifactBundle(message = {}) {
    const bundles = getMessageArtifactBundles(message).filter((bundle) => (
        String(bundle.bundle_path || bundle.bundlePath || '').trim()
    ));
    return bundles.length ? bundles[bundles.length - 1] : null;
}

function messageHasStructuredRenderSurface(message = {}) {
    if (!message || typeof message !== 'object') return false;
    const artifacts = sanitizeResponseArtifacts(message.artifacts);
    if (artifacts.some((artifact) => {
        if (isArtifactUnavailable(artifact)) return false;
        return Boolean(String(artifact.path || artifact.image_data_url || '').trim());
    })) {
        return true;
    }
    const outputs = sanitizeResponseOutputs(message.outputs || message.canonical_outputs || message.canonicalOutputs);
    if (outputs.length >= 2) return true;
    if (outputs.some((output) => (
        String(output.artifact_ref || output.ref || '').trim()
        || sanitizeResponseArtifacts(output.artifacts).some((artifact) => (
            !isArtifactUnavailable(artifact)
            && Boolean(String(artifact.path || artifact.image_data_url || '').trim())
        ))
    ))) {
        return true;
    }
    const outputSlots = sanitizeResponseOutputSlots(message.outputSlots || message.output_slots);
    if (outputSlots.some((slot) => String(slot.artifact_ref || slot.ref || '').trim())) {
        return true;
    }
    return getMessageArtifactBundles(message).length > 0;
}

function getResponseArtifactBundleActionState(message = {}, artifacts = []) {
    const existingBundle = getLatestOpenableArtifactBundle(message);
    const visible = Boolean(existingBundle) || canBundleAssistantMessage(message, artifacts);
    const enabled = visible && !responseArtifactBundleHasActiveWork(message);
    const mode = existingBundle ? 'open' : 'create';
    return {
        visible,
        enabled,
        mode,
        bundle: existingBundle,
        label: existingBundle ? 'Open Bundle' : 'Bundle',
        icon: existingBundle ? 'fas fa-folder-open' : 'fas fa-box-archive',
        title: enabled
            ? (
                existingBundle
                    ? 'Open the latest response artifact bundle folder'
                    : 'Create a portable folder from this response\'s saved artifacts'
            )
            : 'Bundle action becomes available after this response finishes active work',
    };
}

function getMessageArtifactBundles(message = {}) {
    const candidates = [];
    if (Array.isArray(message.artifactBundles)) candidates.push(...message.artifactBundles);
    if (Array.isArray(message.artifact_bundles)) candidates.push(...message.artifact_bundles);
    if (message.artifactBundle && typeof message.artifactBundle === 'object') candidates.push(message.artifactBundle);
    const seen = new Set();
    return candidates.filter((bundle) => {
        if (!bundle || typeof bundle !== 'object') return false;
        const key = String(bundle.bundle_id || bundle.bundleId || bundle.bundle_path || bundle.bundlePath || '').trim();
        if (!key || seen.has(key)) return false;
        seen.add(key);
        return true;
    });
}

function attachArtifactBundleToMessage(message = {}, bundle = {}) {
    if (!message || typeof message !== 'object' || !bundle || typeof bundle !== 'object') {
        return;
    }
    const bundles = getMessageArtifactBundles(message);
    const bundleId = String(bundle.bundle_id || bundle.bundleId || '').trim();
    const bundlePath = String(bundle.bundle_path || bundle.bundlePath || '').trim();
    const nextBundles = bundles.filter((item) => {
        const existingId = String(item.bundle_id || item.bundleId || '').trim();
        const existingPath = String(item.bundle_path || item.bundlePath || '').trim();
        return !(
            (bundleId && existingId === bundleId)
            || (bundlePath && existingPath === bundlePath)
        );
    });
    nextBundles.push(bundle);
    message.artifactBundle = bundle;
    message.artifactBundles = nextBundles;
    message.artifact_bundles = nextBundles;
}

function findConversationMessageForBundle(conversationId = '', message = {}) {
    const conversation = state.conversations?.[conversationId];
    if (!Array.isArray(conversation)) return message;
    const responseId = String(message.responseId || message.response_id || '').trim();
    const clientMessageId = String(message.clientMessageId || '').trim();
    return conversation.find((item) => (
        String(item?.role || '').trim().toLowerCase() === 'assistant'
        && (
            (responseId && String(item.responseId || item.response_id || '').trim() === responseId)
            || (clientMessageId && String(item.clientMessageId || '').trim() === clientMessageId)
        )
    )) || message;
}

function formatArtifactBundleName(bundle = {}) {
    const bundlePath = String(bundle.bundle_path || bundle.bundlePath || '').trim();
    return basenameFromPath(bundlePath) || 'Response artifact bundle';
}

function formatArtifactBundleStatus(bundle = {}) {
    const linkStatus = String(bundle.link_check?.status || bundle.linkCheck?.status || '').trim().toLowerCase();
    const missing = Array.isArray(bundle.link_check?.missing)
        ? bundle.link_check.missing.length
        : (Array.isArray(bundle.linkCheck?.missing) ? bundle.linkCheck.missing.length : 0);
    if (linkStatus === 'passed') return 'Links verified';
    if (missing > 0) return `${missing} missing link${missing === 1 ? '' : 's'}`;
    return String(bundle.status || '').trim() || 'Bundled';
}

function appendResponseArtifactBundleCards(bubble, message = {}) {
    getMessageArtifactBundles(message).forEach((bundle) => {
        const card = document.createElement('div');
        const linkStatus = String(bundle.link_check?.status || bundle.linkCheck?.status || '').trim().toLowerCase();
        card.className = `chat-artifact-bundle-card${linkStatus === 'failed' ? ' chat-artifact-bundle-card--failed' : ''}`;

        const summary = document.createElement('div');
        summary.className = 'chat-artifact-bundle-card__summary';
        summary.innerHTML = `
            <span class="chat-artifact-bundle-card__icon"><i class="fas fa-box-archive"></i></span>
            <span class="chat-artifact-bundle-card__name">${escapeHtml(formatArtifactBundleName(bundle))}</span>
            <span class="chat-artifact-bundle-card__status">${escapeHtml(formatArtifactBundleStatus(bundle))}</span>
        `;
        card.appendChild(summary);

        const actions = document.createElement('div');
        actions.className = 'chat-image-actions chat-artifact-bundle-card__actions';
        const bundlePath = String(bundle.bundle_path || bundle.bundlePath || '').trim();
        const entrypoint = String(bundle.entrypoint || '').trim();
        if (bundlePath) {
            const openFolderButton = createChatMessageActionButton('<i class="fas fa-folder-open"></i> Open Folder', (event) => {
                openSavedArtifactLocation(bundlePath, event.currentTarget);
            });
            actions.appendChild(openFolderButton);
        }
        if (entrypoint) {
            const openEntryButton = createChatMessageActionButton('<i class="fas fa-up-right-from-square"></i> Open Entry', (event) => {
                openSavedArtifactEntry(entrypoint, event.currentTarget);
            });
            actions.appendChild(openEntryButton);
        }
        if (actions.childElementCount) {
            card.appendChild(actions);
        }
        bubble.appendChild(card);
    });
}

async function bundleResponseArtifactsForMessage(message = {}, conversationId = '', triggerButton = null) {
    const responseId = String(message.responseId || message.response_id || '').trim();
    if (!responseId) {
        updateGlobalModelStatus('Response id unavailable for bundle.');
        return;
    }
    if (responseArtifactBundleHasActiveWork(message)) {
        updateGlobalModelStatus('Bundle becomes available after this response finishes active work.');
        return;
    }
    if (triggerButton) {
        triggerButton.disabled = true;
    }
    updateGlobalModelStatus('Bundling response artifacts...');
    try {
        const response = await axios.post(
            `${state.flaskServerUrl}/api/responses/${encodeURIComponent(responseId)}/bundle_artifacts`,
            {}
        );
        const bundle = response.data?.bundle;
        if (!bundle || typeof bundle !== 'object') {
            throw new Error('Bundle response did not include bundle metadata.');
        }
        const targetMessage = findConversationMessageForBundle(conversationId, message);
        attachArtifactBundleToMessage(targetMessage, bundle);
        if (isConversationVisible(conversationId)) {
            renderConversation(conversationId);
        }
        persistChatHistory(conversationId);
        const statusText = String(bundle.link_check?.status || '').trim().toLowerCase() === 'failed'
            ? 'Bundle created with missing local links.'
            : 'Bundle created.';
        updateGlobalModelStatus(statusText);
        setTimeout(() => updateGlobalModelStatus(''), 2200);
    } catch (error) {
        const messageText = error.response?.data?.error || error.message || 'Could not bundle response artifacts.';
        updateGlobalModelStatus(`Could not bundle response artifacts: ${messageText}`);
    } finally {
        if (triggerButton) {
            triggerButton.disabled = false;
        }
    }
}

function buildMessageArtifactReferencePayload(artifact = {}, message = {}) {
    const path = String(artifact.path || '').trim();
    if (!path) return null;
    const requestSnapshot = sanitizeRequestSnapshot(message.requestSnapshot || message.request_snapshot);
    return sanitizeSelectedReferenceArtifact({
        type: String(artifact.type || '').trim().toLowerCase() || 'document',
        path,
        message_id: String(message.clientMessageId || '').trim() || null,
        name: artifact.name || null,
        kind: artifact.kind || null,
        origin: artifact.origin || null,
        source_path: artifact.source_path || null,
        mime_type: artifact.mime_type || null,
        prompt: String(
            artifact.prompt
            || requestSnapshot?.prompt_text
            || requestSnapshot?.promptText
            || message.content
            || ''
        ).trim() || null,
        seed: artifact.seed,
        image_state: artifact.image_state || null,
    });
}

function inferArtifactFileName(savedArtifactPath = '', artifact = {}) {
    const explicitName = String(artifact.name || '').trim();
    if (explicitName) {
        return explicitName;
    }
    const pathName = basenameFromPath(savedArtifactPath);
    if (pathName) {
        return pathName;
    }
    const artifactType = String(artifact.type || '').trim().toLowerCase();
    return artifactType === 'image' ? 'image-artifact.png' : 'artifact';
}

function inferArtifactMimeType(savedArtifactPath = '', artifact = {}) {
    const explicitMimeType = String(artifact.mime_type || artifact.mimeType || '').trim().toLowerCase();
    if (explicitMimeType) {
        return explicitMimeType;
    }
    const normalizedPath = String(savedArtifactPath || '').trim().toLowerCase();
    if (normalizedPath.endsWith('.jpg') || normalizedPath.endsWith('.jpeg')) return 'image/jpeg';
    if (normalizedPath.endsWith('.webp')) return 'image/webp';
    if (normalizedPath.endsWith('.gif')) return 'image/gif';
    if (normalizedPath.endsWith('.bmp')) return 'image/bmp';
    if (normalizedPath.endsWith('.svg')) return 'image/svg+xml';
    return 'image/png';
}

function artifactIsTextLike(artifact = {}, savedArtifactPath = '') {
    const type = String(artifact.type || artifact.kind || '').trim().toLowerCase();
    if (type === 'text' || type === 'document') return true;
    const mimeType = String(artifact.mime_type || artifact.mimeType || '').trim().toLowerCase();
    if (
        mimeType.startsWith('text/')
        || mimeType.includes('html')
        || mimeType.includes('json')
        || mimeType.includes('xml')
        || mimeType.includes('javascript')
    ) {
        return true;
    }
    const path = String(savedArtifactPath || artifact.path || artifact.source_path || '').trim().toLowerCase();
    return /\.(?:html?|css|js|mjs|json|md|markdown|txt|xml|csv|tsv|yaml|yml|svg)$/.test(path);
}

async function copySavedArtifactContent(savedArtifactPath = '', triggerButton = null) {
    const targetPath = String(savedArtifactPath || '').trim();
    if (!targetPath) {
        showCopyStatus('Artifact path unavailable.');
        return;
    }
    if (triggerButton) {
        triggerButton.disabled = true;
    }
    try {
        const response = await fetch(buildSavedArtifactViewUrl(targetPath), {
            method: 'GET',
            credentials: 'same-origin',
        });
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        const text = await response.text();
        if (!String(text || '').trim()) {
            showCopyStatus('Artifact is empty.');
            return;
        }
        await copyMessageToClipboard(text, {
            successMessage: 'Artifact copied to clipboard.',
            failureMessage: 'Unable to copy artifact.',
        });
    } catch (error) {
        console.error('Artifact copy failed:', error);
        showCopyStatus('Unable to copy artifact.');
    } finally {
        if (triggerButton) {
            triggerButton.disabled = false;
        }
    }
}

function buildArtifactImagePreviewMeta({
    fileName = '',
    savedArtifactPath = '',
    naturalWidth = 0,
    naturalHeight = 0,
} = {}) {
    const parts = [];
    const normalizedFileName = String(fileName || '').trim();
    if (normalizedFileName) {
        parts.push(normalizedFileName);
    } else {
        const pathName = basenameFromPath(savedArtifactPath);
        if (pathName) {
            parts.push(pathName);
        }
    }
    if (Number.isFinite(naturalWidth) && naturalWidth > 0 && Number.isFinite(naturalHeight) && naturalHeight > 0) {
        parts.push(`${naturalWidth} x ${naturalHeight}px`);
    }
    return parts.join(' • ');
}

function setArtifactImageDragPayload(event, {
    dragUrl = '',
    fallbackText = '',
    fileName = '',
    mimeType = '',
} = {}) {
    if (!event?.dataTransfer) return;
    const normalizedDragUrl = String(dragUrl || '').trim();
    const normalizedFallbackText = String(fallbackText || '').trim();
    const normalizedFileName = String(fileName || '').trim() || 'image-artifact.png';
    const normalizedMimeType = String(mimeType || '').trim() || 'image/png';
    event.dataTransfer.effectAllowed = 'copy';
    try {
        if (normalizedDragUrl) {
            event.dataTransfer.setData('text/uri-list', normalizedDragUrl);
        }
    } catch (_error) {
        // Ignore drag payload support differences across browsers.
    }
    try {
        if (normalizedFallbackText || normalizedDragUrl) {
            event.dataTransfer.setData('text/plain', normalizedFallbackText || normalizedDragUrl);
        }
    } catch (_error) {
        // Ignore drag payload support differences across browsers.
    }
    try {
        if (normalizedDragUrl) {
            event.dataTransfer.setData('DownloadURL', `${normalizedMimeType}:${normalizedFileName}:${normalizedDragUrl}`);
        }
    } catch (_error) {
        // Ignore drag payload support differences across browsers.
    }
}

function bindArtifactImageDrag(element, {
    savedArtifactPath = '',
    viewUrl = '',
    fileName = '',
    mimeType = '',
} = {}) {
    if (!element) return;
    const normalizedViewUrl = String(viewUrl || '').trim();
    const normalizedSavedArtifactPath = String(savedArtifactPath || '').trim();
    const dragUrl = normalizedSavedArtifactPath
        ? buildSavedArtifactViewUrl(normalizedSavedArtifactPath)
        : normalizedViewUrl;
    if (!dragUrl) return;
    element.draggable = true;
    element.dataset.dragArtifactUrl = dragUrl;
    element.dataset.dragArtifactFallback = normalizedSavedArtifactPath || normalizedViewUrl;
    element.dataset.dragArtifactName = String(fileName || '').trim() || 'image-artifact.png';
    element.dataset.dragArtifactMimeType = String(mimeType || '').trim() || 'image/png';
    if (element.dataset.dragArtifactBound === '1') {
        return;
    }
    element.dataset.dragArtifactBound = '1';
    element.addEventListener('dragstart', (event) => {
        setArtifactImageDragPayload(event, {
            dragUrl: element.dataset.dragArtifactUrl || '',
            fallbackText: element.dataset.dragArtifactFallback || '',
            fileName: element.dataset.dragArtifactName || '',
            mimeType: element.dataset.dragArtifactMimeType || '',
        });
    });
}

function updateArtifactImagePreviewMeta() {
    if (!elements.artifactImagePreviewMeta) return;
    const previewMeta = buildArtifactImagePreviewMeta({
        fileName: state.imagePreview?.name || '',
        savedArtifactPath: state.imagePreview?.path || '',
        naturalWidth: Number(state.imagePreview?.naturalWidth || 0),
        naturalHeight: Number(state.imagePreview?.naturalHeight || 0),
    });
    const totalItems = Array.isArray(state.imagePreview?.items) ? state.imagePreview.items.length : 0;
    const currentIndex = Number(state.imagePreview?.index ?? -1);
    const counterText = totalItems > 1 && currentIndex >= 0 ? `${currentIndex + 1}/${totalItems}` : '';
    elements.artifactImagePreviewMeta.textContent = counterText && previewMeta
        ? `${counterText} • ${previewMeta}`
        : (counterText || previewMeta);
}

function resetArtifactImagePreviewLayout() {
    if (elements.artifactImagePreviewImage) {
        elements.artifactImagePreviewImage.style.width = '';
        elements.artifactImagePreviewImage.style.height = '';
    }
}

function buildPreviewableConversationImageItems(conversationId = '') {
    const normalizedConversationId = String(conversationId || '').trim();
    const conversation = state.conversations?.[normalizedConversationId];
    if (!normalizedConversationId || !Array.isArray(conversation)) {
        return [];
    }
    const items = [];
    const seenKeys = new Set();
    conversation.forEach((message) => {
        sanitizeResponseArtifacts(message?.artifacts).forEach((artifact) => {
            const artifactType = String(artifact?.type || '').trim().toLowerCase();
            if (artifactType !== 'image') return;
            const savedArtifactPath = String(artifact.path || '').trim();
            const artifactUnavailable = isArtifactUnavailable(artifact);
            const canAccessSavedArtifact = Boolean(savedArtifactPath) && !artifactUnavailable;
            const viewUrl = String(
                artifact.image_data_url
                || (canAccessSavedArtifact ? buildSavedArtifactViewUrl(savedArtifactPath) : '')
            ).trim();
            if (!viewUrl) return;
            const item = {
                path: savedArtifactPath,
                src: viewUrl,
                name: inferArtifactFileName(savedArtifactPath, artifact),
                mimeType: inferArtifactMimeType(savedArtifactPath, artifact),
                alt: String(
                    artifact.prompt
                    || artifact.name
                    || artifact.image_state?.summary
                    || 'Saved image artifact'
                ).trim(),
            };
            const key = item.path || item.src;
            if (seenKeys.has(key)) return;
            seenKeys.add(key);
            items.push(item);
        });
    });
    return items;
}

function resolveArtifactImagePreviewItems({
    conversationId = '',
    savedArtifactPath = '',
    viewUrl = '',
    fileName = '',
    mimeType = '',
    alt = '',
} = {}) {
    const normalizedConversationId = String(conversationId || '').trim();
    const fallbackItem = {
        path: String(savedArtifactPath || '').trim(),
        src: String(viewUrl || '').trim(),
        name: String(fileName || '').trim() || inferArtifactFileName(savedArtifactPath, {}),
        mimeType: String(mimeType || '').trim() || inferArtifactMimeType(savedArtifactPath, {}),
        alt: String(alt || fileName || 'Image preview').trim(),
    };
    let items = buildPreviewableConversationImageItems(normalizedConversationId);
    if (!items.length) {
        return {
            items: fallbackItem.src ? [fallbackItem] : [],
            index: fallbackItem.src ? 0 : -1,
            conversationId: normalizedConversationId,
        };
    }
    let index = items.findIndex((item) => {
        if (fallbackItem.path && item.path && item.path === fallbackItem.path) return true;
        return fallbackItem.src && item.src === fallbackItem.src;
    });
    if (index < 0 && fallbackItem.src) {
        items = items.concat([fallbackItem]);
        index = items.length - 1;
    }
    return {
        items,
        index,
        conversationId: normalizedConversationId,
    };
}

function updateArtifactImagePreviewNavigation() {
    const items = Array.isArray(state.imagePreview?.items) ? state.imagePreview.items : [];
    const currentIndex = Number(state.imagePreview?.index ?? -1);
    const hasMultipleItems = items.length > 1;
    const prevButton = elements.artifactImagePreviewPrevBtn;
    const nextButton = elements.artifactImagePreviewNextBtn;
    if (prevButton) {
        prevButton.hidden = !hasMultipleItems;
        prevButton.disabled = !hasMultipleItems || currentIndex <= 0;
    }
    if (nextButton) {
        nextButton.hidden = !hasMultipleItems;
        nextButton.disabled = !hasMultipleItems || currentIndex >= items.length - 1;
    }
}

function showArtifactImagePreviewItem(index) {
    const items = Array.isArray(state.imagePreview?.items) ? state.imagePreview.items : [];
    const normalizedIndex = Number(index);
    if (!items.length || !Number.isInteger(normalizedIndex) || normalizedIndex < 0 || normalizedIndex >= items.length) {
        return;
    }
    const item = items[normalizedIndex];
    if (!item || !elements.artifactImagePreviewImage) {
        return;
    }
    state.imagePreview = {
        ...(state.imagePreview || {}),
        index: normalizedIndex,
        path: String(item.path || '').trim(),
        src: String(item.src || '').trim(),
        name: String(item.name || '').trim(),
        mimeType: String(item.mimeType || '').trim(),
        naturalWidth: 0,
        naturalHeight: 0,
    };
    updateArtifactImagePreviewNavigation();
    updateArtifactImagePreviewMeta();
    if (elements.artifactImagePreviewOpenFolderBtn) {
        const hasSavedPath = Boolean(state.imagePreview.path);
        elements.artifactImagePreviewOpenFolderBtn.hidden = !hasSavedPath;
        elements.artifactImagePreviewOpenFolderBtn.disabled = !hasSavedPath;
    }
    elements.artifactImagePreviewImage.alt = String(item.alt || item.name || 'Image preview').trim();
    resetArtifactImagePreviewLayout();
    elements.artifactImagePreviewImage.onload = () => {
        state.imagePreview = {
            ...(state.imagePreview || {}),
            naturalWidth: Number(elements.artifactImagePreviewImage?.naturalWidth || 0),
            naturalHeight: Number(elements.artifactImagePreviewImage?.naturalHeight || 0),
        };
        updateArtifactImagePreviewMeta();
        syncArtifactImagePreviewLayout();
    };
    bindArtifactImageDrag(elements.artifactImagePreviewImage, {
        savedArtifactPath: state.imagePreview.path,
        viewUrl: state.imagePreview.src,
        fileName: state.imagePreview.name,
        mimeType: state.imagePreview.mimeType,
    });
    elements.artifactImagePreviewImage.src = state.imagePreview.src;
}

function stepArtifactImagePreview(direction = 0) {
    const delta = Number(direction);
    if (!state.imagePreview?.open || !Number.isFinite(delta) || delta === 0) {
        return;
    }
    const items = Array.isArray(state.imagePreview?.items) ? state.imagePreview.items : [];
    const currentIndex = Number(state.imagePreview?.index ?? -1);
    if (!items.length || !Number.isInteger(currentIndex)) {
        return;
    }
    const nextIndex = Math.max(0, Math.min(items.length - 1, currentIndex + delta));
    if (nextIndex === currentIndex) {
        return;
    }
    showArtifactImagePreviewItem(nextIndex);
}

function syncArtifactImagePreviewLayout() {
    const image = elements.artifactImagePreviewImage;
    const card = elements.artifactImagePreviewCard;
    if (!image || !card || !state.imagePreview?.open) {
        return;
    }
    const naturalWidth = Number(image.naturalWidth || state.imagePreview?.naturalWidth || 0);
    const naturalHeight = Number(image.naturalHeight || state.imagePreview?.naturalHeight || 0);
    if (!Number.isFinite(naturalWidth) || naturalWidth <= 0 || !Number.isFinite(naturalHeight) || naturalHeight <= 0) {
        resetArtifactImagePreviewLayout();
        return;
    }
    const viewportWidth = Math.max(window.innerWidth || document.documentElement?.clientWidth || 0, 320);
    const viewportHeight = Math.max(window.innerHeight || document.documentElement?.clientHeight || 0, 240);
    const headerHeight = Number(card.querySelector('.artifact-image-preview-card__header')?.offsetHeight || 0);
    const overlayMargin = 32;
    const cardChromeWidth = 34;
    const cardChromeHeight = 34;
    const headerGap = 14;
    const maxImageWidth = Math.max(120, viewportWidth - overlayMargin - cardChromeWidth);
    const maxImageHeight = Math.max(120, viewportHeight - overlayMargin - cardChromeHeight - headerHeight - headerGap);
    const scale = Math.min(1, maxImageWidth / naturalWidth, maxImageHeight / naturalHeight);
    const displayWidth = Math.max(1, Math.round(naturalWidth * scale));
    const displayHeight = Math.max(1, Math.round(naturalHeight * scale));
    image.style.width = `${displayWidth}px`;
    image.style.height = `${displayHeight}px`;
}

function closeArtifactImagePreview() {
    if (!elements.artifactImagePreviewOverlay) return;
    elements.artifactImagePreviewOverlay.classList.remove('open');
    elements.artifactImagePreviewOverlay.hidden = true;
    if (elements.artifactImagePreviewImage) {
        elements.artifactImagePreviewImage.onload = null;
        elements.artifactImagePreviewImage.removeAttribute('src');
        elements.artifactImagePreviewImage.alt = '';
    }
    resetArtifactImagePreviewLayout();
    if (elements.artifactImagePreviewMeta) {
        elements.artifactImagePreviewMeta.textContent = '';
    }
    if (elements.artifactImagePreviewOpenFolderBtn) {
        elements.artifactImagePreviewOpenFolderBtn.hidden = false;
        elements.artifactImagePreviewOpenFolderBtn.disabled = false;
    }
    state.imagePreview = {
        ...(state.imagePreview || {}),
        open: false,
        path: '',
        src: '',
        name: '',
        mimeType: '',
        naturalWidth: 0,
        naturalHeight: 0,
        items: [],
        index: -1,
        conversationId: '',
    };
    updateArtifactImagePreviewNavigation();
}

function ensureArtifactImagePreviewBindings() {
    if (state.imagePreview?.bindingsInitialized) {
        return;
    }
    if (elements.artifactImagePreviewOverlay) {
        elements.artifactImagePreviewOverlay.addEventListener('click', (event) => {
            if (event.target === elements.artifactImagePreviewOverlay) {
                closeArtifactImagePreview();
            }
        });
    }
    if (elements.artifactImagePreviewCloseBtn) {
        elements.artifactImagePreviewCloseBtn.addEventListener('click', () => {
            closeArtifactImagePreview();
        });
    }
    if (elements.artifactImagePreviewOpenFolderBtn) {
        elements.artifactImagePreviewOpenFolderBtn.addEventListener('click', () => {
            if (!state.imagePreview?.path) return;
            openSavedArtifactLocation(state.imagePreview.path, elements.artifactImagePreviewOpenFolderBtn);
        });
    }
    if (elements.artifactImagePreviewPrevBtn) {
        elements.artifactImagePreviewPrevBtn.addEventListener('click', () => {
            stepArtifactImagePreview(-1);
        });
    }
    if (elements.artifactImagePreviewNextBtn) {
        elements.artifactImagePreviewNextBtn.addEventListener('click', () => {
            stepArtifactImagePreview(1);
        });
    }
    document.addEventListener('keydown', (event) => {
        if (!state.imagePreview?.open) return;
        if (event.key === 'Escape') {
            event.preventDefault();
            closeArtifactImagePreview();
            return;
        }
        if (event.key === 'ArrowLeft') {
            event.preventDefault();
            stepArtifactImagePreview(-1);
            return;
        }
        if (event.key === 'ArrowRight') {
            event.preventDefault();
            stepArtifactImagePreview(1);
        }
    });
    window.addEventListener('resize', () => {
        if (!state.imagePreview?.open) return;
        syncArtifactImagePreviewLayout();
    });
    state.imagePreview = {
        ...(state.imagePreview || {}),
        bindingsInitialized: true,
    };
}

function openArtifactImagePreview({
    savedArtifactPath = '',
    viewUrl = '',
    fileName = '',
    mimeType = '',
    alt = '',
    conversationId = '',
} = {}) {
    if (!elements.artifactImagePreviewOverlay || !elements.artifactImagePreviewImage) {
        return;
    }
    const normalizedViewUrl = String(viewUrl || '').trim();
    if (!normalizedViewUrl) {
        return;
    }
    ensureArtifactImagePreviewBindings();
    const normalizedSavedArtifactPath = String(savedArtifactPath || '').trim();
    const normalizedFileName = String(fileName || '').trim() || inferArtifactFileName(normalizedSavedArtifactPath, {});
    const normalizedMimeType = String(mimeType || '').trim() || inferArtifactMimeType(normalizedSavedArtifactPath, {});
    const previewItems = resolveArtifactImagePreviewItems({
        conversationId,
        savedArtifactPath: normalizedSavedArtifactPath,
        viewUrl: normalizedViewUrl,
        fileName: normalizedFileName,
        mimeType: normalizedMimeType,
        alt,
    });
    if (!previewItems.items.length || previewItems.index < 0) {
        return;
    }
    state.imagePreview = {
        ...(state.imagePreview || {}),
        open: true,
        items: previewItems.items,
        index: previewItems.index,
        conversationId: previewItems.conversationId,
    };
    showArtifactImagePreviewItem(previewItems.index);
    elements.artifactImagePreviewOverlay.hidden = false;
    requestAnimationFrame(() => {
        elements.artifactImagePreviewOverlay.classList.add('open');
        elements.artifactImagePreviewCloseBtn?.focus();
    });
}

function appendRenderableArtifactToBubble(
    bubble,
    artifact,
    conversationId,
    message,
    {
        allowReference = false,
        deleteLabel = '',
    } = {}
) {
    const artifactType = String(artifact.type || '').trim().toLowerCase();
    const savedArtifactPath = String(artifact.path || '').trim();
    const artifactUnavailable = isArtifactUnavailable(artifact);
    const canAccessSavedArtifact = Boolean(savedArtifactPath) && !artifactUnavailable;
    const viewUrl = artifact.image_data_url || (canAccessSavedArtifact ? buildSavedArtifactViewUrl(savedArtifactPath) : '');
    const artifactFileName = inferArtifactFileName(savedArtifactPath, artifact);
    const artifactMimeType = inferArtifactMimeType(savedArtifactPath, artifact);
    const artifactActions = document.createElement('div');
    artifactActions.className = 'chat-image-actions';
    let artifactHasActions = false;
    const appendArtifactAction = (node) => {
        artifactActions.appendChild(node);
        artifactHasActions = true;
    };

    if (artifactType === 'image' && viewUrl) {
        const image = document.createElement('img');
        image.className = 'chat-inline-image';
        image.src = viewUrl;
        image.alt = String(
            artifact.prompt
            || artifact.name
            || artifact.image_state?.summary
            || 'Saved image artifact'
        ).trim();
        image.loading = 'lazy';
        image.tabIndex = 0;
        image.setAttribute('role', 'button');
        image.setAttribute('aria-label', 'Open image preview');
        image.title = 'Open image preview';
        image.addEventListener('click', () => {
            openArtifactImagePreview({
                savedArtifactPath,
                viewUrl,
                fileName: artifactFileName,
                mimeType: artifactMimeType,
                alt: image.alt,
                conversationId,
            });
        });
        image.addEventListener('keydown', (event) => {
            if (event.key !== 'Enter' && event.key !== ' ') {
                return;
            }
            event.preventDefault();
            openArtifactImagePreview({
                savedArtifactPath,
                viewUrl,
                fileName: artifactFileName,
                mimeType: artifactMimeType,
                alt: image.alt,
                conversationId,
            });
        });
        bindArtifactImageDrag(image, {
            savedArtifactPath,
            viewUrl,
            fileName: artifactFileName,
            mimeType: artifactMimeType,
        });
        bindConversationBottomAnchorToMedia(image, conversationId);
        bubble.appendChild(image);
    } else if (artifactType === 'audio' && viewUrl) {
        const audio = document.createElement('audio');
        audio.className = 'chat-inline-audio';
        audio.controls = true;
        audio.preload = 'metadata';
        audio.src = viewUrl;
        bindConversationBottomAnchorToMedia(audio, conversationId);
        bubble.appendChild(audio);
    }

    if (canAccessSavedArtifact) {
        const openArtifactButton = createChatMessageActionButton(
            '<i class="fas fa-folder-open"></i> Open Folder',
            () => {
                openSavedArtifactLocation(savedArtifactPath, openArtifactButton);
            }
        );
        appendArtifactAction(openArtifactButton);

        if (artifactIsTextLike(artifact, savedArtifactPath)) {
            const viewArtifactButton = createChatMessageActionButton(
                '<i class="fas fa-up-right-from-square"></i> View',
                () => {
                    window.open(buildSavedArtifactViewUrl(savedArtifactPath), '_blank', 'noopener,noreferrer');
                }
            );
            appendArtifactAction(viewArtifactButton);

            const copyArtifactButton = createChatMessageActionButton(
                '<i class="fas fa-copy"></i> Copy',
                () => {
                    copySavedArtifactContent(savedArtifactPath, copyArtifactButton);
                }
            );
            appendArtifactAction(copyArtifactButton);
        }

        if (allowReference && canUseArtifactAsReference(artifact)) {
            const referencePayload = buildMessageArtifactReferencePayload(artifact, message);
            if (referencePayload) {
                const referenceButton = createChatMessageActionButton('', () => {
                    setSelectedReferenceArtifact(referencePayload, { conversationId });
                });
                const isSelectedReference = isSelectedArtifactReferencePath(savedArtifactPath, conversationId);
                referenceButton.innerHTML = isSelectedReference
                    ? '<i class="fas fa-code-branch"></i> Pinned'
                    : `<i class="fas fa-code-branch"></i> ${escapeHtml(formatArtifactReferenceActionLabel(artifactType))}`;
                referenceButton.disabled = isSelectedReference;
                appendArtifactAction(referenceButton);
            }
        }
    }

    const availabilityText = formatArtifactAvailabilityText(artifact);
    if (availabilityText) {
        const status = document.createElement('div');
        status.className = 'chat-message-meta';
        status.textContent = availabilityText;
        bubble.appendChild(status);
    }

    if (artifactHasActions) {
        bubble.appendChild(artifactActions);
    }
}

function getChatArtifactDefaultLabel(type = '', index = 1) {
    switch (String(type || '').trim()) {
    case 'image':
        return `Image ${index}`;
    case 'audio':
        return `Audio ${index}`;
    case 'text':
        return `Text Artifact ${index}`;
    default:
        return `Artifact ${index}`;
    }
}

function appendChatArtifactCaption(bubble, text = '') {
    const captionText = String(text || '').trim();
    if (!captionText) return;
    const label = document.createElement('div');
    label.className = 'chat-artifact-caption';
    label.textContent = captionText;
    bubble.appendChild(label);
}

function chatArtifactIdentity(artifact = {}) {
    return String(
        artifact.artifact_ref
        || artifact.ref
        || artifact.artifact_id
        || artifact.path
        || artifact.image_data_url
        || ''
    ).trim();
}

function renderOrderedAssistantOutputs(body, message, artifacts, conversationId, { allowReference = false } = {}) {
    const lateFill = sanitizeMessageLateFill(message.lateFill || message.late_fill);
    const outputs = sanitizeResponseOutputs(message.outputs || message.canonical_outputs || message.canonicalOutputs);
    if (String(message.role || '').trim().toLowerCase() === 'user' || outputs.length < 2) {
        return false;
    }
    const visibleOutputs = typeof filterUserVisibleResponseWorkItems === 'function'
        ? filterUserVisibleResponseWorkItems(outputs)
        : outputs;
    const displayTextOutputs = visibleOutputs.filter((output) => shouldRenderAssistantOutputText(output, visibleOutputs, lateFill));
    const displayTextOutputSet = new Set(displayTextOutputs);
    const hasArtifactOutput = visibleOutputs.some((output) => (
        sanitizeResponseArtifacts(output.artifacts).length
        || String(output.artifact_ref || '').trim()
    ));
    if (!hasArtifactOutput && !artifacts.length) {
        return false;
    }
    const preservedPreviewText = getPreservedAssistantPreviewText(message, visibleOutputs, {
        displayTextOutputCount: displayTextOutputs.length,
    });

    const artifactsByIdentity = new Map();
    artifacts.forEach((artifact) => {
        const identity = chatArtifactIdentity(artifact);
        if (identity && !artifactsByIdentity.has(identity)) {
            artifactsByIdentity.set(identity, artifact);
        }
    });
    const renderedArtifacts = new Set();
    let renderedAny = false;

    const appendOutputArtifact = (artifact, output = {}) => {
        const identity = chatArtifactIdentity(artifact);
        if (identity && renderedArtifacts.has(identity)) {
            return;
        }
        if (identity) {
            renderedArtifacts.add(identity);
        }
        const artifactType = String(artifact.type || '').trim();
        const labelText = String(artifact.prompt || artifact.name || '').trim()
            || getChatArtifactDefaultLabel(artifactType, artifact.batch_index || output.batch_index || 1);
        appendChatArtifactCaption(body, labelText);
        appendRenderableArtifactToBubble(body, artifact, conversationId, message, {
            allowReference,
            deleteLabel: artifactType === 'text' ? 'text artifact' : artifactType || 'artifact',
        });
        renderedAny = true;
    };

    if (preservedPreviewText) {
        const block = document.createElement('div');
        block.className = 'chat-message__output-text chat-message__output-text--preview';
        block.innerHTML = formatMessageHtml(preservedPreviewText, { trustedHtml: Boolean(message.trustedHtml) });
        bindMarkdownCodeCopyButtons(block);
        bindMarkdownCodePreviewToggles(block);
        bindMarkdownBlockquoteCopyButtons(block);
        body.appendChild(block);
        renderedAny = true;
    }

    visibleOutputs.forEach((output) => {
        if (displayTextOutputSet.has(output)) {
            const value = String(output.value || '').trim();
            if (value) {
                const block = document.createElement('div');
                block.className = 'chat-message__output-text';
                block.innerHTML = formatMessageHtml(value, { trustedHtml: Boolean(message.trustedHtml) });
                bindMarkdownCodeCopyButtons(block);
                bindMarkdownCodePreviewToggles(block);
                bindMarkdownBlockquoteCopyButtons(block);
                body.appendChild(block);
                renderedAny = true;
            }
        }

        const outputArtifacts = sanitizeResponseArtifacts(output.artifacts);
        const artifactRef = String(output.artifact_ref || '').trim();
        if (artifactRef && !outputArtifacts.length && artifactsByIdentity.has(artifactRef)) {
            outputArtifacts.push(artifactsByIdentity.get(artifactRef));
        }
        outputArtifacts.forEach((artifact) => {
            appendOutputArtifact(artifact, output);
        });
    });
    artifacts.forEach((artifact) => {
        appendOutputArtifact(artifact, {});
    });

    return renderedAny || !displayTextOutputs.length;
}

function assistantPreviewTextIsGenericSummary(value = '') {
    const normalized = String(value || '').trim().toLowerCase();
    if (!normalized) return true;
    return /^(?:generated \d+ images?\.?|image generated(?: from the reference image)?\.?|audio generated\.?|text artifact saved\.?|transcript saved\.?|artifact generated\.?|artifacts generated\.?|completed\.?|received empty response\.?)$/.test(normalized)
        || /^generating \d+ (?:images|audio outputs)\.\.\.$/.test(normalized)
        || /^(?:image|audio) generation in progress\.\.\.$/.test(normalized);
}

function assistantPreviewTextHasStrongInternalMarker(value = '') {
    const normalized = String(value || '').trim().toLowerCase().replace(/\r\n/g, '\n');
    if (!normalized) return true;
    const firstLine = String(normalized.split('\n').find((line) => String(line || '').trim()) || '')
        .trim()
        .replace(/^#{1,6}\s+/, '')
        .replace(/^[*_`~\s]+|[*_`~\s]+$/g, '');
    return firstLine.startsWith('image generation prompts')
        || normalized.includes('original user request for bounded intent context')
        || normalized.includes('target text artifact:')
        || normalized.includes('deterministic syntax sanity issues')
        || normalized.includes('materialize only the requested text artifact')
        || normalized.includes('return only the complete file payload')
        || normalized.includes('update only the target text artifact')
        || (
            normalized.includes('unresolved linked artifact binding')
            && normalized.includes('resolved runtime artifacts')
        );
}

function assistantPreviewTextIsWorthPreserving(value = '') {
    const text = String(value || '').trim();
    if (!text) return false;
    if (assistantPreviewTextIsGenericSummary(text)) return false;
    if (assistantPreviewTextHasStrongInternalMarker(text)) return false;
    return text.length >= 24
        || text.includes('\n')
        || /[`*_#<>{}\[\]]/.test(text);
}

function getPreservedAssistantPreviewText(message = {}, visibleOutputs = [], options = {}) {
    if (Number(options.displayTextOutputCount || 0) > 0) return '';
    const content = String(message.content || '').trim();
    if (!assistantPreviewTextIsWorthPreserving(content)) return '';
    const normalizedContent = normalizeAssistantOutputTextForComparison(content);
    if (!normalizedContent) return '';
    const duplicatesVisibleText = sanitizeResponseOutputs(visibleOutputs).some((output) => {
        const value = normalizeAssistantOutputTextForComparison(
            output.value
            || output.content_payload
            || output.contentPayload
            || ''
        );
        return Boolean(value) && value === normalizedContent;
    });
    return duplicatesVisibleText ? '' : content;
}

function createChatMessageActionButton(innerHTML, onClick, className = 'chat-image-action') {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = className;
    button.innerHTML = innerHTML;
    if (typeof onClick === 'function') {
        button.addEventListener('click', onClick);
    }
    return button;
}

function formatArtifactReferenceActionLabel(artifactType = '') {
    const normalized = String(artifactType || '').trim().toLowerCase();
    const labels = {
        audio: 'Reference Audio',
        document: 'Reference Document',
        image: 'Reference Image',
        text: 'Reference Text',
    };
    return labels[normalized] || 'Reference Artifact';
}

function formatCompactWorkStatusLabel(lines = []) {
    const joined = (Array.isArray(lines) ? lines : []).join(' ').toLowerCase();
    if (!joined.trim()) return 'Status';
    if (
        joined.includes('failed')
        || joined.includes('blocked')
        || joined.includes('partial_failed')
        || joined.includes('partial failure')
        || joined.includes('repair')
        || joined.includes('syntax')
        || joined.includes('unmet')
        || joined.includes('open check')
    ) {
        return 'Needs Attention';
    }
    if (joined.includes('cancelled') || joined.includes('waived') || joined.includes('superseded') || joined.includes('resolved')) return 'Resolved';
    if (
        joined.includes('accepted')
        || joined.includes('queued')
        || joined.includes('generating')
        || joined.includes('running')
        || joined.includes('backend')
        || joined.includes('runtime')
        || joined.includes('branch')
        || joined.includes('follow-up work')
        || joined.includes('open')
        || joined.includes('pending')
    ) {
        return 'Working';
    }
    return 'Status';
}

function normalizeWorkGlowTone(value = '') {
    const token = String(value || '').trim().toLowerCase();
    if (token === 'active' || token === 'resolved') return token;
    return 'pending';
}

function workGlowToneForBranch(branch = {}) {
    const phase = normalizeCurrentWorkItemPhase(branch);
    return phase === 'running' ? 'active' : 'pending';
}

function inferMessageWorkGlowTone(message = {}, lines = []) {
    const lateFill = sanitizeMessageLateFill(message.lateFill || message.late_fill);
    if (typeof lateFillNeedsRepairAttention === 'function' && lateFillNeedsRepairAttention(lateFill)) {
        return 'pending';
    }
    const branchItems = [
        ...(lateFill?.activeBranches || []),
        ...(lateFill?.pendingBranches || []),
    ];
    if (branchItems.some((branch) => workGlowToneForBranch(branch) === 'active')) {
        return 'active';
    }
    if (branchItems.length) {
        return 'pending';
    }
    const joined = (Array.isArray(lines) ? lines : []).join(' ').toLowerCase();
    if (
        joined.includes('failed')
        || joined.includes('blocked')
        || joined.includes('partial_failed')
        || joined.includes('partial failure')
        || joined.includes('repair')
        || joined.includes('syntax')
        || joined.includes('unmet')
        || joined.includes('open check')
    ) {
        return 'pending';
    }
    if (joined.includes('resolved') || joined.includes('cancelled') || joined.includes('waived') || joined.includes('superseded')) {
        return 'resolved';
    }
    return joined.includes('generating') || joined.includes('running') ? 'active' : 'pending';
}

function shouldPrependAssistantWorkStatusMeta(workTone = '') {
    const tone = normalizeWorkGlowTone(workTone);
    return tone === 'active' || tone === 'pending';
}

function getTransientResolvedWorkStatusStore() {
    state.transientResolvedWorkStatusDismissed = state.transientResolvedWorkStatusDismissed || {};
    return state.transientResolvedWorkStatusDismissed;
}

function getTransientResolvedWorkStatusKey(message = {}, conversationId = '', lines = []) {
    const responseId = String(message.responseId || message.response_id || '').trim();
    const messageId = String(message.id || message.messageId || message.message_id || '').trim();
    const timestamp = String(message.timestamp || message.created_at || message.createdAt || '').trim();
    const stableId = responseId || messageId || timestamp || (Array.isArray(lines) ? lines.join('|') : '');
    if (!stableId) return '';
    return `${String(conversationId || 'conversation').trim() || 'conversation'}:${stableId}`;
}

function transientResolvedWorkStatusIsDismissed(key = '') {
    const normalized = String(key || '').trim();
    if (!normalized) return false;
    return Boolean(getTransientResolvedWorkStatusStore()[normalized]);
}

function markTransientResolvedWorkStatusDismissed(key = '') {
    const normalized = String(key || '').trim();
    if (!normalized) return;
    getTransientResolvedWorkStatusStore()[normalized] = true;
}

function scheduleTransientResolvedWorkStatusDismissal(meta, options = {}) {
    if (!(meta instanceof HTMLElement)) return;
    const transientKey = String(options.transientKey || '').trim();
    if (!transientKey) return;
    meta.classList.add('chat-message-work-compact--transient');
    meta.dataset.transientWorkStatus = 'resolved';
    const timerHost = typeof window !== 'undefined' ? window : null;
    if (!timerHost?.setTimeout) return;
    timerHost.setTimeout(() => {
        markTransientResolvedWorkStatusDismissed(transientKey);
        if (!meta.isConnected) return;
        meta.classList.add('is-expiring');
        timerHost.setTimeout(() => {
            if (meta.isConnected) meta.remove();
        }, TRANSIENT_RESOLVED_WORK_STATUS_FADE_MS);
    }, TRANSIENT_RESOLVED_WORK_STATUS_VISIBLE_MS);
}

function appendCompactWorkStatusMeta(bubble, lines = [], options = {}) {
    const cleanLines = Array.from(new Set(
        (Array.isArray(lines) ? lines : [])
            .map((line) => String(line || '').trim())
            .filter(Boolean)
    ));
    if (!cleanLines.length) return false;
    const label = formatCompactWorkStatusLabel(cleanLines);
    const tone = normalizeWorkGlowTone(options.tone);
    const meta = document.createElement('div');
    meta.className = `chat-message-meta chat-message-work-compact chat-message-work-compact--${tone}`;
    meta.dataset.workTone = tone;
    meta.title = cleanLines.join('\n');
    meta.setAttribute('aria-label', `${label}: ${cleanLines.join(' ')}`);
    const branchControls = Array.isArray(options.branchControls)
        ? options.branchControls.filter((item) => item instanceof HTMLElement)
        : [];
    branchControls.forEach((control) => meta.appendChild(control));
    if (!branchControls.length) {
        const indicator = document.createElement('span');
        indicator.className = 'chat-work-indicator';
        indicator.setAttribute('aria-hidden', 'true');
        meta.appendChild(indicator);
    }
    const labelElement = document.createElement('span');
    labelElement.textContent = label;
    meta.appendChild(labelElement);
    const setBranchControlHighlight = (enabled) => {
        bubble.querySelectorAll('[data-late-fill-control="cancel"]').forEach((button) => {
            button.classList.toggle('chat-image-action--work-highlight', Boolean(enabled));
        });
    };
    meta.addEventListener('mouseenter', () => setBranchControlHighlight(true));
    meta.addEventListener('mouseleave', () => setBranchControlHighlight(false));
    meta.addEventListener('focus', () => setBranchControlHighlight(true));
    meta.addEventListener('blur', () => setBranchControlHighlight(false));
    meta.tabIndex = 0;
    bubble.appendChild(meta);
    if (tone === 'resolved' && options.transientResolved === true && !branchControls.length) {
        scheduleTransientResolvedWorkStatusDismissal(meta, {
            transientKey: options.transientKey,
        });
    }
    return true;
}

function createCompactBranchControlDot(controlButton, branch = {}, options = {}) {
    if (!(controlButton instanceof HTMLElement)) return null;
    const tone = normalizeWorkGlowTone(options.tone || controlButton.dataset.workTone || workGlowToneForBranch(branch));
    const host = document.createElement('span');
    host.className = 'chat-work-branch-control';
    host.dataset.workTone = tone;
    host.dataset.branchId = String(branch.branchId || branch.branch_id || controlButton.dataset.branchId || '').trim();
    host.tabIndex = 0;
    host.title = controlButton.title || controlButton.getAttribute('aria-label') || '';
    host.setAttribute('aria-label', host.title || 'Branch action');

    const dot = document.createElement('span');
    dot.className = 'chat-work-branch-dot';
    dot.setAttribute('aria-hidden', 'true');
    host.appendChild(dot);

    const popover = document.createElement('span');
    popover.className = 'chat-work-branch-popover';
    popover.appendChild(controlButton);
    host.appendChild(popover);
    return host;
}

function createLateFillCancelControlButton(message, branch, conversationId) {
    const cancelButton = createChatMessageActionButton('', (event) => {
        controlLateFillBranch(
            message.responseId,
            branch.branchId,
            conversationId,
            {
                triggerButton: event.currentTarget,
                action: 'cancel',
            }
        );
    });
    const target = describeLateFillBranchControlTarget(branch, {
        actionLabel: 'Cancel',
        fallbackIndex: branch.queue_index,
    });
    cancelButton.innerHTML = `<i class="fas fa-ban"></i> ${escapeHtml(target.label)}`;
    cancelButton.title = target.title || 'Stop this queued or running late-fill branch';
    cancelButton.setAttribute('aria-label', target.ariaLabel || cancelButton.title);
    const workTone = workGlowToneForBranch(branch);
    cancelButton.classList.add(`chat-image-action--work-${workTone}`);
    cancelButton.dataset.workTone = workTone;
    cancelButton.dataset.branchId = branch.branchId;
    cancelButton.dataset.lateFillControl = 'cancel';
    return cancelButton;
}

function createLateFillRetryControlButton(message, branch, conversationId) {
    const retryButton = createChatMessageActionButton('', (event) => {
        retryLateFillBranch(
            message.responseId,
            branch.branchId,
            conversationId,
            {
                triggerButton: event.currentTarget,
                excludeInstanceIds: branch.excludeInstanceIds,
            }
        );
    });
    const label = branch.type === 'audio'
        ? 'Retry Audio'
        : branch.type === 'image'
            ? 'Retry Image'
            : 'Retry Output';
    retryButton.innerHTML = `<i class="fas fa-rotate-right"></i> ${escapeHtml(label)}`;
    retryButton.title = branch.reason || 'Retry this failed output branch';
    retryButton.setAttribute('aria-label', retryButton.title);
    retryButton.dataset.branchId = branch.branchId;
    retryButton.dataset.lateFillControl = 'retry';
    return retryButton;
}

function createCompactLateFillBranchControlsForMessage(message = {}, conversationId = '') {
    if (!message || message.role === 'user') return [];
    const controls = [];
    getControllableLateFillBranchesForMessage(message).forEach((branch) => {
        const button = createLateFillCancelControlButton(message, branch, conversationId);
        const dot = createCompactBranchControlDot(button, branch, { tone: button.dataset.workTone });
        if (dot) controls.push(dot);
    });
    getRecoverableLateFillBranchesForMessage(message).forEach((branch) => {
        const button = createLateFillRetryControlButton(message, branch, conversationId);
        const dot = createCompactBranchControlDot(button, branch, { tone: button.dataset.workTone || 'pending' });
        if (dot) controls.push(dot);
    });
    return controls;
}

function getRecoverableLateFillBranchesForMessage(message = {}) {
    const responseId = String(message.responseId || message.response_id || '').trim();
    if (!responseId) return [];
    const candidates = [
        ...sanitizeResponseOutputs(message.outputs || message.canonical_outputs || message.canonicalOutputs),
        ...sanitizeResponseOutputBranches(message.outputBranches || message.output_branches),
        ...sanitizeResponseOutputSlots(message.outputSlots || message.output_slots),
    ];
    const seen = new Set();
    const branches = [];
    candidates.forEach((item) => {
        const status = String(item.status || '').trim().toLowerCase();
        const recoveryContext = item.recovery_context && typeof item.recovery_context === 'object'
            ? item.recovery_context
            : null;
        const errorRef = item.error_ref && typeof item.error_ref === 'object'
            ? item.error_ref
            : null;
        const branchId = String(errorRef?.branch_id || item.branch_id || item.phase_id || '').trim();
        if (status !== 'blocked' || !branchId || recoveryContext?.can_retry !== true || seen.has(branchId)) {
            return;
        }
        seen.add(branchId);
        branches.push({
            branchId,
            type: String(item.type || '').trim().toLowerCase() || 'output',
            reason: String(item.blocked_reason || '').trim(),
            excludeInstanceIds: Array.isArray(recoveryContext.exclude_instance_ids)
                ? recoveryContext.exclude_instance_ids
                : [],
        });
    });
    return branches;
}

async function retryLateFillBranch(responseId, branchId, conversationId, options = {}) {
    const targetResponseId = String(responseId || '').trim();
    const targetBranchId = String(branchId || '').trim();
    if (!targetResponseId || !targetBranchId) return;
    const triggerButton = options.triggerButton || null;
    if (triggerButton) {
        triggerButton.disabled = true;
    }
    try {
        const body = {
            branch_id: targetBranchId,
        };
        if (Array.isArray(options.excludeInstanceIds) && options.excludeInstanceIds.length) {
            body.exclude_instance_ids = options.excludeInstanceIds;
        }
        const response = await axios.post(
            `${state.flaskServerUrl}/api/responses/${encodeURIComponent(targetResponseId)}/late_fill/retry`,
            body
        );
        const payload = response.data?.response || {};
        if (payload && typeof payload === 'object') {
            updateAssistantResponseByResponseId(conversationId, targetResponseId, payload, null);
            maybeWatchLateFillResponse(conversationId, payload, null);
        }
        updateGlobalModelStatus('Retrying failed output branch.');
    } catch (error) {
        const message = error.response?.data?.error || error.message || 'Could not retry output branch.';
        updateGlobalModelStatus(`Could not retry output branch: ${message}`);
    } finally {
        if (triggerButton) {
            triggerButton.disabled = false;
        }
    }
}

function getControllableLateFillBranchesForMessage(message = {}) {
    const responseId = String(message.responseId || message.response_id || '').trim();
    if (!responseId) return [];
    const lateFill = sanitizeMessageLateFill(message.lateFill || message.late_fill);
    if (!lateFillStatusIsActive(lateFill)) return [];
    const candidates = [
        ...(lateFill?.activeBranches || []),
        ...(lateFill?.pendingBranches || []),
    ];
    const seen = new Set();
    const branches = [];
    candidates.forEach((item) => {
        const branchId = String(item.branch_id || item.branchId || item.phase_id || item.phaseId || '').trim();
        if (!branchId || seen.has(branchId)) return;
        const status = String(item.status || '').trim().toLowerCase();
        if (!['pending', 'queued', 'scheduled', 'accepted', 'running'].includes(status)) return;
        seen.add(branchId);
        branches.push({
            branchId,
            branch_id: branchId,
            phaseId: String(item.phase_id || item.phaseId || '').trim(),
            phase_id: String(item.phase_id || item.phaseId || '').trim(),
            type: String(item.type || '').trim().toLowerCase() || normalizeCurrentWorkItemType(item) || 'output',
            capability: normalizeCapability(item.capability || item.follow_up_capability || item.followUpCapability || ''),
            output_type: String(item.output_type || item.outputType || item.type || '').trim().toLowerCase(),
            status,
            depends_on: Array.isArray(item.depends_on || item.dependsOn)
                ? (item.depends_on || item.dependsOn).map((value) => String(value || '').trim()).filter(Boolean)
                : [],
            batch_index: Number.isFinite(Number(item.batch_index ?? item.batchIndex))
                ? Number(item.batch_index ?? item.batchIndex)
                : null,
            queue_index: branches.length + 1,
            role: String(item.role || '').trim(),
            source: String(item.source || '').trim(),
            source_name: String(item.source_name || item.sourceName || '').trim(),
            content_payload_source: String(item.content_payload_source || item.contentPayloadSource || '').trim(),
        });
    });
    return branches;
}

async function controlLateFillBranch(responseId, branchId, conversationId, options = {}) {
    const targetResponseId = String(responseId || '').trim();
    const targetBranchId = String(branchId || '').trim();
    if (!targetResponseId || !targetBranchId) return;
    const triggerButton = options.triggerButton || null;
    if (triggerButton) {
        triggerButton.disabled = true;
    }
    try {
        const response = await axios.post(
            `${state.flaskServerUrl}/api/responses/${encodeURIComponent(targetResponseId)}/late_fill/control`,
            {
                branch_id: targetBranchId,
                action: options.action || 'cancel',
                reason: options.reason || 'user stopped this late-fill branch',
            }
        );
        const payload = response.data?.response || {};
        if (payload && typeof payload === 'object') {
            updateAssistantResponseByResponseId(conversationId, targetResponseId, payload, null);
            maybeWatchLateFillResponse(conversationId, payload, null);
        }
        updateGlobalModelStatus('Late-fill branch marked for cancellation.');
    } catch (error) {
        const message = error.response?.data?.error || error.message || 'Could not control late-fill branch.';
        updateGlobalModelStatus(`Could not stop late-fill branch: ${message}`);
    } finally {
        if (triggerButton) {
            triggerButton.disabled = false;
        }
    }
}

function bindMarkdownCodeCopyButtons(root) {
    if (!root) return;
    root.querySelectorAll('[data-copy-code-block]').forEach((button) => {
        if (button.dataset.copyBound === '1') return;
        button.dataset.copyBound = '1';
        button.addEventListener('click', async (event) => {
            event.stopPropagation();
            const codeElement = button.closest('.chat-markdown__code-block')?.querySelector('pre code');
            const codeText = String(codeElement?.textContent || '');
            if (!codeText.trim()) {
                showCopyStatus('No code to copy.');
                return;
            }
            await copyMessageToClipboard(codeText, {
                successMessage: 'Code copied to clipboard.',
                failureMessage: 'Unable to copy code.',
            });
            const originalLabel = button.dataset.copyLabel || 'Copy';
            button.dataset.copyLabel = originalLabel;
            button.classList.add('is-copied');
            button.innerHTML = '<i class="fas fa-check"></i> Copied';
            if (button._copyResetTimer) {
                window.clearTimeout(button._copyResetTimer);
            }
            button._copyResetTimer = window.setTimeout(() => {
                button.classList.remove('is-copied');
                button.innerHTML = `<i class="fas fa-copy"></i> ${escapeHtml(originalLabel)}`;
                button._copyResetTimer = null;
            }, 1500);
        });
    });
}

function bindMarkdownCodePreviewToggles(root) {
    if (!root) return;
    root.querySelectorAll('[data-toggle-code-preview]').forEach((button) => {
        if (button.dataset.toggleBound === '1') return;
        const block = button.closest('.chat-markdown__code-block--collapsible');
        if (!block) return;
        button.dataset.toggleBound = '1';
        const totalLines = Number(block.dataset.codeLineCount || 0);
        const update = () => {
            const expanded = block.classList.contains('is-expanded');
            block.classList.toggle('is-collapsed', !expanded);
            button.setAttribute('aria-expanded', String(expanded));
            button.innerHTML = expanded
                ? '<i class="fas fa-compress-alt"></i> Collapse'
                : `<i class="fas fa-expand-alt"></i> Show all${totalLines ? ` ${totalLines}` : ''} lines`;
        };
        button.addEventListener('click', (event) => {
            event.stopPropagation();
            block.classList.toggle('is-expanded');
            update();
        });
        update();
    });
}

function bindMarkdownBlockquoteCopyButtons(root) {
    if (!root) return;
    root.querySelectorAll('blockquote').forEach((quote) => {
        if (quote.querySelector('[data-copy-blockquote]')) {
            return;
        }
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'chat-markdown__quote-copy';
        button.setAttribute('aria-label', 'Copy quote');
        button.setAttribute('title', 'Copy quote');
        button.setAttribute('data-copy-blockquote', '1');
        button.innerHTML = '<i class="fas fa-copy"></i>';
        button.addEventListener('click', async (event) => {
            event.stopPropagation();
            const quoteText = String(quote.textContent || '').trim();
            if (!quoteText) {
                showCopyStatus('No quote to copy.');
                return;
            }
            await copyMessageToClipboard(quoteText, {
                successMessage: 'Quote copied to clipboard.',
                failureMessage: 'Unable to copy quote.',
            });
        });
        quote.appendChild(button);
    });
}

function createChatMessageElement(message, conversationId = '') {
    const isUser = message.role === 'user';
    const isAssistantMessage = !isUser
        && !message.ephemeralUiNotice;
    const allowReferenceActions = !state.arena.enabled && !message.isLoading && !message.ephemeralUiNotice;
    const requestSnapshot = sanitizeRequestSnapshot(message.requestSnapshot || message.request_snapshot);
    const artifacts = buildRenderableMessageArtifacts(message);
    const wrapper = document.createElement('div');
    wrapper.className = `chat-message flex ${isUser ? 'justify-end' : 'justify-start'} mb-3`;
    if (message.clientMessageId) {
        wrapper.dataset.clientMessageId = message.clientMessageId;
    }

    const bubble = document.createElement('div');
    bubble.className = `p-3 rounded-lg shadow ${isUser ? 'user-message ml-auto' : 'bot-message mr-auto'}`;
    const body = document.createElement('div');
    body.className = 'chat-message__body';
    body.dataset.messageBody = '1';
    const orderedOutputsRendered = renderOrderedAssistantOutputs(body, message, artifacts, conversationId, {
        allowReference: allowReferenceActions,
    });
    if (!orderedOutputsRendered) {
        body.innerHTML = formatMessageHtml(
            getRenderableMessageContent(message, artifacts),
            { trustedHtml: Boolean(message.trustedHtml) }
        );
        bindMarkdownCodeCopyButtons(body);
        bindMarkdownCodePreviewToggles(body);
        bindMarkdownBlockquoteCopyButtons(body);
    }
    let assistantWorkStatusMeta = null;
    if (isAssistantMessage) {
        const lateFillStatusLines = formatAssistantCurrentWorkStatusLines(message);
        const compactBranchControls = createCompactLateFillBranchControlsForMessage(message, conversationId);
        if (lateFillStatusLines.length) {
            const workTone = inferMessageWorkGlowTone(message, lateFillStatusLines);
            const transientResolved = workTone === 'resolved' && !compactBranchControls.length;
            const transientKey = transientResolved
                ? getTransientResolvedWorkStatusKey(message, conversationId, lateFillStatusLines)
                : '';
            if (!transientResolved || !transientResolvedWorkStatusIsDismissed(transientKey)) {
                assistantWorkStatusMeta = {
                    lines: lateFillStatusLines,
                    tone: workTone,
                    branchControls: compactBranchControls,
                    transientResolved,
                    transientKey,
                };
            }
        }
    }
    bubble.appendChild(body);
    if (isAssistantMessage) {
        const provenanceText = formatAssistantProvenanceText(message, conversationId);
        if (provenanceText) {
            const meta = document.createElement('div');
            meta.className = 'chat-message-meta';
            meta.textContent = provenanceText;
            bubble.appendChild(meta);
        }
        if (assistantWorkStatusMeta) {
            appendCompactWorkStatusMeta(bubble, assistantWorkStatusMeta.lines, {
                tone: assistantWorkStatusMeta.tone,
                branchControls: assistantWorkStatusMeta.branchControls,
                transientResolved: assistantWorkStatusMeta.transientResolved,
                transientKey: assistantWorkStatusMeta.transientKey,
            });
        }
        const artifactProvenanceText = formatAssistantArtifactProvenanceText(message);
        if (artifactProvenanceText) {
            const artifactMeta = document.createElement('div');
            artifactMeta.className = 'chat-message-meta';
            artifactMeta.textContent = artifactProvenanceText;
            bubble.appendChild(artifactMeta);
        }
        appendResponseArtifactBundleCards(bubble, message);
    }

    const actions = document.createElement('div');
    actions.className = 'chat-image-actions';
    let hasActions = false;
    const artifactCountsByType = artifacts.reduce((counts, artifact) => {
        const type = String(artifact.type || '').trim();
        if (!type) return counts;
        counts[type] = (counts[type] || 0) + 1;
        return counts;
    }, {});
    const artifactIndexByType = {};
    if (!orderedOutputsRendered) artifacts.forEach((artifact) => {
        const artifactType = String(artifact.type || '').trim();
        if (!artifactType) return;
        artifactIndexByType[artifactType] = (artifactIndexByType[artifactType] || 0) + 1;
        const artifactIndex = artifactIndexByType[artifactType];
        const artifactCount = artifactCountsByType[artifactType] || 1;
        const labelText = isUser
            ? (
                (() => {
                    const baseLabel = String(
                        artifact.name
                        || basenameFromPath(artifact.source_path || artifact.path || '')
                        || artifact.prompt
                        || ''
                    ).trim() || (artifactCount > 1 ? getChatArtifactDefaultLabel(artifactType, artifactIndex) : '');
                    return baseLabel ? `Input: ${baseLabel}` : '';
                })()
            )
            : (
                String(artifact.prompt || artifact.name || '').trim()
                || (artifactCount > 1 ? getChatArtifactDefaultLabel(artifactType, artifact.batch_index || artifactIndex) : '')
            );
        if (labelText) {
            appendChatArtifactCaption(bubble, labelText);
        }
        appendRenderableArtifactToBubble(bubble, artifact, conversationId, message, {
            allowReference: allowReferenceActions,
            deleteLabel: isUser
                ? 'input artifact'
                : (artifactType === 'text' ? 'text artifact' : artifactType || 'artifact'),
        });
    });

    const bundleAction = getResponseArtifactBundleActionState(message, artifacts);
    if (isAssistantMessage && allowReferenceActions && bundleAction.visible) {
        const bundleButton = createChatMessageActionButton(`<i class="${bundleAction.icon}"></i> ${bundleAction.label}`, (event) => {
            if (bundleAction.mode === 'open') {
                const bundlePath = String(bundleAction.bundle?.bundle_path || bundleAction.bundle?.bundlePath || '').trim();
                openSavedArtifactLocation(bundlePath, event.currentTarget);
                return;
            }
            bundleResponseArtifactsForMessage(message, conversationId, event.currentTarget);
        });
        bundleButton.disabled = !bundleAction.enabled;
        bundleButton.title = bundleAction.title;
        bundleButton.setAttribute('aria-label', bundleButton.title);
        actions.appendChild(bundleButton);
        hasActions = true;
    }

    if (isAssistantMessage && allowReferenceActions && String(message.content || '').trim()) {
        const isPinnedReply = isSelectedMessageReferenceForMessage(message);
        const referenceButton = createChatMessageActionButton('', () => {
            setSelectedReferenceArtifact({
                type: 'message',
                content: String(message.content || '').trim(),
                message_role: 'assistant',
                message_id: String(message.clientMessageId || '').trim() || null,
                response_model: String(message.responseModel || '').trim() || null,
                response_instance_id: String(message.responseInstanceId || '').trim() || null,
                timestamp: String(message.timestamp || '').trim() || null,
            }, { conversationId });
        });
        referenceButton.innerHTML = isPinnedReply
            ? '<i class="fas fa-code-branch"></i> Pinned Reply'
            : '<i class="fas fa-code-branch"></i> Reference Reply';
        referenceButton.disabled = isPinnedReply;
        actions.appendChild(referenceButton);
        hasActions = true;
    }

    const userPromptText = isUser
        ? String(requestSnapshot?.prompt_text || requestSnapshot?.promptText || '').trim()
        : '';
    if (isUser && allowReferenceActions && userPromptText) {
        const isPinnedPrompt = isSelectedMessageReferenceForMessage(message);
        const referenceButton = createChatMessageActionButton('', () => {
            setSelectedReferenceArtifact({
                type: 'message',
                content: userPromptText,
                message_role: 'user',
                message_id: String(message.clientMessageId || '').trim() || null,
                timestamp: String(message.timestamp || '').trim() || null,
            }, { conversationId });
        });
        referenceButton.innerHTML = isPinnedPrompt
            ? '<i class="fas fa-code-branch"></i> Pinned Prompt'
            : '<i class="fas fa-code-branch"></i> Use Prompt As Reference';
        referenceButton.disabled = isPinnedPrompt;
        actions.appendChild(referenceButton);
        hasActions = true;
    }

    if (hasActions) {
        bubble.appendChild(actions);
    }

    const copyBtn = document.createElement('button');
    copyBtn.type = 'button';
    copyBtn.className = 'message-copy-btn';
    copyBtn.setAttribute('aria-label', 'Copy message');
    copyBtn.innerHTML = '<i class="fas fa-copy"></i>';
    copyBtn.addEventListener('click', (event) => {
        event.stopPropagation();
        copyMessageToClipboard(message.content || '');
    });
    bubble.appendChild(copyBtn);

    wrapper.appendChild(bubble);
    return wrapper;
}

function escapeHtml(value = '') {
    return String(value || '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function escapeHtmlAttribute(value = '') {
    return escapeHtml(value).replace(/`/g, '&#96;');
}

function sanitizeMarkdownUrl(value = '') {
    const raw = String(value || '').trim();
    if (!raw) return '';
    if (/^mailto:/i.test(raw)) {
        return raw.replace(/\s+/g, '');
    }
    try {
        const parsed = new URL(raw, window.location.origin);
        if (parsed.protocol === 'http:' || parsed.protocol === 'https:') {
            return parsed.href;
        }
    } catch (_error) {
        return '';
    }
    return '';
}

function createMarkdownTokenStore() {
    const tokens = [];
    return {
        add(html) {
            const token = `@@MDTOKEN${tokens.length}@@`;
            tokens.push({ token, html });
            return token;
        },
        restore(text) {
            return tokens.reduce(
                (result, entry) => result.replaceAll(entry.token, entry.html),
                String(text || '')
            );
        },
    };
}

function parseMarkdownTableRow(line = '') {
    let working = String(line || '').trim();
    if (working.startsWith('|')) working = working.slice(1);
    if (working.endsWith('|')) working = working.slice(0, -1);
    return working.split('|').map(cell => cell.trim());
}

function isMarkdownTableSeparator(line = '') {
    const cells = parseMarkdownTableRow(line);
    return cells.length > 0 && cells.every(cell => /^:?-{3,}:?$/.test(cell));
}

function markdownTableAlignment(separatorCell = '') {
    const cell = String(separatorCell || '').trim();
    if (cell.startsWith(':') && cell.endsWith(':')) return 'center';
    if (cell.endsWith(':')) return 'right';
    if (cell.startsWith(':')) return 'left';
    return '';
}

function renderMarkdownInline(text = '') {
    const escaped = escapeHtml(String(text || ''));
    const tokens = createMarkdownTokenStore();
    let output = escaped;

    output = output.replace(/`([^`\n]+)`/g, (_match, code) => tokens.add(`<code>${code}</code>`));

    output = output.replace(/\[([^\]]+)\]\(([^)\s]+)(?:\s+\"([^\"]*)\")?\)/g, (_match, label, href, title) => {
        const safeHref = sanitizeMarkdownUrl(href);
        if (!safeHref) return label;
        const titleAttr = title ? ` title="${escapeHtmlAttribute(title)}"` : '';
        return tokens.add(
            `<a href="${escapeHtmlAttribute(safeHref)}" target="_blank" rel="noopener noreferrer"${titleAttr}>${label}</a>`
        );
    });

    output = output.replace(/(^|[\s(])((?:https?:\/\/|mailto:)[^\s<]+)/g, (match, prefix, href) => {
        const safeHref = sanitizeMarkdownUrl(href);
        if (!safeHref) return match;
        return `${prefix}${tokens.add(
            `<a href="${escapeHtmlAttribute(safeHref)}" target="_blank" rel="noopener noreferrer">${escapeHtml(href)}</a>`
        )}`;
    });

    output = output.replace(/\*\*\*([^*\n]+)\*\*\*/g, '<strong><em>$1</em></strong>');
    output = output.replace(/___([^_\n]+)___/g, '<strong><em>$1</em></strong>');
    output = output.replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>');
    output = output.replace(/__([^_\n]+)__/g, '<strong>$1</strong>');
    output = output.replace(/~~([^~\n]+)~~/g, '<del>$1</del>');
    output = output.replace(/(^|[\s(])\*([^*\n][^*\n]*?)\*(?=$|[\s),.!?;:])/g, '$1<em>$2</em>');
    output = output.replace(/(^|[\s(])_([^_\n][^_\n]*?)_(?=$|[\s),.!?;:])/g, '$1<em>$2</em>');

    return tokens.restore(output);
}

function renderMarkdownTable(lines = []) {
    if (lines.length < 2) return '';
    const headers = parseMarkdownTableRow(lines[0]);
    const alignments = parseMarkdownTableRow(lines[1]).map(markdownTableAlignment);
    const rows = lines.slice(2).map(parseMarkdownTableRow);
    const thead = `<thead><tr>${headers.map((cell, index) => {
        const align = alignments[index] ? ` style="text-align:${alignments[index]}"` : '';
        return `<th${align}>${renderMarkdownInline(cell)}</th>`;
    }).join('')}</tr></thead>`;
    const tbody = rows.length
        ? `<tbody>${rows.map((row) => `<tr>${headers.map((_cell, index) => {
            const align = alignments[index] ? ` style="text-align:${alignments[index]}"` : '';
            return `<td${align}>${renderMarkdownInline(row[index] || '')}</td>`;
        }).join('')}</tr>`).join('')}</tbody>`
        : '';
    return `<table>${thead}${tbody}</table>`;
}

const COLLAPSIBLE_CODE_PREVIEW_LINE_LIMIT = 15;

function normalizeMarkdownCodeLanguage(language = '') {
    return String(language || '')
        .trim()
        .toLowerCase()
        .split(/\s+/)[0]
        .replace(/[^a-z0-9_+-]/g, '');
}

function lineCountForCodeBlock(code = '') {
    const normalized = String(code || '').replace(/\r\n?/g, '\n');
    if (!normalized) return 0;
    return normalized.split('\n').length;
}

function isHtmlCssCodePreview(language = '', code = '') {
    const normalizedLanguage = normalizeMarkdownCodeLanguage(language);
    if (['html', 'htm', 'css', 'scss', 'sass'].includes(normalizedLanguage)) {
        return true;
    }
    if (normalizedLanguage) return false;
    const sample = String(code || '').trimStart().slice(0, 1600);
    if (/^(<!doctype\s+html\b|<html\b|<head\b|<body\b|<style\b)/i.test(sample)) {
        return true;
    }
    if (/^<[a-z][\w:-]*(?:\s[^>]*)?>[\s\S]*<\/[a-z][\w:-]*>/i.test(sample)) {
        return true;
    }
    return /^(?:\/\*[\s\S]*?\*\/\s*)?(?::root|html|body|[#.][\w-]+|[a-z][\w-]*(?:\s|,|>|\.|#|\[|:))[^{]{0,120}\{/i.test(sample);
}

function renderMarkdownToHtml(markdown = '') {
    const source = String(markdown || '').replace(/\r\n?/g, '\n').trim();
    if (!source) return '';

    const blockTokens = createMarkdownTokenStore();
    let normalized = source.replace(/```([^\n`]*)\n([\s\S]*?)```/g, (_match, language, code) => {
        const label = String(language || '').trim();
        const lineCount = lineCountForCodeBlock(code.replace(/\n$/, ''));
        const shouldCollapse = lineCount > COLLAPSIBLE_CODE_PREVIEW_LINE_LIMIT
            && isHtmlCssCodePreview(label, code);
        const labelHtml = label
            ? `<div class="chat-markdown__code-label">${escapeHtml(label)}</div>`
            : '';
        const toolbarLead = labelHtml || '<span class="chat-markdown__code-toolbar-spacer" aria-hidden="true"></span>';
        const toggleButton = shouldCollapse
            ? '<button type="button" class="chat-markdown__code-toggle" data-toggle-code-preview="1" aria-expanded="false"><i class="fas fa-expand-alt"></i> Show all lines</button>'
            : '';
        const blockClass = shouldCollapse
            ? 'chat-markdown__code-block chat-markdown__code-block--collapsible is-collapsed'
            : 'chat-markdown__code-block';
        const blockMeta = shouldCollapse
            ? ` data-code-line-count="${lineCount}" data-code-preview-limit="${COLLAPSIBLE_CODE_PREVIEW_LINE_LIMIT}"`
            : '';
        return blockTokens.add(
            `<div class="${blockClass}"${blockMeta}>`
            + `<div class="chat-markdown__code-toolbar">${toolbarLead}<div class="chat-markdown__code-actions">${toggleButton}<button type="button" class="chat-markdown__code-copy" data-copy-code-block="1" aria-label="Copy code block"><i class="fas fa-copy"></i> Copy</button></div></div>`
            + `<pre><code>${escapeHtml(code.replace(/\n$/, ''))}</code></pre>`
            + `</div>`
        );
    });

    const lines = normalized.split('\n');
    const htmlParts = [];
    let paragraphLines = [];
    let listState = null;

    const flushParagraph = () => {
        if (!paragraphLines.length) return;
        htmlParts.push(`<p>${paragraphLines.map(line => renderMarkdownInline(line)).join('<br>')}</p>`);
        paragraphLines = [];
    };

    const flushList = () => {
        if (!listState || !listState.items.length) {
            listState = null;
            return;
        }
        const tag = listState.ordered ? 'ol' : 'ul';
        htmlParts.push(`<${tag}>${listState.items.map(item => `<li>${renderMarkdownInline(item)}</li>`).join('')}</${tag}>`);
        listState = null;
    };

    for (let index = 0; index < lines.length; index += 1) {
        const rawLine = lines[index];
        const trimmed = rawLine.trim();

        if (!trimmed) {
            flushParagraph();
            flushList();
            continue;
        }

        if (trimmed.startsWith('@@MDTOKEN') && trimmed.endsWith('@@')) {
            flushParagraph();
            flushList();
            htmlParts.push(trimmed);
            continue;
        }

        if (trimmed.includes('|') && index + 1 < lines.length && isMarkdownTableSeparator(lines[index + 1])) {
            flushParagraph();
            flushList();
            const tableLines = [trimmed, lines[index + 1].trim()];
            index += 2;
            while (index < lines.length && lines[index].trim() && lines[index].includes('|')) {
                tableLines.push(lines[index].trim());
                index += 1;
            }
            index -= 1;
            htmlParts.push(renderMarkdownTable(tableLines));
            continue;
        }

        if (/^ {0,3}(?:[-*_]\s*){3,}$/.test(trimmed)) {
            flushParagraph();
            flushList();
            htmlParts.push('<hr>');
            continue;
        }

        const headingMatch = trimmed.match(/^(#{1,6})\s+(.*)$/);
        if (headingMatch) {
            flushParagraph();
            flushList();
            const level = headingMatch[1].length;
            htmlParts.push(`<h${level}>${renderMarkdownInline(headingMatch[2])}</h${level}>`);
            continue;
        }

        if (trimmed.startsWith('>')) {
            flushParagraph();
            flushList();
            const quoteLines = [trimmed.replace(/^>\s?/, '')];
            while (index + 1 < lines.length && lines[index + 1].trim().startsWith('>')) {
                index += 1;
                quoteLines.push(lines[index].trim().replace(/^>\s?/, ''));
            }
            htmlParts.push(`<blockquote>${renderMarkdownToHtml(quoteLines.join('\n'))}</blockquote>`);
            continue;
        }

        const orderedMatch = trimmed.match(/^(\d+)\.\s+(.*)$/);
        const unorderedMatch = trimmed.match(/^[-*+]\s+(.*)$/);
        if (orderedMatch || unorderedMatch) {
            flushParagraph();
            const ordered = Boolean(orderedMatch);
            const itemText = ordered ? orderedMatch[2] : unorderedMatch[1];
            if (!listState || listState.ordered !== ordered) {
                flushList();
                listState = { ordered, items: [] };
            }
            listState.items.push(itemText);
            continue;
        }

        flushList();
        paragraphLines.push(trimmed);
    }

    flushParagraph();
    flushList();
    return blockTokens.restore(htmlParts.join(''));
}

function formatMessageHtml(content = '', options = {}) {
    if (options.trustedHtml) {
        return String(content || '');
    }
    return renderMarkdownToHtml(content);
}

function buildSavedArtifactViewUrl(savedPath = '') {
    return buildSavedArtifactUrl(savedPath, 'view_saved_artifact');
}

function buildSavedArtifactUrl(savedPath = '', endpoint = 'view_saved_artifact') {
    const targetPath = String(savedPath || '').trim();
    if (!targetPath) return '';
    if (/^(?:https?:|data:|blob:)/i.test(targetPath)) {
        return targetPath;
    }
    if (targetPath.startsWith(`${state.flaskServerUrl}/api/`)) {
        return targetPath;
    }
    if (targetPath.startsWith('/api/')) {
        return `${state.flaskServerUrl}${targetPath}`;
    }
    return `${state.flaskServerUrl}/api/${endpoint}?path=${encodeURIComponent(targetPath)}`;
}

async function copyMessageToClipboard(text, options = {}) {
    const successMessage = String(options.successMessage || 'Message copied to clipboard.');
    const failureMessage = String(options.failureMessage || 'Unable to copy message.');
    try {
        if (navigator.clipboard && navigator.clipboard.writeText) {
            await navigator.clipboard.writeText(text);
        } else {
            const textarea = document.createElement('textarea');
            textarea.value = text;
            textarea.style.position = 'fixed';
            textarea.style.opacity = '0';
            document.body.appendChild(textarea);
            textarea.focus();
            textarea.select();
            document.execCommand('copy');
            document.body.removeChild(textarea);
        }
        showCopyStatus(successMessage);
    } catch (error) {
        console.error('Clipboard copy failed:', error);
        showCopyStatus(failureMessage);
    }
}

async function openSavedImageLocation(savedImagePath, triggerButton = null) {
    const targetPath = String(savedImagePath || '').trim();
    if (!targetPath) {
        updateGlobalModelStatus('Image path unavailable for local open.');
        return;
    }
    if (triggerButton) {
        triggerButton.disabled = true;
    }
    try {
        await axios.post(`${state.flaskServerUrl}/api/open_saved_image`, { path: targetPath });
        updateGlobalModelStatus('Opened image location in file manager.');
        setTimeout(() => updateGlobalModelStatus(''), 1800);
    } catch (error) {
        const message = error.response?.data?.error || error.message || 'Unknown error';
        updateGlobalModelStatus(`Could not open image location: ${message}`);
    } finally {
        if (triggerButton) {
            triggerButton.disabled = false;
        }
    }
}

async function openSavedArtifactLocation(savedPath, triggerButton = null) {
    const targetPath = String(savedPath || '').trim();
    if (!targetPath) {
        updateGlobalModelStatus('Artifact path unavailable for local open.');
        return;
    }
    if (triggerButton) {
        triggerButton.disabled = true;
    }
    try {
        await axios.post(`${state.flaskServerUrl}/api/open_saved_artifact`, { path: targetPath });
        updateGlobalModelStatus('Opened artifact location in file manager.');
        setTimeout(() => updateGlobalModelStatus(''), 1800);
    } catch (error) {
        const message = error.response?.data?.error || error.message || 'Unknown error';
        updateGlobalModelStatus(`Could not open artifact location: ${message}`);
    } finally {
        if (triggerButton) {
            triggerButton.disabled = false;
        }
    }
}

async function openSavedArtifactEntry(savedPath, triggerButton = null) {
    const targetPath = String(savedPath || '').trim();
    if (!targetPath) {
        updateGlobalModelStatus('Bundle entry path unavailable for local open.');
        return;
    }
    if (triggerButton) {
        triggerButton.disabled = true;
    }
    try {
        await axios.post(`${state.flaskServerUrl}/api/open_saved_artifact`, {
            path: targetPath,
            open_file: true,
        });
        updateGlobalModelStatus('Opened bundle entry.');
        setTimeout(() => updateGlobalModelStatus(''), 1800);
    } catch (error) {
        const message = error.response?.data?.error || error.message || 'Unknown error';
        updateGlobalModelStatus(`Could not open bundle entry: ${message}`);
    } finally {
        if (triggerButton) {
            triggerButton.disabled = false;
        }
    }
}

function removeDeletedArtifactFromConversation(conversationId, savedPath, artifactType = 'artifact') {
    const targetPath = String(savedPath || '').trim();
    if (!targetPath || !state.conversations[conversationId]) return;
    if (String(state.imagePreview?.path || '').trim() === targetPath) {
        closeArtifactImagePreview();
    }
    if (isSelectedArtifactReferencePath(targetPath, conversationId)) {
        removeSelectedReferenceArtifactByPath(targetPath, { quiet: true, conversationId });
    }
    const purgedAt = new Date().toISOString();
    let changed = false;
    state.conversations[conversationId].forEach((message) => {
        syncMessageArtifactLedger(message);
        const currentArtifacts = sanitizeResponseArtifacts(message.artifacts);
        let messageChanged = false;
        const nextArtifacts = currentArtifacts.map((artifact) => {
            if (String(artifact.path || '').trim() !== targetPath) {
                return artifact;
            }
            messageChanged = true;
            return sanitizeResponseArtifacts([
                {
                    ...artifact,
                    availability: 'purged',
                    purged_at: purgedAt,
                    purge_reason: String(artifactType || 'artifact').trim() || 'artifact',
                    availability_checked_at: purgedAt,
                },
            ])[0] || artifact;
        });
        if (!messageChanged) {
            return;
        }
        message.artifacts = nextArtifacts;
        const requestSnapshot = sanitizeRequestSnapshot(message.requestSnapshot || message.request_snapshot);
        if (String(message.role || '').trim().toLowerCase() === 'user') {
            message.requestSnapshot = mergeRequestSnapshotInputArtifacts(requestSnapshot, nextArtifacts);
        }
        changed = true;
    });
    if (!changed) return;
    if (isConversationVisible(conversationId)) {
        renderConversation(conversationId);
    }
    persistChatHistory(conversationId);
}

async function deleteSavedArtifact(savedPath, conversationId, artifactType = 'artifact', triggerButton = null) {
    const targetPath = String(savedPath || '').trim();
    if (!targetPath) {
        updateGlobalModelStatus('Artifact path unavailable for delete.');
        return;
    }
    if (triggerButton) {
        triggerButton.disabled = true;
    }
    try {
        await axios.post(`${state.flaskServerUrl}/api/delete_saved_artifact`, { path: targetPath });
        removeDeletedArtifactFromConversation(conversationId, targetPath, artifactType);
        updateGlobalModelStatus(`Deleted ${artifactType}.`);
        setTimeout(() => updateGlobalModelStatus(''), 1800);
    } catch (error) {
        const message = error.response?.data?.error || error.message || 'Unknown error';
        updateGlobalModelStatus(`Could not delete ${artifactType}: ${message}`);
    } finally {
        if (triggerButton) {
            triggerButton.disabled = false;
        }
    }
}

function showCopyStatus(message) {
    updateGlobalModelStatus(message);
    if (copyStatusTimeout) {
        clearTimeout(copyStatusTimeout);
    }
    copyStatusTimeout = setTimeout(() => {
        if (elements.modelStatus && elements.modelStatus.textContent?.includes(message)) {
            updateGlobalModelStatus('');
        }
    }, 1500);
}
