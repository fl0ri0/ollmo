function requestLifecycleTargetIsExternal(instance) {
    return Boolean(
        instance
        && (
            String(instance.target_kind || '').trim().toLowerCase() === 'external'
            || String(instance.instance_id || '').trim() === 'external:codex'
        )
    );
}

function isInstanceRequestPending(instanceId) {
    return Boolean(state.inference?.pendingByInstance?.[instanceId]);
}

function listPendingRequests() {
    return Object.values(state.inference?.pendingRequests || {});
}

function persistPendingRequestsSnapshot() {
    try {
        const requests = listPendingRequests()
            .filter((request) => request && request.requestId && request.conversationId)
            .map((request) => ({
                requestId: String(request.requestId || '').trim(),
                conversationId: String(request.conversationId || '').trim(),
                instanceId: String(request.instanceId || '').trim() || null,
                capability: normalizeCapability(request.capability || '') || null,
                phase: String(request.phase || 'running').trim() || 'running',
                responseId: String(request.responseId || '').trim() || null,
                loadingMessageId: String(request.loadingMessageId || '').trim() || null,
                userMessageId: String(request.userMessageId || '').trim() || null,
                userContent: String(request.userContent || ''),
                userTimestamp: String(request.userTimestamp || '').trim() || null,
                requestSnapshot: sanitizeRequestSnapshot(request.requestSnapshot),
            }));
        if (!requests.length) {
            sessionStorage.removeItem(PENDING_REQUEST_SNAPSHOT_STORAGE_KEY);
            return;
        }
        sessionStorage.setItem(PENDING_REQUEST_SNAPSHOT_STORAGE_KEY, JSON.stringify({
            savedAt: new Date().toISOString(),
            requests,
        }));
    } catch (error) {
        console.warn('Could not persist pending request snapshot:', error);
    }
}

function queueInterruptedPendingRequest(request = {}, resumeFailureReason = '') {
    const conversationId = String(request?.conversationId || '').trim();
    const requestId = String(request?.requestId || '').trim();
    if (!conversationId || !requestId) return;
    if (!state.inference.interruptedRequestsByConversation[conversationId]) {
        state.inference.interruptedRequestsByConversation[conversationId] = [];
    }
    state.inference.interruptedRequestsByConversation[conversationId].push({
        requestId,
        conversationId,
        instanceId: String(request?.instanceId || '').trim() || null,
        capability: normalizeCapability(request?.capability || '') || null,
        phase: String(request?.phase || 'running').trim() || 'running',
        responseId: String(request?.responseId || '').trim() || null,
        userMessageId: String(request?.userMessageId || '').trim() || null,
        userContent: String(request?.userContent || ''),
        userTimestamp: String(request?.userTimestamp || '').trim() || null,
        requestSnapshot: sanitizeRequestSnapshot(request?.requestSnapshot),
        resumeFailureReason: String(resumeFailureReason || request?.resumeFailureReason || '').trim() || null,
    });
}

function restoreInterruptedPendingRequestsSnapshot() {
    try {
        const raw = sessionStorage.getItem(PENDING_REQUEST_SNAPSHOT_STORAGE_KEY);
        if (!raw) return;
        sessionStorage.removeItem(PENDING_REQUEST_SNAPSHOT_STORAGE_KEY);
        const parsed = JSON.parse(raw);
        const savedAt = Date.parse(String(parsed?.savedAt || ''));
        if (!Number.isFinite(savedAt) || (Date.now() - savedAt) > (10 * 60 * 1000)) {
            return;
        }
        const requests = Array.isArray(parsed?.requests) ? parsed.requests : [];
        let restoredPendingCount = 0;
        requests.forEach((request) => {
            const conversationId = String(request?.conversationId || '').trim();
            const requestId = String(request?.requestId || '').trim();
            if (!conversationId || !requestId) return;
            const restoredMeta = {
                requestId,
                conversationId,
                instanceId: String(request?.instanceId || '').trim() || null,
                capability: normalizeCapability(request?.capability || '') || null,
                phase: String(request?.phase || 'running').trim() || 'running',
                responseId: String(request?.responseId || '').trim() || null,
                loadingMessageId: String(request?.loadingMessageId || '').trim() || null,
                userMessageId: String(request?.userMessageId || '').trim() || null,
                userContent: String(request?.userContent || ''),
                userTimestamp: String(request?.userTimestamp || '').trim() || null,
                requestSnapshot: sanitizeRequestSnapshot(request?.requestSnapshot),
                restoredAfterReload: true,
            };
            if ((restoredMeta.phase === 'running' || restoredMeta.phase === 'recovering') && restoredMeta.responseId) {
                if (!state.inference.pendingRequests) {
                    state.inference.pendingRequests = {};
                }
                state.inference.pendingRequests[requestId] = restoredMeta;
                restoredPendingCount += 1;
                return;
            }
            queueInterruptedPendingRequest(restoredMeta);
        });
        if (requests.length) {
            updateSendButtonState();
            const count = restoredPendingCount || requests.length;
            updateGlobalModelStatus(
                restoredPendingCount
                    ? (
                        restoredPendingCount === 1
                            ? 'Recovered 1 pending response after reload.'
                            : `Recovered ${restoredPendingCount} pending responses after reload.`
                    )
                    : (
                        count === 1
                            ? 'A page reload interrupted 1 running request.'
                            : `A page reload interrupted ${count} running requests.`
                    )
            );
        }
    } catch (error) {
        console.warn('Could not restore interrupted pending requests:', error);
        try {
            sessionStorage.removeItem(PENDING_REQUEST_SNAPSHOT_STORAGE_KEY);
        } catch (_storageError) {
            // ignore cleanup failures
        }
    }
}

function buildInterruptedPendingRequestNotice(request = {}) {
    const explicitReason = String(request.resumeFailureReason || '').trim();
    if (explicitReason) {
        return explicitReason;
    }
    const capability = normalizeCapability(request.capability || '') || 'request';
    const instanceLabel = request.instanceId
        ? formatModelDisplayName(getInstanceMeta(request.instanceId)?.model || request.instanceId)
        : 'this conversation';
    return `The browser connection to a running ${capability} request for ${instanceLabel} was interrupted. The previous connection cannot resume automatically, so resend it if no answer arrives.`;
}

function appendInterruptedPendingRequestNotices(conversationId) {
    const key = String(conversationId || '').trim();
    if (!key) return;
    const queue = Array.isArray(state.inference?.interruptedRequestsByConversation?.[key])
        ? state.inference.interruptedRequestsByConversation[key]
        : [];
    if (!queue.length) return;
    if (!state.conversations[key]) {
        state.conversations[key] = [];
    }
    queue.forEach((request) => {
        const clientMessageId = `request-interrupted-${request.requestId}`;
        const alreadyExists = state.conversations[key].some((message) => (
            message?.ephemeralUiNotice && message.clientMessageId === clientMessageId
        ));
        if (alreadyExists) return;
        addMessageToConversation(
            key,
            'assistant',
            buildInterruptedPendingRequestNotice(request),
            false,
            {
                clientMessageId,
                ephemeralUiNotice: true,
            }
        );
    });
    delete state.inference.interruptedRequestsByConversation[key];
}

function buildConversationRequestId(prefix = 'req') {
    return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function buildCanonicalResponseId() {
    return `resp_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

function registerPendingRequest(meta = {}) {
    const requestId = String(meta.requestId || '').trim();
    if (!requestId) return null;
    if (!state.inference.pendingRequests) {
        state.inference.pendingRequests = {};
    }
    const normalizedCapability = normalizeCapability(meta.capability || '') || null;
    const pendingMeta = {
        requestId,
        conversationId: String(meta.conversationId || '').trim() || null,
        instanceId: String(meta.instanceId || '').trim() || null,
        capability: normalizedCapability,
        phase: String(meta.phase || 'running').trim() || 'running',
        responseId: String(meta.responseId || '').trim() || null,
        loadingMessageId: String(meta.loadingMessageId || '').trim() || null,
        userMessageId: String(meta.userMessageId || '').trim() || null,
        userContent: String(meta.userContent || ''),
        userTimestamp: String(meta.userTimestamp || '').trim() || null,
        requestSnapshot: sanitizeRequestSnapshot(meta.requestSnapshot),
        restoredAfterReload: Boolean(meta.restoredAfterReload),
    };
    state.inference.pendingRequests[requestId] = pendingMeta;
    updateSendButtonState();
    persistPendingRequestsSnapshot();
    return pendingMeta;
}

function updatePendingRequest(requestId, patch = {}) {
    const key = String(requestId || '').trim();
    if (!key || !state.inference?.pendingRequests?.[key]) return null;
    const next = {
        ...state.inference.pendingRequests[key],
        ...patch,
    };
    next.requestId = key;
    next.conversationId = String(next.conversationId || '').trim() || null;
    next.instanceId = String(next.instanceId || '').trim() || null;
    next.capability = normalizeCapability(next.capability || '') || null;
    next.phase = String(next.phase || 'running').trim() || 'running';
    next.responseId = String(next.responseId || '').trim() || null;
    next.loadingMessageId = String(next.loadingMessageId || '').trim() || null;
    next.userMessageId = String(next.userMessageId || '').trim() || null;
    next.userContent = String(next.userContent || '');
    next.userTimestamp = String(next.userTimestamp || '').trim() || null;
    next.requestSnapshot = sanitizeRequestSnapshot(next.requestSnapshot);
    next.restoredAfterReload = Boolean(next.restoredAfterReload);
    state.inference.pendingRequests[key] = next;
    updateSendButtonState();
    persistPendingRequestsSnapshot();
    return next;
}

function isRecoverableResponsesConnectionError(error, responseId = '') {
    const normalizedResponseId = String(responseId || '').trim();
    if (!normalizedResponseId) return false;
    if (error?.response) return false;
    const code = String(error?.code || '').trim().toUpperCase();
    if (code === 'ECONNABORTED') {
        return true;
    }
    const message = String(error?.message || '').trim().toLowerCase();
    if (!message) return false;
    return [
        'network error',
        'failed to fetch',
        'load failed',
        'network request failed',
        'network connection was lost',
        'the internet connection appears to be offline',
        'the network connection was lost',
        'fetch failed',
    ].some((token) => message.includes(token));
}

function buildPendingResponseRecoveryMessage() {
    return '<span class="loading-dots">Reconnecting</span> <span>Trying to resume the local response</span>';
}

function ensurePendingRequestUserPreview(request = {}) {
    const conversationId = String(request.conversationId || '').trim();
    const userContent = String(request.userContent || '');
    if (!conversationId || !userContent) return '';
    if (!state.conversations[conversationId]) {
        state.conversations[conversationId] = [];
    }
    const expectedTimestamp = String(request.userTimestamp || '').trim();
    const expectedClientMessageId = String(request.userMessageId || '').trim();
    const alreadyExists = state.conversations[conversationId].some((message) => (
        message?.role === 'user'
        && (
            (expectedClientMessageId && message.clientMessageId === expectedClientMessageId)
            || (
                String(message.content || '') === userContent
                && (
                    !expectedTimestamp
                    || String(message.timestamp || '').trim() === expectedTimestamp
                )
            )
        )
    ));
    if (alreadyExists) {
        if (expectedClientMessageId && request.requestSnapshot) {
            updateMessageRequestSnapshot(conversationId, expectedClientMessageId, request.requestSnapshot);
        }
        return expectedClientMessageId;
    }
    return addMessageToConversation(
        conversationId,
        'user',
        userContent,
        false,
        {
            clientMessageId: expectedClientMessageId || undefined,
            timestamp: expectedTimestamp || undefined,
            requestSnapshot: request.requestSnapshot || null,
        }
    );
}

function clearPendingRequest(requestId) {
    const key = String(requestId || '').trim();
    if (!key || !state.inference?.pendingRequests?.[key]) return;
    stopPendingResponseResumePoller(key);
    delete state.inference.pendingRequests[key];
    updateSendButtonState();
    persistPendingRequestsSnapshot();
}

function stopPendingResponseResumePoller(requestId) {
    const key = String(requestId || '').trim();
    if (!key || !state.inference?.resumePollers?.[key]) return;
    clearTimeout(state.inference.resumePollers[key]);
    delete state.inference.resumePollers[key];
}

function ensurePendingRequestLoadingMessage(request = {}) {
    const conversationId = String(request.conversationId || '').trim();
    if (!conversationId) return '';
    if (!state.conversations[conversationId]) {
        state.conversations[conversationId] = [];
    }
    const existingId = String(request.loadingMessageId || '').trim();
    if (existingId) {
        const existing = state.conversations[conversationId].find((message) => (
            message?.isLoading && message.clientMessageId === existingId
        ));
        if (existing) {
            return existingId;
        }
    }
    const loadingMessageId = existingId || `reload-recovery-${request.requestId || Date.now()}`;
    addLoadingAssistantMessage(conversationId, request.requestSnapshot || null, loadingMessageId);
    updatePendingRequest(request.requestId, { loadingMessageId });
    return loadingMessageId;
}

async function fetchResponseLookupPayload(responseId, options = {}) {
    const params = {};
    const view = String(options.view || 'ui').trim();
    if (view) {
        params.view = view;
    }
    const response = await axios.get(
        `${state.flaskServerUrl}/api/responses/${encodeURIComponent(String(responseId || '').trim())}`,
        Object.keys(params).length ? { params } : undefined
    );
    return response.data || {};
}

async function fetchResponseLookupStatusPayload(responseId) {
    return fetchResponseLookupPayload(responseId, { view: 'status' });
}

const LATE_FILL_INITIAL_POLL_DELAY_MS = 250;
const LATE_FILL_ACTIVE_POLL_DELAY_MS = 650;
const LATE_FILL_ERROR_RETRY_DELAY_MS = 1500;
const LATE_FILL_MIN_POLL_DELAY_MS = 250;
const LATE_FILL_MAX_POLL_DELAY_MS = 30000;
const LATE_FILL_UNCHANGED_POLL_DELAYS_MS = [650, 900, 1200, 1500, 2000];

function getLateFillPollState(responseId = '') {
    const key = String(responseId || '').trim();
    state.inference.lateFillPollStates = state.inference.lateFillPollStates || {};
    if (!key) return {};
    if (!state.inference.lateFillPollStates[key]) {
        state.inference.lateFillPollStates[key] = {
            stateVersion: '',
            unchangedCount: 0,
            errorCount: 0,
        };
    }
    return state.inference.lateFillPollStates[key];
}

function clearLateFillPollState(responseId = '') {
    const key = String(responseId || '').trim();
    if (!key || !state.inference?.lateFillPollStates) return;
    delete state.inference.lateFillPollStates[key];
}

function getResponseLookupStateVersion(payload = {}) {
    const responseFrame = payload?.response_frame && typeof payload.response_frame === 'object'
        ? payload.response_frame
        : {};
    const statusLookupSource = payload?.status_lookup || payload?.statusLookup;
    const statusLookup = statusLookupSource && typeof statusLookupSource === 'object'
        ? statusLookupSource
        : {};
    const frameSequence = (
        responseFrame.frame_sequence
        ?? responseFrame.frameSequence
        ?? payload.frame_sequence
        ?? payload.frameSequence
        ?? payload.latest_frame_sequence
        ?? payload.latestFrameSequence
        ?? statusLookup.frame_sequence
        ?? statusLookup.frameSequence
        ?? statusLookup.latest_frame_sequence
        ?? statusLookup.latestFrameSequence
    );
    const frameId = String(
        responseFrame.frame_id
        || responseFrame.frameId
        || payload.frame_id
        || payload.frameId
        || payload.latest_frame_id
        || payload.latestFrameId
        || statusLookup.frame_id
        || statusLookup.frameId
        || statusLookup.latest_frame_id
        || statusLookup.latestFrameId
        || ''
    ).trim();
    return String(
        payload?.state_version
        || payload?.stateVersion
        || statusLookup.state_version
        || statusLookup.stateVersion
        || (frameSequence !== undefined && frameSequence !== null && frameSequence !== '' ? `frame:${frameSequence}` : '')
        || (frameId ? `frame:${frameId}` : '')
        || ''
    ).trim();
}

function rememberLateFillPollVersion(responseId = '', payload = {}, { changed = false, error = false } = {}) {
    const pollState = getLateFillPollState(responseId);
    if (error) {
        pollState.errorCount = Math.min(8, Number(pollState.errorCount || 0) + 1);
        return pollState;
    }
    pollState.errorCount = 0;
    const stateVersion = getResponseLookupStateVersion(payload);
    if (stateVersion) {
        pollState.stateVersion = stateVersion;
    }
    if (changed) {
        pollState.unchangedCount = 0;
    } else {
        pollState.unchangedCount = Math.min(32, Number(pollState.unchangedCount || 0) + 1);
    }
    return pollState;
}

function getLateFillPollDelayMs(responseId = '', { changed = false, error = false } = {}) {
    const pollState = getLateFillPollState(responseId);
    if (error) {
        const errorCount = Math.max(1, Number(pollState.errorCount || 1));
        return Math.min(LATE_FILL_MAX_POLL_DELAY_MS, LATE_FILL_ERROR_RETRY_DELAY_MS * (2 ** Math.min(4, errorCount - 1)));
    }
    if (changed) {
        return LATE_FILL_ACTIVE_POLL_DELAY_MS;
    }
    const unchangedCount = Math.max(0, Number(pollState.unchangedCount || 0));
    const index = Math.min(LATE_FILL_UNCHANGED_POLL_DELAYS_MS.length - 1, unchangedCount);
    return LATE_FILL_UNCHANGED_POLL_DELAYS_MS[index] || LATE_FILL_ACTIVE_POLL_DELAY_MS;
}

function getLateFillState(payload = {}) {
    if (!payload || typeof payload !== 'object') {
        return null;
    }
    const runtime = payload.runtime && typeof payload.runtime === 'object'
        ? payload.runtime
        : {};
    const lateFill = payload.late_fill || payload.lateFill || runtime.late_fill || runtime.lateFill || null;
    return lateFill && typeof lateFill === 'object' ? lateFill : null;
}

function responseLifecycleStateIsActive(value = '') {
    const lifecycleState = String(value || '').trim().toLowerCase();
    return [
        'accepted',
        'active',
        'in_progress',
        'late_fill_pending',
        'late_fill_running',
        'pending',
        'queued',
        'running',
        'streaming',
    ].includes(lifecycleState);
}

function coerceResponseOptionalBoolean(value) {
    if (value === true || value === false) return value;
    const normalized = String(value ?? '').trim().toLowerCase();
    if (normalized === 'true' || normalized === '1' || normalized === 'yes') return true;
    if (normalized === 'false' || normalized === '0' || normalized === 'no') return false;
    return null;
}

function responseStatusSemanticsIndicatesTerminal(statusSemantics = {}, lifecycleState = '') {
    const semantics = statusSemantics && typeof statusSemantics === 'object'
        ? statusSemantics
        : {};
    const normalizedLifecycle = String(
        lifecycleState
        || semantics.canonical_lifecycle_state
        || semantics.canonicalLifecycleState
        || ''
    ).trim().toLowerCase();
    const hasOpenContinuation = coerceResponseOptionalBoolean(
        semantics.has_open_continuation ?? semantics.hasOpenContinuation
    );
    if (hasOpenContinuation === true) return false;
    const terminalFlag = coerceResponseOptionalBoolean(
        semantics.is_terminal ?? semantics.isTerminal ?? semantics.terminal
    );
    if (terminalFlag === true) return true;
    const terminalLifecycle = ['completed', 'cancelled', 'failed', 'incomplete'].includes(normalizedLifecycle);
    return terminalLifecycle && hasOpenContinuation === false;
}

function responseHasPendingLateFill(payload = {}) {
    const statusSemantics = payload?.status_semantics || payload?.statusSemantics || {};
    const hasCanonicalLifecycle = Boolean(
        String(payload?.lifecycle_state || payload?.lifecycleState || '').trim()
        || String(statusSemantics?.canonical_lifecycle_state || statusSemantics?.canonicalLifecycleState || '').trim()
    );
    const lifecycleState = String(
        payload?.lifecycle_state
        || payload?.lifecycleState
        || statusSemantics?.canonical_lifecycle_state
        || statusSemantics?.canonicalLifecycleState
        || ''
    ).trim().toLowerCase();
    if (responseStatusSemanticsIndicatesTerminal(statusSemantics, lifecycleState)) {
        return false;
    }
    if (responseLifecycleStateIsActive(lifecycleState)) {
        return true;
    }
    if (statusSemantics?.has_open_continuation || statusSemantics?.hasOpenContinuation) {
        return true;
    }
    if (hasCanonicalLifecycle) {
        return false;
    }
    const status = String(getLateFillState(payload)?.status || '').trim().toLowerCase();
    return status === 'pending' || status === 'queued' || status === 'scheduled' || status === 'accepted' || status === 'running' || status === 'in_progress';
}

function responseNeedsTerminalTruthHydration(payload = {}) {
    const responseId = String(payload?.id || payload?.response_id || '').trim();
    if (!responseId) return false;
    if (responseHasPendingLateFill(payload)) return false;
    const statusSemantics = payload?.status_semantics || payload?.statusSemantics || {};
    const lifecycleState = String(
        payload?.lifecycle_state
        || payload?.lifecycleState
        || statusSemantics?.canonical_lifecycle_state
        || statusSemantics?.canonicalLifecycleState
        || ''
    ).trim().toLowerCase();
    const compatibilityStatus = String(
        statusSemantics?.compatibility_status
        || statusSemantics?.compatibilityStatus
        || payload?.status
        || ''
    ).trim().toLowerCase();
    if (statusSemantics?.has_actionable_repair === true || statusSemantics?.hasActionableRepair === true) {
        return false;
    }
    if (lifecycleState && !['completed', 'late_fill_completed'].includes(lifecycleState)) {
        return false;
    }
    return responseStatusSemanticsIndicatesTerminal(statusSemantics, lifecycleState)
        || lifecycleState === 'completed'
        || compatibilityStatus === 'completed';
}

function getPublicResponseArtifactCount(payload = {}) {
    const artifacts = typeof sanitizeResponseArtifacts === 'function'
        ? sanitizeResponseArtifacts(payload?.artifacts)
        : (Array.isArray(payload?.artifacts) ? payload.artifacts : []);
    return artifacts.filter((artifact) => (
        artifact
        && typeof artifact === 'object'
        && String(artifact.type || artifact.kind || '').trim()
        && (
            String(
                artifact.path
                || artifact.source_path
                || artifact.sourcePath
                || artifact.image_data_url
                || artifact.imageDataUrl
                || artifact.saved_image_path
                || artifact.savedImagePath
                || artifact.saved_audio_path
                || artifact.savedAudioPath
                || artifact.saved_text_path
                || artifact.savedTextPath
                || artifact.artifact_ref
                || artifact.artifactRef
                || artifact.ref
                || artifact.artifact_id
                || artifact.artifactId
                || ''
            ).trim()
        )
    )).length;
}

function getDeclaredPublicResponseArtifactCount(payload = {}) {
    const statusLookup = payload?.status_lookup || payload?.statusLookup || {};
    const counts = payload?.output_counts || payload?.outputCounts || statusLookup?.output_counts || statusLookup?.outputCounts || {};
    const value = Number(counts.artifact_count ?? counts.artifactCount);
    return Number.isFinite(value) && value > 0 ? value : 0;
}

function responseOutputItemDeclaresPublicArtifact(item = {}, index = 0) {
    if (!item || typeof item !== 'object') return '';
    const status = String(item.status || item.state || item.lifecycle || '').trim().toLowerCase().replace(/[-\s]+/g, '_');
    if ([
        'blocked',
        'cancelled',
        'failed',
        'open',
        'partial_failed',
        'pending',
        'queued',
        'repair_needed',
        'rejected',
        'running',
        'skipped',
        'superseded',
        'waived',
    ].includes(status)) {
        return '';
    }
    const nestedArtifacts = typeof sanitizeResponseArtifacts === 'function'
        ? sanitizeResponseArtifacts(item.artifacts)
        : (Array.isArray(item.artifacts) ? item.artifacts : []);
    const nestedIdentity = nestedArtifacts
        .map((artifact) => String(
            artifact?.path
            || artifact?.source_path
            || artifact?.sourcePath
            || artifact?.artifact_ref
            || artifact?.artifactRef
            || artifact?.ref
            || artifact?.artifact_id
            || artifact?.artifactId
            || ''
        ).trim())
        .find(Boolean);
    const identity = String(
        item.artifact_ref
        || item.artifactRef
        || item.ref
        || item.artifact_id
        || item.artifactId
        || item.path
        || item.saved_image_path
        || item.savedImagePath
        || item.saved_audio_path
        || item.savedAudioPath
        || item.saved_text_path
        || item.savedTextPath
        || nestedIdentity
        || ''
    ).trim();
    if (!identity) return '';
    const type = String(item.type || item.output_type || item.outputType || nestedArtifacts[0]?.type || '').trim().toLowerCase();
    return [type || 'artifact', identity || `artifact-${index}`].join('\u0000');
}

function responsePayloadHasPublicArtifactProjectionGap(payload = {}) {
    if (!payload || typeof payload !== 'object') return false;
    const publicArtifactCount = getPublicResponseArtifactCount(payload);
    const declaredArtifactCount = getDeclaredPublicResponseArtifactCount(payload);
    if (declaredArtifactCount > publicArtifactCount) return true;
    const artifacts = typeof sanitizeResponseArtifacts === 'function'
        ? sanitizeResponseArtifacts(payload.artifacts)
        : (Array.isArray(payload.artifacts) ? payload.artifacts : []);
    const outputSlots = typeof extractResponseOutputSlots === 'function'
        ? extractResponseOutputSlots(payload)
        : (Array.isArray(payload.output_slots || payload.outputSlots) ? (payload.output_slots || payload.outputSlots) : []);
    const outputs = typeof extractResponseOutputs === 'function'
        ? extractResponseOutputs(payload, { artifacts, outputSlots })
        : (Array.isArray(payload.outputs) ? payload.outputs : []);
    const requiredArtifactKeys = new Set();
    outputs.forEach((output, index) => {
        const key = responseOutputItemDeclaresPublicArtifact(output, index);
        if (key) requiredArtifactKeys.add(key);
    });
    outputSlots.forEach((slot, index) => {
        const key = responseOutputItemDeclaresPublicArtifact(slot, index + outputs.length);
        if (key) requiredArtifactKeys.add(key);
    });
    return requiredArtifactKeys.size > publicArtifactCount;
}

function responsePayloadHasOpenPublicArtifactProjectionGap(payload = {}) {
    if (!responsePayloadHasPublicArtifactProjectionGap(payload)) return false;
    if (responseHasPendingLateFill(payload)) return true;
    const statusSemantics = payload?.status_semantics || payload?.statusSemantics || {};
    const lifecycleState = String(
        payload?.lifecycle_state
        || payload?.lifecycleState
        || statusSemantics?.canonical_lifecycle_state
        || statusSemantics?.canonicalLifecycleState
        || ''
    ).trim().toLowerCase();
    if (
        responseStatusSemanticsIndicatesTerminal(statusSemantics, lifecycleState)
        || ['completed', 'late_fill_completed', 'cancelled', 'failed', 'incomplete'].includes(lifecycleState)
    ) {
        return false;
    }
    return true;
}

function stopLateFillResponsePoller(responseId = '') {
    const key = String(responseId || '').trim();
    if (!key) return;
    state.inference.lateFillPollers = state.inference.lateFillPollers || {};
    if (state.inference.lateFillPollers[key]) {
        window.clearTimeout(state.inference.lateFillPollers[key]);
        delete state.inference.lateFillPollers[key];
    }
}

function markActiveResponseSseStream(responseId = '', active = true) {
    const key = String(responseId || '').trim();
    if (!key) return;
    state.inference.activeResponseStreams = state.inference.activeResponseStreams || {};
    if (active) {
        state.inference.activeResponseStreams[key] = true;
    } else {
        delete state.inference.activeResponseStreams[key];
    }
}

function responseHasActiveSseStream(responseId = '') {
    const key = String(responseId || '').trim();
    return Boolean(key && state.inference?.activeResponseStreams?.[key]);
}

function scheduleLateFillResponsePoll(conversationId, responseId, requestSnapshot = null, delayMs = LATE_FILL_ACTIVE_POLL_DELAY_MS) {
    const targetConversationId = String(conversationId || '').trim();
    const targetResponseId = String(responseId || '').trim();
    if (!targetConversationId || !targetResponseId) return;
    if (responseHasActiveSseStream(targetResponseId)) {
        return;
    }
    state.inference.lateFillPollers = state.inference.lateFillPollers || {};
    stopLateFillResponsePoller(targetResponseId);
    state.inference.lateFillPollers[targetResponseId] = window.setTimeout(() => {
        pollLateFillResponse(targetConversationId, targetResponseId, requestSnapshot);
    }, Math.max(LATE_FILL_MIN_POLL_DELAY_MS, Number(delayMs) || LATE_FILL_ACTIVE_POLL_DELAY_MS));
}

async function pollLateFillResponse(conversationId, responseId, requestSnapshot = null) {
    const targetConversationId = String(conversationId || '').trim();
    const targetResponseId = String(responseId || '').trim();
    if (!targetConversationId || !targetResponseId) {
        stopLateFillResponsePoller(targetResponseId);
        return;
    }
    try {
        const statusPayload = await fetchResponseLookupStatusPayload(targetResponseId);
        const pollState = getLateFillPollState(targetResponseId);
        const statusVersion = getResponseLookupStateVersion(statusPayload);
        const statusChanged = !statusVersion || !pollState.stateVersion || pollState.stateVersion !== statusVersion;
        const compactStatusStillPending = responseHasPendingLateFill(statusPayload);
        const needsTerminalFullFrame = !statusChanged && !compactStatusStillPending;
        let payload = statusPayload;
        let mergedSnapshot = requestSnapshot;
        if (statusChanged || needsTerminalFullFrame) {
            payload = await fetchResponseLookupPayload(targetResponseId);
            mergedSnapshot = mergeRequestSnapshotInputArtifacts(requestSnapshot, payload?.input_artifacts);
            updateAssistantResponseByResponseId(
                targetConversationId,
                targetResponseId,
                payload,
                mergedSnapshot || requestSnapshot || null
            );
        }
        rememberLateFillPollVersion(targetResponseId, payload, { changed: statusChanged || needsTerminalFullFrame });
        if (
            responseHasPendingLateFill(payload)
            || responsePayloadHasOpenPublicArtifactProjectionGap(payload)
        ) {
            scheduleLateFillResponsePoll(
                targetConversationId,
                targetResponseId,
                mergedSnapshot || requestSnapshot || null,
                getLateFillPollDelayMs(targetResponseId, { changed: statusChanged || needsTerminalFullFrame })
            );
            return;
        }
        stopLateFillResponsePoller(targetResponseId);
        clearLateFillPollState(targetResponseId);
    } catch (error) {
        const statusCode = Number(error?.response?.status || 0);
        if (statusCode === 404) {
            stopLateFillResponsePoller(targetResponseId);
            clearLateFillPollState(targetResponseId);
            return;
        }
        console.warn('Could not refresh late-fill response state:', error?.message || error);
        rememberLateFillPollVersion(targetResponseId, {}, { error: true });
        scheduleLateFillResponsePoll(
            targetConversationId,
            targetResponseId,
            requestSnapshot,
            getLateFillPollDelayMs(targetResponseId, { error: true })
        );
    }
}

async function hydrateTerminalResponseLookupTruth(conversationId, payload = {}, requestSnapshot = null) {
    const targetConversationId = String(conversationId || '').trim();
    const responseId = String(payload?.id || payload?.response_id || '').trim();
    if (!targetConversationId || !responseId) {
        return null;
    }
    try {
        const hydratedPayload = await fetchResponseLookupPayload(responseId);
        const mergedSnapshot = mergeRequestSnapshotInputArtifacts(
            requestSnapshot,
            hydratedPayload?.input_artifacts
        );
        updateAssistantResponseByResponseId(
            targetConversationId,
            responseId,
            hydratedPayload,
            mergedSnapshot || requestSnapshot || null
        );
        rememberLateFillPollVersion(responseId, hydratedPayload, { changed: true });
        if (
            responseHasPendingLateFill(hydratedPayload)
            || responsePayloadHasOpenPublicArtifactProjectionGap(hydratedPayload)
        ) {
            scheduleLateFillResponsePoll(
                targetConversationId,
                responseId,
                mergedSnapshot || requestSnapshot || null,
                getLateFillPollDelayMs(responseId, { changed: true })
            );
        } else {
            stopLateFillResponsePoller(responseId);
            clearLateFillPollState(responseId);
        }
        return hydratedPayload;
    } catch (error) {
        console.warn('Could not hydrate completed stream response truth:', error?.message || error);
        maybeWatchLateFillResponse(targetConversationId, payload, requestSnapshot);
        return null;
    }
}

function maybeWatchLateFillResponse(conversationId, payload = {}, requestSnapshot = null) {
    const targetConversationId = String(conversationId || '').trim();
    const responseId = String(payload?.id || payload?.response_id || '').trim();
    if (!targetConversationId || !responseId) {
        return;
    }
    if (responseHasPendingLateFill(payload)) {
        scheduleLateFillResponsePoll(targetConversationId, responseId, requestSnapshot, LATE_FILL_INITIAL_POLL_DELAY_MS);
        return;
    }
    if (responseNeedsTerminalTruthHydration(payload)) {
        scheduleLateFillResponsePoll(targetConversationId, responseId, requestSnapshot, LATE_FILL_INITIAL_POLL_DELAY_MS);
        return;
    }
    if (responsePayloadHasOpenPublicArtifactProjectionGap(payload)) {
        scheduleLateFillResponsePoll(targetConversationId, responseId, requestSnapshot, LATE_FILL_INITIAL_POLL_DELAY_MS);
        return;
    }
    stopLateFillResponsePoller(responseId);
}

function historyMessageHasOpenResponseWork(message = {}) {
    if (!message || String(message.role || '').trim().toLowerCase() === 'user') {
        return false;
    }
    const responseId = String(message.responseId || message.response_id || '').trim();
    if (!responseId) return false;
    const statusSemantics = message.statusSemantics || message.status_semantics || {};
    const lifecycleState = String(
        message.lifecycleState
        || message.lifecycle_state
        || statusSemantics?.canonicalLifecycleState
        || statusSemantics?.canonical_lifecycle_state
        || ''
    ).trim().toLowerCase();
    if (responseStatusSemanticsIndicatesTerminal(statusSemantics, lifecycleState)) {
        return false;
    }
    const lateFill = sanitizeMessageLateFill(message.lateFill || message.late_fill);
    if (lateFillStatusIsActive(lateFill)) return true;
    const lines = formatAssistantCurrentWorkStatusLines(message);
    return lines.some((line) => /(?:still open|still generating|queued for late fill|pending)/i.test(String(line || '')));
}

function historyMessageNeedsResponseTruthHydration(message = {}) {
    if (!message || String(message.role || '').trim().toLowerCase() === 'user') {
        return false;
    }
    const responseId = String(message.responseId || message.response_id || '').trim();
    if (!responseId) return false;
    const statusSemantics = message.statusSemantics || message.status_semantics || null;
    const canonicalLifecycleState = String(
        message.lifecycleState
        || message.lifecycle_state
        || statusSemantics?.canonicalLifecycleState
        || statusSemantics?.canonical_lifecycle_state
        || ''
    ).trim();
    const lateFill = message.lateFill || message.late_fill || null;
    const hasStatusSemantics = Boolean(statusSemantics && typeof statusSemantics === 'object' && Object.keys(statusSemantics).length);
    const hasLateFill = Boolean(lateFill && typeof lateFill === 'object' && Object.keys(lateFill).length);
    if (!canonicalLifecycleState || !hasStatusSemantics) {
        return true;
    }
    return responseLifecycleStateIsActive(canonicalLifecycleState) && !hasLateFill;
}

function historyMessageNeedsResponseLookup(message = {}) {
    return historyMessageHasOpenResponseWork(message) || historyMessageNeedsResponseTruthHydration(message);
}

function resumeHydratedLateFillResponses(conversationId = '') {
    const targetConversationId = String(conversationId || '').trim();
    if (!targetConversationId) return;
    const conversation = Array.isArray(state.conversations?.[targetConversationId])
        ? state.conversations[targetConversationId]
        : [];
    const seen = new Set();
    conversation.forEach((message) => {
        if (!historyMessageNeedsResponseLookup(message)) return;
        const responseId = String(message.responseId || message.response_id || '').trim();
        if (!responseId || seen.has(responseId)) return;
        seen.add(responseId);
        scheduleLateFillResponsePoll(
            targetConversationId,
            responseId,
            message.requestSnapshot || message.request_snapshot || null,
            LATE_FILL_INITIAL_POLL_DELAY_MS
        );
    });
}

function schedulePendingResponseResumePoll(requestId, delayMs = 1200) {
    const key = String(requestId || '').trim();
    if (!key) return;
    stopPendingResponseResumePoller(key);
    state.inference.resumePollers[key] = window.setTimeout(() => {
        pollPendingResponseAfterReload(key);
    }, Math.max(250, Number(delayMs) || 1200));
}

async function pollPendingResponseAfterReload(requestId) {
    const key = String(requestId || '').trim();
    const request = state.inference?.pendingRequests?.[key];
    if (!key || !request) {
        stopPendingResponseResumePoller(key);
        return;
    }
    const conversationId = String(request.conversationId || '').trim();
    const responseId = String(request.responseId || '').trim();
    if (!conversationId || !responseId) {
        queueInterruptedPendingRequest(
            request,
            'The browser connection to a running request was interrupted, and there was no resumable response ID available. Resend it if no answer arrives.'
        );
        clearPendingRequest(key);
        if (conversationId) {
            appendInterruptedPendingRequestNotices(conversationId);
            if (isConversationVisible(conversationId)) {
                renderConversation(conversationId);
            }
        }
        return;
    }

    const loadingMessageId = ensurePendingRequestLoadingMessage(request);
    try {
        const statusPayload = await fetchResponseLookupStatusPayload(responseId);
        if (
            responseHasPendingLateFill(statusPayload)
            || responsePayloadHasOpenPublicArtifactProjectionGap(statusPayload)
        ) {
            const lateFill = getLateFillState(statusPayload);
            const liveStatusLines = buildResponseSseLiveStatusLines(
                'response.state.updated',
                { response: statusPayload, late_fill: lateFill || undefined },
                statusPayload
            );
            updateLoadingAssistantMessage(
                conversationId,
                '<span class="loading-dots">Working</span>',
                {
                    clientMessageId: loadingMessageId,
                    trustedHtml: true,
                    responseId,
                    lifecycleState: statusPayload.lifecycle_state || statusPayload.lifecycleState || null,
                    statusSemantics: statusPayload.status_semantics || statusPayload.statusSemantics || null,
                    lateFill,
                    ...(liveStatusLines.length ? { liveStatusLines } : {}),
                    requestSnapshot: request.requestSnapshot || null,
                    forceRender: true,
                }
            );
            maybeWatchLateFillResponse(
                conversationId,
                statusPayload,
                request.requestSnapshot || null
            );
            clearPendingRequest(key);
            return;
        }
        const payload = await fetchResponseLookupPayload(responseId);
        const status = String(payload?.status || '').trim().toLowerCase();
        const text = extractAssistantResponseText(payload);
        const displayText = text ? getAssistantDisplayContent(payload, text) : '';
        if (displayText) {
            updateLoadingAssistantMessage(
                conversationId,
                displayText,
                {
                    clientMessageId: loadingMessageId,
                    requestSnapshot: request.requestSnapshot || null,
                }
            );
        }
        if (status === 'completed') {
            finalizeLoadingAssistantResponse(
                conversationId,
                loadingMessageId,
                payload,
                text,
                request.requestSnapshot || null
            );
            maybeWatchLateFillResponse(
                conversationId,
                payload,
                request.requestSnapshot || null
            );
            clearPendingRequest(key);
            return;
        }
        if (status === 'failed') {
            const errorMessage = extractResponseErrorMessage(payload, 'Request failed.');
            finalizeLoadingAssistantResponse(
                conversationId,
                loadingMessageId,
                payload,
                `Sorry, error: ${errorMessage}`,
                request.requestSnapshot || null
            );
            clearPendingRequest(key);
            return;
        }
        schedulePendingResponseResumePoll(key, text ? 800 : 1200);
    } catch (error) {
        const statusCode = Number(error?.response?.status || 0);
        if (statusCode === 404) {
            removeLoadingAssistantMessage(conversationId, loadingMessageId);
            queueInterruptedPendingRequest(
                request,
                'The browser connection to a running request was interrupted, but the server no longer has resumable state for it. Resend it if no answer arrives.'
            );
            clearPendingRequest(key);
            appendInterruptedPendingRequestNotices(conversationId);
            if (isConversationVisible(conversationId)) {
                renderConversation(conversationId);
            }
            return;
        }
        console.warn('Could not resume pending response after reload:', error?.message || error);
        schedulePendingResponseResumePoll(key, 1500);
    }
}

async function resumeRestoredPendingRequests() {
    const restoredRequests = listPendingRequests().filter((request) => (
        request?.restoredAfterReload
        && request.phase === 'running'
        && request.responseId
    ));
    if (!restoredRequests.length) return;
    const conversationIds = Array.from(new Set(
        restoredRequests
            .map((request) => String(request.conversationId || '').trim())
            .filter(Boolean)
    ));
    await Promise.allSettled(conversationIds.map((conversationId) => fetchChatHistory(conversationId)));
    restoredRequests.forEach((request) => {
        const requestId = String(request.requestId || '').trim();
        if (!requestId || state.inference.resumePollers?.[requestId]) return;
        ensurePendingRequestUserPreview(request);
        ensurePendingRequestLoadingMessage(request);
        pollPendingResponseAfterReload(requestId);
    });
}

function hasPendingConversationPreview(conversationId, { excludeRequestId = '' } = {}) {
    const targetConversationId = String(conversationId || '').trim();
    const excluded = String(excludeRequestId || '').trim();
    return listPendingRequests().some((request) => (
        request.phase === 'preview'
        && request.conversationId === targetConversationId
        && request.requestId !== excluded
    ));
}

function hasPendingConversationChatRequest(conversationId, { excludeRequestId = '' } = {}) {
    const targetConversationId = String(conversationId || '').trim();
    const excluded = String(excludeRequestId || '').trim();
    return listPendingRequests().some((request) => (
        request.phase === 'running'
        && request.conversationId === targetConversationId
        && request.capability === 'chat'
        && request.requestId !== excluded
    ));
}

function hasPendingConversationRequest(conversationId, { excludeRequestId = '' } = {}) {
    const targetConversationId = String(conversationId || '').trim();
    const excluded = String(excludeRequestId || '').trim();
    return listPendingRequests().some((request) => (
        request.phase === 'running'
        && request.conversationId === targetConversationId
        && request.requestId !== excluded
    ));
}

function hasPendingRequestForInstance(instanceId, { excludeRequestId = '' } = {}) {
    const targetInstanceId = String(instanceId || '').trim();
    const excluded = String(excludeRequestId || '').trim();
    if (!targetInstanceId) return false;
    return listPendingRequests().some((request) => (
        request.phase === 'running'
        && request.instanceId === targetInstanceId
        && request.requestId !== excluded
    ));
}

function setInstanceRequestPending(instanceId, isPending) {
    if (!instanceId) return;
    if (!state.inference.pendingByInstance) {
        state.inference.pendingByInstance = {};
    }
    if (isPending) {
        state.inference.pendingByInstance[instanceId] = true;
    } else {
        delete state.inference.pendingByInstance[instanceId];
    }
    updateSendButtonState();
}

function getLoadingMessageResponseId(message = {}) {
    return String(message?.responseId || message?.response_id || '').trim();
}

function getLoadingMessageClientId(message = {}) {
    return String(message?.clientMessageId || message?.client_message_id || message?.message_id || '').trim();
}

function pendingRequestMatchesLoadingMessage(request = {}, conversationId = '', message = {}) {
    if (!request || request.phase !== 'running') return false;
    const targetConversationId = String(conversationId || '').trim();
    if (targetConversationId && String(request.conversationId || '').trim() !== targetConversationId) {
        return false;
    }
    const messageResponseId = getLoadingMessageResponseId(message);
    const requestResponseId = String(request.responseId || '').trim();
    if (messageResponseId && requestResponseId && messageResponseId === requestResponseId) {
        return true;
    }
    const messageClientId = getLoadingMessageClientId(message);
    const requestLoadingMessageId = String(request.loadingMessageId || '').trim();
    return Boolean(messageClientId && requestLoadingMessageId && messageClientId === requestLoadingMessageId);
}

function loadingMessageHasActiveRequestOwner(conversationId = '', message = {}) {
    if (!message || !message.isLoading) return false;
    const responseId = getLoadingMessageResponseId(message);
    if (responseId && responseHasActiveSseStream(responseId)) {
        return true;
    }
    return listPendingRequests().some((request) => pendingRequestMatchesLoadingMessage(request, conversationId, message));
}

function clearStaleLoadingMessages(instanceId = null, force = false) {
    const now = Date.now();
    const ttl = Math.max(10000, Number(state.inference.staleLoadingMs || 180000));
    const targets = instanceId
        ? [instanceId]
        : Object.keys(state.conversations || {});
    let changedCurrent = false;
    targets.forEach((id) => {
        const conversation = state.conversations[id];
        if (!Array.isArray(conversation) || conversation.length === 0) return;
        const before = conversation.length;
        state.conversations[id] = conversation.filter((message) => {
            if (!message || !message.isLoading) return true;
            if (force) return false;
            if (loadingMessageHasActiveRequestOwner(id, message)) return true;
            const ts = Date.parse(message.timestamp || '');
            if (!Number.isFinite(ts)) return false;
            return (now - ts) <= ttl;
        });
        if (isConversationVisible(id) && state.conversations[id].length !== before) {
            changedCurrent = true;
        }
    });
    if (changedCurrent && !state.arena.enabled) {
        renderConversation(getActiveConversationId());
    }
}


function canSendPrompt() {
    const hasMessage = Boolean(elements.userInput?.value.trim());
    const queuedItems = getPendingInputItems();
    const hasAttachment = queuedItems.length > 0;
    if (state.arena.enabled) {
        if (isInstanceRequestPending(state.arena.modelA) || isInstanceRequestPending(state.arena.modelB)) {
            return false;
        }
        return hasMessage && !hasAttachment && arenaSelectionsAreChatCapable();
    }
    if (isResponsesWorkbenchActive() && isResponsesWorkbenchAutoTarget()) {
        const conversationId = getResponsesWorkbenchConversationId();
        if (hasPendingConversationPreview(conversationId)) return false;
        if (hasPendingConversationChatRequest(conversationId)) return false;
        return hasMessage || hasAttachment;
    }
    const instance = getActivePromptTargetInstance();
    if (!instance) return false;
    const capability = normalizeCapability(instance.capability || 'chat');
    const conversationId = getActiveConversationId();
    if (requestLifecycleTargetIsExternal(instance)) {
        if (hasPendingConversationChatRequest(conversationId)) return false;
        return hasMessage;
    }
    if (capability === 'chat') {
        if (hasPendingConversationChatRequest(conversationId)) return false;
    } else if (hasPendingRequestForInstance(instance.instance_id)) {
        return false;
    }
    if (capability === 'speech_to_text') {
        if (!queuedItems.length) return false;
        return queuedItems.every((item) => {
            if (item.source === 'upload') {
                return inferFileKindFromName(item.file?.name || '') === 'audio';
            }
            return inferFileKindFromName(item.localPath || '') === 'audio';
        });
    }
    if (capability === 'image_generation') {
        return hasMessage;
    }
    return hasMessage || hasAttachment;
}

function updateSendButtonState() {
    if (!elements.sendBtn) return;
    elements.sendBtn.disabled = !canSendPrompt();
}


function buildResponsesInputForHistory(instanceId, limit = 10, pendingMessage = '') {
    const history = buildHistoryForApi(instanceId, limit);
    const input = history.map((message) => ({
        type: 'message',
        role: message.role === 'assistant' ? 'assistant' : 'user',
        content: [
            {
                type: 'input_text',
                text: String(message.content || '')
            }
        ]
    }));
    const trimmedPendingMessage = String(pendingMessage || '').trim();
    if (!trimmedPendingMessage) {
        return input;
    }
    const lastEntry = input[input.length - 1];
    const lastEntryText = Array.isArray(lastEntry?.content)
        ? String(lastEntry.content[0]?.text || '').trim()
        : '';
    const lastEntryRole = String(lastEntry?.role || '').trim();
    if (lastEntryRole === 'user' && lastEntryText === trimmedPendingMessage) {
        return input;
    }
    input.push({
        type: 'message',
        role: 'user',
        content: [
            {
                type: 'input_text',
                text: trimmedPendingMessage
            }
        ]
    });
    return input;
}

function buildGhostRoutingConversationSnapshot(instanceId, limit = 14) {
    const history = buildHistoryForApi(instanceId, limit);
    return history.map((message) => {
        const requestSnapshot = sanitizeRequestSnapshot(
            message.requestSnapshot
            || message.request_snapshot
        );
        const artifacts = sanitizeResponseArtifacts(
            buildCanonicalMessageArtifacts(message, { requestSnapshot })
        );
        return {
            role: message.role === 'assistant' ? 'assistant' : 'user',
            content: String(message.content || ''),
            timestamp: message.timestamp || new Date().toISOString(),
            ...(artifacts.length ? { artifacts } : {}),
            ...(message.role === 'assistant' && message.savedImagePath ? { saved_image_path: String(message.savedImagePath) } : {}),
            ...(message.role === 'assistant' && message.savedAudioPath ? { saved_audio_path: String(message.savedAudioPath) } : {}),
            ...(message.role === 'assistant' && message.savedTextPath ? { saved_text_path: String(message.savedTextPath) } : {}),
            ...(message.responseModel ? { response_model: String(message.responseModel) } : {}),
            ...(message.responseBackend ? { response_backend: String(message.responseBackend) } : {}),
            ...(message.responseCapability ? { response_capability: String(message.responseCapability) } : {}),
            ...(message.responseInstanceId ? { response_instance_id: String(message.responseInstanceId) } : {}),
            ...(message.routeSource ? { route_source: String(message.routeSource) } : {}),
            ...(message.routeReason ? { route_reason: String(message.routeReason) } : {}),
            ...(message.contextMode ? { context_mode: String(message.contextMode) } : {}),
            ...(message.contextReason ? { context_reason: String(message.contextReason) } : {}),
        };
    });
}

function buildGhostRoutePreviewPayload(message = '', attachment = null, localPath = '', conversationId = '') {
    const selectedReferenceArtifact = buildSelectedReferenceArtifactPayload(conversationId);
    const ghostPreferences = getResponsesGhostPreferencesPayload();
    const requestMeta = getResponsesGhostRequestMetaPayload();
    return {
        ...(message ? { prompt: message } : {}),
        ...(attachment?.name ? { upload_filename: attachment.name } : {}),
        ...(localPath ? { file_path: localPath } : {}),
        compute_semantics: false,
        conversation_id: conversationId,
        ghost_messages: buildGhostRoutingConversationSnapshot(conversationId),
        ...(ghostPreferences ? { ghost_preferences: ghostPreferences } : {}),
        ...(requestMeta ? { request_meta: requestMeta } : {}),
        ...(selectedReferenceArtifact ? { reference_artifacts: selectedReferenceArtifact } : {}),
    };
}

function buildGhostExecutionPreviewPayload(
    resolvedInstance = getGhostResolvedTargetInstance(),
    route = state.responsesWorkbench.ghostResolvedRoute,
    conversationId = ''
) {
    if (!resolvedInstance || !route) return null;
    return {
        instance_id: String(resolvedInstance.instance_id || '').trim() || null,
        capability: String(resolvedInstance.capability || '').trim() || null,
        reuse_last_artifact: Boolean(route.reuse_last_artifact),
        artifact_ref: String(route.artifact_ref || '').trim() || null,
        artifact_path: String(route.artifact_path || '').trim() || null,
        confidence: Number(route.confidence || 0) || 0,
        reason: String(route.reason || '').trim() || 'ghost preview route',
        route_source: String(route.source || '').trim() || 'router',
        route_router_instance_id: String(route.router_instance_id || '').trim() || null,
        route_router_model: String(route.router_model || '').trim() || null,
        reference_artifacts: buildSelectedReferenceArtifactPayload(conversationId),
    };
}

function buildRequestExecutionContext(
    requestInstance,
    {
        ghostPreview = null,
        requestMeta = null,
        requestId = '',
        transport = 'auto',
        message = '',
        attachment = null,
        localPath = '',
        conversationId = '',
        responseId = '',
        batchPrompts = [],
    } = {}
) {
    if (!requestInstance?.instance_id) return null;
    const requestControlFields = buildSessionControlRequestFields(requestInstance);
    return {
        requestInstance: { ...requestInstance },
        requestControlFields,
        ghostPreview: ghostPreview ? { ...ghostPreview } : null,
        requestSnapshot: buildRequestSnapshot(
            requestInstance,
            {
                requestId,
                transport,
                message,
                attachment,
                localPath,
                conversationId,
                responseId,
                requestControlFields,
                ghostPreview,
                requestMeta,
                batchPrompts,
            }
        ),
    };
}

async function ensureGhostAutoResolvedTarget(message = '', attachment = null, localPath = '', conversationId = '') {
    if (!isResponsesWorkbenchAutoTarget()) return null;
    const previousOwner = getGhostResolvedTargetInstance();
    const previousKey = getInstanceSettingsKey(previousOwner);
    const response = await axios.post(
        `${state.flaskServerUrl}/api/ghost_route_preview`,
        buildGhostRoutePreviewPayload(message, attachment, localPath, conversationId)
    );
    const payload = response.data || {};
    const resolvedInstance = payload.instance && typeof payload.instance === 'object'
        ? payload.instance
        : null;
    const nextKey = getInstanceSettingsKey(resolvedInstance);
    if (previousKey !== nextKey) {
        persistSettingsForCurrentInstance();
    }
    state.responsesWorkbench.ghostResolvedTarget = resolvedInstance;
    state.responsesWorkbench.ghostResolvedRoute = payload.route || null;
    state.responsesWorkbench.ghostResolvedRuntime = payload.runtime || null;
    if (previousKey !== nextKey) {
        loadSettingsForInstance(resolvedInstance);
    }
    applyRuntimeControlHintsToLiveSettings(payload.runtime, resolvedInstance);
    refreshTtsSettingOptions(resolvedInstance);
    updateSessionControlMode();
    updateActiveModelToolbar();
    renderResponsesWorkbenchTargetOptions();
    updatePromptPlaceholder();
    return payload;
}

function parseSseEventBlock(rawBlock) {
    const lines = String(rawBlock || '').split('\n');
    let eventName = '';
    const dataLines = [];
    lines.forEach((line) => {
        if (line.startsWith('event:')) {
            eventName = line.slice(6).trim();
        } else if (line.startsWith('data:')) {
            dataLines.push(line.slice(5).trim());
        }
    });
    if (!eventName && dataLines.length === 0) return null;
    let data = null;
    if (dataLines.length > 0) {
        try {
            data = JSON.parse(dataLines.join('\n'));
        } catch (error) {
            return {
                event: eventName || 'invalid',
                data: null,
                invalid: true,
                rawData: dataLines.join('\n')
            };
        }
    }
    return { event: eventName, data };
}

function formatSseCapabilityLabel(value = '') {
    const token = String(value || '').trim().toLowerCase();
    const labels = {
        chat: 'Text',
        image_generation: 'Image',
        speech_to_text: 'Audio transcript',
        text_to_speech: 'Audio',
        vision_analysis: 'Vision',
        embedding: 'Embedding',
    };
    if (labels[token]) return labels[token];
    return token ? token.replace(/_/g, ' ') : 'Runtime';
}

function sseLateFillStatusIsActive(value = '') {
    return ['accepted', 'pending', 'queued', 'running', 'scheduled'].includes(
        String(value || '').trim().toLowerCase()
    );
}

function buildResponseSseLiveStatusLines(eventName = '', data = {}, responsePayload = {}) {
    const event = String(eventName || '').trim();
    const payload = data && typeof data === 'object' ? data : {};
    const response = responsePayload && typeof responsePayload === 'object'
        ? responsePayload
        : (payload.response && typeof payload.response === 'object' ? payload.response : {});
    const lateFill = payload.late_fill && typeof payload.late_fill === 'object'
        ? payload.late_fill
        : (response.late_fill && typeof response.late_fill === 'object' ? response.late_fill : {});
    if (event === 'response.completed') return [];
    if (event === 'response.failed') return ['Request failed.'];
    if (event === 'response.requires_action') return ['Needs attention.'];
    if (event === 'response.created') return ['Request accepted; backend pending...'];
    if (event === 'response.route.resolved') return ['Route resolved; backend pending...'];
    if (event === 'response.backend.started' || event === 'response.in_progress') return ['Backend running...'];
    if (event === 'response.output_text.done') return ['Text done; follow-up work pending...'];
    if (event === 'response.artifact.created') return ['Artifact created; remaining work pending...'];
    if (event === 'response.late_fill.branch.updated') {
        const branch = payload.branch && typeof payload.branch === 'object' ? payload.branch : {};
        const capability = formatSseCapabilityLabel(branch.capability || response.capability || '');
        const status = String(branch.status || '').trim().toLowerCase();
        const lateStatus = String(payload.late_fill_status || lateFill.status || '').trim().toLowerCase();
        if (['failed', 'blocked', 'repair_needed'].includes(status)) {
            return [`${capability} needs attention.`];
        }
        if (['completed', 'fulfilled', 'waived', 'superseded', 'cancelled'].includes(status)) {
            return sseLateFillStatusIsActive(lateStatus)
                ? [`${capability} completed; remaining work pending...`]
                : [`${capability} resolved.`];
        }
        if (status === 'pending' || status === 'queued') return [`${capability} queued...`];
        return [`${capability} running...`];
    }
    if (event === 'response.late_fill.updated' || event === 'response.state.updated') {
        const status = String(lateFill.status || response.lifecycle_state || '').trim().toLowerCase();
        const statusSemantics = response.status_semantics && typeof response.status_semantics === 'object'
            ? response.status_semantics
            : {};
        if (
            statusSemantics.has_actionable_repair === true
            || statusSemantics.hasActionableRepair === true
            || status === 'repair_needed'
        ) {
            return ['Needs attention.'];
        }
        if (sseLateFillStatusIsActive(status) || statusSemantics.has_open_continuation === true) {
            const activeCount = Number(lateFill.active_count ?? lateFill.activeBranchCount ?? lateFill.active_branch_count ?? 0);
            const pendingCount = Number(lateFill.pending_count ?? lateFill.pendingBranchCount ?? lateFill.pending_branch_count ?? 0);
            const completedCount = Number(lateFill.completed_count ?? lateFill.completedBranchCount ?? lateFill.completed_branch_count ?? 0);
            if (activeCount > 0) return [`${activeCount} branch${activeCount === 1 ? '' : 'es'} running...`];
            if (pendingCount > 0) return [`${pendingCount} branch${pendingCount === 1 ? '' : 'es'} pending...`];
            if (completedCount > 0) return [`${completedCount} branch${completedCount === 1 ? '' : 'es'} completed; remaining work pending...`];
            return ['Runtime work running...'];
        }
    }
    return [];
}

async function sendViaResponsesStream(
    currentInstance,
    targetInstanceId,
    conversationId = targetInstanceId,
    clientMessageId = '',
    requestContext = null,
    pendingMessage = '',
    responseId = '',
    pendingRequestId = ''
) {
    const requestInstance = requestContext?.requestInstance || getRequestExecutionInstance(currentInstance);
    const input = buildResponsesInputForHistory(conversationId, 10, pendingMessage);
    const ghostMessages = currentInstance?.ghostAuto
        ? buildGhostRoutingConversationSnapshot(conversationId)
        : [];
    const ghostPreferences = currentInstance?.ghostAuto ? getResponsesGhostPreferencesPayload() : null;
    const requestMeta = currentInstance?.ghostAuto ? getResponsesGhostRequestMetaPayload() : null;
    const sessionFields = requestContext?.requestControlFields || buildSessionControlRequestFields(requestInstance);
    const selectedReferenceArtifact = buildSelectedReferenceArtifactPayload(conversationId);
    const payload = {
        stream: true,
        input,
        ...sessionFields,
        ...(selectedReferenceArtifact ? { reference_artifacts: selectedReferenceArtifact } : {}),
    };
    const resolvedResponseId = String(responseId || '').trim() || buildCanonicalResponseId();
    payload.response_id = resolvedResponseId;
    if (currentInstance?.ghostAuto) {
        payload.ghost_route = true;
        payload.conversation_id = conversationId;
        payload.ghost_messages = ghostMessages;
        if (ghostPreferences) {
            payload.ghost_preferences = ghostPreferences;
        }
        if (requestMeta) {
            payload.request_meta = requestMeta;
        }
        const ghostPreview = requestContext?.ghostPreview || buildGhostExecutionPreviewPayload(undefined, undefined, conversationId);
        if (ghostPreview) {
            payload.ghost_preview = ghostPreview;
        }
    } else {
        payload.instance_id = targetInstanceId;
    }
    const response = await fetch(`${state.flaskServerUrl}/api/responses`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
    });

    if (!response.ok) {
        let errorMessage = `HTTP ${response.status}`;
        let errorPayload = null;
        try {
            const rawText = await response.text();
            if (rawText) {
                try {
                    errorPayload = JSON.parse(rawText);
                    errorMessage = errorPayload?.error || errorPayload?.message || rawText || errorMessage;
                } catch (_parseError) {
                    errorMessage = rawText;
                }
            }
        } catch (_error) {
            // Keep the original HTTP status fallback when the response body is unreadable.
        }
        const responseError = new Error(errorMessage);
        responseError.response = {
            status: response.status,
            data: errorPayload,
        };
        throw responseError;
    }

    if (!response.body) {
        throw new Error('Streaming response body missing.');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let assembledText = '';
    let finalizedText = '';
    let finalResponse = null;
    let finalResponseFinalized = false;
    let streamResponseId = resolvedResponseId;

    const ensureLoadingResponseBinding = (nextResponseId = '') => {
        const normalizedResponseId = String(nextResponseId || streamResponseId || '').trim();
        if (!normalizedResponseId) return '';
        if (streamResponseId && streamResponseId !== normalizedResponseId) {
            markActiveResponseSseStream(streamResponseId, false);
        }
        streamResponseId = normalizedResponseId;
        markActiveResponseSseStream(streamResponseId, true);
        const visibleContent = finalizedText || assembledText || '<span class="loading-dots">Thinking</span>';
        updateLoadingAssistantMessage(
            conversationId,
            visibleContent,
            {
                clientMessageId,
                responseId: streamResponseId,
                trustedHtml: !(finalizedText || assembledText),
                requestSnapshot: requestContext?.requestSnapshot || null,
            }
        );
        return streamResponseId;
    };

    const applySseLiveStatus = (eventName = '', data = {}, nextResponseId = '') => {
        const lines = buildResponseSseLiveStatusLines(eventName, data, data?.response || {});
        if (!lines.length) return false;
        const normalizedResponseId = ensureLoadingResponseBinding(nextResponseId || data?.response_id || data?.response?.id || '');
        const visibleContent = finalizedText || assembledText || '<span class="loading-dots">Thinking</span>';
        updateLoadingAssistantMessage(
            conversationId,
            visibleContent,
            {
                clientMessageId,
                responseId: normalizedResponseId || streamResponseId,
                trustedHtml: visibleContent === '<span class="loading-dots">Thinking</span>',
                liveStatusLines: lines,
                requestSnapshot: requestContext?.requestSnapshot || null,
                forceRender: true,
            }
        );
        return true;
    };

    const applyPushedResponsePayload = (responsePayload = {}, eventContext = {}) => {
        if (!responsePayload || typeof responsePayload !== 'object') return false;
        const compactStatusPayload = responsePayload.object === 'response.status'
            || (responsePayload.compact === true && responsePayload.ui_compact === true);
        const pushedResponseId = String(
            responsePayload.id
            || responsePayload.response_id
            || streamResponseId
            || ''
        ).trim();
        if (!pushedResponseId) return false;
        ensureLoadingResponseBinding(pushedResponseId);
        if (compactStatusPayload) {
            return applySseLiveStatus(
                eventContext.event || 'response.state.updated',
                eventContext.data || { response: responsePayload },
                pushedResponseId
            );
        }
        const finalRequestSnapshot = mergeRequestSnapshotInputArtifacts(
            requestContext?.requestSnapshot || null,
            responsePayload.input_artifacts
        );
        if (requestContext) {
            requestContext.requestSnapshot = finalRequestSnapshot;
        }
        finalResponse = responsePayload;
        const streamedText = finalizedText || assembledText;
        const visibleContent = getAssistantDisplayContent(responsePayload, streamedText)
            || streamedText
            || '<span class="loading-dots">Thinking</span>';
        const eventLiveStatusLines = buildResponseSseLiveStatusLines(
            eventContext.event || '',
            eventContext.data || {},
            responsePayload
        );
        updateLoadingAssistantMessageFromResponsePayload(
            conversationId,
            pushedResponseId,
            responsePayload,
            finalRequestSnapshot,
            {
                clientMessageId,
                content: visibleContent,
                trustedHtml: visibleContent === '<span class="loading-dots">Thinking</span>',
                ...(eventLiveStatusLines.length ? { liveStatusLines: eventLiveStatusLines } : {}),
                forceRender: true,
            }
        );
        return updateAssistantResponseByResponseId(
            conversationId,
            pushedResponseId,
            responsePayload,
            finalRequestSnapshot
        );
    };

    const applyBlock = (rawBlock) => {
        const parsed = parseSseEventBlock(rawBlock);
        if (!parsed) return;
        if (parsed.invalid) {
            console.warn('Skipping invalid SSE block:', parsed.rawData || rawBlock);
            return;
        }
        if (parsed.event === 'response.output_text.delta') {
            const delta = String(parsed.data?.delta || '');
            if (delta) {
                assembledText += delta;
                updateLoadingAssistantMessage(conversationId, assembledText, { clientMessageId });
            }
            return;
        }
        if (parsed.event === 'response.output_text.done') {
            const text = String(parsed.data?.text || '').trim();
            if (text) {
                finalizedText = text;
                assembledText = text;
                updateLoadingAssistantMessage(conversationId, assembledText, { clientMessageId });
            }
            applySseLiveStatus(parsed.event, parsed.data || {});
            return;
        }
        if (parsed.event === 'response.created') {
            const createdResponseId = String(parsed.data?.response?.id || '').trim();
            if (createdResponseId && pendingRequestId) {
                updatePendingRequest(pendingRequestId, { responseId: createdResponseId });
            }
            ensureLoadingResponseBinding(createdResponseId);
            applySseLiveStatus(parsed.event, parsed.data || {}, createdResponseId);
            return;
        }
        if (
            parsed.event === 'response.in_progress'
            || parsed.event === 'response.route.resolved'
            || parsed.event === 'response.backend.started'
            || parsed.event === 'response.late_fill.branch.updated'
        ) {
            applySseLiveStatus(parsed.event, parsed.data || {});
            return;
        }
        if (
            parsed.event === 'response.state.updated'
            || parsed.event === 'response.late_fill.updated'
            || parsed.event === 'response.artifact.created'
            || parsed.event === 'response.requires_action'
        ) {
            applyPushedResponsePayload(parsed.data?.response || parsed.data || {}, {
                event: parsed.event,
                data: parsed.data || {},
            });
            return;
        }
        if (parsed.event === 'response.failed') {
            applySseLiveStatus(parsed.event, parsed.data || {});
            throw new Error(parsed.data?.error?.message || 'Streaming request failed.');
        }
        if (parsed.event === 'response.completed') {
            finalResponse = parsed.data?.response || null;
            if (finalResponse) {
                const finalRequestSnapshot = mergeRequestSnapshotInputArtifacts(
                    requestContext?.requestSnapshot || null,
                    finalResponse.input_artifacts
                );
                if (requestContext) {
                    requestContext.requestSnapshot = finalRequestSnapshot;
                }
                applyPushedResponsePayload(finalResponse, {
                    event: parsed.event,
                    data: parsed.data || {},
                });
                finalizeLoadingAssistantResponse(
                    conversationId,
                    clientMessageId,
                    finalResponse,
                    finalizedText || assembledText,
                    finalRequestSnapshot
                );
                finalResponseFinalized = true;
                maybeWatchLateFillResponse(
                    conversationId,
                    finalResponse,
                    finalRequestSnapshot
                );
            }
        }
    };

    markActiveResponseSseStream(streamResponseId, true);

    try {
        while (true) {
            const { value, done } = await reader.read();
            buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
            let boundaryIndex = buffer.indexOf('\n\n');
            while (boundaryIndex >= 0) {
                const rawBlock = buffer.slice(0, boundaryIndex);
                buffer = buffer.slice(boundaryIndex + 2);
                applyBlock(rawBlock);
                boundaryIndex = buffer.indexOf('\n\n');
            }
            if (done) break;
        }
    } finally {
        markActiveResponseSseStream(streamResponseId, false);
    }

    if (buffer.trim()) {
        applyBlock(buffer);
    }

    if (finalResponse && !finalResponseFinalized) {
        const finalRequestSnapshot = mergeRequestSnapshotInputArtifacts(
            requestContext?.requestSnapshot || null,
            finalResponse.input_artifacts
        );
        if (requestContext) {
            requestContext.requestSnapshot = finalRequestSnapshot;
        }
        finalizeLoadingAssistantResponse(
            conversationId,
            clientMessageId,
            finalResponse,
            finalizedText || assembledText,
            finalRequestSnapshot
        );
        maybeWatchLateFillResponse(
            conversationId,
            finalResponse,
            finalRequestSnapshot
        );
        finalResponseFinalized = true;
    }

    if (!finalResponse && assembledText) {
        finalResponse = { id: resolvedResponseId, output_text: assembledText };
        updateLoadingAssistantMessage(
            conversationId,
            assembledText,
            {
                clientMessageId,
                requestSnapshot: requestContext?.requestSnapshot || null,
            },
            true
        );
    }
    if (finalResponseFinalized && finalResponse) {
        await hydrateTerminalResponseLookupTruth(
            conversationId,
            finalResponse,
            requestContext?.requestSnapshot || null
        );
    }
    return finalResponse;
}

function stopRequestProgressMonitor(instanceId) {
    const existing = state.requestProgressTimers?.[instanceId];
    if (!existing) return;
    clearInterval(existing.intervalId);
    delete state.requestProgressTimers[instanceId];
    const currentMeta = getCurrentInstanceMeta();
    const responsesTargetId = state.responsesWorkbench.targetInstanceId;
    if (!state.arena.enabled && (currentMeta?.instance_id === instanceId || (isResponsesWorkbenchActive() && responsesTargetId === instanceId))) {
        updateGlobalModelStatus('');
    }
}

function buildRequestProgressMessage(instance, elapsedSec, context = 'chat', slot = '') {
    const displayModel = requestLifecycleTargetIsExternal(instance)
        ? 'ChatGPT'
        : formatModelDisplayName(instance?.model || instance?.instance_id || 'model');
    const prefix = context === 'arena' ? `Arena ${slot}` : displayModel;
    if (elapsedSec < 10) {
        return '<span class="loading-dots">Thinking</span>';
    }
    if (elapsedSec < 30) {
        return '<span class="loading-dots">Still thinking</span>';
    }
    if (elapsedSec < 120) {
        return '<span class="loading-dots">Still generating</span> <span>Heavy model may take a while</span>';
    }
    return `<span class="loading-dots">Still generating</span> <span>${prefix} may be cold-loading or handling a long request</span>`;
}

function startRequestProgressMonitor(instanceId, instance, context = 'chat', slot = '') {
    stopRequestProgressMonitor(instanceId);
    const startedAt = Date.now();
    const tick = () => {
        const elapsedSec = Math.max(1, Math.floor((Date.now() - startedAt) / 1000));
        const messageHtml = buildRequestProgressMessage(instance, elapsedSec, context, slot);
        if (context === 'arena') {
            setLoadingMessage(instanceId, messageHtml);
        }
        if (
            context === 'chat' &&
            (
                state.currentInstanceId === instanceId ||
                (isResponsesWorkbenchActive() && state.responsesWorkbench.targetInstanceId === instanceId)
            )
        ) {
            const displayModel = requestLifecycleTargetIsExternal(instance)
                ? 'ChatGPT'
                : formatModelDisplayName(instance?.model || instance?.instance_id || 'model');
            if (elapsedSec >= 10) {
                updateGlobalModelStatus(`${displayModel}: still generating...`);
            }
        }
    };
    tick();
    if (context === 'arena') {
        state.requestProgressTimers[instanceId] = {
            intervalId: null,
        };
        return;
    }
    state.requestProgressTimers[instanceId] = {
        intervalId: setInterval(tick, 1000),
    };
}


function buildGhostRoutingStatusMessage(payload = {}) {
    const resolvedInstanceId = String(payload.instance_id || '').trim();
    if (!resolvedInstanceId) return '';
    const externalTarget = requestLifecycleTargetIsExternal({
        instance_id: resolvedInstanceId,
        target_kind: payload.target_kind,
    });
    const modelLabel = externalTarget
        ? 'ChatGPT'
        : formatModelDisplayName(payload.model || resolvedInstanceId);
    const backendLabel = externalTarget
        ? 'external provider'
        : formatBackendLabel(payload.backend || 'runtime');
    const capabilityLabel = normalizeCapability(payload.capability || payload.mode || 'chat').replace(/_/g, ' ');
    const routeSource = String(payload.route_source || '').trim().toLowerCase();
    const routeLabel = routeSource === 'router'
        ? 'semantic router'
        : routeSource === 'heuristic'
            ? 'heuristic fallback'
            : 'backend routing';
    const helper = payload.runtime && typeof payload.runtime === 'object'
        ? payload.runtime.embedding_helper
        : null;
    let helperText = '';
    if (helper && typeof helper === 'object') {
        if (helper.available && helper.model) {
            helperText = ` Helper: ${formatModelDisplayName(helper.model)}.`;
        } else if (!helper.available && helper.reason) {
            helperText = ` Helper unavailable: ${String(helper.reason).replace(/_/g, ' ')}.`;
        }
    }
    const reason = String(payload.route_reason || payload.reason || '').trim();
    const contextMode = String(payload.context_mode || payload?.runtime?.context_strategy?.mode || '').trim().toLowerCase();
    const contextText = contextMode
        ? ` Context: ${contextMode === 'compressed_history'
            ? 'compressed history'
            : contextMode === 'bounded_file_context'
                ? 'file context'
                : contextMode === 'raw_history'
                    ? 'raw history'
                    : contextMode.replace(/_/g, ' ')
        }.`
        : '';
    const reasonText = reason ? ` Reason: ${reason}` : '';
    return `Ollmo routed this ${capabilityLabel} request to ${modelLabel} (${backendLabel}) via ${routeLabel}.${helperText}${contextText}${reasonText}`;
}

// Send a message via Flask backend
async function sendMessage() {
    const message = elements.userInput.value.trim();
    const queuedItems = getPendingInputItems();
    if (!message && !queuedItems.length) return;
    if (message === '/new' && !queuedItems.length) {
        elements.userInput.value = '';
        if (typeof syncUserInputComposer === 'function') {
            syncUserInputComposer();
        }
        updateSendButtonState();
        await clearCurrentChat();
        return;
    }
    if (!state.arena.enabled) {
        const pendingConversationId = getActiveConversationId();
        const pendingExplicitTarget = (
            isResponsesWorkbenchActive() && isResponsesWorkbenchAutoTarget()
        )
            ? null
            : getActivePromptTargetInstance();
        if (requestLifecycleTargetIsExternal(pendingExplicitTarget)) {
            const hasExternalFiles = (
                queuedItems.length > 0
                || getSelectedReferenceArtifacts(pendingConversationId).length > 0
            );
            if (
                hasExternalFiles
                && typeof ensureCodexFileConsent === 'function'
                && !(await ensureCodexFileConsent())
            ) {
                updateGlobalModelStatus('The selected files were not sent to ChatGPT.');
                return;
            }
        }
    }
    if (!canSendPrompt()) return;

    if (state.arena.enabled) {
        if (queuedItems.length) {
            updateArenaWarning('Arena mode supports text prompts only.');
            return;
        }
        await sendArenaMessage(message);
        return;
    }

    const conversationId = getActiveConversationId();
    const useResponsesTransport = isResponsesWorkbenchActive();
    const useGhostAuto = useResponsesTransport && isResponsesWorkbenchAutoTarget();
    const explicitTarget = useGhostAuto ? null : getActivePromptTargetInstance();
    if (!useGhostAuto && !explicitTarget) return;
    if (!queuedItems.length) {
        const target = useGhostAuto
            ? buildGhostAutoTargetInstance()
            : explicitTarget;
        await sendSingleMessage(target.instance_id, message, null, true, '', {
            conversationId,
            transport: useResponsesTransport ? 'responses' : 'auto',
            currentInstance: target,
        });
        return;
    }
    if (requestLifecycleTargetIsExternal(explicitTarget)) {
        const externalAttachments = queuedItems
            .filter((item) => item.source === 'upload' && item.file)
            .map((item) => item.file);
        const externalLocalPaths = queuedItems
            .filter((item) => item.source === 'path' && item.localPath)
            .map((item) => String(item.localPath || '').trim())
            .filter(Boolean);
        const primaryAttachment = externalAttachments[0] || null;
        // The legacy snapshot accepts one source. The complete selection travels in the plural context below.
        const primaryLocalPath = primaryAttachment ? '' : (externalLocalPaths[0] || '');
        clearPendingAttachment();
        await sendSingleMessage(
            explicitTarget.instance_id,
            message,
            primaryAttachment,
            false,
            primaryLocalPath,
            {
                conversationId,
                transport: useResponsesTransport ? 'responses' : 'auto',
                currentInstance: explicitTarget,
                externalAttachments,
                externalLocalPaths,
            }
        );
        return;
    }
    // Clear attachment UI state immediately after submit so the next prompt starts clean.
    clearPendingAttachment();
    for (let idx = 0; idx < queuedItems.length; idx += 1) {
        const item = queuedItems[idx];
        const attachment = item.source === 'upload' ? item.file : null;
        const localPath = item.source === 'path' ? item.localPath : '';
        const label = item.label || attachment?.name || basenameFromPath(localPath) || `item ${idx + 1}`;
        updateGlobalModelStatus(`Processing ${idx + 1}/${queuedItems.length}: ${label}`);
        const target = useGhostAuto
            ? buildGhostAutoTargetInstance()
            : explicitTarget;
        await sendSingleMessage(target.instance_id, message, attachment, false, localPath, {
            conversationId,
            transport: useResponsesTransport ? 'responses' : 'auto',
            currentInstance: target,
        });
    }
    updateGlobalModelStatus(`Completed ${queuedItems.length} item(s).`);
    setTimeout(() => updateGlobalModelStatus(''), 2000);
}

function buildHistoryForApi(instanceId, limit = 10) {
    const history = state.conversations[instanceId] || [];
    return history.filter(m => !m.isLoading).slice(-limit);
}

function getRequestedImageCount(instance) {
    if (!isImageGenerationInstance(instance)) return 1;
    return Math.min(8, Math.max(1, parseInt(state.settings.imageCount, 10) || 1));
}

function getExplicitGhostAutoImageCount(message = '') {
    const text = String(message || '').trim().toLowerCase();
    if (!text) return 1;
    const match = text.match(/\b([1-8])\s+(images?|variations?|versions?|renders?|pictures?|shots|bilder|varianten|versionen)\b/);
    if (!match) return 1;
    return Math.min(8, Math.max(1, parseInt(match[1], 10) || 1));
}

function parseExplicitBatchPrompts(message = '') {
    const rawMessage = String(message || '').trim();
    if (!rawMessage || !rawMessage.startsWith('[')) {
        return [];
    }
    try {
        const parsed = JSON.parse(rawMessage);
        if (!Array.isArray(parsed)) {
            return [];
        }
        return parsed
            .map((item) => {
                if (item && typeof item === 'object' && !Array.isArray(item)) {
                    const prompt = String(
                        item.prompt
                            ?? item.input
                            ?? item.text
                            ?? item.content
                            ?? ''
                    ).trim();
                    if (!prompt) {
                        return null;
                    }
                    const normalized = { prompt };
                    const aspectRatio = String(item.aspect_ratio ?? item.aspectRatio ?? '').trim();
                    if (aspectRatio) {
                        normalized.aspect_ratio = aspectRatio;
                    }
                    if (item.width !== undefined && item.width !== null && item.width !== '') {
                        normalized.width = item.width;
                    }
                    if (item.height !== undefined && item.height !== null && item.height !== '') {
                        normalized.height = item.height;
                    }
                    return normalized;
                }
                const prompt = String(item || '').trim();
                return prompt ? prompt : null;
            })
            .filter(Boolean);
    } catch (_error) {
        if (!rawMessage.endsWith(']')) {
            return [];
        }
        const inner = rawMessage.slice(1, -1);
        const matches = [...inner.matchAll(/"([\s\S]*?)"/g)];
        if (!matches.length) {
            return [];
        }
        return matches
            .map((match) => String(match[1] || '').replace(/\s+/g, ' ').trim())
            .filter(Boolean);
    }
}

function clearLoadingMessages(instanceId) {
    if (!state.conversations[instanceId]) return;
    state.conversations[instanceId] = state.conversations[instanceId].filter(m => !m.isLoading);
}

async function sendSingleMessage(instanceId, message, attachment = null, clearAttachmentOnSuccess = true, localPath = '', options = {}) {
    const currentInstance = options.currentInstance || state.runningInstances.find(inst => inst.instance_id === instanceId);
    if (!currentInstance) {
        alert('Current instance not found!');
        return;
    }
    const conversationId = options.conversationId || instanceId;
    const transport = options.transport || 'auto';
    const requestId = options.requestId || buildConversationRequestId('msg');
    const requestKey = options.requestKey || requestId;
    let requestRegistered = false;

    let previewMonitorStarted = false;
    let shouldResetGhostAutoRoute = false;
    let autoRouteReleasedForNextPrompt = false;
    let loadingMessageId = '';
    let activeLoadingMessageId = '';
    let responseId = '';
    let requestContext = null;
    let keepPendingRequestForRecovery = false;
    const releaseActiveRequestUiState = () => {
        setInstanceRequestPending(requestKey, false);
        stopRequestProgressMonitor(requestKey);
    };
    const releasePendingRequestState = () => {
        clearPendingRequest(requestId);
        releaseActiveRequestUiState();
        requestRegistered = false;
    };
    if (currentInstance.ghostAuto) {
        if (hasPendingConversationPreview(conversationId, { excludeRequestId: requestId })) {
            updateGlobalModelStatus('Ollmo is already resolving another request.');
            return;
        }
        registerPendingRequest({
            requestId,
            conversationId,
            phase: 'preview',
        });
        requestRegistered = true;
        setInstanceRequestPending(requestKey, true);
        startRequestProgressMonitor(requestKey, currentInstance, 'chat');
        previewMonitorStarted = true;
        updateGlobalModelStatus('Ollmo is resolving the route...');
        try {
            await ensureGhostAutoResolvedTarget(message, attachment, localPath, conversationId);
        } catch (error) {
            releasePendingRequestState();
            resetResponsesWorkbenchAutoRoute();
            const errorMsg = error.response?.data?.error || error.message || 'Ollmo preview failed.';
            updateGlobalModelStatus(errorMsg);
            addMessageToConversation(conversationId, 'assistant', `Sorry, error: ${errorMsg}`);
            return;
        }
    }

    const requestInstance = getRequestExecutionInstance(currentInstance);
    if (!requestInstance?.instance_id) {
        if (requestRegistered) {
            releasePendingRequestState();
        }
        if (currentInstance.ghostAuto) {
            resetResponsesWorkbenchAutoRoute();
        }
        updateGlobalModelStatus('No resolved target instance is available for this request.');
        return;
    }
    const capability = normalizeCapability(requestInstance?.capability || 'chat');
    const explicitBatchPrompts = !attachment && !localPath
        && (currentInstance.ghostAuto || isImageGenerationInstance(requestInstance))
        ? parseExplicitBatchPrompts(message)
        : [];
    const usesResponsesStream = capability === 'chat'
        && !attachment
        && !localPath
        && !explicitBatchPrompts.length;
    const effectiveTransport = 'responses';
    responseId = buildCanonicalResponseId();
    if (requestRegistered) {
        updatePendingRequest(requestId, {
            conversationId,
            instanceId: requestInstance.instance_id,
            capability,
            phase: 'running',
            responseId,
        });
    } else {
        registerPendingRequest({
            requestId,
            conversationId,
            instanceId: requestInstance.instance_id,
            capability,
            phase: 'running',
            responseId,
        });
        requestRegistered = true;
    }
    if (capability === 'chat' && hasPendingConversationChatRequest(conversationId, { excludeRequestId: requestId })) {
        releasePendingRequestState();
        if (currentInstance.ghostAuto) {
            resetResponsesWorkbenchAutoRoute();
        }
        updateGlobalModelStatus('A chat request is already running in this conversation.');
        return;
    }
    if (capability !== 'chat' && hasPendingRequestForInstance(requestInstance.instance_id, { excludeRequestId: requestId })) {
        const displayModel = formatModelDisplayName(requestInstance.model || requestInstance.instance_id);
        releasePendingRequestState();
        if (currentInstance.ghostAuto) {
            resetResponsesWorkbenchAutoRoute();
        }
        updateGlobalModelStatus(`A request is already running for ${displayModel}.`);
        return;
    }
    const fileNameForKind = attachment?.name || localPath;
    const fileKind = fileNameForKind ? inferFileKindFromName(fileNameForKind) : '';
    if (capability === 'speech_to_text' && fileNameForKind && fileKind !== 'audio') {
        releasePendingRequestState();
        addMessageToConversation(
            conversationId,
            'assistant',
            `Please attach an audio file for speech_to_text (received: ${fileKind || 'unknown'}).`
        );
        return;
    }
    const sessionControlValidation = validateRequiredSessionControls(requestInstance);
    if (sessionControlValidation) {
        releasePendingRequestState();
        focusSessionControlField(sessionControlValidation.fieldKey);
        updateGlobalModelStatus(sessionControlValidation.message);
        return;
    }
    requestContext = buildRequestExecutionContext(
        requestInstance,
        {
            ghostPreview: currentInstance.ghostAuto
                ? buildGhostExecutionPreviewPayload(requestInstance, state.responsesWorkbench.ghostResolvedRoute, conversationId)
                : null,
            requestMeta: currentInstance.ghostAuto ? getResponsesGhostRequestMetaPayload() : null,
            requestId,
            transport: effectiveTransport,
            message,
            attachment,
            localPath,
            conversationId,
            responseId,
            batchPrompts: explicitBatchPrompts,
        }
    );
    if (requestContext && !currentInstance.ghostAuto && requestLifecycleTargetIsExternal(requestInstance)) {
        requestContext.externalAttachments = Array.isArray(options.externalAttachments)
            ? options.externalAttachments.filter(Boolean)
            : attachment
                ? [attachment]
                : [];
        requestContext.externalLocalPaths = Array.isArray(options.externalLocalPaths)
            ? options.externalLocalPaths.map((item) => String(item || '').trim()).filter(Boolean)
            : localPath
                ? [localPath]
                : [];
    }
    if (currentInstance.ghostAuto) {
        resetResponsesWorkbenchAutoRoute();
        autoRouteReleasedForNextPrompt = true;
    }

    let userPreview = buildUserPromptPreview(message, attachment, capability, localPath);
    if (requestContext?.externalAttachments?.length + requestContext?.externalLocalPaths?.length > 1) {
        const allNames = [
            ...requestContext.externalAttachments.map((item) => item?.name || 'attachment'),
            ...requestContext.externalLocalPaths.map((item) => basenameFromPath(item) || item),
        ];
        userPreview = `${message}\n${allNames.map((name) => `[File: ${name}]`).join('\n')}`.trim();
    }
    const userMessageId = addMessageToConversation(
        conversationId,
        'user',
        userPreview || message || '[Attachment]',
        false,
        {
            requestSnapshot: requestContext?.requestSnapshot || null,
        }
    );
    const userMessage = (state.conversations[conversationId] || []).find((entry) => entry?.clientMessageId === userMessageId) || null;
    updatePendingRequest(requestId, {
        userMessageId,
        userContent: String(userMessage?.content || userPreview || message || '[Attachment]'),
        userTimestamp: String(userMessage?.timestamp || '').trim() || null,
        requestSnapshot: requestContext?.requestSnapshot || null,
    });
    const applyResolvedRequestSnapshot = (payload = null, targetMessageId = '') => {
        const mergedSnapshot = mergeRequestSnapshotInputArtifacts(
            requestContext?.requestSnapshot || null,
            payload?.input_artifacts
        );
        const nextSnapshot = sanitizeRequestSnapshot({
            ...(mergedSnapshot || requestContext?.requestSnapshot || {}),
            ...(payload?.request_meta ? { request_meta: payload.request_meta } : {}),
            ...(payload?.runtime?.developer_diagnostics ? { developer_diagnostics: payload.runtime.developer_diagnostics } : {}),
        });
        if (!nextSnapshot) {
            return null;
        }
        if (requestContext) {
            requestContext.requestSnapshot = nextSnapshot;
        }
        updatePendingRequest(requestId, { requestSnapshot: nextSnapshot });
        updateMessageRequestSnapshot(conversationId, userMessageId, nextSnapshot);
        const resolvedTargetId = String(targetMessageId || '').trim();
        if (resolvedTargetId && resolvedTargetId !== userMessageId) {
            updateMessageRequestSnapshot(conversationId, resolvedTargetId, nextSnapshot);
        }
        return nextSnapshot;
    };
    elements.userInput.value = '';
    if (typeof syncUserInputComposer === 'function') {
        syncUserInputComposer();
    }
    clearStaleLoadingMessages(conversationId);
    updateSendButtonState();
    updatePromptPlaceholder();

    loadingMessageId = addLoadingAssistantMessage(conversationId, requestContext?.requestSnapshot || null);
    activeLoadingMessageId = loadingMessageId;
    updatePendingRequest(requestId, { loadingMessageId, responseId });
    scrollConversationToBottom(conversationId);
    if (!previewMonitorStarted) {
        setInstanceRequestPending(requestKey, true);
        startRequestProgressMonitor(requestKey, requestInstance, 'chat');
    }

    let ghostRoutingStatusMessage = '';
    try {
        const imageBatchCount = explicitBatchPrompts.length
            ? 1
            : (
                !attachment && !localPath
                    ? (
                        currentInstance.ghostAuto && isImageGenerationInstance(requestInstance)
                            ? getExplicitGhostAutoImageCount(message)
                            : getRequestedImageCount(requestInstance)
                    )
                    : 1
            );
        let responsePayload = null;
        let handledStreaming = false;
        if (effectiveTransport === 'responses' || explicitBatchPrompts.length) {
            if (isImageGenerationInstance(requestInstance) && imageBatchCount > 1) {
                handledStreaming = true;
                for (let idx = 0; idx < imageBatchCount; idx += 1) {
                    updateGlobalModelStatus(`Generating image ${idx + 1}/${imageBatchCount}...`);
                    const responseResult = await sendViaResponsesTransport(
                        currentInstance,
                        instanceId,
                        conversationId,
                        message,
                        attachment,
                        localPath,
                        activeLoadingMessageId,
                        requestContext,
                        responseId,
                        requestId
                    );
                    responsePayload = responseResult.payload || {};
                    const resolvedRequestSnapshot = applyResolvedRequestSnapshot(responsePayload, activeLoadingMessageId);
                    finalizeRequestResponse(
                        conversationId,
                        activeLoadingMessageId,
                        responsePayload,
                        resolvedRequestSnapshot || requestContext?.requestSnapshot || null
                    );
                    maybeWatchLateFillResponse(
                        conversationId,
                        responsePayload,
                        resolvedRequestSnapshot || requestContext?.requestSnapshot || null
                    );
                    if (idx + 1 < imageBatchCount) {
                        activeLoadingMessageId = addLoadingAssistantMessage(conversationId, requestContext?.requestSnapshot || null);
                        updatePendingRequest(requestId, { loadingMessageId: activeLoadingMessageId });
                    }
                }
            } else {
                const responseResult = await sendViaResponsesTransport(
                    currentInstance,
                    instanceId,
                    conversationId,
                    message,
                    attachment,
                    localPath,
                    loadingMessageId,
                    requestContext,
                    responseId,
                    requestId
                );
                responsePayload = responseResult.payload || {};
                handledStreaming = Boolean(responseResult.streamed);
            }
        }
        if (!handledStreaming) {
            const resolvedRequestSnapshot = applyResolvedRequestSnapshot(responsePayload, loadingMessageId);
            finalizeRequestResponse(
                conversationId,
                loadingMessageId,
                responsePayload,
                resolvedRequestSnapshot || requestContext?.requestSnapshot || null
            );
            maybeWatchLateFillResponse(
                conversationId,
                responsePayload,
                resolvedRequestSnapshot || requestContext?.requestSnapshot || null
            );
        } else {
            applyResolvedRequestSnapshot(responsePayload, loadingMessageId);
        }
        if (currentInstance.ghostAuto) {
            ghostRoutingStatusMessage = buildGhostRoutingStatusMessage(responsePayload);
            shouldResetGhostAutoRoute = !autoRouteReleasedForNextPrompt;
        }
        if ((attachment || localPath) && clearAttachmentOnSuccess) {
            clearPendingAttachment();
        }
    } catch (error) {
        console.error('Error sending message:', error);
        let missingSessionControls = error.response?.data?.missing_session_controls || null;
        if (currentInstance.ghostAuto && missingSessionControls?.instance) {
            let syncResult = syncGhostMissingSessionControlsState(missingSessionControls);
            if (!syncResult.validation) {
                try {
                    requestContext = buildRequestExecutionContext(
                        missingSessionControls.instance,
                        {
                            ghostPreview: buildGhostExecutionPreviewPayload(
                                missingSessionControls.instance,
                                state.responsesWorkbench.ghostResolvedRoute,
                                conversationId
                            ),
                            requestMeta: currentInstance.ghostAuto ? getResponsesGhostRequestMetaPayload() : null,
                            requestId,
                            transport: effectiveTransport,
                            message,
                            attachment,
                            localPath,
                            conversationId,
                            responseId,
                            batchPrompts: explicitBatchPrompts,
                        }
                    );
                    updatePendingRequest(requestId, { requestSnapshot: requestContext?.requestSnapshot || null });
                    updateMessageRequestSnapshot(conversationId, userMessageId, requestContext?.requestSnapshot || null);
                    updateLoadingAssistantMessage(
                        conversationId,
                        '<span class="loading-dots">Continuing with inferred controls</span>',
                        {
                            clientMessageId: activeLoadingMessageId,
                            trustedHtml: true,
                            requestSnapshot: requestContext?.requestSnapshot || null,
                        }
                    );
                    const responseResult = await sendViaResponsesTransport(
                        currentInstance,
                        instanceId,
                        conversationId,
                        message,
                        attachment,
                        localPath,
                        activeLoadingMessageId,
                        requestContext,
                        responseId,
                        requestId
                    );
                    const retryPayload = responseResult.payload || {};
                    const resolvedRetrySnapshot = applyResolvedRequestSnapshot(retryPayload, activeLoadingMessageId);
                    finalizeRequestResponse(
                        conversationId,
                        activeLoadingMessageId,
                        retryPayload,
                        resolvedRetrySnapshot || requestContext?.requestSnapshot || null
                    );
                    maybeWatchLateFillResponse(
                        conversationId,
                        retryPayload,
                        resolvedRetrySnapshot || requestContext?.requestSnapshot || null
                    );
                    if (currentInstance.ghostAuto) {
                        ghostRoutingStatusMessage = buildGhostRoutingStatusMessage(retryPayload);
                        shouldResetGhostAutoRoute = !autoRouteReleasedForNextPrompt;
                    }
                    if ((attachment || localPath) && clearAttachmentOnSuccess) {
                        clearPendingAttachment();
                    }
                    return;
                } catch (retryError) {
                    console.error('Ghost auto-retry after inferred controls failed:', retryError);
                    error = retryError;
                    missingSessionControls = retryError.response?.data?.missing_session_controls || null;
                    if (currentInstance.ghostAuto && missingSessionControls?.instance) {
                        syncResult = syncGhostMissingSessionControlsState(missingSessionControls);
                    }
                }
            }
            if (currentInstance.ghostAuto && missingSessionControls?.instance) {
                const validation = syncResult.validation;
                const effectiveMissingFields = validation?.fieldKey
                    ? [{ field_key: validation.fieldKey, message: validation.message }]
                    : null;
                updateLoadingAssistantMessage(
                    conversationId,
                    buildMissingSessionControlsAssistantMessage(
                        missingSessionControls,
                        syncResult.appliedHints,
                        effectiveMissingFields
                    ),
                    { clientMessageId: activeLoadingMessageId },
                    true
                );
                if (validation?.fieldKey) {
                    focusSessionControlField(validation.fieldKey);
                }
                updateGlobalModelStatus(
                    String(validation?.message || error.response?.data?.error || 'Required model inputs are missing.')
                );
                return;
            }
        }
        const errorPayload = (error?.response?.data && typeof error.response.data === 'object')
            ? error.response.data
            : null;
        let errorMsg = extractResponseErrorMessage(errorPayload || {}, error.message || 'Unknown error');
        if (isRecoverableResponsesConnectionError(error, responseId)) {
            keepPendingRequestForRecovery = true;
            updatePendingRequest(requestId, {
                phase: 'recovering',
                responseId,
                loadingMessageId: activeLoadingMessageId || loadingMessageId,
            });
            updateLoadingAssistantMessage(
                conversationId,
                buildPendingResponseRecoveryMessage(),
                { clientMessageId: activeLoadingMessageId || loadingMessageId, trustedHtml: true }
            );
            updateGlobalModelStatus('Connection lost. Trying to resume the local response...');
            schedulePendingResponseResumePoll(requestId, 700);
            return;
        }
        if (error?.code === 'ECONNABORTED' && /timeout of \d+ms exceeded/i.test(String(error.message || ''))) {
            const match = String(error.message || '').match(/timeout of (\d+)ms exceeded/i);
            const seconds = match ? Math.round(Number(match[1]) / 1000) : null;
            errorMsg = seconds
                ? `Request timeout after ${seconds}s. Model may still be processing locally.`
                : 'Request timeout. Model may still be processing locally.';
        }
        if (errorPayload) {
            finalizeLoadingAssistantResponse(
                conversationId,
                activeLoadingMessageId,
                errorPayload,
                `Sorry, error: ${errorMsg}`,
                requestContext?.requestSnapshot || null
            );
        } else {
            updateLoadingAssistantMessage(
                conversationId,
                `Sorry, error: ${errorMsg}`,
                { clientMessageId: activeLoadingMessageId },
                true
            );
        }
        if (currentInstance.ghostAuto) {
            shouldResetGhostAutoRoute = !autoRouteReleasedForNextPrompt;
        }
    } finally {
        if (keepPendingRequestForRecovery) {
            releaseActiveRequestUiState();
        } else {
            releasePendingRequestState();
        }
        if (ghostRoutingStatusMessage) {
            updateGlobalModelStatus(ghostRoutingStatusMessage);
        }
        if (shouldResetGhostAutoRoute) {
            resetResponsesWorkbenchAutoRoute();
        }
    }
}

async function sendArenaMessage(message) {
    if (!arenaHasDistinctSelections()) {
        updateArenaWarning('Select two different models for Arena mode.');
        return;
    }
    const arenaReady = await ensureFreshArenaConversations({ forceRotate: false });
    if (!arenaReady) {
        updateSendButtonState();
        return;
    }

    const selections = [
        { slot: 'A', instanceId: state.arena.modelA },
        { slot: 'B', instanceId: state.arena.modelB }
    ];
    const targets = selections
        .map(entry => ({
            ...entry,
            instance: state.runningInstances.find(inst => inst.instance_id === entry.instanceId)
        }))
        .filter(entry => entry.instance);

    if (targets.length < 2) {
        updateArenaWarning('Both arena selections must be running.');
        await fetchRunningInstances();
        return;
    }
    const nonChatTarget = targets.find(({ instance }) => !isTextCapableInstance(instance));
    if (nonChatTarget) {
        updateArenaWarning('Arena mode requires chat-capable instances on both sides.');
        return;
    }

    const preparedTargets = targets.map((entry) => ({
        ...entry,
        conversationId: getArenaConversationId(entry.instanceId) || getInstanceConversationId(entry.instanceId),
        clientMessageId: `arena-${entry.slot}-${Date.now()}-${Math.random().toString(16).slice(2)}`
    }));
    elements.userInput.value = '';
    if (typeof syncUserInputComposer === 'function') {
        syncUserInputComposer();
    }
    updateSendButtonState();

    preparedTargets.forEach(({ conversationId, clientMessageId }) => {
        addMessageToConversation(conversationId, 'user', message);
        addMessageToConversation(
            conversationId,
            'assistant',
            '<span class="loading-dots">Thinking</span>',
            true,
            { clientMessageId, trustedHtml: true }
        );
    });

    const arenaRequests = preparedTargets.map(({ slot, instanceId, conversationId, instance, clientMessageId }) => {
        startRequestProgressMonitor(conversationId, instance, 'arena', slot);
        return sendViaResponsesStream(instance, instanceId, conversationId, clientMessageId, null, message)
        .catch(error => {
            console.error(`Arena ${slot} error:`, error);
            clearLoadingMessages(conversationId);
            const messageText = error.response?.data?.error || error.message || 'Unknown error';
            addMessageToConversation(conversationId, 'assistant', `Arena ${slot} issue: ${messageText}`);
        })
        .finally(() => {
            stopRequestProgressMonitor(conversationId);
        });
    });

    await Promise.allSettled(arenaRequests);
}

// Add a message to the conversation state and render
