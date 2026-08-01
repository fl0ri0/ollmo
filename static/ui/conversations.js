function getDefaultInteractiveInstance() {
    return getUserFacingRunningInstances()[0] || null;
}

function buildInstanceConversationSlotId(instanceId) {
    return `instance:${String(instanceId || '').trim() || 'conversation'}`;
}

function normalizeConversationMessageCount(value) {
    if (value === null || value === undefined || value === '') return null;
    const parsed = Number.parseInt(String(value), 10);
    return Number.isFinite(parsed) && parsed >= 0 ? parsed : null;
}

function normalizeConversationBoolean(value) {
    if (value === true || value === false) return value;
    const normalized = String(value ?? '').trim().toLowerCase();
    if (!normalized) return null;
    if (['1', 'true', 'yes', 'on'].includes(normalized)) return true;
    if (['0', 'false', 'no', 'off'].includes(normalized)) return false;
    return null;
}

function normalizeConversationMetadata(raw = {}, fallbackConversationId = '') {
    const source = raw && typeof raw === 'object' ? raw : {};
    const conversationId = String(fallbackConversationId || '').trim();
    const metadata = {
        workspace: String(source.workspace || '').trim() || null,
        slotId: String(source.slotId || source.slot_id || '').trim() || null,
        sourceInstanceId: String(source.sourceInstanceId || source.source_instance_id || '').trim() || null,
        label: String(source.label || '').trim() || null,
        parentConversationId: String(source.parentConversationId || source.parent_conversation_id || '').trim() || null,
        rootConversationId: String(source.rootConversationId || source.root_conversation_id || '').trim() || null,
        createdAt: String(source.createdAt || source.created_at || '').trim() || null,
        displayTitle: trimConversationPreviewText(source.displayTitle || source.display_title || ''),
        previewText: trimConversationPreviewText(source.previewText || source.preview_text || ''),
        messageCount: normalizeConversationMessageCount(source.messageCount ?? source.message_count),
        lastMessageAt: String(source.lastMessageAt || source.last_message_at || '').trim() || null,
        freshRoot: normalizeConversationBoolean(source.freshRoot ?? source.fresh_root),
        model: String(source.model || '').trim() || null,
        backend: String(source.backend || '').trim() || null,
        capability: String(source.capability || '').trim() || null,
    };
    if (!metadata.workspace && conversationId.startsWith(RESPONSES_WORKBENCH_ID)) {
        metadata.workspace = 'responses';
    } else if (!metadata.workspace && conversationId && !conversationId.startsWith('__instance_chat__--')) {
        metadata.workspace = 'instance';
    }
    if (metadata.workspace === 'responses') {
        metadata.slotId = metadata.slotId || 'responses-workbench';
        metadata.label = metadata.label || 'responses-workbench';
    } else if (metadata.workspace === 'instance') {
        const fallbackSource = metadata.sourceInstanceId || (conversationId && !conversationId.startsWith('__instance_chat__--') ? conversationId : '');
        metadata.sourceInstanceId = fallbackSource || null;
        metadata.slotId = metadata.slotId || buildInstanceConversationSlotId(fallbackSource);
        metadata.label = metadata.label || fallbackSource || metadata.slotId;
    }
    return Object.values(metadata).some(Boolean) ? metadata : null;
}

function serializeConversationMetadata(metadata = null) {
    const normalized = normalizeConversationMetadata(metadata);
    if (!normalized) return null;
    return {
        workspace: normalized.workspace,
        slot_id: normalized.slotId,
        source_instance_id: normalized.sourceInstanceId,
        label: normalized.label,
        parent_conversation_id: normalized.parentConversationId,
        root_conversation_id: normalized.rootConversationId,
        created_at: normalized.createdAt,
        display_title: normalized.displayTitle,
        preview_text: normalized.previewText,
        message_count: normalized.messageCount,
        last_message_at: normalized.lastMessageAt,
        fresh_root: normalized.freshRoot === true ? true : undefined,
    };
}

function registerConversationMetadata(conversationId, metadata = null) {
    const key = String(conversationId || '').trim();
    if (!key) return null;
    const existing = normalizeConversationMetadata(state.conversationMetadataById?.[key], key);
    const incoming = normalizeConversationMetadata(metadata, key);
    const merged = normalizeConversationMetadata({
        ...(existing || {}),
        ...(incoming || {}),
    }, key);
    if (merged) {
        state.conversationMetadataById[key] = merged;
    }
    return merged;
}

function buildConversationSlotHistoryKey(workspace = '', slotId = '') {
    const workspaceKey = String(workspace || '').trim();
    const slotKey = String(slotId || '').trim();
    if (!workspaceKey || !slotKey) return '';
    return `${workspaceKey}:${slotKey}`;
}

function normalizeConversationHistoryIds(ids = []) {
    const next = [];
    const seen = new Set();
    (Array.isArray(ids) ? ids : []).forEach((item) => {
        const value = String(item || '').trim();
        if (!value || seen.has(value)) return;
        seen.add(value);
        next.push(value);
    });
    return next;
}

function getConversationSlotHistoryKey(conversationId) {
    const metadata = getConversationMetadata(conversationId);
    return buildConversationSlotHistoryKey(metadata?.workspace, metadata?.slotId);
}

function setSlotHistoryConversationIds(workspace = '', slotId = '', ids = []) {
    const key = buildConversationSlotHistoryKey(workspace, slotId);
    if (!key) return [];
    const next = normalizeConversationHistoryIds(ids);
    state.slotHistoryConversationIdsByKey[key] = next;
    saveConversationSlots();
    renderConversationHistoryList();
    return next;
}

function mergeConversationHistoryIds(...lists) {
    const merged = [];
    const seen = new Set();
    lists.forEach((list) => {
        (Array.isArray(list) ? list : []).forEach((item) => {
            const value = String(item || '').trim();
            if (!value || seen.has(value)) return;
            seen.add(value);
            merged.push(value);
        });
    });
    return merged;
}

function getPersistedConversationMessages(conversationId = '') {
    const key = String(conversationId || '').trim();
    const conversation = Array.isArray(state.conversations?.[key]) ? state.conversations[key] : [];
    return conversation.filter((message) => message && !message.isLoading && !message.ephemeralUiNotice);
}

function getPersistedConversationMessageCount(conversationId = '') {
    return getPersistedConversationMessages(conversationId).length;
}

function findConversationPreviewCandidate(messages = [], { role = '', direction = 'forward' } = {}) {
    const list = Array.isArray(messages) ? messages : [];
    const normalizedRole = String(role || '').trim().toLowerCase();
    const normalizedDirection = String(direction || '').trim().toLowerCase() === 'reverse' ? 'reverse' : 'forward';
    const iterate = normalizedDirection === 'reverse'
        ? [...list].reverse()
        : list;
    return iterate.find((message) => {
        if (normalizedRole && String(message?.role || '').trim().toLowerCase() !== normalizedRole) {
            return false;
        }
        return Boolean(trimConversationPreviewText(message?.content));
    }) || null;
}

function buildConversationLedgerMetadataFromMessages(messages = []) {
    const persistedMessages = Array.isArray(messages)
        ? messages.filter((message) => message && !message.isLoading && !message.ephemeralUiNotice)
        : [];
    const primaryTitleMessage = findConversationPreviewCandidate(persistedMessages, { role: 'user', direction: 'forward' })
        || findConversationPreviewCandidate(persistedMessages, { direction: 'forward' });
    const latestPreviewMessage = findConversationPreviewCandidate(persistedMessages, { role: 'user', direction: 'reverse' })
        || findConversationPreviewCandidate(persistedMessages, { direction: 'reverse' })
        || primaryTitleMessage;
    let lastMessageAt = '';
    for (let index = persistedMessages.length - 1; index >= 0; index -= 1) {
        const timestamp = String(persistedMessages[index]?.timestamp || '').trim();
        if (!timestamp || Number.isNaN(Date.parse(timestamp))) continue;
        lastMessageAt = timestamp;
        break;
    }
    return {
        display_title: trimConversationPreviewText(primaryTitleMessage?.content || '') || null,
        preview_text: trimConversationPreviewText(latestPreviewMessage?.content || '') || null,
        message_count: persistedMessages.length,
        last_message_at: lastMessageAt || null,
    };
}

function buildConversationLedgerMetadata(conversationId = '') {
    return buildConversationLedgerMetadataFromMessages(getPersistedConversationMessages(conversationId));
}

function buildConversationHistorySnapshot(conversationId = '') {
    const key = String(conversationId || '').trim();
    if (!key) return '[]';
    const slotHistoryKey = getConversationSlotHistoryKey(key);
    return JSON.stringify([
        getPersistedConversationMessages(key).map((message) => buildConversationHistoryMessageIdentity(message)),
        normalizeConversationHistoryIds(state.slotHistoryConversationIdsByKey?.[slotHistoryKey]),
    ]);
}

function buildConversationHistoryMessageIdentity(message = {}) {
    const source = message && typeof message === 'object' ? message : {};
    const requestSnapshot = sanitizeRequestSnapshot(source.request_snapshot || source.requestSnapshot);
    const artifacts = sanitizeResponseArtifacts(source.artifacts)
        .map((artifact) => ({
            type: String(artifact.type || '').trim(),
            path: String(artifact.path || '').trim(),
            name: String(artifact.name || '').trim(),
            source_path: String(artifact.source_path || '').trim(),
            origin: String(artifact.origin || '').trim(),
            mime_type: String(artifact.mime_type || '').trim(),
            availability: String(artifact.availability || '').trim(),
            purged_at: String(artifact.purged_at || '').trim(),
            purge_reason: String(artifact.purge_reason || '').trim(),
        }))
        .filter((artifact) => artifact.type && (artifact.path || artifact.source_path || artifact.name));
    const outputs = sanitizeResponseOutputs(source.outputs || source.canonical_outputs || source.canonicalOutputs)
        .map((output) => ({
            slot_id: String(output.slot_id || '').trim(),
            branch_id: String(output.branch_id || '').trim(),
            phase_id: String(output.phase_id || '').trim(),
            type: String(output.type || '').trim(),
            status: String(output.status || '').trim(),
            lifecycle: String(output.lifecycle || '').trim(),
            artifact_ref: String(output.artifact_ref || '').trim(),
            value: String(output.value || '').trim(),
            artifacts: sanitizeResponseArtifacts(output.artifacts)
                .map((artifact) => ({
                    type: String(artifact.type || '').trim(),
                    path: String(artifact.path || '').trim(),
                    artifact_ref: String(artifact.artifact_ref || artifact.ref || '').trim(),
                })),
        }));
    const outputSlots = sanitizeResponseOutputSlots(source.outputSlots || source.output_slots)
        .map((slot) => ({
            slot_id: String(slot.slot_id || '').trim(),
            branch_id: String(slot.branch_id || '').trim(),
            phase_id: String(slot.phase_id || '').trim(),
            type: String(slot.type || '').trim(),
            status: String(slot.status || '').trim(),
            lifecycle: String(slot.lifecycle || '').trim(),
            artifact_ref: String(slot.artifact_ref || '').trim(),
            placeholder_ref: String(slot.placeholder_ref || '').trim(),
        }));
    const outputBranches = sanitizeResponseOutputBranches(source.outputBranches || source.output_branches)
        .map((branch) => ({
            slot_id: String(branch.slot_id || '').trim(),
            branch_id: String(branch.branch_id || '').trim(),
            phase_id: String(branch.phase_id || '').trim(),
            type: String(branch.type || '').trim(),
            status: String(branch.status || '').trim(),
            lifecycle: String(branch.lifecycle || '').trim(),
            artifact_ref: String(branch.artifact_ref || '').trim(),
            placeholder_ref: String(branch.placeholder_ref || '').trim(),
        }));
    return JSON.stringify([
        String(source.role || '').trim(),
        String(source.content || '').trim(),
        String(source.timestamp || '').trim(),
        String(source.response_id || source.responseId || '').trim(),
        String(source.response_instance_id || source.responseInstanceId || '').trim(),
        String(source.saved_image_path || source.savedImagePath || '').trim(),
        String(source.saved_audio_path || source.savedAudioPath || '').trim(),
        String(source.saved_text_path || source.savedTextPath || '').trim(),
        JSON.stringify(artifacts),
        JSON.stringify(outputs),
        JSON.stringify(outputSlots),
        JSON.stringify(outputBranches),
        JSON.stringify(getRequestSnapshotInputArtifacts(requestSnapshot)),
    ]);
}

function getConversationSlotHistoryUpdate(conversationId, payload = {}) {
    const payloadInstanceId = String(payload?.instance_id || '').trim();
    const metadata = normalizeConversationMetadata(
        payload?.conversation_metadata || getConversationMetadata(payloadInstanceId || conversationId),
        payloadInstanceId || conversationId
    );
    const slotHistoryKey = buildConversationSlotHistoryKey(metadata?.workspace, metadata?.slotId);
    const currentIds = normalizeConversationHistoryIds(state.slotHistoryConversationIdsByKey?.[slotHistoryKey]);
    const nextIds = slotHistoryKey
        ? mergeConversationHistoryIds(
            currentIds,
            Array.isArray(payload?.slot_history_ids) ? payload.slot_history_ids : [],
            [payloadInstanceId || conversationId]
        )
        : [];
    return {
        metadata,
        slotHistoryKey,
        currentIds,
        nextIds,
    };
}

function historyPayloadHasSlotHistoryChanges(conversationId, payload = {}) {
    const { currentIds, nextIds, slotHistoryKey } = getConversationSlotHistoryUpdate(conversationId, payload);
    if (!slotHistoryKey) return false;
    if (currentIds.length !== nextIds.length) return true;
    return currentIds.some((value, index) => value !== nextIds[index]);
}

function syncConversationSlotHistoryFromPayload(conversationId, payload = {}) {
    const { metadata, nextIds, slotHistoryKey } = getConversationSlotHistoryUpdate(conversationId, payload);
    if (metadata?.workspace && metadata?.slotId) {
        registerConversationMetadata(
            String(payload?.instance_id || '').trim() || conversationId,
            metadata
        );
    }
    if (!slotHistoryKey) return [];
    return setSlotHistoryConversationIds(metadata.workspace, metadata.slotId, nextIds);
}

function historyPayloadHasNewMessages(conversationId, payload = {}) {
    const localMessages = getPersistedConversationMessages(conversationId);
    const payloadMessages = Array.isArray(payload?.messages) ? payload.messages : [];
    if (!payloadMessages.length) {
        return false;
    }
    if (!localMessages.length) {
        return true;
    }
    const localKeys = new Set(localMessages.map((message) => buildConversationHistoryMessageIdentity(message)));
    return payloadMessages.some((message) => !localKeys.has(buildConversationHistoryMessageIdentity(message)));
}

function historyPayloadHasAnyNewState(conversationId, payload = {}) {
    return historyPayloadHasSlotHistoryChanges(conversationId, payload)
        || historyPayloadHasNewMessages(conversationId, payload);
}

function appendConversationToSlotHistory(conversationId, metadata = null) {
    const normalized = normalizeConversationMetadata(metadata || getConversationMetadata(conversationId), conversationId);
    const key = buildConversationSlotHistoryKey(normalized?.workspace, normalized?.slotId);
    const value = String(conversationId || '').trim();
    if (!key || !value) return;
    const current = Array.isArray(state.slotHistoryConversationIdsByKey[key])
        ? state.slotHistoryConversationIdsByKey[key]
        : [];
    if (current.includes(value)) return;
    state.slotHistoryConversationIdsByKey[key] = [...current, value];
    saveConversationSlots();
    renderConversationHistoryList();
}

function getConversationMetadata(conversationId) {
    const key = String(conversationId || '').trim();
    if (!key) return null;
    const existing = normalizeConversationMetadata(state.conversationMetadataById?.[key], key);
    if (existing) {
        return existing;
    }
    return registerConversationMetadata(key, null);
}

function getConversationTimelineTimestamp(conversationId) {
    const key = String(conversationId || '').trim();
    if (!key) return 0;
    const metadata = getConversationMetadata(key);
    const ledgerTimestamp = String(metadata?.lastMessageAt || metadata?.createdAt || '').trim();
    if (ledgerTimestamp) {
        const parsed = Date.parse(ledgerTimestamp);
        if (!Number.isNaN(parsed)) {
            return parsed;
        }
    }
    const encodedMatch = key.match(/--(\d{8}T\d{6}Z)-/);
    if (encodedMatch) {
        const stamp = encodedMatch[1];
        const isoStamp = `${stamp.slice(0, 4)}-${stamp.slice(4, 6)}-${stamp.slice(6, 8)}T${stamp.slice(9, 11)}:${stamp.slice(11, 13)}:${stamp.slice(13, 15)}Z`;
        const parsed = Date.parse(isoStamp);
        if (!Number.isNaN(parsed)) {
            return parsed;
        }
    }
    const messages = getPersistedConversationMessages(key);
    for (let index = messages.length - 1; index >= 0; index -= 1) {
        const timestamp = String(messages[index]?.timestamp || '').trim();
        if (!timestamp) continue;
        const parsed = Date.parse(timestamp);
        if (!Number.isNaN(parsed)) {
            return parsed;
        }
    }
    return 0;
}

function getConversationSlotArchiveThresholdMs() {
    const parsedMinutes = Number(state?.conversationSlotArchiveMinutes);
    if (Number.isFinite(parsedMinutes) && parsedMinutes > 0) {
        return parsedMinutes * 60 * 1000;
    }
    const parsedHours = Number(state?.conversationSlotArchiveHours);
    if (Number.isFinite(parsedHours) && parsedHours > 0) {
        return parsedHours * 60 * 60 * 1000;
    }
    return 10 * 60 * 1000;
}

function shouldBootstrapFreshConversationSlot(workspace = 'responses', sourceInstanceId = '') {
    const normalizedWorkspace = String(workspace || '').trim() === 'instance' ? 'instance' : 'responses';
    const bootstrap = state.conversationFreshSlotBootstrap && typeof state.conversationFreshSlotBootstrap === 'object'
        ? state.conversationFreshSlotBootstrap
        : {};
    if (normalizedWorkspace === 'responses') {
        return Boolean(bootstrap.responses);
    }
    const sourceKey = String(sourceInstanceId || '').trim();
    const instanceFlags = bootstrap.instances && typeof bootstrap.instances === 'object'
        ? bootstrap.instances
        : {};
    if (!sourceKey) {
        return true;
    }
    if (Object.prototype.hasOwnProperty.call(instanceFlags, sourceKey)) {
        return Boolean(instanceFlags[sourceKey]);
    }
    return true;
}

function markConversationSlotBootstrapped(workspace = 'responses', sourceInstanceId = '') {
    const normalizedWorkspace = String(workspace || '').trim() === 'instance' ? 'instance' : 'responses';
    if (!state.conversationFreshSlotBootstrap || typeof state.conversationFreshSlotBootstrap !== 'object') {
        state.conversationFreshSlotBootstrap = {
            responses: false,
            instances: {},
        };
    }
    if (normalizedWorkspace === 'responses') {
        state.conversationFreshSlotBootstrap.responses = false;
        return;
    }
    const sourceKey = String(sourceInstanceId || '').trim();
    if (!state.conversationFreshSlotBootstrap.instances || typeof state.conversationFreshSlotBootstrap.instances !== 'object') {
        state.conversationFreshSlotBootstrap.instances = {};
    }
    if (sourceKey) {
        state.conversationFreshSlotBootstrap.instances[sourceKey] = false;
    }
}

function conversationShouldAutoArchive(conversationId = '') {
    const key = String(conversationId || '').trim();
    if (!key) return false;
    const persistedCount = getPersistedConversationMessageCount(key);
    const storedCount = getStoredConversationMessageCount(key);
    const effectiveCount = persistedCount > 0 ? persistedCount : (storedCount ?? 0);
    if (effectiveCount <= 0) {
        return false;
    }
    if (hasPendingConversationRequest(key) || conversationHasLoadingMessage(key)) {
        return false;
    }
    const thresholdMs = getConversationSlotArchiveThresholdMs();
    const lastActiveAt = getConversationTimelineTimestamp(key);
    if (!thresholdMs || !lastActiveAt) {
        return false;
    }
    return (Date.now() - lastActiveAt) >= thresholdMs;
}

async function ensureConversationSlotArchiveWindow({
    workspace = 'responses',
    sourceInstanceId = '',
    conversationId = '',
} = {}) {
    const normalizedWorkspace = String(workspace || '').trim() === 'instance' ? 'instance' : 'responses';
    const sourceKey = String(sourceInstanceId || '').trim();
    const currentConversationId = String(
        conversationId
        || (normalizedWorkspace === 'responses'
            ? getResponsesWorkbenchConversationId()
            : getInstanceConversationId(sourceKey))
        || ''
    ).trim();
    if (!currentConversationId || !conversationShouldAutoArchive(currentConversationId)) {
        return currentConversationId;
    }
    return ensureFreshConversationSlot({
        workspace: normalizedWorkspace,
        sourceInstanceId: normalizedWorkspace === 'instance' ? sourceKey : '',
    });
}

function sortConversationIdsByTimeline(ids = [], { descending = false } = {}) {
    return [...(Array.isArray(ids) ? ids : [])]
        .filter((item, index, all) => {
            const value = String(item || '').trim();
            return value && all.findIndex((candidate) => String(candidate || '').trim() === value) === index;
        })
        .sort((left, right) => {
            const leftId = String(left || '').trim();
            const rightId = String(right || '').trim();
            const leftTimestamp = getConversationTimelineTimestamp(leftId);
            const rightTimestamp = getConversationTimelineTimestamp(rightId);
            if (leftTimestamp !== rightTimestamp) {
                return descending ? rightTimestamp - leftTimestamp : leftTimestamp - rightTimestamp;
            }
            return descending
                ? rightId.localeCompare(leftId)
                : leftId.localeCompare(rightId);
        });
}

function getConversationLineageIds(conversationId, { limit = 12 } = {}) {
    const metadata = getConversationMetadata(conversationId);
    if (metadata?.freshRoot) {
        return [];
    }
    const slotKey = getConversationSlotHistoryKey(conversationId);
    const slotIds = Array.isArray(state.slotHistoryConversationIdsByKey?.[slotKey])
        ? state.slotHistoryConversationIdsByKey[slotKey]
        : [];
    const currentId = String(conversationId || '').trim();
    if (slotIds.length > 0 && currentId) {
        const currentIndex = slotIds.indexOf(currentId);
        const previous = (currentIndex >= 0 ? slotIds.slice(0, currentIndex) : slotIds.filter((item) => item !== currentId))
            .filter((item) => String(item || '').trim());
        return sortConversationIdsByTimeline(previous, { descending: true }).slice(0, limit);
    }
    const chain = [];
    const visited = new Set();
    let currentIdCursor = currentId;
    while (currentIdCursor && !visited.has(currentIdCursor) && chain.length < limit) {
        visited.add(currentIdCursor);
        const parentId = String(getConversationMetadata(currentIdCursor)?.parentConversationId || '').trim();
        if (!parentId || visited.has(parentId)) {
            break;
        }
        chain.push(parentId);
        currentIdCursor = parentId;
    }
    return chain;
}

async function ensureConversationLineageLoaded(conversationId, { limit = 12 } = {}) {
    const lineageIds = getConversationLineageIds(conversationId, { limit });
    for (const lineageId of lineageIds) {
        if (!isConversationEligibleForDurableHistory(lineageId)) continue;
        const shouldForce = !Array.isArray(state.conversations?.[lineageId]) || getPersistedConversationMessageCount(lineageId) === 0;
        await fetchChatHistory(lineageId, { force: shouldForce, suppressVisibleRender: true });
    }
    return lineageIds;
}

function isConversationLineageCollapsed(conversationId, totalPreviousSegments = 0) {
    const key = String(conversationId || '').trim();
    if (!key) return false;
    if (Object.prototype.hasOwnProperty.call(state.lineageCollapsedByConversationId, key)) {
        return Boolean(state.lineageCollapsedByConversationId[key]);
    }
    return totalPreviousSegments > 1;
}

function toggleConversationLineageCollapsed(conversationId, activeConversationId = '', totalPreviousSegments = 0) {
    const key = String(conversationId || '').trim();
    if (!key) return;
    state.lineageCollapsedByConversationId[key] = !isConversationLineageCollapsed(key, totalPreviousSegments);
    const rerenderConversationId = String(activeConversationId || getActiveConversationId() || '').trim();
    if (rerenderConversationId && isConversationVisible(rerenderConversationId)) {
        renderConversation(rerenderConversationId);
    }
}

function saveConversationSlots() {
    try {
        const activeWorkspace = state.activeWorkspace === 'instance' ? 'instance' : 'responses';
        const currentInstanceId = String(state.currentInstanceId || '').trim() || null;
        localStorage.setItem(CONVERSATION_SLOT_STORAGE_KEY, JSON.stringify({
            metadataByConversationId: state.conversationMetadataById || {},
            slotHistoryConversationIdsByKey: state.slotHistoryConversationIdsByKey || {},
        }));
        sessionStorage.setItem(CONVERSATION_SLOT_SESSION_STORAGE_KEY, JSON.stringify({
            responses: state.conversationSlots.responses || RESPONSES_WORKBENCH_ID,
            instances: state.conversationSlots.instances || {},
            ui_state: {
                active_workspace: activeWorkspace,
                current_instance_id: currentInstanceId,
            },
        }));
    } catch (error) {
        console.warn('Could not persist conversation slots:', error);
    }
}

function loadConversationSlots() {
    state.conversationSlots.responses = RESPONSES_WORKBENCH_ID;
    state.conversationSlots.instances = {};
    state.conversationMetadataById = {};
    state.slotHistoryConversationIdsByKey = {};
    state.conversationViewOverrides = {};
    state.conversationFreshSlotBootstrap = {
        responses: true,
        instances: {},
    };
    try {
        const sharedRaw = localStorage.getItem(CONVERSATION_SLOT_STORAGE_KEY);
        if (sharedRaw) {
            const parsed = JSON.parse(sharedRaw);
            if (parsed && typeof parsed === 'object') {
                const metadataByConversationId = parsed.metadataByConversationId && typeof parsed.metadataByConversationId === 'object'
                    ? parsed.metadataByConversationId
                    : {};
                Object.entries(metadataByConversationId).forEach(([conversationId, metadata]) => {
                    registerConversationMetadata(conversationId, metadata);
                });
                const persistedSlotHistory = parsed.slotHistoryConversationIdsByKey && typeof parsed.slotHistoryConversationIdsByKey === 'object'
                    ? parsed.slotHistoryConversationIdsByKey
                    : {};
                Object.entries(persistedSlotHistory).forEach(([slotKey, ids]) => {
                    const normalizedKey = String(slotKey || '').trim();
                    if (!normalizedKey) return;
                    state.slotHistoryConversationIdsByKey[normalizedKey] = normalizeConversationHistoryIds(ids);
                });
            }
        }
        const sessionRaw = sessionStorage.getItem(CONVERSATION_SLOT_SESSION_STORAGE_KEY);
        if (sessionRaw) {
            const parsed = JSON.parse(sessionRaw);
            if (parsed && typeof parsed === 'object') {
                const responsesId = String(parsed.responses || '').trim();
                if (responsesId) {
                    state.conversationSlots.responses = responsesId;
                    state.conversationFreshSlotBootstrap.responses = false;
                }
                const uiState = parsed.ui_state && typeof parsed.ui_state === 'object'
                    ? parsed.ui_state
                    : null;
                const restoredWorkspace = String(uiState?.active_workspace || '').trim().toLowerCase();
                state.activeWorkspace = restoredWorkspace === 'instance' ? 'instance' : 'responses';
                state.currentInstanceId = String(uiState?.current_instance_id || '').trim() || null;
                const instanceMap = parsed.instances && typeof parsed.instances === 'object' ? parsed.instances : {};
                Object.entries(instanceMap).forEach(([instanceId, conversationId]) => {
                    const instanceKey = String(instanceId || '').trim();
                    const conversationKey = String(conversationId || '').trim();
                    if (!instanceKey || !conversationKey) return;
                    state.conversationSlots.instances[instanceKey] = conversationKey;
                    state.conversationFreshSlotBootstrap.instances[instanceKey] = false;
                });
            }
        } else {
            state.activeWorkspace = 'responses';
            state.currentInstanceId = null;
        }
    } catch (error) {
        console.warn('Could not load persisted conversation slots:', error);
        state.conversationSlots.responses = RESPONSES_WORKBENCH_ID;
        state.conversationSlots.instances = {};
        state.conversationMetadataById = {};
        state.slotHistoryConversationIdsByKey = {};
        state.conversationViewOverrides = {};
        state.conversationFreshSlotBootstrap = {
            responses: true,
            instances: {},
        };
    }
    registerConversationMetadata(state.conversationSlots.responses || RESPONSES_WORKBENCH_ID, {
        workspace: 'responses',
        slot_id: 'responses-workbench',
    });
    Object.entries(state.conversationSlots.instances || {}).forEach(([instanceId, conversationId]) => {
        registerConversationMetadata(conversationId, {
            workspace: 'instance',
            slot_id: buildInstanceConversationSlotId(instanceId),
            source_instance_id: instanceId,
            label: instanceId,
        });
    });
}

function getConversationSourceInstanceId(conversationId) {
    const metadata = getConversationMetadata(conversationId);
    if (metadata?.workspace === 'responses') {
        return '';
    }
    return String(metadata?.sourceInstanceId || '').trim() || String(conversationId || '').trim();
}

function getConversationInstanceMeta(conversationId) {
    const sourceInstanceId = getConversationSourceInstanceId(conversationId);
    return sourceInstanceId ? getInstanceMeta(sourceInstanceId) : null;
}

function getConversationViewOverride(workspace = '', slotId = '') {
    const key = buildConversationSlotHistoryKey(workspace, slotId);
    if (!key) return '';
    return String(state.conversationViewOverrides?.[key] || '').trim();
}

function setConversationViewOverride(workspace = '', slotId = '', conversationId = '') {
    const key = buildConversationSlotHistoryKey(workspace, slotId);
    const value = String(conversationId || '').trim();
    if (!key || !value) return '';
    if (!state.conversationViewOverrides || typeof state.conversationViewOverrides !== 'object') {
        state.conversationViewOverrides = {};
    }
    state.conversationViewOverrides[key] = value;
    renderConversationHistoryList();
    return value;
}

function clearConversationViewOverride(workspace = '', slotId = '') {
    const key = buildConversationSlotHistoryKey(workspace, slotId);
    if (!key || !state.conversationViewOverrides || typeof state.conversationViewOverrides !== 'object') {
        return;
    }
    delete state.conversationViewOverrides[key];
    renderConversationHistoryList();
}

function getCurrentWorkspaceSlotContext() {
    if (isResponsesWorkbenchActive()) {
        return {
            workspace: 'responses',
            slotId: 'responses-workbench',
            sourceInstanceId: '',
            conversationId: getResponsesWorkbenchConversationId(),
        };
    }
    const instanceKey = String(state.currentInstanceId || '').trim();
    if (!instanceKey) return null;
    return {
        workspace: 'instance',
        slotId: buildInstanceConversationSlotId(instanceKey),
        sourceInstanceId: instanceKey,
        conversationId: getInstanceConversationId(instanceKey),
    };
}

function getCurrentSlotAssignedConversationIds() {
    return new Set(
        [
            state.conversationSlots.responses,
            ...Object.values(state.conversationSlots.instances || {}),
        ]
            .map((value) => String(value || '').trim())
            .filter(Boolean)
    );
}

function pruneStaleConversationHistoryState(validArchiveIds = []) {
    const keepIds = new Set(normalizeConversationHistoryIds(validArchiveIds));
    getCurrentSlotAssignedConversationIds().forEach((conversationId) => {
        const key = String(conversationId || '').trim();
        if (key) keepIds.add(key);
    });
    Object.keys(state.conversations || {}).forEach((conversationId) => {
        const key = String(conversationId || '').trim();
        if (!key) return;
        if (getPersistedConversationMessageCount(key) > 0) {
            keepIds.add(key);
        }
    });

    let changed = false;
    const nextSlotHistoryConversationIdsByKey = {};
    Object.entries(state.slotHistoryConversationIdsByKey || {}).forEach(([slotKey, ids]) => {
        const normalizedKey = String(slotKey || '').trim();
        if (!normalizedKey) return;
        const currentIds = normalizeConversationHistoryIds(ids);
        const nextIds = currentIds.filter((conversationId) => keepIds.has(conversationId));
        if (nextIds.length !== currentIds.length) {
            changed = true;
        }
        if (nextIds.length) {
            nextSlotHistoryConversationIdsByKey[normalizedKey] = nextIds;
        }
    });

    const referencedConversationIds = new Set(keepIds);
    Object.values(nextSlotHistoryConversationIdsByKey).forEach((ids) => {
        normalizeConversationHistoryIds(ids).forEach((conversationId) => {
            referencedConversationIds.add(conversationId);
        });
    });

    const nextMetadataByConversationId = {};
    Object.entries(state.conversationMetadataById || {}).forEach(([conversationId, metadata]) => {
        const key = String(conversationId || '').trim();
        if (!key) return;
        if (referencedConversationIds.has(key)) {
            nextMetadataByConversationId[key] = metadata;
            return;
        }
        if (getPersistedConversationMessageCount(key) > 0) {
            nextMetadataByConversationId[key] = metadata;
            return;
        }
        changed = true;
    });

    Object.keys(state.chatHistoryLoaded || {}).forEach((conversationId) => {
        const key = String(conversationId || '').trim();
        if (!key) return;
        if (referencedConversationIds.has(key) || getPersistedConversationMessageCount(key) > 0) {
            return;
        }
        delete state.chatHistoryLoaded[key];
        changed = true;
    });
    Object.keys(state.conversationViewOverrides || {}).forEach((slotKey) => {
        const conversationId = String(state.conversationViewOverrides?.[slotKey] || '').trim();
        if (!conversationId || referencedConversationIds.has(conversationId) || getPersistedConversationMessageCount(conversationId) > 0) {
            return;
        }
        delete state.conversationViewOverrides[slotKey];
        changed = true;
    });

    if (!changed) return false;
    state.slotHistoryConversationIdsByKey = nextSlotHistoryConversationIdsByKey;
    state.conversationMetadataById = nextMetadataByConversationId;
    saveConversationSlots();
    return true;
}

function trimConversationPreviewText(value = '') {
    const normalized = String(value || '')
        .replace(/<[^>]+>/g, ' ')
        .replace(/\[[^\]]+:\s*[^\]]+\]/g, ' ')
        .replace(/\s+/g, ' ')
        .trim();
    if (!normalized) return '';
    return normalized.length > 68 ? `${normalized.slice(0, 65)}...` : normalized;
}

function getStoredConversationPreviewText(conversationId = '') {
    const metadata = getConversationMetadata(conversationId);
    return trimConversationPreviewText(metadata?.displayTitle || metadata?.previewText || '');
}

function getStoredConversationMessageCount(conversationId = '') {
    return normalizeConversationMessageCount(getConversationMetadata(conversationId)?.messageCount);
}

function conversationLedgerMetadataNeedsBackfill(conversationId = '', sourceMetadata = null) {
    const derived = buildConversationLedgerMetadata(conversationId);
    const normalized = normalizeConversationMetadata(sourceMetadata, conversationId);
    const currentDisplayTitle = trimConversationPreviewText(normalized?.displayTitle || '');
    const currentPreviewText = trimConversationPreviewText(normalized?.previewText || '');
    const currentMessageCount = normalizeConversationMessageCount(normalized?.messageCount);
    const currentLastMessageAt = String(normalized?.lastMessageAt || '').trim();
    const nextDisplayTitle = trimConversationPreviewText(derived.display_title || '');
    const nextPreviewText = trimConversationPreviewText(derived.preview_text || '');
    const nextMessageCount = normalizeConversationMessageCount(derived.message_count);
    const nextLastMessageAt = String(derived.last_message_at || '').trim();
    return currentDisplayTitle !== nextDisplayTitle
        || currentPreviewText !== nextPreviewText
        || currentMessageCount !== nextMessageCount
        || currentLastMessageAt !== nextLastMessageAt;
}

function getConversationPreviewText(conversationId = '') {
    const storedPreview = getStoredConversationPreviewText(conversationId);
    if (storedPreview) {
        return storedPreview;
    }
    if (!state.chatHistoryLoaded[conversationId]) {
        return 'Preview hydrating...';
    }
    const messages = getPersistedConversationMessages(conversationId);
    const preferredMessage = messages.find((message) => String(message?.role || '').trim().toLowerCase() === 'user' && trimConversationPreviewText(message?.content))
        || messages.find((message) => trimConversationPreviewText(message?.content))
        || null;
    const preview = trimConversationPreviewText(preferredMessage?.content || '');
    if (preview) {
        return preview;
    }
    return isResponsesWorkbenchConversationId(conversationId)
        ? 'Saved Ollmo chat'
        : 'Saved chat';
}

function formatConversationHistoryTimestamp(conversationId = '') {
    const timestamp = getConversationTimelineTimestamp(conversationId);
    if (!timestamp) return 'Saved chat';
    const date = new Date(timestamp);
    if (Number.isNaN(date.getTime())) return 'Saved chat';
    return date.toLocaleString([], {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
    });
}

async function fetchConversationHistoryIndex({ force = false } = {}) {
    if (state.conversationHistoryIndexRequestInFlight) {
        return state.conversationHistoryArchiveIds || [];
    }
    if (state.conversationHistoryIndexLoaded && !force) {
        return state.conversationHistoryArchiveIds || [];
    }
    state.conversationHistoryIndexRequestInFlight = true;
    try {
        const response = await axios.get(`${state.flaskServerUrl}/api/chat_history/index`);
        const items = Array.isArray(response.data?.items) ? response.data.items : [];
        const archiveIds = [];
        items.forEach((item) => {
            const conversationId = String(item?.instance_id || '').trim();
            if (!conversationId) return;
            archiveIds.push(conversationId);
            registerConversationMetadata(conversationId, {
                ...(item?.conversation_metadata || {}),
                model: String(item?.model || '').trim() || null,
                backend: String(item?.backend || '').trim() || null,
                capability: String(item?.capability || '').trim() || null,
            });
        });
        state.conversationHistoryArchiveIds = normalizeConversationHistoryIds(archiveIds);
        pruneStaleConversationHistoryState(state.conversationHistoryArchiveIds);
        state.conversationHistoryIndexLoaded = true;
        renderConversationHistoryList();
    } catch (error) {
        console.warn('Could not load conversation history index:', error);
    } finally {
        state.conversationHistoryIndexRequestInFlight = false;
        if (typeof renderConversationHistoryPanel === 'function') {
            renderConversationHistoryPanel();
        }
    }
    return state.conversationHistoryArchiveIds || [];
}

function getConversationHistoryIds({ limit = 40 } = {}) {
    const archiveIds = normalizeConversationHistoryIds(state.conversationHistoryArchiveIds || []);
    const archiveIdSet = new Set(archiveIds);
    const rawIds = mergeConversationHistoryIds(
        archiveIds,
        ...Object.values(state.slotHistoryConversationIdsByKey || {}),
        [state.conversationSlots.responses],
        Object.values(state.conversationSlots.instances || {})
    );
    const currentSlotIds = getCurrentSlotAssignedConversationIds();
    return sortConversationIdsByTimeline(rawIds, { descending: true })
        .filter((conversationId) => {
            const key = String(conversationId || '').trim();
            if (!key || !isConversationEligibleForDurableHistory(key)) return false;
            const persistedCount = getPersistedConversationMessageCount(key);
            const storedCount = getStoredConversationMessageCount(key);
            const knownCount = persistedCount > 0 ? persistedCount : storedCount;
            const effectiveCount = persistedCount > 0 ? persistedCount : (storedCount ?? 0);
            if (currentSlotIds.has(key) && effectiveCount === 0) {
                return false;
            }
            if (knownCount === 0 && !archiveIdSet.has(key)) {
                return false;
            }
            if (knownCount === null) {
                return archiveIdSet.has(key) || (!state.conversationHistoryIndexLoaded && !state.chatHistoryLoaded[key]);
            }
            return knownCount > 0;
        })
        .slice(0, limit);
}

function conversationNeedsHistoryListHydration(conversationId = '') {
    const key = String(conversationId || '').trim();
    if (!key) return false;
    if (state.chatHistoryLoaded?.[key]) return false;
    if (state.chatHistoryRequestsInFlight?.[key]) return false;
    const metadata = getConversationMetadata(key);
    const hasTitle = Boolean(trimConversationPreviewText(metadata?.displayTitle || metadata?.previewText || ''));
    const hasCount = normalizeConversationMessageCount(metadata?.messageCount) !== null;
    const hasTimestamp = Boolean(String(metadata?.lastMessageAt || metadata?.createdAt || '').trim());
    return !(hasTitle && hasCount && hasTimestamp);
}

async function hydrateConversationHistoryListEntries({ batchSize = 12 } = {}) {
    if (state.conversationHistoryHydrationInFlight) return;
    state.conversationHistoryHydrationInFlight = true;
    try {
        while (true) {
            const pendingIds = getConversationHistoryIds({ limit: Number.MAX_SAFE_INTEGER })
                .filter((conversationId) => conversationNeedsHistoryListHydration(conversationId))
                .slice(0, batchSize);
            if (!pendingIds.length) break;
            for (const conversationId of pendingIds) {
                await fetchChatHistory(conversationId, { force: true, suppressVisibleRender: true });
            }
        }
    } finally {
        state.conversationHistoryHydrationInFlight = false;
        renderConversationHistoryList({ skipHydration: true });
    }
}

function renderConversationHistoryList({ skipHydration = false } = {}) {
    if (!elements.conversationHistoryList) return;
    const conversationIds = getConversationHistoryIds();
    const activeConversationId = String(getActiveConversationId() || '').trim();
    if (!conversationIds.length) {
        elements.conversationHistoryList.innerHTML = '<div class="conversation-history-empty">Saved chats will appear here.</div>';
        if (typeof renderConversationHistoryPanel === 'function') {
            renderConversationHistoryPanel();
        }
        return;
    }
    elements.conversationHistoryList.innerHTML = '';
    conversationIds.forEach((conversationId) => {
        const item = document.createElement('button');
        item.type = 'button';
        item.className = 'conversation-history-item';
        if (conversationId === activeConversationId) {
            item.classList.add('is-active');
        }
        const sourceLabel = getConversationDisplayLabel(conversationId);
        const showSourceLabel = Boolean(sourceLabel);
        const count = getPersistedConversationMessageCount(conversationId);
        const storedCount = getStoredConversationMessageCount(conversationId);
        const effectiveCount = count > 0 ? count : (storedCount ?? 0);
        const countLabel = !state.chatHistoryLoaded[conversationId] && effectiveCount === 0
            ? 'hydrating'
            : (effectiveCount > 0 ? `${effectiveCount} msg${effectiveCount === 1 ? '' : 's'}` : 'saved');
        const metaLabel = [
            showSourceLabel ? sourceLabel : '',
            formatConversationHistoryTimestamp(conversationId),
        ].filter(Boolean).join(' • ');
        item.innerHTML = `
            <span class="conversation-history-item__title">${escapeHtml(getConversationPreviewText(conversationId))}</span>
            <span class="conversation-history-item__meta-row">
                <span class="conversation-history-item__meta">${escapeHtml(metaLabel)}</span>
                <span class="conversation-history-item__aux">${escapeHtml(countLabel)}</span>
            </span>
        `;
        item.addEventListener('click', () => {
            void openConversationFromHistory(conversationId);
        });
        elements.conversationHistoryList.appendChild(item);
    });
    if (!skipHydration) {
        queueMicrotask(() => {
            void hydrateConversationHistoryListEntries();
        });
    }
    if (typeof renderConversationHistoryPanel === 'function') {
        renderConversationHistoryPanel();
    }
}

async function ensureFreshConversationSlot({
    workspace = 'responses',
    sourceInstanceId = '',
} = {}) {
    const normalizedWorkspace = String(workspace || '').trim() === 'instance' ? 'instance' : 'responses';
    const sourceKey = String(sourceInstanceId || '').trim();
    const currentConversationId = normalizedWorkspace === 'responses'
        ? getResponsesWorkbenchConversationId()
        : getInstanceConversationId(sourceKey);
    if (!currentConversationId) return '';
    if (!state.chatHistoryLoaded[currentConversationId] && getPersistedConversationMessageCount(currentConversationId) === 0) {
        await fetchChatHistory(currentConversationId, { suppressVisibleRender: true });
    }
    if (getPersistedConversationMessageCount(currentConversationId) === 0) {
        return currentConversationId;
    }
    if (hasPendingConversationRequest(currentConversationId) || conversationHasLoadingMessage(currentConversationId)) {
        return currentConversationId;
    }
    const rotated = await rotateConversationToFreshSuccessor({
        conversationId: currentConversationId,
        workspace: normalizedWorkspace,
        sourceInstanceId: normalizedWorkspace === 'instance' ? sourceKey : '',
        label: normalizedWorkspace === 'responses' ? 'responses-workbench' : sourceKey,
        assignSlot: true,
    });
    return rotated?.conversationId || currentConversationId;
}

async function returnToCurrentDraftConversation({
    focusInput = true,
    statusText = '',
} = {}) {
    const slotContext = getCurrentWorkspaceSlotContext();
    if (!slotContext) return '';
    clearConversationViewOverride(slotContext.workspace, slotContext.slotId);
    const draftConversationId = String(await ensureConversationSlotArchiveWindow({
        workspace: slotContext.workspace,
        sourceInstanceId: slotContext.sourceInstanceId,
        conversationId: slotContext.conversationId,
    }) || '').trim();
    if (!draftConversationId) return '';
    if (!state.chatHistoryLoaded[draftConversationId]) {
        await fetchChatHistory(draftConversationId, { suppressVisibleRender: true });
    }
    renderConversation(draftConversationId || slotContext.conversationId);
    updatePromptPlaceholder();
    updateSendButtonState();
    if (statusText) {
        updateGlobalModelStatus(statusText);
    }
    if (focusInput) {
        elements.userInput?.focus();
    }
    return draftConversationId || slotContext.conversationId;
}

async function openConversationFromHistory(conversationId = '') {
    const key = String(conversationId || '').trim();
    if (!key) return;
    const metadata = getConversationMetadata(key);
    if (!metadata?.workspace || !metadata?.slotId) return;
    if (metadata.workspace === 'responses') {
        await switchToResponsesWorkbench({
            focusInput: false,
            historyConversationId: key,
        });
        return;
    }
    const sourceInstanceId = String(metadata.sourceInstanceId || '').trim();
    const instance = sourceInstanceId ? getInstanceMeta(sourceInstanceId) : null;
    if (!instance || !isUserFacingInstance(instance)) {
        updateGlobalModelStatus('That saved chat belongs to a model that is not currently running.');
        return;
    }
    await switchToInstance(sourceInstanceId, {
        focusInput: false,
        historyConversationId: key,
    });
}

function getInstanceConversationId(instanceId) {
    const instanceKey = String(instanceId || '').trim();
    if (!instanceKey) return '';
    const assigned = String(state.conversationSlots.instances?.[instanceKey] || '').trim();
    if (assigned) {
        registerConversationMetadata(assigned, {
            workspace: 'instance',
            slot_id: buildInstanceConversationSlotId(instanceKey),
            source_instance_id: instanceKey,
            label: instanceKey,
        });
        return assigned;
    }
    registerConversationMetadata(instanceKey, {
        workspace: 'instance',
        slot_id: buildInstanceConversationSlotId(instanceKey),
        source_instance_id: instanceKey,
        label: instanceKey,
    });
    return instanceKey;
}

function getArenaConversationId(instanceId) {
    const instanceKey = String(instanceId || '').trim();
    if (!instanceKey) return '';
    return String(state.arena?.conversationIds?.[instanceKey] || '').trim();
}

function setArenaConversationId(instanceId, conversationId, metadata = null) {
    const instanceKey = String(instanceId || '').trim();
    if (!instanceKey) return '';
    const nextId = String(conversationId || '').trim();
    if (!nextId) return '';
    if (!state.arena.conversationIds || typeof state.arena.conversationIds !== 'object') {
        state.arena.conversationIds = {};
    }
    state.arena.conversationIds[instanceKey] = nextId;
    const mergedMetadata = registerConversationMetadata(nextId, {
        workspace: 'instance',
        slot_id: buildInstanceConversationSlotId(instanceKey),
        source_instance_id: instanceKey,
        label: instanceKey,
        ...(metadata && typeof metadata === 'object' ? metadata : {}),
    });
    appendConversationToSlotHistory(nextId, mergedMetadata);
    return nextId;
}

function setResponsesWorkbenchConversationId(conversationId, metadata = null) {
    const nextId = String(conversationId || '').trim() || RESPONSES_WORKBENCH_ID;
    state.conversationSlots.responses = nextId;
    const mergedMetadata = registerConversationMetadata(nextId, {
        workspace: 'responses',
        slot_id: 'responses-workbench',
        ...(metadata && typeof metadata === 'object' ? metadata : {}),
    });
    appendConversationToSlotHistory(nextId, mergedMetadata);
    saveConversationSlots();
    renderConversationHistoryList();
    return nextId;
}

async function recoverResponsesWorkbenchConversationSlot({
    force = false,
    expectedConversationId = '',
    preservePopulatedLocalState = false,
    suppressVisibleRender = false,
} = {}) {
    const currentId = getResponsesWorkbenchConversationId();
    const currentLoaded = Boolean(state.chatHistoryLoaded[currentId]);
    const currentHasMessages = getPersistedConversationMessageCount(currentId) > 0;
    const slotHistoryKey = buildConversationSlotHistoryKey('responses', 'responses-workbench');
    const hasSlotHistory = Array.isArray(state.slotHistoryConversationIdsByKey?.[slotHistoryKey])
        && state.slotHistoryConversationIdsByKey[slotHistoryKey].length > 0;
    if (!force && currentLoaded && currentHasMessages && hasSlotHistory) {
        return ensureConversationSlotArchiveWindow({
            workspace: 'responses',
            conversationId: currentId,
        });
    }
    try {
        const needsFreshWindowSession = shouldBootstrapFreshConversationSlot('responses');
        const response = await axios.get(`${state.flaskServerUrl}/api/chat_history/slot`, {
            params: {
                workspace: 'responses',
                slot_id: 'responses-workbench',
                fallback_instance_id: currentId || RESPONSES_WORKBENCH_ID,
            },
        });
        const payload = response.data || {};
        const resolvedId = String(payload.instance_id || '').trim() || currentId || RESPONSES_WORKBENCH_ID;
        const activeConversationId = String(getResponsesWorkbenchConversationId() || expectedConversationId || currentId).trim() || currentId;
        const preserveActiveConversation = !force
            && isResponsesWorkbenchActive()
            && (preservePopulatedLocalState || getPersistedConversationMessageCount(activeConversationId) > 0)
            && getPersistedConversationMessageCount(activeConversationId) > 0
            && !historyPayloadHasAnyNewState(activeConversationId, payload);
        if (preserveActiveConversation) {
            syncConversationSlotHistoryFromPayload(activeConversationId, payload);
            return ensureConversationSlotArchiveWindow({
                workspace: 'responses',
                conversationId: activeConversationId,
            });
        }
        syncConversationSlotHistoryFromPayload(resolvedId, payload);
        hydrateConversationFromHistoryPayload(resolvedId, payload);
        setResponsesWorkbenchConversationId(resolvedId, payload.conversation_metadata || {
            workspace: 'responses',
            slot_id: 'responses-workbench',
            label: 'responses-workbench',
        });
        let activeResolvedId = String(await ensureConversationSlotArchiveWindow({
            workspace: 'responses',
            conversationId: resolvedId,
        }) || resolvedId).trim() || resolvedId;
        if (needsFreshWindowSession) {
            activeResolvedId = String(await ensureFreshConversationSlot({
                workspace: 'responses',
            }) || activeResolvedId).trim() || activeResolvedId;
        }
        markConversationSlotBootstrapped('responses');
        if (!state.conversations[activeResolvedId]) {
            state.conversations[activeResolvedId] = [];
        }
        if (!suppressVisibleRender && isResponsesWorkbenchActive()) {
            renderConversation(activeResolvedId);
            updatePromptPlaceholder();
            updateSendButtonState();
        }
        renderConversationHistoryList();
        return activeResolvedId;
    } catch (error) {
        console.warn('Unable to recover responses workbench slot:', error?.message || error);
        if (!state.chatHistoryLoaded[currentId]) {
            await fetchChatHistory(currentId, { suppressVisibleRender });
        }
        return currentId;
    }
}

async function recoverInstanceConversationSlot(instanceId, {
    force = false,
    expectedConversationId = '',
    preservePopulatedLocalState = false,
    suppressVisibleRender = false,
} = {}) {
    const instanceKey = String(instanceId || '').trim();
    if (!instanceKey) return '';
    const slotId = buildInstanceConversationSlotId(instanceKey);
    const currentId = getInstanceConversationId(instanceKey);
    const currentLoaded = Boolean(state.chatHistoryLoaded[currentId]);
    const currentHasMessages = getPersistedConversationMessageCount(currentId) > 0;
    const slotHistoryKey = buildConversationSlotHistoryKey('instance', slotId);
    const hasSlotHistory = Array.isArray(state.slotHistoryConversationIdsByKey?.[slotHistoryKey])
        && state.slotHistoryConversationIdsByKey[slotHistoryKey].length > 0;
    if (!force && currentLoaded && currentHasMessages && hasSlotHistory) {
        return ensureConversationSlotArchiveWindow({
            workspace: 'instance',
            sourceInstanceId: instanceKey,
            conversationId: currentId,
        });
    }
    try {
        const needsFreshWindowSession = shouldBootstrapFreshConversationSlot('instance', instanceKey);
        const response = await axios.get(`${state.flaskServerUrl}/api/chat_history/slot`, {
            params: {
                workspace: 'instance',
                slot_id: slotId,
                fallback_instance_id: currentId || instanceKey,
            },
        });
        const payload = response.data || {};
        const resolvedId = String(payload.instance_id || '').trim() || currentId || instanceKey;
        const activeConversationId = String(getInstanceConversationId(instanceKey) || expectedConversationId || currentId).trim() || currentId;
        const preserveActiveConversation = !force
            && state.activeWorkspace === 'instance'
            && state.currentInstanceId === instanceKey
            && (preservePopulatedLocalState || getPersistedConversationMessageCount(activeConversationId) > 0)
            && getPersistedConversationMessageCount(activeConversationId) > 0
            && !historyPayloadHasAnyNewState(activeConversationId, payload);
        if (preserveActiveConversation) {
            syncConversationSlotHistoryFromPayload(activeConversationId, payload);
            return ensureConversationSlotArchiveWindow({
                workspace: 'instance',
                sourceInstanceId: instanceKey,
                conversationId: activeConversationId,
            });
        }
        syncConversationSlotHistoryFromPayload(resolvedId, payload);
        hydrateConversationFromHistoryPayload(resolvedId, payload);
        setInstanceConversationId(instanceKey, resolvedId, payload.conversation_metadata || {
            workspace: 'instance',
            slot_id: slotId,
            source_instance_id: instanceKey,
            label: instanceKey,
        });
        let activeResolvedId = String(await ensureConversationSlotArchiveWindow({
            workspace: 'instance',
            sourceInstanceId: instanceKey,
            conversationId: resolvedId,
        }) || resolvedId).trim() || resolvedId;
        if (needsFreshWindowSession) {
            activeResolvedId = String(await ensureFreshConversationSlot({
                workspace: 'instance',
                sourceInstanceId: instanceKey,
            }) || activeResolvedId).trim() || activeResolvedId;
        }
        markConversationSlotBootstrapped('instance', instanceKey);
        if (!state.conversations[activeResolvedId]) {
            state.conversations[activeResolvedId] = [];
        }
        if (!suppressVisibleRender && state.activeWorkspace === 'instance' && state.currentInstanceId === instanceKey) {
            renderConversation(activeResolvedId);
            updatePromptPlaceholder();
            updateSendButtonState();
        }
        renderConversationHistoryList();
        return activeResolvedId;
    } catch (error) {
        console.warn(`Unable to recover instance slot for ${instanceKey}:`, error?.message || error);
        if (!state.chatHistoryLoaded[currentId]) {
            await fetchChatHistory(currentId, { suppressVisibleRender });
        }
        return currentId;
    }
}

function setInstanceConversationId(instanceId, conversationId, metadata = null) {
    const instanceKey = String(instanceId || '').trim();
    if (!instanceKey) return '';
    const nextId = String(conversationId || '').trim() || instanceKey;
    state.conversationSlots.instances[instanceKey] = nextId;
    const mergedMetadata = registerConversationMetadata(nextId, {
        workspace: 'instance',
        slot_id: buildInstanceConversationSlotId(instanceKey),
        source_instance_id: instanceKey,
        label: instanceKey,
        ...(metadata && typeof metadata === 'object' ? metadata : {}),
    });
    appendConversationToSlotHistory(nextId, mergedMetadata);
    saveConversationSlots();
    renderConversationHistoryList();
    return nextId;
}

function getConversationDisplayLabel(conversationId) {
    if (isResponsesWorkbenchConversationId(conversationId)) {
        return 'Ollmo';
    }
    const metadata = getConversationMetadata(conversationId);
    const instance = getConversationInstanceMeta(conversationId);
    const sourceInstanceId = getConversationSourceInstanceId(conversationId);
    if (isExternalConversationTarget(instance)) {
        return String(instance.label || 'ChatGPT').replace(/\s*\(automatic\)\s*$/i, '');
    }
    return formatModelDisplayName(instance?.model || metadata?.model || sourceInstanceId || conversationId);
}

function isResponsesWorkbenchActive() {
    return state.activeWorkspace === 'responses';
}

function isResponsesWorkbenchConversationId(instanceId) {
    return getConversationMetadata(instanceId)?.workspace === 'responses';
}

function getResponsesWorkbenchConversationId() {
    return String(state.conversationSlots.responses || RESPONSES_WORKBENCH_ID).trim() || RESPONSES_WORKBENCH_ID;
}

function isConversationEligibleForDurableHistory(instanceId) {
    if (!instanceId) return false;
    if (isResponsesWorkbenchConversationId(instanceId)) return true;
    const sourceInstanceId = getConversationSourceInstanceId(instanceId);
    return Boolean(sourceInstanceId);
}

function getConversationHistoryMetadata(instanceId) {
    if (!instanceId) return null;
    const conversationMetadata = serializeConversationMetadata(getConversationMetadata(instanceId));
    if (isResponsesWorkbenchConversationId(instanceId)) {
        return {
            model: null,
            backend: null,
            capability: null,
            conversationMetadata,
        };
    }
    const instance = getConversationInstanceMeta(instanceId);
    return {
        model: instance?.model || null,
        backend: instance?.backend || null,
        capability: instance?.capability || null,
        conversationMetadata,
    };
}

function isResponsesWorkbenchAutoTargetId(value) {
    return String(value || '') === RESPONSES_GHOST_AUTO_ID;
}

function isResponsesWorkbenchAutoTarget() {
    return isResponsesWorkbenchAutoTargetId(state.responsesWorkbench.targetInstanceId);
}

function getResponsesWorkbenchTargetInstance() {
    if (isResponsesWorkbenchAutoTarget()) return null;
    return getInstanceMeta(state.responsesWorkbench.targetInstanceId || '');
}

function getGhostResolvedTargetInstance() {
    const resolved = state.responsesWorkbench.ghostResolvedTarget;
    if (!resolved || typeof resolved !== 'object') return null;
    const instanceId = String(resolved.instance_id || '').trim();
    if (!instanceId) return null;
    return getInstanceMeta(instanceId) || resolved;
}

function getRequestExecutionInstance(instance) {
    if (instance?.ghostAuto) {
        return getGhostResolvedTargetInstance() || instance;
    }
    return instance;
}

function clearGhostResolvedTarget() {
    state.responsesWorkbench.ghostResolvedTarget = null;
    state.responsesWorkbench.ghostResolvedRoute = null;
    state.responsesWorkbench.ghostResolvedRuntime = null;
}

function resetResponsesWorkbenchAutoRoute() {
    const previousOwner = getGhostResolvedTargetInstance();
    if (previousOwner) {
        persistSettingsForCurrentInstance();
    }
    clearGhostResolvedTarget();
    if (isResponsesWorkbenchActive() && isResponsesWorkbenchAutoTarget()) {
        loadSettingsForInstance(null);
        refreshTtsSettingOptions(null);
        updateSessionControlMode();
        updateActiveModelToolbar();
        renderResponsesWorkbenchTargetOptions();
        updatePromptPlaceholder();
        updateSendButtonState();
    }
}

function syncGhostResolvedTargetWithRunningInstances() {
    const resolved = state.responsesWorkbench.ghostResolvedTarget;
    if (!resolved || typeof resolved !== 'object') return;
    const live = getInstanceMeta(resolved.instance_id || '');
    if (!live) {
        clearGhostResolvedTarget();
        return;
    }
    state.responsesWorkbench.ghostResolvedTarget = live;
}

function formatResponsesAutoLabel() {
    const resolved = getGhostResolvedTargetInstance();
    if (!resolved) return 'Ollmo';
    const modelLabel = isExternalConversationTarget(resolved)
        ? 'ChatGPT'
        : formatModelDisplayName(resolved.model || resolved.instance_id);
    const backendLabel = isExternalConversationTarget(resolved)
        ? 'external provider'
        : formatBackendLabel(resolved.backend || 'runtime');
    return `Ollmo -> ${modelLabel} (${backendLabel})`;
}

function buildGhostHelperStatusText() {
    const runtime = state.responsesWorkbench.ghostResolvedRuntime;
    const helper = runtime && typeof runtime === 'object' ? runtime.embedding_helper : null;
    if (!helper || typeof helper !== 'object') return '';
    if (!helper.available) {
        const reason = String(helper.reason || '').trim();
        if (!reason) return 'Embedding helper unavailable.';
        return `Embedding helper unavailable (${reason.replace(/_/g, ' ')}).`;
    }
    const modelLabel = helper.model
        ? formatModelDisplayName(helper.model)
        : String(helper.instance_id || '').trim();
    const transportLabel = String(helper.transport || '').trim();
    const suffix = helper.attached
        ? transportLabel ? ` via ${transportLabel}` : ''
        : helper.reason ? ` (${String(helper.reason).replace(/_/g, ' ')})` : '';
    return `Embedding helper: ${modelLabel || 'available'}${suffix}.`;
}

function buildResponsesAutoStatusText() {
    const resolved = getGhostResolvedTargetInstance();
    const route = state.responsesWorkbench.ghostResolvedRoute;
    const helperText = buildGhostHelperStatusText();
    if (!resolved) {
        return helperText
            ? `Ollmo is ready. ${helperText}`
            : 'Ollmo is ready.';
    }
    const modelLabel = isExternalConversationTarget(resolved)
        ? 'ChatGPT'
        : formatModelDisplayName(resolved.model || resolved.instance_id);
    const backendLabel = isExternalConversationTarget(resolved)
        ? 'external provider'
        : formatBackendLabel(resolved.backend || 'runtime');
    const routeSource = String(route?.source || '').trim().toLowerCase();
    const routeLabel = routeSource === 'router'
        ? 'semantic router'
        : routeSource === 'heuristic'
            ? 'heuristic fallback'
            : '';
    const reason = String(route?.reason || '').trim();
    const contextMode = String(route?.context_mode || '').trim().toLowerCase();
    const parts = [
        `Ollmo is ready. Current route: ${modelLabel} (${backendLabel})`,
        routeLabel ? `via ${routeLabel}.` : '.',
        helperText,
        contextMode
            ? `Context: ${contextMode === 'compressed_history'
                ? 'compressed history'
                : contextMode === 'bounded_file_context'
                    ? 'file context'
                    : contextMode === 'raw_history'
                        ? 'raw history'
                        : contextMode.replace(/_/g, ' ')
            }.`
            : '',
        reason ? `Reason: ${reason}` : '',
    ].filter(Boolean);
    return parts.join(' ').replace(/\s+\./g, '.');
}

function buildGhostAutoTargetInstance() {
    return {
        instance_id: RESPONSES_GHOST_AUTO_ID,
        model: 'Ollmo',
        backend: 'ghost',
        ghostAuto: true,
    };
}

function getCurrentSettingsOwnerInstance() {
    if (isResponsesWorkbenchActive()) {
        if (isResponsesWorkbenchAutoTarget()) {
            return getGhostResolvedTargetInstance();
        }
        return getResponsesWorkbenchTargetInstance();
    }
    return getCurrentInstanceMeta();
}

function getActivePromptTargetInstance() {
    return isResponsesWorkbenchActive() ? getResponsesWorkbenchTargetInstance() : getCurrentInstanceMeta();
}

function getActiveConversationId() {
    const slotContext = getCurrentWorkspaceSlotContext();
    if (!slotContext) return '';
    return getConversationViewOverride(slotContext.workspace, slotContext.slotId) || slotContext.conversationId;
}

function getActiveConversationLabel() {
    const activeConversationId = String(getActiveConversationId() || '').trim();
    if (!activeConversationId) {
        return isResponsesWorkbenchActive() ? 'responses-workbench' : (state.currentInstanceId || 'conversation');
    }
    return getConversationDisplayLabel(activeConversationId);
}

function isConversationVisible(instanceId) {
    const candidateId = String(instanceId || '').trim();
    if (state.arena.enabled) return false;
    if (!candidateId) return false;
    const activeConversationId = String(getActiveConversationId() || '').trim();
    return Boolean(activeConversationId) && candidateId === activeConversationId;
}

function scrollConversationToBottom(instanceId = getActiveConversationId()) {
    if (!instanceId || !elements.chatArea || !isConversationVisible(instanceId) || state.arena.enabled) {
        return;
    }
    elements.chatArea.scrollTop = elements.chatArea.scrollHeight;
}

function hasActiveConversationBottomAnchor(instanceId) {
    return Number(state.conversationBottomAnchors?.[instanceId] || 0) > Date.now();
}

function clearConversationBottomAnchor(instanceId) {
    if (!instanceId) return;
    const timerId = state.conversationBottomAnchorTimers?.[instanceId];
    if (timerId) {
        clearInterval(timerId);
    }
    delete state.conversationBottomAnchorTimers[instanceId];
    delete state.conversationBottomAnchors[instanceId];
}

function maintainConversationBottomAnchor(instanceId) {
    if (!hasActiveConversationBottomAnchor(instanceId)) {
        clearConversationBottomAnchor(instanceId);
        return;
    }
    requestAnimationFrame(() => scrollConversationToBottom(instanceId));
}

function activateConversationBottomAnchor(instanceId, durationMs = 1800) {
    if (!instanceId) return;
    const nextDeadline = Date.now() + Math.max(250, Number(durationMs) || 0);
    state.conversationBottomAnchors[instanceId] = Math.max(
        Number(state.conversationBottomAnchors?.[instanceId] || 0),
        nextDeadline
    );
    if (state.conversationBottomAnchorTimers?.[instanceId]) {
        maintainConversationBottomAnchor(instanceId);
        return;
    }
    maintainConversationBottomAnchor(instanceId);
    state.conversationBottomAnchorTimers[instanceId] = setInterval(() => {
        maintainConversationBottomAnchor(instanceId);
    }, 120);
}

function bindConversationBottomAnchorToMedia(mediaElement, conversationId) {
    if (!mediaElement || !conversationId) return;
    const maintain = () => {
        if (hasActiveConversationBottomAnchor(conversationId)) {
            maintainConversationBottomAnchor(conversationId);
        }
    };
    mediaElement.addEventListener('load', maintain, { once: true });
    mediaElement.addEventListener('loadedmetadata', maintain, { once: true });
    mediaElement.addEventListener('error', maintain, { once: true });
}

function conversationHasLoadingMessage(instanceId) {
    const conversation = Array.isArray(state.conversations?.[instanceId]) ? state.conversations[instanceId] : [];
    return conversation.some((message) => Boolean(message?.isLoading));
}

function ensureResponsesWorkbenchTarget() {
    const options = getDirectConversationTargets();
    if (
        !isResponsesWorkbenchAutoTargetId(state.responsesWorkbench.targetInstanceId)
        && !options.some(inst => inst.instance_id === state.responsesWorkbench.targetInstanceId)
    ) {
        state.responsesWorkbench.targetInstanceId = RESPONSES_GHOST_AUTO_ID;
    }
    return getResponsesWorkbenchTargetInstance();
}

function renderResponsesWorkbenchTargetOptions() {
    if (!elements.responsesTargetSelect) return;
    const options = getDirectConversationTargets();
    ensureResponsesWorkbenchTarget();
    elements.responsesTargetBar.hidden = !isResponsesWorkbenchActive() || state.arena.enabled;
    elements.responsesTargetSelect.innerHTML = '';
    const autoOption = document.createElement('option');
    autoOption.value = RESPONSES_GHOST_AUTO_ID;
    autoOption.textContent = formatResponsesAutoLabel();
    autoOption.selected = isResponsesWorkbenchAutoTarget();
    elements.responsesTargetSelect.appendChild(autoOption);
    options.forEach((instance) => {
        const option = document.createElement('option');
        option.value = instance.instance_id;
        if (isExternalConversationTarget(instance)) {
            option.textContent = `ChatGPT (automatic model · ${formatCodexSourceLabel(instance)})`;
        } else {
            const backendLabel = formatBackendLabel(instance.backend || 'ollama');
            const truth = getInstanceRuntimeTruthSummary(instance);
            option.textContent = truth
                ? `${formatModelDisplayName(instance.model || instance.instance_id)} (${backendLabel} • ${truth})`
                : `${formatModelDisplayName(instance.model || instance.instance_id)} (${backendLabel})`;
        }
        option.selected = option.value === state.responsesWorkbench.targetInstanceId;
        elements.responsesTargetSelect.appendChild(option);
    });
    elements.responsesTargetSelect.disabled = false;
    if (typeof renderInputUtilityDrawer === 'function') {
        renderInputUtilityDrawer();
    }
}
