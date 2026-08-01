function setLoadingMessage(instanceId, html) {
    const conversation = state.conversations[instanceId] || [];
    let targetMessage = null;
    for (let idx = conversation.length - 1; idx >= 0; idx -= 1) {
        if (conversation[idx].isLoading) {
            conversation[idx].content = html;
            conversation[idx].trustedHtml = true;
            targetMessage = conversation[idx];
            break;
        }
    }
    const canPatchDom = Boolean(
        targetMessage?.clientMessageId
        && isRenderableConversationVisible(instanceId)
    );
    if (canPatchDom) {
        queueStreamDomPatch(
            instanceId,
            targetMessage.clientMessageId,
            formatMessageHtml(targetMessage.content, { trustedHtml: Boolean(targetMessage.trustedHtml) })
        );
    } else if (state.arena.enabled) {
        renderArenaConversations();
    } else if (isConversationVisible(instanceId)) {
        renderConversation(instanceId);
    }
}

function resolveStreamingMessageBody(instanceId, clientMessageId = '') {
    const selector = clientMessageId
        ? `[data-client-message-id="${clientMessageId}"] [data-message-body="1"]`
        : '';
    if (selector) {
        const directMatch = elements.chatArea?.querySelector(selector);
        if (directMatch) return directMatch;
    }
    return elements.chatArea?.querySelector('.chat-message:last-child [data-message-body="1"]') || null;
}

function isArenaConversationVisible(instanceId = '') {
    if (!state.arena.enabled) return false;
    const targetId = String(instanceId || '').trim();
    if (!targetId) return false;
    return [
        getArenaConversationId(state.arena.modelA),
        getArenaConversationId(state.arena.modelB),
    ].includes(targetId);
}

function isRenderableConversationVisible(instanceId = '') {
    return isConversationVisible(instanceId) || isArenaConversationVisible(instanceId);
}

function scrollRenderableConversationToBottom(instanceId = '', body = null) {
    if (isConversationVisible(instanceId)) {
        scrollConversationToBottom(instanceId);
        return;
    }
    if (!isArenaConversationVisible(instanceId)) return;
    const targetBody = body || resolveStreamingMessageBody(instanceId);
    const scroller = targetBody?.closest('.arena-column__body');
    if (scroller) {
        scroller.scrollTop = scroller.scrollHeight;
    }
}

function flushStreamDomPatch(instanceId) {
    const patchState = state.streamDomPatches?.[instanceId];
    if (!patchState) return;
    if (patchState.timerId) {
        clearTimeout(patchState.timerId);
    }
    delete state.streamDomPatches[instanceId];
    if (!isRenderableConversationVisible(instanceId)) return;
    const body = resolveStreamingMessageBody(instanceId, patchState.clientMessageId);
    if (body) {
        body.innerHTML = patchState.html;
        if (typeof bindMarkdownCodeCopyButtons === 'function') {
            bindMarkdownCodeCopyButtons(body);
        }
        if (typeof bindMarkdownCodePreviewToggles === 'function') {
            bindMarkdownCodePreviewToggles(body);
        }
        if (typeof bindMarkdownBlockquoteCopyButtons === 'function') {
            bindMarkdownBlockquoteCopyButtons(body);
        }
        scrollRenderableConversationToBottom(instanceId, body);
    } else {
        if (state.arena.enabled) {
            renderArenaConversations();
        } else {
            renderConversation(instanceId);
        }
    }
}

function queueStreamDomPatch(instanceId, clientMessageId, html) {
    if (!isRenderableConversationVisible(instanceId)) return;
    const existing = state.streamDomPatches?.[instanceId];
    if (existing?.timerId) {
        existing.clientMessageId = clientMessageId;
        existing.html = html;
        return;
    }
    state.streamDomPatches[instanceId] = {
        clientMessageId,
        html,
        timerId: setTimeout(() => flushStreamDomPatch(instanceId), 45),
    };
}

function removeLoadingAssistantMessage(instanceId, clientMessageId = '') {
    const targetId = String(clientMessageId || '').trim();
    const conversation = state.conversations[instanceId] || [];
    let removed = false;
    state.conversations[instanceId] = conversation.filter((message) => {
        if (!message?.isLoading) return true;
        if (targetId && message.clientMessageId !== targetId) return true;
        if (!targetId && removed) return true;
        removed = true;
        return false;
    });
    if (!removed) return;
    const patchState = state.streamDomPatches?.[instanceId];
    if (patchState && (!targetId || patchState.clientMessageId === targetId)) {
        if (patchState.timerId) {
            clearTimeout(patchState.timerId);
        }
        delete state.streamDomPatches[instanceId];
    }
    if (state.arena.enabled) {
        renderArenaConversations();
    } else if (isConversationVisible(instanceId)) {
        renderConversation(instanceId);
    }
    persistChatHistory(instanceId);
    renderConversationHistoryList();
}

function updateLoadingAssistantMessage(instanceId, content, extras = {}, finalize = false) {
    const conversation = state.conversations[instanceId] || [];
    let targetMessage = null;
    const targetClientMessageId = String(extras.clientMessageId || '').trim();
    for (let idx = conversation.length - 1; idx >= 0; idx -= 1) {
        if (conversation[idx].isLoading && (!targetClientMessageId || conversation[idx].clientMessageId === targetClientMessageId)) {
            conversation[idx].content = content;
            conversation[idx].trustedHtml = extras.trustedHtml !== undefined ? Boolean(extras.trustedHtml) : false;
            if (extras.imageDataUrl !== undefined) conversation[idx].imageDataUrl = extras.imageDataUrl || null;
            if (extras.savedImagePath !== undefined) conversation[idx].savedImagePath = extras.savedImagePath || null;
            if (extras.savedAudioPath !== undefined) conversation[idx].savedAudioPath = extras.savedAudioPath || null;
            if (extras.savedTextPath !== undefined) conversation[idx].savedTextPath = extras.savedTextPath || null;
            if (extras.artifacts !== undefined) conversation[idx].artifacts = sanitizeResponseArtifacts(extras.artifacts);
            if (extras.responseId !== undefined) conversation[idx].responseId = extras.responseId || null;
            if (extras.responseStateVersion !== undefined || extras.response_state_version !== undefined) {
                conversation[idx].responseStateVersion = String(extras.responseStateVersion || extras.response_state_version || '').trim() || null;
            }
            if (extras.responseFrameSequence !== undefined || extras.response_frame_sequence !== undefined) {
                conversation[idx].responseFrameSequence = coerceResponseFrameSequence(extras.responseFrameSequence ?? extras.response_frame_sequence);
            }
            if (extras.responseFrameId !== undefined || extras.response_frame_id !== undefined) {
                conversation[idx].responseFrameId = String(extras.responseFrameId || extras.response_frame_id || '').trim() || null;
            }
            if (extras.responseCapability !== undefined) conversation[idx].responseCapability = extras.responseCapability || null;
            if (extras.responseModel !== undefined) conversation[idx].responseModel = extras.responseModel || null;
            if (extras.responseBackend !== undefined) conversation[idx].responseBackend = extras.responseBackend || null;
            if (extras.responseInstanceId !== undefined) conversation[idx].responseInstanceId = extras.responseInstanceId || null;
            if (extras.routeSource !== undefined) conversation[idx].routeSource = extras.routeSource || null;
            if (extras.routeReason !== undefined) conversation[idx].routeReason = extras.routeReason || null;
            if (extras.routeRouterInstanceId !== undefined) conversation[idx].routeRouterInstanceId = extras.routeRouterInstanceId || null;
            if (extras.routeRouterModel !== undefined) conversation[idx].routeRouterModel = extras.routeRouterModel || null;
            if (extras.routeArtifactRef !== undefined) conversation[idx].routeArtifactRef = extras.routeArtifactRef || null;
            if (extras.routeArtifactPath !== undefined) conversation[idx].routeArtifactPath = extras.routeArtifactPath || null;
            if (extras.routeReuseLastArtifact !== undefined) conversation[idx].routeReuseLastArtifact = Boolean(extras.routeReuseLastArtifact);
            if (extras.referenceImageCount !== undefined) conversation[idx].referenceImageCount = Number.isFinite(Number(extras.referenceImageCount)) ? Number(extras.referenceImageCount) : null;
            if (extras.referenceImageKind !== undefined) conversation[idx].referenceImageKind = extras.referenceImageKind || null;
            if (extras.contextMode !== undefined) conversation[idx].contextMode = extras.contextMode || null;
            if (extras.contextReason !== undefined) conversation[idx].contextReason = extras.contextReason || null;
            if (extras.lifecycleState !== undefined || extras.lifecycle_state !== undefined) {
                conversation[idx].lifecycleState = String(extras.lifecycleState || extras.lifecycle_state || '').trim().toLowerCase() || null;
            }
            if (extras.statusSemantics !== undefined || extras.status_semantics !== undefined) {
                conversation[idx].statusSemantics = sanitizeResponseStatusSemantics(extras.statusSemantics || extras.status_semantics);
            }
            if (extras.lateFill !== undefined) conversation[idx].lateFill = sanitizeMessageLateFill(extras.lateFill);
            if (extras.surfaceState !== undefined || extras.surface_state !== undefined) {
                conversation[idx].surfaceState = sanitizeSurfaceState(extras.surfaceState || extras.surface_state);
            }
            if (extras.liveStatusLines !== undefined || extras.live_status_lines !== undefined) {
                conversation[idx].liveStatusLines = sanitizeAssistantLiveStatusLines(extras.liveStatusLines || extras.live_status_lines);
                conversation[idx].live_status_lines = conversation[idx].liveStatusLines;
            }
            if (extras.outputs !== undefined) conversation[idx].outputs = sanitizeResponseOutputs(extras.outputs);
            if (extras.outputSlots !== undefined || extras.output_slots !== undefined) {
                conversation[idx].outputSlots = sanitizeResponseOutputSlots(extras.outputSlots || extras.output_slots);
            }
            if (extras.outputBranches !== undefined || extras.output_branches !== undefined) {
                conversation[idx].outputBranches = sanitizeResponseOutputBranches(extras.outputBranches || extras.output_branches);
            }
            if (extras.requestSnapshot !== undefined) conversation[idx].requestSnapshot = sanitizeRequestSnapshot(extras.requestSnapshot);
            syncMessageArtifactLedger(conversation[idx]);
            if (finalize) {
                conversation[idx].isLoading = false;
                conversation[idx].timestamp = new Date().toISOString();
            }
            targetMessage = conversation[idx];
            break;
        }
    }

    if (!targetMessage && finalize) {
        addMessageToConversation(
            instanceId,
            'assistant',
            content,
            false,
            {
                clientMessageId: targetClientMessageId || undefined,
                trustedHtml: Boolean(extras.trustedHtml),
                imageDataUrl: extras.imageDataUrl || null,
                savedImagePath: extras.savedImagePath || null,
                savedAudioPath: extras.savedAudioPath || null,
                savedTextPath: extras.savedTextPath || null,
                artifacts: sanitizeResponseArtifacts(extras.artifacts),
                responseId: extras.responseId || null,
                responseStateVersion: extras.responseStateVersion || extras.response_state_version || null,
                responseFrameSequence: coerceResponseFrameSequence(extras.responseFrameSequence ?? extras.response_frame_sequence),
                responseFrameId: extras.responseFrameId || extras.response_frame_id || null,
                responseCapability: extras.responseCapability || null,
                responseModel: extras.responseModel || null,
                responseBackend: extras.responseBackend || null,
                responseInstanceId: extras.responseInstanceId || null,
                routeSource: extras.routeSource || null,
                routeReason: extras.routeReason || null,
                routeRouterInstanceId: extras.routeRouterInstanceId || null,
                routeRouterModel: extras.routeRouterModel || null,
                routeArtifactRef: extras.routeArtifactRef || null,
                routeArtifactPath: extras.routeArtifactPath || null,
                routeReuseLastArtifact: extras.routeReuseLastArtifact !== undefined ? Boolean(extras.routeReuseLastArtifact) : null,
                referenceImageCount: Number.isFinite(Number(extras.referenceImageCount)) ? Number(extras.referenceImageCount) : null,
                referenceImageKind: extras.referenceImageKind || null,
                contextMode: extras.contextMode || null,
                contextReason: extras.contextReason || null,
                lifecycleState: extras.lifecycleState || extras.lifecycle_state || null,
                statusSemantics: sanitizeResponseStatusSemantics(extras.statusSemantics || extras.status_semantics),
                lateFill: sanitizeMessageLateFill(extras.lateFill),
                surfaceState: sanitizeSurfaceState(extras.surfaceState || extras.surface_state),
                liveStatusLines: sanitizeAssistantLiveStatusLines(extras.liveStatusLines || extras.live_status_lines),
                outputs: sanitizeResponseOutputs(extras.outputs),
                outputSlots: sanitizeResponseOutputSlots(extras.outputSlots || extras.output_slots),
                outputBranches: sanitizeResponseOutputBranches(extras.outputBranches || extras.output_branches),
                requestSnapshot: extras.requestSnapshot || null,
            }
        );
        return;
    }

    const streamPatchWouldDropStructuredControls = Boolean(
        targetMessage
        && !finalize
        && !extras.forceRender
        && typeof messageHasStructuredRenderSurface === 'function'
        && messageHasStructuredRenderSurface(targetMessage)
    );
    const canPatchDom =
        !finalize &&
        !extras.forceRender &&
        !streamPatchWouldDropStructuredControls &&
        isRenderableConversationVisible(instanceId) &&
        targetMessage &&
        targetMessage.clientMessageId;

    if (canPatchDom) {
        queueStreamDomPatch(
            instanceId,
            targetMessage.clientMessageId,
            formatMessageHtml(targetMessage.content, { trustedHtml: Boolean(targetMessage.trustedHtml) })
        );
    } else if (state.arena.enabled) {
        renderArenaConversations();
    } else if (isRenderableConversationVisible(instanceId)) {
        if (finalize) {
            activateConversationBottomAnchor(instanceId, 2200);
        }
        if (finalize && state.streamDomPatches?.[instanceId]) {
            flushStreamDomPatch(instanceId);
        }
        if (state.arena.enabled) {
            renderArenaConversations();
        } else {
            renderConversation(instanceId);
        }
    }
    if (finalize) {
        persistChatHistory(instanceId);
        renderConversationHistoryList();
    }
}

function addLoadingAssistantMessage(instanceId, requestSnapshot = null, clientMessageId = '') {
    return addMessageToConversation(
        instanceId,
        'assistant',
        '<span class="loading-dots">Thinking</span>',
        true,
        {
            clientMessageId: String(clientMessageId || '').trim() || undefined,
            trustedHtml: true,
            requestSnapshot,
        }
    );
}

function finalizeRequestResponse(instanceId, clientMessageId, responsePayload = {}, requestSnapshot = null) {
    finalizeLoadingAssistantResponse(
        instanceId,
        clientMessageId,
        responsePayload,
        '',
        requestSnapshot
    );
}

function extractResponsesContentText(value) {
    if (typeof value === 'string') {
        return value.trim();
    }
    if (Array.isArray(value)) {
        const parts = [];
        value.forEach((item) => {
            const chunk = extractResponsesContentText(item);
            if (chunk) {
                parts.push(chunk);
            }
        });
        return parts.join('\n').trim();
    }
    if (value && typeof value === 'object') {
        const itemType = String(value.type || '').trim();
        if (itemType === 'message') {
            return extractResponsesContentText(value.content);
        }
        if (itemType === 'output_text' || itemType === 'input_text' || itemType === 'text') {
            const text = String(value.text || '').trim();
            if (text) {
                return text;
            }
        }
        for (const key of ['text', 'content', 'response', 'output']) {
            if (key in value) {
                const chunk = extractResponsesContentText(value[key]);
                if (chunk) {
                    return chunk;
                }
            }
        }
    }
    return '';
}

function extractAssistantResponseText(payload = {}, preferredText = '') {
    const explicitPreferred = String(preferredText || '').trim();
    if (explicitPreferred) return explicitPreferred;
    if (!payload || typeof payload !== 'object') return '';
    return extractResponsesContentText(
        payload.output_text
        || payload.outputText
        || payload.text
        || payload.transcript
        || payload.transcription
        || payload.content
        || payload.output
        || payload.response
        || ''
    );
}

function extractResponseErrorMessage(payload = {}, fallback = 'Request failed.') {
    if (!payload || typeof payload !== 'object') {
        return String(fallback || 'Request failed.').trim() || 'Request failed.';
    }
    const candidates = [
        payload?.error_detail?.message,
        payload?.error?.message,
        payload?.error,
        payload?.message,
    ];
    for (const candidate of candidates) {
        const text = String(candidate || '').trim();
        if (text) return text;
    }
    return String(fallback || 'Request failed.').trim() || 'Request failed.';
}

function collectResponseArtifactCandidates(value, depth = 0) {
    if (!value || depth > 4) return [];
    if (Array.isArray(value)) {
        return value.flatMap((item) => collectResponseArtifactCandidates(item, depth + 1));
    }
    if (typeof value !== 'object') return [];

    const candidates = [];
    const directArtifact = value.artifact && typeof value.artifact === 'object'
        ? value.artifact
        : null;
    if (directArtifact) {
        candidates.push({
            ...directArtifact,
            availability: directArtifact.availability || value.metadata?.availability || null,
            artifact_id: directArtifact.artifact_id || value.artifact_id || value.artifactId || null,
            artifact_ref: directArtifact.artifact_ref || directArtifact.artifactRef || value.artifact_ref || value.artifactRef || null,
            kind: directArtifact.kind || value.type || value.kind || null,
            origin: directArtifact.origin || value.origin || null,
        });
    } else if (
        (value.type || value.kind)
        && (
            value.path
            || value.source_path
            || value.sourcePath
            || value.image_data_url
            || value.imageDataUrl
            || value.saved_image_path
            || value.savedImagePath
            || value.artifact_ref
            || value.artifactRef
            || value.ref
            || value.artifact_id
            || value.artifactId
        )
    ) {
        candidates.push(value);
    }

    // Dossiers are diagnostic truth, not public assistant output artifacts.
    // Public artifact rendering must stay bound to payload.artifacts or
    // response_frame.artifacts.output so repair/intermediate records do not leak.
    if (value.output !== undefined) {
        candidates.push(...collectResponseArtifactCandidates(value.output, depth + 1));
    }
    if (value.outputs !== undefined) {
        candidates.push(...collectResponseArtifactCandidates(value.outputs, depth + 1));
    }
    if (value.artifacts !== undefined) {
        candidates.push(...collectResponseArtifactCandidates(value.artifacts, depth + 1));
    }
    return candidates;
}

function sanitizeResponseArtifacts(value) {
    const candidates = collectResponseArtifactCandidates(value);
    const normalizedArtifacts = candidates
        .filter((artifact) => artifact && typeof artifact === 'object')
        .map((artifact) => {
            const rawType = String(artifact.type || artifact.kind || '').trim().toLowerCase();
            const path = artifact.path
                || artifact.saved_image_path
                || artifact.savedImagePath
                || null;
            return {
                type: rawType === 'pdf'
                ? 'document'
                : rawType,
                path,
                name: String(artifact.name || '').trim() || null,
                kind: String(artifact.kind || '').trim().toLowerCase() || null,
                origin: String(artifact.origin || '').trim().toLowerCase() || null,
                artifact_id: String(artifact.artifact_id || artifact.artifactId || '').trim() || null,
                artifact_ref: String(artifact.artifact_ref || artifact.artifactRef || artifact.ref || '').trim() || null,
                ref: String(artifact.ref || artifact.artifact_ref || artifact.artifactRef || '').trim() || null,
                provenance_id: String(artifact.provenance_id || artifact.provenanceId || '').trim() || null,
                source_response_id: String(artifact.source_response_id || artifact.sourceResponseId || '').trim() || null,
                response_model: String(artifact.response_model || artifact.responseModel || '').trim() || null,
                response_instance_id: String(artifact.response_instance_id || artifact.responseInstanceId || '').trim() || null,
                source_path: artifact.source_path || artifact.sourcePath || null,
                image_data_url: artifact.image_data_url || artifact.imageDataUrl || null,
                mime_type: artifact.mime_type || artifact.mimeType || null,
                seed: Number.isFinite(Number(artifact.seed)) ? Number(artifact.seed) : null,
                batch_index: Number.isFinite(Number(artifact.batch_index)) ? Number(artifact.batch_index) : null,
                prompt: String(artifact.prompt || artifact.content || '').trim() || null,
                availability: normalizeArtifactAvailability(artifact.availability),
                purged_at: String(artifact.purged_at || artifact.purgedAt || '').trim() || null,
                purge_reason: String(artifact.purge_reason || artifact.purgeReason || '').trim() || null,
                availability_checked_at: String(
                    artifact.availability_checked_at
                    || artifact.availabilityCheckedAt
                    || ''
                ).trim() || null,
                image_state: artifact.image_state && typeof artifact.image_state === 'object'
                    ? artifact.image_state
                    : null,
            };
        })
        .filter((artifact) => artifact.type && (
            artifact.path
            || artifact.source_path
            || artifact.image_data_url
            || artifact.artifact_id
            || artifact.artifact_ref
            || artifact.ref
            || artifact.name
        ));
    const seen = new Set();
    const deduped = [];
    normalizedArtifacts.forEach((artifact, index) => {
        const durableIdentity = String(artifact.path || artifact.source_path || artifact.image_data_url || '').trim();
        const fallbackIdentity = String(
            artifact.artifact_id
            || artifact.artifact_ref
            || artifact.ref
            || ''
        ).trim();
        const key = durableIdentity
            ? [
                String(artifact.type || '').trim().toLowerCase(),
                durableIdentity,
            ].join('\u0000')
            : fallbackIdentity
                ? [
                    String(artifact.type || '').trim().toLowerCase(),
                    fallbackIdentity,
                    String(artifact.origin || '').trim().toLowerCase(),
                    String(artifact.mime_type || '').trim().toLowerCase(),
                ].join('\u0000')
            : [
                String(artifact.type || '').trim().toLowerCase(),
                'unidentified',
                String(index),
            ].join('\u0000');
        if (seen.has(key)) return;
        seen.add(key);
        deduped.push(artifact);
    });
    return deduped;
}

const NON_PUBLIC_OUTPUT_ARTIFACT_STATUSES = new Set([
    'blocked',
    'cancelled',
    'failed',
    'open',
    'partial_failed',
    'pending',
    'repair_needed',
    'rejected',
    'skipped',
    'superseded',
    'waived',
]);

function artifactPublicIdentityKeys(artifact = {}) {
    if (!artifact || typeof artifact !== 'object') return [];
    const keys = [];
    const artifactRef = String(artifact.artifact_ref || artifact.artifactRef || artifact.ref || '').trim();
    const artifactId = String(artifact.artifact_id || artifact.artifactId || '').trim();
    const path = String(artifact.path || artifact.source_path || artifact.sourcePath || '').trim();
    if (artifactRef) keys.push(`ref:${artifactRef}`);
    if (artifactId) {
        keys.push(`ref:${artifactId}`);
        keys.push(`ref:artifact:${artifactId}`);
    }
    if (path) keys.push(`path:${path}`);
    return keys;
}

function isGeneratedImageTextMisbindingArtifact(artifact = {}) {
    if (!artifact || typeof artifact !== 'object') return false;
    const artifactType = String(artifact.type || artifact.kind || '').trim().toLowerCase();
    if (artifactType !== 'text' && artifactType !== 'document') return false;
    const artifactRef = String(artifact.artifact_ref || artifact.artifactRef || artifact.ref || '').trim();
    const artifactId = String(artifact.artifact_id || artifact.artifactId || '').trim();
    return artifactRef.startsWith('artifact:text_generated_image_')
        || artifactId.startsWith('text_generated_image_');
}

function isPublicArtifactOutputItem(output = {}) {
    if (!output || typeof output !== 'object') return false;
    if (isGeneratedImageTextMisbindingArtifact(output)) return false;
    const status = String(output.status || output.state || '').trim().toLowerCase().replace(/[-\s]+/g, '_');
    if (NON_PUBLIC_OUTPUT_ARTIFACT_STATUSES.has(status)) return false;
    if (output.compatibility_derived === true || output.compatibilityDerived === true) return false;
    const source = String(output.source || '').trim().toLowerCase();
    if (source === 'compatibility_derived' || source === 'raw_saved_artifact_fallback') return false;
    return artifactPublicIdentityKeys(output).length > 0
        || sanitizeResponseArtifacts(output.artifacts).length > 0;
}

function filterPublicArtifactsForOutputs(artifacts = [], outputs = [], outputSlots = []) {
    const normalizedArtifacts = sanitizeResponseArtifacts(artifacts);
    const normalizedOutputs = sanitizeResponseOutputs(outputs);
    const normalizedSlots = sanitizeResponseOutputSlots(outputSlots);
    const hasOutputSurface = normalizedOutputs.length > 0 || normalizedSlots.length > 0;
    if (!hasOutputSurface) return normalizedArtifacts;
    const publicKeys = new Set();
    normalizedOutputs.forEach((output) => {
        if (!isPublicArtifactOutputItem(output)) return;
        artifactPublicIdentityKeys(output).forEach((key) => publicKeys.add(key));
        sanitizeResponseArtifacts(output.artifacts).forEach((artifact) => {
            artifactPublicIdentityKeys(artifact).forEach((key) => publicKeys.add(key));
        });
    });
    normalizedSlots.forEach((slot) => {
        if (!isPublicArtifactOutputItem(slot)) return;
        artifactPublicIdentityKeys(slot).forEach((key) => publicKeys.add(key));
    });
    if (!publicKeys.size) return [];
    return normalizedArtifacts.filter((artifact) => (
        !isGeneratedImageTextMisbindingArtifact(artifact)
        &&
        artifactPublicIdentityKeys(artifact).some((key) => publicKeys.has(key))
    ));
}

function sanitizeResponseErrorRef(value) {
    if (!value || typeof value !== 'object') return null;
    const payload = {
        branch_id: String(value.branch_id || value.branchId || '').trim() || null,
        code: String(value.code || '').trim() || null,
        stage: String(value.stage || '').trim() || null,
    };
    return Object.values(payload).some(Boolean) ? payload : null;
}

function sanitizeResponseRecoveryContext(value) {
    if (!value || typeof value !== 'object') return null;
    const payload = {
        can_retry: typeof value.can_retry === 'boolean'
            ? value.can_retry
            : typeof value.canRetry === 'boolean'
                ? value.canRetry
                : null,
        retry_scope: String(value.retry_scope || value.retryScope || '').trim() || null,
        suggested_action: String(value.suggested_action || value.suggestedAction || '').trim() || null,
        preserve_intent: typeof value.preserve_intent === 'boolean'
            ? value.preserve_intent
            : typeof value.preserveIntent === 'boolean'
                ? value.preserveIntent
                : null,
        exclude_instance_ids: Array.isArray(value.exclude_instance_ids || value.excludeInstanceIds)
            ? (value.exclude_instance_ids || value.excludeInstanceIds)
                .map((item) => String(item || '').trim())
                .filter(Boolean)
            : [],
    };
    return Object.values(payload).some((item) => {
        if (Array.isArray(item)) return item.length > 0;
        return item !== null && item !== '';
    }) ? payload : null;
}

const TERMINAL_OUTPUT_STATUSES = new Set([
    'blocked',
    'cancelled',
    'completed',
    'failed',
    'fulfilled',
    'skipped',
    'superseded',
    'waived',
]);
const TERMINAL_OUTPUT_LIFECYCLES = new Set([
    'blocked_output',
    'cancelled_output',
    'materialized_output',
    'superseded_output',
    'waived_output',
]);

function stripTerminalPlaceholderRef(payload = {}) {
    if (!payload || typeof payload !== 'object') return payload;
    const status = String(payload.status || '').trim().toLowerCase();
    const lifecycle = String(payload.lifecycle || '').trim().toLowerCase();
    if (TERMINAL_OUTPUT_STATUSES.has(status) || TERMINAL_OUTPUT_LIFECYCLES.has(lifecycle)) {
        return {
            ...payload,
            placeholder_ref: null,
        };
    }
    return payload;
}

function sanitizeResponseOutputSlots(value) {
    if (!Array.isArray(value)) return [];
    return value
        .filter((slot) => slot && typeof slot === 'object')
        .map((slot) => {
            const errorRefSource = slot.error_ref && typeof slot.error_ref === 'object'
                ? slot.error_ref
                : slot.errorRef && typeof slot.errorRef === 'object'
                    ? slot.errorRef
                    : null;
            const recoverySource = slot.recovery_context && typeof slot.recovery_context === 'object'
                ? slot.recovery_context
                : slot.recoveryContext && typeof slot.recoveryContext === 'object'
                    ? slot.recoveryContext
                    : null;
            const payload = {
                slot_id: String(slot.slot_id || slot.slotId || '').trim() || null,
                type: String(slot.type || '').trim().toLowerCase() || null,
                status: String(slot.status || '').trim().toLowerCase() || null,
                lifecycle: String(slot.lifecycle || '').trim().toLowerCase() || null,
                artifact_ref: String(slot.artifact_ref || slot.artifactRef || '').trim() || null,
                placeholder_ref: String(slot.placeholder_ref || slot.placeholderRef || '').trim() || null,
                blocked_reason: String(slot.blocked_reason || slot.blockedReason || '').trim() || null,
                follow_up_capability: normalizeCapability(slot.follow_up_capability || slot.followUpCapability || '') || null,
                follow_up_source: String(slot.follow_up_source || slot.followUpSource || '').trim() || null,
                branch_id: String(slot.branch_id || slot.branchId || '').trim() || null,
                phase_id: String(slot.phase_id || slot.phaseId || '').trim() || null,
                parent_slot_id: String(slot.parent_slot_id || slot.parentSlotId || '').trim() || null,
                child_slot_ids: Array.isArray(slot.child_slot_ids || slot.childSlotIds)
                    ? (slot.child_slot_ids || slot.childSlotIds).map((item) => String(item || '').trim()).filter(Boolean)
                    : [],
                batch_index: Number.isFinite(Number(slot.batch_index ?? slot.batchIndex))
                    ? Number(slot.batch_index ?? slot.batchIndex)
                    : null,
                error_ref: sanitizeResponseErrorRef(errorRefSource),
                recovery_context: sanitizeResponseRecoveryContext(recoverySource),
            };
            const normalized = stripTerminalPlaceholderRef(payload);
            return Object.values(normalized).some((item) => {
                if (Array.isArray(item)) return item.length > 0;
                return item !== null && item !== '';
            }) ? normalized : null;
        })
        .filter(Boolean);
}

function sanitizeResponseOutputBranches(value) {
    if (!Array.isArray(value)) return [];
    return value
        .filter((branch) => branch && typeof branch === 'object')
        .map((branch) => {
            const errorRefSource = branch.error_ref && typeof branch.error_ref === 'object'
                ? branch.error_ref
                : branch.errorRef && typeof branch.errorRef === 'object'
                    ? branch.errorRef
                    : null;
            const recoverySource = branch.recovery_context && typeof branch.recovery_context === 'object'
                ? branch.recovery_context
                : branch.recoveryContext && typeof branch.recoveryContext === 'object'
                    ? branch.recoveryContext
                    : null;
            const payload = {
                slot_id: String(branch.slot_id || branch.slotId || '').trim() || null,
                branch_id: String(branch.branch_id || branch.branchId || '').trim() || null,
                phase_id: String(branch.phase_id || branch.phaseId || '').trim() || null,
                type: String(branch.type || '').trim().toLowerCase() || null,
                capability: normalizeCapability(branch.capability || branch.follow_up_capability || branch.followUpCapability || '') || null,
                output_type: String(branch.output_type || branch.outputType || branch.type || '').trim().toLowerCase() || null,
                status: String(branch.status || '').trim().toLowerCase() || null,
                lifecycle: String(branch.lifecycle || '').trim().toLowerCase() || null,
                follow_up_capability: normalizeCapability(branch.follow_up_capability || branch.followUpCapability || '') || null,
                role: String(branch.role || '').trim() || null,
                phase_summary: String(branch.phase_summary || branch.phaseSummary || '').trim() || null,
                objective: String(branch.objective || '').trim() || null,
                deliverable: String(branch.deliverable || '').trim() || null,
                semantic_intent: String(branch.semantic_intent || branch.semanticIntent || '').trim() || null,
                artifact_prompt: String(branch.artifact_prompt || branch.artifactPrompt || '').trim() || null,
                content_payload: String(branch.content_payload || branch.contentPayload || '').trim() || null,
                source: String(branch.source || '').trim() || null,
                source_name: String(branch.source_name || branch.sourceName || '').trim() || null,
                content_payload_source: String(branch.content_payload_source || branch.contentPayloadSource || '').trim() || null,
                review_criteria: Array.isArray(branch.review_criteria || branch.reviewCriteria)
                    ? (branch.review_criteria || branch.reviewCriteria).map((item) => String(item || '').trim()).filter(Boolean)
                    : [],
                artifact_ref: String(branch.artifact_ref || branch.artifactRef || '').trim() || null,
                placeholder_ref: String(branch.placeholder_ref || branch.placeholderRef || '').trim() || null,
                blocked_reason: String(branch.blocked_reason || branch.blockedReason || '').trim() || null,
                cancel_reason: String(branch.cancel_reason || branch.cancelReason || '').trim() || null,
                cancelled_at: String(branch.cancelled_at || branch.cancelledAt || '').trim() || null,
                cancelled_by: String(branch.cancelled_by || branch.cancelledBy || '').trim() || null,
                waiver_reason: String(branch.waiver_reason || branch.waiverReason || '').trim() || null,
                supersession_reason: String(branch.supersession_reason || branch.supersessionReason || '').trim() || null,
                parent_slot_id: String(branch.parent_slot_id || branch.parentSlotId || '').trim() || null,
                child_slot_ids: Array.isArray(branch.child_slot_ids || branch.childSlotIds)
                    ? (branch.child_slot_ids || branch.childSlotIds).map((item) => String(item || '').trim()).filter(Boolean)
                    : [],
                depends_on: Array.isArray(branch.depends_on || branch.dependsOn)
                    ? (branch.depends_on || branch.dependsOn).map((item) => String(item || '').trim()).filter(Boolean)
                    : [],
                batch_index: Number.isFinite(Number(branch.batch_index ?? branch.batchIndex))
                    ? Number(branch.batch_index ?? branch.batchIndex)
                    : null,
                queue_index: Number.isFinite(Number(branch.queue_index ?? branch.queueIndex))
                    ? Number(branch.queue_index ?? branch.queueIndex)
                    : null,
                error_ref: sanitizeResponseErrorRef(errorRefSource),
                recovery_context: sanitizeResponseRecoveryContext(recoverySource),
            };
            const normalized = stripTerminalPlaceholderRef(payload);
            return Object.values(normalized).some((item) => {
                if (Array.isArray(item)) return item.length > 0;
                return item !== null && item !== '';
            }) ? normalized : null;
        })
        .filter(Boolean);
}

function sanitizeResponseOutputs(value) {
    if (!Array.isArray(value)) return [];
    return value
        .filter((output) => output && typeof output === 'object')
        .map((output) => {
            const errorRefSource = output.error_ref && typeof output.error_ref === 'object'
                ? output.error_ref
                : output.errorRef && typeof output.errorRef === 'object'
                    ? output.errorRef
                    : null;
            const recoverySource = output.recovery_context && typeof output.recovery_context === 'object'
                ? output.recovery_context
                : output.recoveryContext && typeof output.recoveryContext === 'object'
                    ? output.recoveryContext
                    : null;
            const payload = {
                slot_id: String(output.slot_id || output.slotId || '').trim() || null,
                branch_id: String(output.branch_id || output.branchId || '').trim() || null,
                phase_id: String(output.phase_id || output.phaseId || '').trim() || null,
                type: String(output.type || '').trim().toLowerCase() || null,
                status: String(output.status || '').trim().toLowerCase() || null,
                lifecycle: String(output.lifecycle || '').trim().toLowerCase() || null,
                artifact_ref: String(output.artifact_ref || output.artifactRef || '').trim() || null,
                placeholder_ref: String(output.placeholder_ref || output.placeholderRef || '').trim() || null,
                blocked_reason: String(output.blocked_reason || output.blockedReason || '').trim() || null,
                parent_slot_id: String(output.parent_slot_id || output.parentSlotId || '').trim() || null,
                follow_up_capability: normalizeCapability(output.follow_up_capability || output.followUpCapability || '') || null,
                value: String(output.value || '').trim() || null,
                child_slot_ids: Array.isArray(output.child_slot_ids || output.childSlotIds)
                    ? (output.child_slot_ids || output.childSlotIds).map((item) => String(item || '').trim()).filter(Boolean)
                    : [],
                artifacts: sanitizeResponseArtifacts(output.artifacts),
                batch_index: Number.isFinite(Number(output.batch_index ?? output.batchIndex))
                    ? Number(output.batch_index ?? output.batchIndex)
                    : null,
                error_ref: sanitizeResponseErrorRef(errorRefSource),
                recovery_context: sanitizeResponseRecoveryContext(recoverySource),
            };
            const normalized = stripTerminalPlaceholderRef(payload);
            return Object.values(normalized).some((item) => {
                if (Array.isArray(item)) return item.length > 0;
                return item !== null && item !== '';
            }) ? normalized : null;
        })
        .filter(Boolean);
}

function extractResponseOutputSlots(payload = {}) {
    const direct = sanitizeResponseOutputSlots(payload.output_slots || payload.outputSlots);
    if (direct.length) return direct;
    const responseFrame = payload.response_frame && typeof payload.response_frame === 'object'
        ? payload.response_frame
        : null;
    const frameSource = responseFrame || (payload && typeof payload === 'object' ? payload : null);
    const frameSlots = frameSource?.planning?.artifact_flow?.output_slots;
    return sanitizeResponseOutputSlots(frameSlots);
}

function buildOutputsFromSlots(payload = {}, outputSlots = [], artifacts = []) {
    const normalizedArtifacts = sanitizeResponseArtifacts(artifacts);
    return sanitizeResponseOutputSlots(outputSlots)
        .map((slot) => {
            const slotArtifactRef = String(slot.artifact_ref || '').trim();
            const slotArtifacts = normalizedArtifacts.filter((artifact) => {
                const artifactRef = String(artifact.artifact_ref || artifact.ref || artifact.artifact_id || '').trim();
                if (slotArtifactRef) {
                    return artifactRef === slotArtifactRef;
                }
                return String(artifact.type || '').trim().toLowerCase() === String(slot.type || '').trim().toLowerCase()
                    && (!Number.isFinite(Number(slot.batch_index)) || Number(artifact.batch_index) === Number(slot.batch_index));
            });
            const output = {
                slot_id: slot.slot_id || null,
                branch_id: slot.branch_id || slot.phase_id || null,
                phase_id: slot.phase_id || slot.branch_id || null,
                type: slot.type || null,
                status: slot.status || null,
                lifecycle: slot.lifecycle || null,
                artifact_ref: slot.artifact_ref || null,
                placeholder_ref: slot.placeholder_ref || null,
                blocked_reason: slot.blocked_reason || null,
                parent_slot_id: slot.parent_slot_id || null,
                follow_up_capability: slot.follow_up_capability || null,
                child_slot_ids: Array.isArray(slot.child_slot_ids) ? slot.child_slot_ids : [],
                artifacts: slotArtifacts,
                batch_index: Number.isFinite(Number(slot.batch_index)) ? Number(slot.batch_index) : null,
                error_ref: sanitizeResponseErrorRef(slot.error_ref),
                recovery_context: sanitizeResponseRecoveryContext(slot.recovery_context),
            };
            if (slot.type === 'text' || slot.type === 'document') {
                const explicitValue = String(payload.content_payload || payload.output_text || payload.outputText || '').trim();
                if (explicitValue) {
                    output.value = explicitValue;
                }
            }
            return output;
        });
}

function extractResponseOutputs(payload = {}, { artifacts = null, outputSlots = null } = {}) {
    const outputPayload = payload.output && typeof payload.output === 'object'
        ? payload.output
        : null;
    const direct = sanitizeResponseOutputs(
        payload.outputs
        || payload.canonical_outputs
        || payload.canonicalOutputs
        || outputPayload?.outputs
    );
    if (direct.length) return direct;
    const responseFrame = payload.response_frame && typeof payload.response_frame === 'object'
        ? payload.response_frame
        : null;
    const frameOutputs = sanitizeResponseOutputs(responseFrame?.outputs || responseFrame?.output?.outputs);
    if (frameOutputs.length) return frameOutputs;
    return buildOutputsFromSlots(
        payload,
        Array.isArray(outputSlots) ? outputSlots : extractResponseOutputSlots(payload),
        Array.isArray(artifacts) ? artifacts : sanitizeResponseArtifacts(payload.artifacts)
    );
}

function buildOutputBranchesFromSlots(outputSlots = []) {
    return outputSlots
        .filter((slot) => slot && typeof slot === 'object')
        .map((slot) => {
            const payload = {
                slot_id: String(slot.slot_id || '').trim() || null,
                branch_id: String(slot.branch_id || slot.phase_id || '').trim() || null,
                phase_id: String(slot.phase_id || slot.branch_id || '').trim() || null,
                type: String(slot.type || '').trim().toLowerCase() || null,
                status: String(slot.status || '').trim().toLowerCase() || null,
                lifecycle: String(slot.lifecycle || '').trim().toLowerCase() || null,
                follow_up_capability: normalizeCapability(slot.follow_up_capability || '') || null,
                artifact_ref: String(slot.artifact_ref || '').trim() || null,
                placeholder_ref: String(slot.placeholder_ref || '').trim() || null,
                blocked_reason: String(slot.blocked_reason || '').trim() || null,
                parent_slot_id: String(slot.parent_slot_id || '').trim() || null,
                child_slot_ids: Array.isArray(slot.child_slot_ids) ? slot.child_slot_ids.map((item) => String(item || '').trim()).filter(Boolean) : [],
                error_ref: sanitizeResponseErrorRef(slot.error_ref),
                recovery_context: sanitizeResponseRecoveryContext(slot.recovery_context),
            };
            const normalized = stripTerminalPlaceholderRef(payload);
            return Object.values(normalized).some((item) => {
                if (Array.isArray(item)) return item.length > 0;
                return item !== null && item !== '';
            }) ? normalized : null;
        })
        .filter(Boolean);
}

function extractResponseOutputBranches(payload = {}, { outputSlots = null } = {}) {
    const direct = sanitizeResponseOutputBranches(payload.output_branches || payload.outputBranches);
    if (direct.length) return direct;
    const responseFrame = payload.response_frame && typeof payload.response_frame === 'object'
        ? payload.response_frame
        : null;
    const frameSource = responseFrame || (payload && typeof payload === 'object' ? payload : null);
    const frameBranches = sanitizeResponseOutputBranches(frameSource?.output_branches || frameSource?.outputBranches);
    if (frameBranches.length) return frameBranches;
    return buildOutputBranchesFromSlots(Array.isArray(outputSlots) ? outputSlots : extractResponseOutputSlots(payload));
}

function normalizeArtifactAvailability(value) {
    const normalized = String(value || '').trim().toLowerCase();
    if (normalized === 'available' || normalized === 'missing' || normalized === 'purged') {
        return normalized;
    }
    return null;
}

function buildCanonicalArtifactIdentityKeys(artifact = {}) {
    const type = String(artifact.type || '').trim().toLowerCase();
    if (!type) return [];
    const path = String(artifact.path || '').trim();
    const imageDataUrl = String(artifact.image_data_url || '').trim();
    const sourcePath = String(artifact.source_path || '').trim();
    const artifactRef = String(artifact.artifact_ref || artifact.ref || '').trim();
    const artifactId = String(artifact.artifact_id || '').trim();
    const keys = [];
    if (artifactRef) {
        keys.push(JSON.stringify([type, 'artifact_ref', artifactRef]));
    }
    if (artifactId) {
        keys.push(JSON.stringify([type, 'artifact_id', artifactId]));
    }
    if (path) {
        keys.push(JSON.stringify([type, 'path', path]));
    }
    if (imageDataUrl) {
        keys.push(JSON.stringify([type, 'image_data_url', imageDataUrl]));
    }
    if (sourcePath) {
        keys.push(JSON.stringify([type, 'source_path', sourcePath]));
    }
    return keys;
}

function findCanonicalArtifactIndex(targetArtifacts = [], artifact = {}) {
    const candidateKeys = buildCanonicalArtifactIdentityKeys(artifact);
    if (!candidateKeys.length) return -1;
    return targetArtifacts.findIndex((item) => {
        const existingKeys = buildCanonicalArtifactIdentityKeys(item);
        return existingKeys.some((key) => candidateKeys.includes(key));
    });
}

function mergeCanonicalArtifactRecords(existing = {}, incoming = {}) {
    const merged = { ...existing };
    Object.entries(incoming).forEach(([key, value]) => {
        if (value === null || value === '' || typeof value === 'undefined') {
            return;
        }
        if (merged[key] === null || merged[key] === '' || typeof merged[key] === 'undefined') {
            merged[key] = value;
            return;
        }
        if (key === 'availability') {
            const current = normalizeArtifactAvailability(merged[key]);
            const nextValue = normalizeArtifactAvailability(value);
            if (current !== 'purged' && nextValue === 'purged') {
                merged[key] = nextValue;
            } else if (current === 'available' && nextValue === 'missing') {
                merged[key] = nextValue;
            }
        }
    });
    return merged;
}

function appendCanonicalArtifact(targetArtifacts, artifact) {
    const normalized = sanitizeResponseArtifacts([artifact])[0] || null;
    if (!normalized) return;
    const existingIndex = findCanonicalArtifactIndex(targetArtifacts, normalized);
    if (existingIndex >= 0) {
        targetArtifacts[existingIndex] = sanitizeResponseArtifacts([
            mergeCanonicalArtifactRecords(targetArtifacts[existingIndex], normalized),
        ])[0] || targetArtifacts[existingIndex];
        return;
    }
    targetArtifacts.push(normalized);
}

function buildCanonicalMessageArtifacts(message = {}, { requestSnapshot = null } = {}) {
    const source = message && typeof message === 'object' ? message : {};
    const role = String(source.role || '').trim().toLowerCase();
    const artifacts = [];
    const normalizedRequestSnapshot = sanitizeRequestSnapshot(
        requestSnapshot
        ?? source.requestSnapshot
        ?? source.request_snapshot
    );
    const sourceOutputs = sanitizeResponseOutputs(source.outputs || source.canonical_outputs || source.canonicalOutputs);
    const sourceOutputSlots = sanitizeResponseOutputSlots(source.outputSlots || source.output_slots);
    const sourceArtifacts = filterPublicArtifactsForOutputs(
        source.artifacts,
        sourceOutputs,
        sourceOutputSlots
    );
    if (role !== 'user') {
        sourceArtifacts.forEach((artifact) => {
            appendCanonicalArtifact(artifacts, artifact);
        });
        sourceOutputs.forEach((output) => {
            if (!isPublicArtifactOutputItem(output)) return;
            sanitizeResponseArtifacts(output.artifacts).forEach((artifact) => {
                appendCanonicalArtifact(artifacts, artifact);
            });
        });
    }
    if (role === 'user') {
        const inputArtifacts = getRequestSnapshotInputArtifacts(normalizedRequestSnapshot);
        (inputArtifacts.length ? inputArtifacts : sourceArtifacts).forEach((artifact) => {
            appendCanonicalArtifact(artifacts, artifact);
        });
    }
    const appendFallbackArtifact = (type, { path = null, imageDataUrl = null } = {}) => {
        const normalizedPath = String(path || '').trim() || null;
        const normalizedImageDataUrl = String(imageDataUrl || '').trim() || null;
        if (!normalizedPath && !normalizedImageDataUrl) {
            return;
        }
        appendCanonicalArtifact(artifacts, {
            type,
            path: normalizedPath,
            image_data_url: normalizedImageDataUrl,
        });
    };
    if (role !== 'user') {
        appendFallbackArtifact('image', {
            path: source.savedImagePath || source.saved_image_path || null,
            imageDataUrl: source.imageDataUrl || source.image_data_url || null,
        });
        appendFallbackArtifact('audio', {
            path: source.savedAudioPath || source.saved_audio_path || null,
        });
        appendFallbackArtifact('text', {
            path: source.savedTextPath || source.saved_text_path || null,
        });
    }
    return artifacts;
}

function syncMessageArtifactLedger(message = {}, { requestSnapshot = undefined } = {}) {
    if (!message || typeof message !== 'object') {
        return message;
    }
    const role = String(message.role || '').trim().toLowerCase();
    const normalizedRequestSnapshot = requestSnapshot === undefined
        ? sanitizeRequestSnapshot(message.requestSnapshot || message.request_snapshot)
        : sanitizeRequestSnapshot(requestSnapshot);
    message.artifacts = buildCanonicalMessageArtifacts(message, { requestSnapshot: normalizedRequestSnapshot });
    if (role === 'user') {
        message.requestSnapshot = mergeRequestSnapshotInputArtifacts(normalizedRequestSnapshot);
    } else if (requestSnapshot !== undefined) {
        message.requestSnapshot = normalizedRequestSnapshot;
    }
    return message;
}

function sanitizeAssistantLiveStatusLines(value) {
    if (!Array.isArray(value)) return [];
    return Array.from(new Set(
        value
            .map((line) => String(line || '').trim())
            .filter(Boolean)
    ));
}

function buildAssistantMessageRenderableState(message = {}) {
    return {
        content: String(message.content || ''),
        trustedHtml: Boolean(message.trustedHtml),
        imageDataUrl: String(message.imageDataUrl || '').trim() || null,
        savedImagePath: String(message.savedImagePath || '').trim() || null,
        savedAudioPath: String(message.savedAudioPath || '').trim() || null,
        savedTextPath: String(message.savedTextPath || '').trim() || null,
        artifacts: sanitizeResponseArtifacts(message.artifacts),
        responseId: String(message.responseId || message.response_id || '').trim() || null,
        responseStateVersion: String(message.responseStateVersion || message.response_state_version || '').trim() || null,
        responseFrameSequence: coerceResponseFrameSequence(message.responseFrameSequence ?? message.response_frame_sequence),
        responseFrameId: String(message.responseFrameId || message.response_frame_id || '').trim() || null,
        responseCapability: normalizeCapability(message.responseCapability || '') || null,
        responseModel: String(message.responseModel || '').trim() || null,
        responseBackend: normalizeBackend(message.responseBackend || '') || null,
        responseInstanceId: String(message.responseInstanceId || '').trim() || null,
        routeSource: String(message.routeSource || '').trim().toLowerCase() || null,
        routeReason: String(message.routeReason || '').trim() || null,
        routeRouterInstanceId: String(message.routeRouterInstanceId || '').trim() || null,
        routeRouterModel: String(message.routeRouterModel || '').trim() || null,
        routeArtifactRef: String(message.routeArtifactRef || '').trim() || null,
        routeArtifactPath: String(message.routeArtifactPath || '').trim() || null,
        routeReuseLastArtifact: message.routeReuseLastArtifact === null || message.routeReuseLastArtifact === undefined
            ? null
            : Boolean(message.routeReuseLastArtifact),
        referenceImageCount: Number.isFinite(Number(message.referenceImageCount))
            ? Number(message.referenceImageCount)
            : null,
        referenceImageKind: String(message.referenceImageKind || '').trim() || null,
        contextMode: String(message.contextMode || '').trim().toLowerCase() || null,
        contextReason: String(message.contextReason || '').trim() || null,
        lifecycleState: String(message.lifecycleState || message.lifecycle_state || '').trim().toLowerCase() || null,
        statusSemantics: sanitizeResponseStatusSemantics(message.statusSemantics || message.status_semantics),
        lateFill: sanitizeMessageLateFill(message.lateFill || message.late_fill),
        surfaceState: sanitizeSurfaceState(message.surfaceState || message.surface_state),
        liveStatusLines: sanitizeAssistantLiveStatusLines(message.liveStatusLines || message.live_status_lines),
        outputs: sanitizeResponseOutputs(message.outputs || message.canonical_outputs || message.canonicalOutputs),
        outputSlots: sanitizeResponseOutputSlots(message.outputSlots || message.output_slots),
        outputBranches: sanitizeResponseOutputBranches(message.outputBranches || message.output_branches),
        artifactBundles: Array.isArray(message.artifactBundles)
            ? message.artifactBundles
            : (Array.isArray(message.artifact_bundles) ? message.artifact_bundles : []),
    };
}

function buildAssistantMessagePersistedState(message = {}) {
    return {
        ...buildAssistantMessageRenderableState(message),
        requestSnapshot: sanitizeRequestSnapshot(message.requestSnapshot || message.request_snapshot),
    };
}

function fingerprintAssistantMessageState(message = {}, { includeRequestSnapshot = false } = {}) {
    const normalized = includeRequestSnapshot
        ? buildAssistantMessagePersistedState(message)
        : buildAssistantMessageRenderableState(message);
    return JSON.stringify(normalized);
}

function mapInputArtifactKindToType(kind = '') {
    const normalizedKind = String(kind || '').trim().toLowerCase();
    if (normalizedKind === 'image') return 'image';
    if (normalizedKind === 'audio') return 'audio';
    if (normalizedKind === 'text') return 'text';
    if (normalizedKind === 'pdf') return 'document';
    return normalizedKind || 'binary';
}

function sanitizeRequestSnapshotInputArtifact(value) {
    if (!value || typeof value !== 'object') return null;
    const kind = String(value.kind || '').trim().toLowerCase()
        || inferFileKindFromName(value.name || value.path || value.source_path || value.sourcePath || value.local_path || value.localPath || '');
    const rawType = String(value.type || '').trim().toLowerCase();
    const type = rawType === 'pdf'
        ? 'document'
        : (rawType || mapInputArtifactKindToType(kind));
    const path = String(value.path || '').trim() || null;
    const sourcePath = String(
        value.source_path
        || value.sourcePath
        || value.local_path
        || value.localPath
        || ''
    ).trim() || null;
    const name = String(value.name || '').trim()
        || basenameFromPath(path || sourcePath || '')
        || null;
    const origin = String(value.origin || '').trim().toLowerCase() || null;
    const mimeType = String(value.mime_type || value.mimeType || '').trim() || null;
    const payload = {
        type: type || null,
        path,
        name,
        kind: kind || null,
        origin,
        source_path: sourcePath,
        mime_type: mimeType,
        availability: normalizeArtifactAvailability(value.availability),
        purged_at: String(value.purged_at || value.purgedAt || '').trim() || null,
        purge_reason: String(value.purge_reason || value.purgeReason || '').trim() || null,
        availability_checked_at: String(
            value.availability_checked_at
            || value.availabilityCheckedAt
            || ''
        ).trim() || null,
    };
    return Object.values(payload).some((item) => {
        if (Array.isArray(item)) return item.length > 0;
        return item !== null && item !== '';
    }) ? payload : null;
}

function buildRequestSnapshotInputArtifactIdentityKeys(item = {}) {
    const type = String(item.type || '').trim().toLowerCase();
    if (!type) return [];
    const path = String(item.path || '').trim();
    const sourcePath = String(item.source_path || '').trim();
    const origin = String(item.origin || '').trim().toLowerCase();
    const kind = String(item.kind || '').trim().toLowerCase();
    const name = String(item.name || '').trim().toLowerCase();
    const keys = [];
    if (path) {
        keys.push(JSON.stringify([type, 'path', path]));
    }
    if (sourcePath) {
        keys.push(JSON.stringify([type, 'source_path', sourcePath]));
    }
    if (name) {
        keys.push(JSON.stringify([type, 'named_input', origin, kind, name]));
    }
    return keys;
}

function sanitizeRequestSnapshotInputArtifacts(value) {
    const items = Array.isArray(value) ? value : [value];
    const mergedItems = [];
    const mergedIndexByKey = new Map();
    items
        .map((item) => sanitizeRequestSnapshotInputArtifact(item))
        .filter(Boolean)
        .forEach((item) => {
            const identityKeys = buildRequestSnapshotInputArtifactIdentityKeys(item);
            let existingIndex = -1;
            identityKeys.some((key) => {
                if (!mergedIndexByKey.has(key)) {
                    return false;
                }
                existingIndex = mergedIndexByKey.get(key);
                return true;
            });
            const mergedItem = sanitizeRequestSnapshotInputArtifact(
                existingIndex >= 0
                    ? mergeCanonicalArtifactRecords(mergedItems[existingIndex], item)
                    : item
            );
            if (!mergedItem) {
                return;
            }
            if (existingIndex >= 0) {
                mergedItems[existingIndex] = mergedItem;
            } else {
                existingIndex = mergedItems.push(mergedItem) - 1;
            }
            buildRequestSnapshotInputArtifactIdentityKeys(mergedItem).forEach((key) => {
                mergedIndexByKey.set(key, existingIndex);
            });
        });
    return mergedItems.filter(Boolean);
}

function getRequestSnapshotInputArtifacts(snapshot) {
    return sanitizeRequestSnapshotInputArtifacts(
        snapshot?.input_artifacts
        ?? snapshot?.inputArtifacts
    );
}

function buildPendingRequestInputArtifacts({ attachment = null, localPath = '' } = {}) {
    if (!attachment && !localPath) {
        return [];
    }
    const sourceName = String(attachment?.name || basenameFromPath(localPath) || '').trim();
    const kind = inferFileKindFromName(sourceName || localPath);
    return sanitizeRequestSnapshotInputArtifacts([
        {
            type: mapInputArtifactKindToType(kind),
            name: sourceName || null,
            kind,
            origin: attachment ? 'upload' : 'local_path',
            source_path: localPath || null,
        },
    ]);
}

function mergeRequestSnapshotInputArtifacts(requestSnapshot, inputArtifacts = []) {
    const baseSnapshot = sanitizeRequestSnapshot(requestSnapshot) || {};
    const mergedInputArtifacts = sanitizeRequestSnapshotInputArtifacts(
        Array.isArray(inputArtifacts) && inputArtifacts.length
            ? inputArtifacts
            : getRequestSnapshotInputArtifacts(baseSnapshot)
    );
    if (!Object.keys(baseSnapshot).length && !mergedInputArtifacts.length) {
        return null;
    }
    return sanitizeRequestSnapshot({
        ...baseSnapshot,
        input_artifacts: mergedInputArtifacts,
    });
}

function sanitizeSnapshotScalarMap(value, { keepNull = false } = {}) {
    if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
    const payload = {};
    Object.entries(value).forEach(([rawKey, rawValue]) => {
        const key = String(rawKey || '').trim();
        if (!key) return;
        if (rawValue === null) {
            if (keepNull) {
                payload[key] = null;
            }
            return;
        }
        if (typeof rawValue === 'undefined') return;
        if (typeof rawValue === 'string') {
            payload[key] = rawValue;
            return;
        }
        if (typeof rawValue === 'number') {
            if (Number.isFinite(rawValue)) {
                payload[key] = rawValue;
            }
            return;
        }
        if (typeof rawValue === 'boolean') {
            payload[key] = rawValue;
        }
    });
    return Object.keys(payload).length ? payload : null;
}

function sanitizeRequestSnapshotTarget(value) {
    if (!value || typeof value !== 'object') return null;
    const backendRaw = String(value.backend || '').trim();
    const capabilityRaw = String(value.capability || '').trim();
    const payload = {
        instance_id: String(value.instance_id || value.instanceId || '').trim() || null,
        model: String(value.model || value.modelName || '').trim() || null,
        backend: backendRaw ? normalizeBackend(backendRaw) : null,
        capability: capabilityRaw ? normalizeCapability(capabilityRaw) : null,
    };
    return Object.values(payload).some((item) => item !== null && item !== '') ? payload : null;
}

function sanitizeRequestSnapshotAttachment(value) {
    if (!value || typeof value !== 'object') return null;
    const localPath = String(value.local_path || value.localPath || '').trim();
    const name = String(value.name || '').trim() || basenameFromPath(localPath) || '';
    const kind = String(value.kind || '').trim().toLowerCase() || (name ? inferFileKindFromName(name) : '');
    const payload = {
        name: name || null,
        kind: kind || null,
        local_path: localPath || null,
    };
    return Object.values(payload).some((item) => item !== null && item !== '') ? payload : null;
}

function sanitizeRequestSnapshotGhostPreview(value) {
    if (!value || typeof value !== 'object') return null;
    const payload = {
        instance_id: String(value.instance_id || value.instanceId || '').trim() || null,
        capability: String(value.capability || '').trim() ? normalizeCapability(value.capability) : null,
        reuse_last_artifact: Boolean(value.reuse_last_artifact),
        artifact_ref: String(value.artifact_ref || value.artifactRef || '').trim() || null,
        artifact_path: String(value.artifact_path || '').trim() || null,
        confidence: Number.isFinite(Number(value.confidence)) ? Number(value.confidence) : null,
        reason: String(value.reason || '').trim() || null,
        route_source: String(value.route_source || value.routeSource || '').trim() || null,
        route_router_instance_id: String(value.route_router_instance_id || '').trim() || null,
        route_router_model: String(value.route_router_model || '').trim() || null,
    };
    return Object.values(payload).some((item) => item !== null && item !== '') ? payload : null;
}

function sanitizeRequestSnapshotRequestMeta(value) {
    if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
    const payload = {
        ghost_mode: String(value.ghost_mode || value.ghostMode || '').trim().toLowerCase() || null,
        capability_hint: String(value.capability_hint || value.capabilityHint || '').trim().toLowerCase() || null,
        language_hint: String(value.language_hint || value.languageHint || '').trim().toLowerCase() || null,
        developer_flags: sanitizeSnapshotScalarMap(
            value.developer_flags || value.developerFlags,
            { keepNull: true }
        ),
    };
    return Object.entries(payload).some(([key, item]) => (
        Array.isArray(item) ? item.length > 0 : item && (typeof item !== 'object' || Object.keys(item).length > 0)
    )) ? payload : null;
}

function sanitizeRequestSnapshotDeveloperDiagnostics(value) {
    return sanitizeSnapshotScalarMap(
        value,
        { keepNull: true }
    );
}

function sanitizeRequestSnapshot(value) {
    if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
    const batchPrompts = Array.isArray(value.batch_prompts || value.batchPrompts)
        ? (value.batch_prompts || value.batchPrompts)
            .map((item) => String(item || '').trim())
            .filter(Boolean)
        : [];
    const inputArtifacts = sanitizeRequestSnapshotInputArtifacts(
        value.input_artifacts
        ?? value.inputArtifacts
    );
    const referenceArtifacts = compactRequestSnapshotReferenceArtifacts(
        value.reference_artifacts
        ?? value.referenceArtifacts
        ?? value.selected_reference_artifacts
        ?? value.selectedReferenceArtifacts
        ?? value.selected_reference_artifact
        ?? value.selectedReferenceArtifact
    );
    const payload = {
        request_id: String(value.request_id || value.requestId || '').trim() || null,
        created_at: String(value.created_at || value.createdAt || '').trim() || null,
        conversation_id: String(value.conversation_id || value.conversationId || '').trim() || null,
        response_id: String(value.response_id || value.responseId || '').trim() || null,
        transport: String(value.transport || '').trim() || null,
        prompt_text: String(value.prompt_text || value.promptText || '').trim() || null,
        prompt_preview: String(value.prompt_preview || value.promptPreview || '').trim() || null,
        target: sanitizeRequestSnapshotTarget(value.target),
        attachment: sanitizeRequestSnapshotAttachment(value.attachment),
        input_artifacts: inputArtifacts,
        session_controls: sanitizeSnapshotScalarMap(value.session_controls || value.sessionControls),
        settings: sanitizeSnapshotScalarMap(
            value.settings && typeof value.settings === 'object'
                ? sanitizeSettingsObject(value.settings)
                : null,
            { keepNull: true }
        ),
        request_meta: sanitizeRequestSnapshotRequestMeta(value.request_meta || value.requestMeta),
        developer_diagnostics: sanitizeRequestSnapshotDeveloperDiagnostics(
            value.developer_diagnostics
            || value.developerDiagnostics
        ),
        ghost_preview: sanitizeRequestSnapshotGhostPreview(value.ghost_preview || value.ghostPreview),
    };
    if (batchPrompts.length) {
        payload.batch_prompts = batchPrompts;
    }
    if (referenceArtifacts.length) {
        payload.reference_artifacts = referenceArtifacts;
    }
    return Object.entries(payload).some(([key, item]) => (
        Array.isArray(item) ? item.length > 0 : item && (typeof item !== 'object' || Object.keys(item).length > 0)
    )) ? payload : null;
}

function buildRequestSnapshot(requestInstance, {
    requestId = '',
    transport = 'auto',
    message = '',
    attachment = null,
    localPath = '',
    conversationId = '',
    responseId = '',
    requestControlFields = null,
    ghostPreview = null,
    requestMeta = null,
    batchPrompts = [],
} = {}) {
    const sessionControls = requestControlFields || buildSessionControlRequestFields(requestInstance);
    const selectedReferenceArtifacts = getSelectedReferenceArtifacts(conversationId);
    const inputArtifacts = buildPendingRequestInputArtifacts({ attachment, localPath });
    return sanitizeRequestSnapshot({
        request_id: requestId,
        created_at: new Date().toISOString(),
        conversation_id: conversationId,
        response_id: responseId,
        transport,
        prompt_text: String(message || '').trim() || null,
        prompt_preview: buildUserPromptPreview(
            message,
            attachment,
            normalizeCapability(requestInstance?.capability || 'chat'),
            localPath
        ),
        target: requestInstance
            ? {
                instance_id: requestInstance.instance_id,
                model: requestInstance.model || requestInstance.modelName || null,
                backend: requestInstance.backend || null,
                capability: requestInstance.capability || null,
            }
            : null,
        attachment: (attachment || localPath)
            ? {
                name: attachment?.name || basenameFromPath(localPath) || null,
                kind: attachment?.name || localPath ? inferFileKindFromName(attachment?.name || localPath) : null,
                local_path: localPath || null,
            }
            : null,
        input_artifacts: inputArtifacts,
        session_controls: sessionControls,
        settings: sanitizeSettingsObject(state.settings),
        request_meta: requestMeta,
        reference_artifacts: selectedReferenceArtifacts,
        ghost_preview: ghostPreview,
        batch_prompts: batchPrompts,
    });
}

function sanitizeSelectedReferenceArtifact(value) {
    if (!value || typeof value !== 'object') return null;
    const type = String(value.type || '').trim().toLowerCase();
    if (!type) return null;
    if (type === 'message') {
        const content = String(value.content || value.text || value.prompt || '').trim();
        const messageId = String(value.message_id || value.messageId || '').trim() || null;
        if (!content && !messageId) return null;
        const messageRole = String(value.message_role || value.messageRole || value.role || 'assistant').trim().toLowerCase() || 'assistant';
        const normalized = {
            type: 'message',
            content: content || null,
            message_role: ['user', 'assistant', 'system'].includes(messageRole) ? messageRole : 'assistant',
            message_id: messageId,
            response_model: String(value.response_model || value.responseModel || '').trim() || null,
            response_instance_id: String(value.response_instance_id || value.responseInstanceId || '').trim() || null,
            timestamp: String(value.timestamp || '').trim() || null,
            artifact_id: String(value.artifact_id || value.artifactId || '').trim() || null,
            artifact_ref: String(value.artifact_ref || value.artifactRef || value.ref || '').trim() || messageId || null,
        };
        return normalized;
    }
    const path = String(value.path || '').trim();
    if (!path) return null;
    const normalizedType = type === 'pdf' ? 'document' : type;
    const normalized = {
        type: normalizedType,
        path,
        artifact_id: String(value.artifact_id || value.artifactId || '').trim() || null,
        artifact_ref: String(value.artifact_ref || value.artifactRef || value.ref || '').trim() || path,
        message_id: String(value.message_id || value.messageId || '').trim() || null,
        name: String(value.name || '').trim() || null,
        kind: String(value.kind || '').trim().toLowerCase() || null,
        origin: String(value.origin || '').trim().toLowerCase() || null,
        source_path: String(value.source_path || value.sourcePath || '').trim() || null,
        mime_type: String(value.mime_type || value.mimeType || '').trim() || null,
        prompt: String(value.prompt || '').trim() || null,
        seed: Number.isFinite(Number(value.seed)) ? Number(value.seed) : null,
        image_state: value.image_state && typeof value.image_state === 'object'
            ? value.image_state
            : null,
    };
    return normalized;
}

function sanitizeSelectedReferenceArtifacts(value) {
    const items = Array.isArray(value) ? value : [value];
    let messageReference = null;
    let artifactReference = null;
    items.forEach((item) => {
        const normalized = sanitizeSelectedReferenceArtifact(item);
        if (!normalized) return;
        if (normalized.type === 'message') {
            messageReference = normalized;
            return;
        }
        artifactReference = normalized;
    });
    return [messageReference, artifactReference].filter(Boolean);
}

function compactRequestSnapshotReferenceArtifacts(value) {
    return sanitizeSelectedReferenceArtifacts(value).map((item, index) => {
        const type = String(item.type || '').trim().toLowerCase() || 'artifact';
        const fallbackRef = `selected_reference:${index + 1}:${type}`;
        if (type === 'message') {
            const payload = {
                type: 'message',
                artifact_id: String(item.artifact_id || '').trim() || null,
                artifact_ref: String(item.artifact_ref || item.message_id || fallbackRef).trim() || fallbackRef,
                message_id: String(item.message_id || '').trim() || null,
                message_role: String(item.message_role || '').trim().toLowerCase() || 'assistant',
                response_model: String(item.response_model || '').trim() || null,
                response_instance_id: String(item.response_instance_id || '').trim() || null,
                timestamp: String(item.timestamp || '').trim() || null,
            };
            if (!payload.message_id) {
                payload.content = String(item.content || '').trim() || null;
            }
            return payload;
        }
        return {
            type,
            artifact_id: String(item.artifact_id || '').trim() || null,
            artifact_ref: String(item.artifact_ref || item.path || item.message_id || fallbackRef).trim() || fallbackRef,
            path: String(item.path || '').trim() || null,
            message_id: String(item.message_id || '').trim() || null,
            name: String(item.name || '').trim() || null,
            kind: String(item.kind || '').trim().toLowerCase() || null,
            mime_type: String(item.mime_type || '').trim() || null,
            seed: Number.isFinite(Number(item.seed)) ? Number(item.seed) : null,
        };
    }).filter((item) => item && Object.values(item).some((value) => value !== null && value !== ''));
}

function ensureSelectedReferenceArtifactStore() {
    if (!state.responsesWorkbench || typeof state.responsesWorkbench !== 'object') {
        return {};
    }
    if (
        !state.responsesWorkbench.selectedReferenceArtifactsByConversation
        || typeof state.responsesWorkbench.selectedReferenceArtifactsByConversation !== 'object'
        || Array.isArray(state.responsesWorkbench.selectedReferenceArtifactsByConversation)
    ) {
        state.responsesWorkbench.selectedReferenceArtifactsByConversation = {};
    }
    const legacySelectedReference = state.responsesWorkbench.selectedReferenceArtifact;
    const activeConversationId = String(getActiveConversationId() || '').trim();
    if (
        legacySelectedReference
        && activeConversationId
        && !state.responsesWorkbench.selectedReferenceArtifactsByConversation[activeConversationId]
    ) {
        state.responsesWorkbench.selectedReferenceArtifactsByConversation[activeConversationId] = legacySelectedReference;
    }
    state.responsesWorkbench.selectedReferenceArtifact = null;
    return state.responsesWorkbench.selectedReferenceArtifactsByConversation;
}

function resolveSelectedReferenceConversationId(conversationId = '') {
    return String(conversationId || getActiveConversationId() || '').trim();
}

function getSelectedReferenceArtifacts(conversationId = '') {
    const targetConversationId = resolveSelectedReferenceConversationId(conversationId);
    if (!targetConversationId) {
        return [];
    }
    const scopedStore = ensureSelectedReferenceArtifactStore();
    return sanitizeSelectedReferenceArtifacts(scopedStore[targetConversationId]);
}

function getSelectedReferenceArtifact(conversationId = '') {
    return getSelectedReferenceArtifacts(conversationId)[0] || null;
}

function findConversationMessageById(conversationId = '', messageId = '') {
    const targetConversationId = String(conversationId || '').trim();
    const targetMessageId = String(messageId || '').trim();
    if (!targetConversationId || !targetMessageId) {
        return null;
    }
    return (state.conversations?.[targetConversationId] || []).find((message) => (
        String(message?.clientMessageId || '').trim() === targetMessageId
    )) || null;
}

function expandSelectedReferenceArtifactForPayload(artifact, conversationId = '') {
    const normalized = sanitizeSelectedReferenceArtifact(artifact);
    if (!normalized) return null;
    if (normalized.type !== 'message' || normalized.content) {
        return normalized;
    }
    const resolvedMessage = findConversationMessageById(conversationId, normalized.message_id);
    if (!resolvedMessage) {
        return normalized;
    }
    const requestSnapshot = sanitizeRequestSnapshot(
        resolvedMessage.requestSnapshot
        || resolvedMessage.request_snapshot
    );
    const resolvedContent = normalized.message_role === 'user'
        ? String(requestSnapshot?.prompt_text || requestSnapshot?.promptText || resolvedMessage.content || '').trim()
        : String(resolvedMessage.content || '').trim();
    if (!resolvedContent) {
        return normalized;
    }
    return sanitizeSelectedReferenceArtifact({
        ...normalized,
        content: resolvedContent,
        timestamp: normalized.timestamp || resolvedMessage.timestamp || null,
        response_model: normalized.response_model || resolvedMessage.responseModel || resolvedMessage.response_model || null,
        response_instance_id: normalized.response_instance_id || resolvedMessage.responseInstanceId || resolvedMessage.response_instance_id || null,
    });
}

function buildSelectedReferenceArtifactPayload(conversationId = '') {
    const selected = getSelectedReferenceArtifacts(conversationId)
        .map((item) => expandSelectedReferenceArtifactForPayload(item, conversationId))
        .filter(Boolean);
    if (!selected.length) return null;
    return selected.length === 1 ? selected[0] : selected;
}

function areSelectedReferenceArtifactsEqual(left, right) {
    return JSON.stringify(sanitizeSelectedReferenceArtifacts(left))
        === JSON.stringify(sanitizeSelectedReferenceArtifacts(right));
}

function formatSelectedReferenceArtifactLabel(artifact) {
    const item = sanitizeSelectedReferenceArtifact(artifact);
    if (!item) return '';
    if (item.type === 'message') {
        const detail = String(item.content || '').trim().replace(/\s+/g, ' ');
        return `message: ${detail.slice(0, 120)}${detail.length > 120 ? '...' : ''}`;
    }
    const typeLabel = item.type === 'image'
        ? 'image'
        : item.type === 'audio'
            ? 'audio'
            : item.type === 'text'
                ? 'text'
                : item.type === 'document'
                    ? 'document'
                : item.type;
    const detail = item.prompt
        || item.name
        || basenameFromPath(item.path)
        || item.source_path
        || item.image_state?.summary
        || item.image_state?.subject
        || item.path;
    return `${typeLabel}: ${String(detail || '').trim()}`;
}

function formatSelectedReferenceArtifactsLabel(artifacts) {
    const items = sanitizeSelectedReferenceArtifacts(artifacts);
    if (!items.length) return '';
    return items.map((item) => formatSelectedReferenceArtifactLabel(item)).filter(Boolean).join(' + ');
}

function renderSelectedReferenceArtifact() {
    const selected = getSelectedReferenceArtifacts();
    const visible = selected.length > 0 && !state.arena.enabled;
    if (elements.referenceArtifactChip) {
        elements.referenceArtifactChip.hidden = !visible;
    }
    if (elements.referenceArtifactLabel) {
        elements.referenceArtifactLabel.textContent = visible
            ? formatSelectedReferenceArtifactsLabel(selected)
            : '';
    }
    if (typeof renderInputUtilityDrawer === 'function') {
        renderInputUtilityDrawer();
    }
}

function rerenderReferenceAwareConversation() {
    const conversationId = String(getActiveConversationId() || '').trim();
    if (!conversationId || !isConversationVisible(conversationId)) {
        return;
    }
    renderConversation(conversationId);
}

function clearSelectedReferenceArtifact({ quiet = false, conversationId = '' } = {}) {
    const targetConversationId = resolveSelectedReferenceConversationId(conversationId);
    const scopedStore = ensureSelectedReferenceArtifactStore();
    if (targetConversationId) {
        delete scopedStore[targetConversationId];
    }
    renderSelectedReferenceArtifact();
    updatePromptPlaceholder();
    rerenderReferenceAwareConversation();
    if (!quiet) {
        updateGlobalModelStatus('Reference cleared. Following the latest context again.');
        setTimeout(() => updateGlobalModelStatus(''), 1800);
    }
}

function isSelectedArtifactReferencePath(path, conversationId = '') {
    const targetPath = String(path || '').trim();
    if (!targetPath) return false;
    return getSelectedReferenceArtifacts(conversationId).some((item) => (
        item.type !== 'message' && String(item.path || '').trim() === targetPath
    ));
}

function removeSelectedReferenceArtifactByPath(path, { quiet = false, conversationId = '' } = {}) {
    const targetPath = String(path || '').trim();
    if (!targetPath) return;
    const targetConversationId = resolveSelectedReferenceConversationId(conversationId);
    const current = getSelectedReferenceArtifacts(targetConversationId);
    const remaining = current.filter((item) => (
        item.type === 'message' || String(item.path || '').trim() !== targetPath
    ));
    if (remaining.length === current.length) {
        return;
    }
    const scopedStore = ensureSelectedReferenceArtifactStore();
    if (targetConversationId) {
        if (remaining.length) {
            scopedStore[targetConversationId] = remaining.length <= 1
                ? (remaining[0] || null)
                : remaining;
        } else {
            delete scopedStore[targetConversationId];
        }
    }
    renderSelectedReferenceArtifact();
    updatePromptPlaceholder();
    rerenderReferenceAwareConversation();
    if (!quiet) {
        updateGlobalModelStatus('Reference updated after artifact removal.');
        setTimeout(() => updateGlobalModelStatus(''), 1800);
    }
}

function setSelectedReferenceArtifact(artifact, { quiet = false, conversationId = '' } = {}) {
    const normalized = sanitizeSelectedReferenceArtifact(artifact);
    if (!normalized) return;
    const targetConversationId = resolveSelectedReferenceConversationId(conversationId);
    const current = getSelectedReferenceArtifacts(targetConversationId);
    const next = normalized.type === 'message'
        ? [
            normalized,
            ...current.filter((item) => item.type !== 'message'),
        ]
        : [
            ...current.filter((item) => item.type === 'message'),
            normalized,
        ];
    const scopedStore = ensureSelectedReferenceArtifactStore();
    if (targetConversationId) {
        scopedStore[targetConversationId] = next.length === 1 ? next[0] : next;
    }
    renderSelectedReferenceArtifact();
    updatePromptPlaceholder();
    rerenderReferenceAwareConversation();
    if (!quiet) {
        updateGlobalModelStatus(
            next.length > 1
                ? `Next request will use ${formatSelectedReferenceArtifactsLabel(next)} as references.`
                : `Next request will use ${formatSelectedReferenceArtifactsLabel(next)} as reference.`
        );
        setTimeout(() => updateGlobalModelStatus(''), 2200);
    }
}

function isSelectedMessageReferenceForMessage(message) {
    if (!message) return false;
    return getSelectedReferenceArtifacts().some((selected) => {
        if (!selected || selected.type !== 'message') return false;
        const selectedMessageId = String(selected.message_id || '').trim();
        const messageId = String(message.clientMessageId || '').trim();
        if (selectedMessageId && messageId) {
            return selectedMessageId === messageId;
        }
        const selectedTimestamp = String(selected.timestamp || '').trim();
        const messageTimestamp = String(message.timestamp || '').trim();
        const selectedContent = String(selected.content || '').trim();
        const messageContent = String(message.content || '').trim();
        const requestSnapshot = sanitizeRequestSnapshot(message.requestSnapshot || message.request_snapshot);
        const requestPromptText = String(requestSnapshot?.prompt_text || requestSnapshot?.promptText || '').trim();
        return Boolean(selectedContent)
            && (selectedContent === messageContent || selectedContent === requestPromptText)
            && selectedTimestamp === messageTimestamp;
    });
}

function getAssistantDisplayContent(payload = {}, preferredText = '') {
    const preferredPreviewText = String(preferredText || '').trim();
    if (
        preferredPreviewText
        && typeof assistantPreviewTextIsWorthPreserving === 'function'
        && assistantPreviewTextIsWorthPreserving(preferredPreviewText)
    ) {
        return preferredPreviewText;
    }
    const imageDataUrl = payload.image_data_url || payload.imageDataUrl || null;
    const savedAudioPath = payload.saved_audio_path || payload.savedAudioPath || null;
    const savedTextPath = payload.saved_text_path || payload.savedTextPath || null;
    const mode = String(payload.mode || '').trim().toLowerCase();
    const artifacts = sanitizeResponseArtifacts(payload.artifacts);
    const outputSlots = extractResponseOutputSlots(payload);
    const outputBranches = extractResponseOutputBranches(payload, { outputSlots });
    const outputs = extractResponseOutputs(payload, { artifacts, outputSlots });
    const lateFill = sanitizeMessageLateFill(payload.late_fill || payload.lateFill || payload?.runtime?.late_fill || payload?.runtime?.lateFill);
    const surfaceState = extractSurfaceState(payload, lateFill);
    const statusSemantics = extractResponseStatusSemantics(payload);
    const lifecycleState = extractResponseLifecycleState(payload, statusSemantics);
    const activeCanonicalWork = lateFillStatusIsActive(lateFill)
        || statusSemantics?.hasOpenContinuation === true
        || lifecycleState === 'late_fill_running'
        || lifecycleState === 'late_fill_pending';
    const autoExecutableRepairWork = lateFillHasAutoExecutableRepairWork(lateFill);
    const actionableRepair = (
        statusSemantics?.hasActionableRepair === true
        && !autoExecutableRepairWork
    ) || lateFillNeedsRepairAttention(lateFill);
    const terminalResponseTruth = responseTruthIndicatesNoOpenWork(lateFill, statusSemantics, lifecycleState);
    const cleanTerminalResolution = responseTruthIndicatesCleanResolution(lateFill, statusSemantics, lifecycleState);
    const staleWorkOptions = {
        terminalResponseTruth,
        cleanTerminalResolution,
        actionableRepair,
    };
    const visibleWorkOptions = {
        keepActionableInternal: actionableRepair && !cleanTerminalResolution,
    };
    const displayOutputSlots = filterUserVisibleResponseWorkItems(
        filterStaleWorkItemsByLateFill(outputSlots, lateFill, staleWorkOptions),
        visibleWorkOptions
    );
    const displayOutputBranches = filterUserVisibleResponseWorkItems(
        filterStaleWorkItemsByLateFill(outputBranches, lateFill, staleWorkOptions),
        visibleWorkOptions
    );
    const displayOutputs = filterUserVisibleResponseWorkItems(
        filterStaleWorkItemsByLateFill(outputs, lateFill, staleWorkOptions),
        visibleWorkOptions
    );
    const imageArtifacts = artifacts.filter((artifact) => artifact.type === 'image');
    const audioArtifacts = artifacts.filter((artifact) => artifact.type === 'audio');
    const textArtifacts = artifacts.filter((artifact) => artifact.type === 'text');
    const fulfilledOutputs = displayOutputs.filter((output) => output.status === 'fulfilled' || output.status === 'completed');
    const pendingOutputs = displayOutputs.filter((output) => output.status === 'pending');
    const fulfilledImageOutputs = fulfilledOutputs.filter((output) => output.type === 'image');
    const fulfilledAudioOutputs = fulfilledOutputs.filter((output) => output.type === 'audio');
    const fulfilledTextOutputs = fulfilledOutputs.filter((output) => output.type === 'text' || output.type === 'document');
    const pendingImageOutputs = pendingOutputs.filter((output) => output.type === 'image');
    const pendingAudioOutputs = pendingOutputs.filter((output) => output.type === 'audio');
    const fulfilledOutputSlots = displayOutputSlots.filter((slot) => slot.status === 'fulfilled');
    const pendingOutputSlots = displayOutputSlots.filter((slot) => slot.status === 'pending');
    const fulfilledImageSlots = fulfilledOutputSlots.filter((slot) => slot.type === 'image');
    const fulfilledAudioSlots = fulfilledOutputSlots.filter((slot) => slot.type === 'audio');
    const fulfilledTextLikeSlots = fulfilledOutputSlots.filter((slot) => slot.type === 'text' || slot.type === 'document');
    const pendingImageSlots = pendingOutputSlots.filter((slot) => slot.type === 'image');
    const pendingAudioSlots = pendingOutputSlots.filter((slot) => slot.type === 'audio');
    const renderableTextOutput = fulfilledTextOutputs.find((output) => (
        String(output.value || '').trim()
        && shouldRenderAssistantOutputText(output, displayOutputs, lateFill)
    ));
    const outputArtifactContext = displayOutputs.some((output) => assistantOutputHasArtifactRecord(output))
        || displayOutputSlots.some((slot) => String(slot.artifact_ref || slot.ref || '').trim());
    const fallbackText = extractAssistantResponseText(payload, preferredText);
    const suppressFallbackArtifactDump = (
        (artifacts.length > 0 || outputArtifactContext)
        && assistantOutputTextHasInternalMarker(fallbackText)
    );
    const explicitText = String(renderableTextOutput?.value || '').trim()
        || (suppressFallbackArtifactDump ? '' : fallbackText);
    if (explicitText) return explicitText;
    const activeWorkItems = mergeCurrentWorkItemsWithLateFill(
        [...displayOutputBranches, ...displayOutputSlots, ...displayOutputs],
        lateFill,
        staleWorkOptions
    ).filter((item) => {
        if (normalizeCurrentWorkItemPhase(item)) return true;
        const status = String(item?.status || item?.lifecycle || '').trim().toLowerCase();
        return lateFillStatusIsActive(lateFill) && autoExecutableRepairWork && status === 'repair_needed';
    });
    const activeWorkSummary = formatCurrentWorkStatusLinesFromItems(
        activeWorkItems,
        {
            activeLateFill: lateFillStatusIsActive(lateFill),
            autoExecutableRepairWork,
        }
    ).join(' ');
    const surfaceSummary = (activeCanonicalWork || terminalResponseTruth) && !actionableRepair
        ? ''
        : formatSurfaceStateLines(surfaceState).join(' ');
    const activeWorkTypes = new Set(activeWorkItems.map((item) => normalizeCurrentWorkItemType(item)));
    if (activeWorkSummary && (
        activeWorkTypes.size > 1
        || (!fulfilledOutputs.length && !imageArtifacts.length && !audioArtifacts.length && !textArtifacts.length)
    )) {
        return activeWorkSummary;
    }
    if (surfaceSummary && !fulfilledOutputs.length && !imageArtifacts.length && !audioArtifacts.length && !textArtifacts.length) {
        return surfaceSummary;
    }
    if (fulfilledImageOutputs.length > 1 || fulfilledImageSlots.length > 1 || imageArtifacts.length > 1) {
        return `Generated ${Math.max(fulfilledImageOutputs.length, fulfilledImageSlots.length, imageArtifacts.length)} images.`;
    }
    if (imageDataUrl) {
        if (mode === 'image_generation_edit' || Number(payload.reference_image_count || 0) > 0) {
            return 'Image generated from the reference image.';
        }
        return 'Image generated.';
    }
    if (fulfilledImageSlots.length === 1) {
        return 'Image generated.';
    }
    if (!imageDataUrl && mode === 'image_generation') {
        return 'Image request completed, but no inline image was returned. Check Ollama server logs/CLI output.';
    }
    if (!imageDataUrl && mode === 'image_generation_edit') {
        return 'Reference-image request completed, but no inline image was returned. Check Ollama server logs/CLI output.';
    }
    if (pendingImageOutputs.length > 1 || pendingImageSlots.length > 1) {
        return `Generating ${Math.max(pendingImageOutputs.length, pendingImageSlots.length)} images...`;
    }
    if (pendingImageOutputs.length === 1 || pendingImageSlots.length === 1) {
        return 'Image generation in progress...';
    }
    if (fulfilledAudioOutputs.length > 1 || fulfilledAudioSlots.length > 1) {
        return `Generated ${Math.max(fulfilledAudioOutputs.length, fulfilledAudioSlots.length)} audio outputs.`;
    }
    if ((savedAudioPath || audioArtifacts.length) && mode === 'text_to_speech') {
        return 'Audio generated.';
    }
    if (fulfilledAudioSlots.length === 1) {
        return 'Audio generated.';
    }
    if (pendingAudioOutputs.length > 1 || pendingAudioSlots.length > 1) {
        return `Generating ${Math.max(pendingAudioOutputs.length, pendingAudioSlots.length)} audio outputs...`;
    }
    if (pendingAudioOutputs.length === 1 || pendingAudioSlots.length === 1) {
        return 'Audio generation in progress...';
    }
    if ((fulfilledTextOutputs.length > 1 || fulfilledTextLikeSlots.length > 1) && outputBranches.length > 1) {
        return `Materialized ${Math.max(fulfilledTextOutputs.length, fulfilledTextLikeSlots.length)} text outputs.`;
    }
    if (savedTextPath || textArtifacts.length) {
        if (mode === 'speech_to_text' || String(payload.capability || '').trim().toLowerCase() === 'speech_to_text') {
            return 'Transcript saved.';
        }
        return 'Text artifact saved.';
    }
    if (savedAudioPath || audioArtifacts.length) {
        return 'Audio generated.';
    }
    return 'Received empty response.';
}

function buildAssistantResponseProvenance(payload = {}, fallbackInstance = null) {
    const explicitModel = String(payload.model || payload.modelName || '').trim();
    const explicitBackendRaw = String(payload.backend || '').trim();
    const explicitBackend = explicitBackendRaw ? normalizeBackend(explicitBackendRaw) : '';
    const explicitInstanceId = String(payload.instance_id || payload.instanceId || '').trim();
    const routeSource = String(payload.route_source || '').trim().toLowerCase();
    const routeReason = String(payload.route_reason || payload.reason || '').trim();
    const contextMode = String(payload.context_mode || payload.contextMode || payload?.runtime?.context_strategy?.mode || '').trim().toLowerCase();
    const contextReason = String(payload.context_reason || payload.contextReason || payload?.runtime?.context_strategy?.reason || '').trim();
    const targetInstance = explicitInstanceId
        ? (getInstanceMeta(explicitInstanceId) || fallbackInstance || null)
        : (fallbackInstance || null);
    const explicitRouterInstanceId = String(payload.route_router_instance_id || payload.routeRouterInstanceId || '').trim() || null;
    const explicitRouterModel = String(payload.route_router_model || payload.routeRouterModel || '').trim();
    const routerMeta = explicitRouterInstanceId ? getInstanceMeta(explicitRouterInstanceId) : null;
    return {
        model: explicitModel || String(targetInstance?.model || targetInstance?.modelName || explicitInstanceId || '').trim() || null,
        backend: explicitBackend || normalizeBackend(targetInstance?.backend || '') || null,
        instanceId: explicitInstanceId || String(targetInstance?.instance_id || '').trim() || null,
        routeSource: routeSource || null,
        routeReason: routeReason || null,
        routeRouterInstanceId: explicitRouterInstanceId,
        routeRouterModel: explicitRouterModel || String(routerMeta?.model || routerMeta?.modelName || '').trim() || null,
        routeArtifactRef: String(payload.route_artifact_ref || payload.routeArtifactRef || '').trim() || null,
        routeArtifactPath: String(payload.route_artifact_path || payload.routeArtifactPath || '').trim() || null,
        routeReuseLastArtifact: 'route_reuse_last_artifact' in payload || 'routeReuseLastArtifact' in payload
            ? Boolean(payload.route_reuse_last_artifact ?? payload.routeReuseLastArtifact)
            : null,
        referenceImageCount: Number.isFinite(Number(payload.reference_image_count ?? payload.referenceImageCount))
            ? Number(payload.reference_image_count ?? payload.referenceImageCount)
            : null,
        referenceImageKind: String(payload.reference_image_kind || payload.referenceImageKind || '').trim() || null,
        contextMode: contextMode || null,
        contextReason: contextReason || null,
    };
}

function sanitizeLateFillRepairExecutionState(value) {
    if (!value || typeof value !== 'object') return null;
    const payload = {
        status: String(value.status || '').trim().toLowerCase() || null,
        autoExecute: sanitizeOptionalBoolean(value.auto_execute ?? value.autoExecute),
        repairWorkAvailable: sanitizeOptionalBoolean(value.repair_work_available ?? value.repairWorkAvailable),
        materializationBlocked: sanitizeOptionalBoolean(value.materialization_blocked ?? value.materializationBlocked),
        needsExternalInput: sanitizeOptionalBoolean(value.needs_external_input ?? value.needsExternalInput),
        blockedScope: String(value.blocked_scope || value.blockedScope || '').trim() || null,
        blockedPrerequisite: String(value.blocked_prerequisite || value.blockedPrerequisite || '').trim() || null,
        reason: String(value.reason || '').trim() || null,
    };
    return Object.values(payload).some((entry) => entry !== null && entry !== '') ? payload : null;
}

function sanitizeLateFillRepairContractList(value) {
    if (!Array.isArray(value)) return [];
    return value
        .filter((item) => item && typeof item === 'object')
        .map((item) => ({
            branchId: String(item.branch_id || item.branchId || item.contract_id || item.contractId || '').trim() || null,
            phaseId: String(item.phase_id || item.phaseId || '').trim() || null,
            capability: normalizeCapability(item.capability || item.expected_capability || item.expectedCapability || '') || null,
            outputType: String(item.output_type || item.outputType || item.type || '').trim().toLowerCase() || null,
            status: String(item.status || '').trim().toLowerCase() || null,
            autoExecute: sanitizeOptionalBoolean(item.auto_execute ?? item.autoExecute),
            repairWorkAvailable: sanitizeOptionalBoolean(item.repair_work_available ?? item.repairWorkAvailable),
            materializationBlocked: sanitizeOptionalBoolean(item.materialization_blocked ?? item.materializationBlocked),
            needsExternalInput: sanitizeOptionalBoolean(item.needs_external_input ?? item.needsExternalInput),
            repairAction: String(item.repair_action || item.recovery_action || item.repairAction || item.recoveryAction || '').trim() || null,
        }))
        .filter((item) => Object.values(item).some((entry) => entry !== null && entry !== ''));
}

function sanitizeMessageLateFill(value) {
    if (!value || typeof value !== 'object') {
        return null;
    }
    const normalizeWorkspacePath = (pathValue) => {
        const normalized = String(pathValue || '').trim().replace(/\\/g, '/');
        if (!normalized) return null;
        const marker = '/ollmo/';
        const markerIndex = normalized.lastIndexOf(marker);
        if (markerIndex >= 0) {
            return normalized.slice(markerIndex + marker.length);
        }
        return normalized;
    };
    const linkedArtifactRebinds = Array.isArray(value.linked_artifact_rebinds || value.linkedArtifactRebinds)
        ? (value.linked_artifact_rebinds || value.linkedArtifactRebinds)
            .filter((item) => item && typeof item === 'object')
            .map((item) => {
                const changeCount = Number(item.change_count ?? item.changeCount);
                const rawChanges = Array.isArray(item.changes) ? item.changes : [];
                return {
                    status: String(item.status || '').trim().toLowerCase() || null,
                    targetPath: normalizeWorkspacePath(item.target_path || item.targetPath),
                    targetExtension: String(item.target_extension || item.targetExtension || '').trim().toLowerCase() || null,
                    changeCount: Number.isFinite(changeCount) && changeCount > 0 ? changeCount : null,
                    changes: rawChanges
                        .filter((change) => change && typeof change === 'object')
                        .map((change) => ({
                            kind: String(change.kind || '').trim() || null,
                            from: String(change.from || '').trim() || null,
                            to: String(change.to || '').trim() || null,
                            linkedPath: normalizeWorkspacePath(change.linked_path || change.linkedPath),
                        }))
                        .filter((change) => Object.values(change).some((entry) => entry !== null && entry !== '')),
                };
            })
            .filter((item) => Object.values(item).some((entry) => (
                Array.isArray(entry) ? entry.length > 0 : entry !== null && entry !== ''
            )))
        : [];
    const linkedArtifactRebindChangeCount = linkedArtifactRebinds.reduce(
        (total, item) => total + (Number.isFinite(Number(item.changeCount)) ? Number(item.changeCount) : 0),
        0
    );
    const branchProgress = sanitizeLateFillBranchList(value.branch_progress || value.branchProgress);
    const terminalProgressBranchIds = new Set(
        branchProgress
            .filter((branch) => ['completed', 'fulfilled', 'failed', 'blocked', 'cancelled', 'waived', 'superseded'].includes(String(branch.status || '').trim().toLowerCase()))
            .map((branch) => String(branch.branch_id || branch.branchId || branch.phase_id || branch.phaseId || '').trim())
            .filter(Boolean)
    );
    const branchHasTerminalProgress = (branch) => {
        const branchId = String(branch?.branch_id || branch?.branchId || branch?.phase_id || branch?.phaseId || '').trim();
        return Boolean(branchId && terminalProgressBranchIds.has(branchId));
    };
    const pendingBranches = sanitizeLateFillBranchList(value.pending_branches || value.pendingBranches, 'pending')
        .filter((branch) => !branchHasTerminalProgress(branch));
    const activeBranches = sanitizeLateFillBranchList(
        value.active_branches || value.activeBranches,
        'running',
        { forceStatus: true }
    ).filter((branch) => !branchHasTerminalProgress(branch));
    const completedBranches = sanitizeLateFillBranchList(
        value.completed_branches || value.completedBranches,
        'fulfilled'
    ).map((branch) => {
        const status = String(branch.status || '').trim().toLowerCase();
        if (['fulfilled', 'completed', 'blocked', 'failed', 'cancelled', 'waived', 'superseded'].includes(status)) {
            return branch;
        }
        return { ...branch, status: 'fulfilled' };
    });
    const failedBranches = sanitizeLateFillBranchList(value.failed_branches || value.failedBranches, 'failed');
    const cancelledBranches = sanitizeLateFillBranchList(value.cancelled_branches || value.cancelledBranches, 'cancelled');
    const fillResults = sanitizeLateFillBranchList(
        value.fill_results || value.fillResults,
        'fulfilled',
        { forceStatus: true }
    );
    const payload = {
        status: String(value.status || '').trim().toLowerCase() || null,
        lifecycleState: String(value.lifecycle_state || value.lifecycleState || '').trim().toLowerCase() || null,
        code: String(value.code || '').trim() || null,
        trigger: String(value.trigger || '').trim() || null,
        expectedCapability: normalizeCapability(value.expected_capability || value.expectedCapability || '') || null,
        missingArtifactType: String(value.missing_artifact_type || value.missingArtifactType || '').trim().toLowerCase() || null,
        fillModel: String(value.fill_model || value.fillModel || '').trim() || null,
        fillBackend: normalizeBackend(value.fill_backend || value.fillBackend || '') || null,
        fillInstanceId: String(value.fill_instance_id || value.fillInstanceId || '').trim() || null,
        routeSource: String(value.route_source || value.routeSource || '').trim().toLowerCase() || null,
        routeReason: String(value.route_reason || value.routeReason || '').trim() || null,
        error: String(value.error || '').trim() || null,
        pendingBranches,
        activeBranches,
        completedBranches,
        failedBranches,
        cancelledBranches,
        fillResults,
        branchProgress,
        surfaceState: sanitizeSurfaceState(value.surface_state || value.surfaceState),
        finalMaterializationContractStatus: String(value.final_materialization_contract_status || value.finalMaterializationContractStatus || '').trim().toLowerCase() || null,
        finalMaterializationContractReason: String(value.final_materialization_contract_reason || value.finalMaterializationContractReason || '').trim() || null,
        materializationContractUnmet: value.materialization_contract_unmet === true || value.materializationContractUnmet === true ? true : null,
        materializationContractOpenCheckCount: Number(value.materialization_contract_open_check_count ?? value.materializationContractOpenCheckCount) > 0
            ? Number(value.materialization_contract_open_check_count ?? value.materializationContractOpenCheckCount)
            : null,
        materializationContractOpenChecks: Array.isArray(value.materialization_contract_open_checks || value.materializationContractOpenChecks)
            ? (value.materialization_contract_open_checks || value.materializationContractOpenChecks)
                .filter((item) => item && typeof item === 'object')
                .map((item) => ({
                    status: String(item.status || '').trim().toLowerCase() || null,
                    checkKind: String(item.check_kind || item.checkKind || '').trim() || null,
                    evidence: String(item.evidence || '').trim() || null,
                    role: String(item.role || '').trim() || null,
                    reason: String(item.reason || '').trim() || null,
                    branchId: String(item.branch_id || item.branchId || '').trim() || null,
                    phaseId: String(item.phase_id || item.phaseId || '').trim() || null,
                    textArtifactExtension: String(item.text_artifact_extension || item.textArtifactExtension || '').trim().toLowerCase() || null,
                    textArtifactSourceName: String(item.text_artifact_source_name || item.textArtifactSourceName || '').trim() || null,
                }))
                .filter((item) => Object.values(item).some((entry) => entry !== null && entry !== ''))
            : [],
        linkedArtifactRebindStatus: String(value.linked_artifact_rebind_status || value.linkedArtifactRebindStatus || '').trim().toLowerCase() || null,
        linkedArtifactRebinds,
        linkedArtifactRebindChangeCount,
        repairLoop: sanitizeLateFillRepairExecutionState(value.repair_loop || value.repairLoop),
        reconsiderationRebuild: sanitizeLateFillRepairExecutionState(value.reconsideration_rebuild || value.reconsiderationRebuild),
        repairRebuildContracts: sanitizeLateFillRepairContractList(value.repair_rebuild_contracts || value.repairRebuildContracts),
    };
    return Object.values(payload).some((item) => item !== null && item !== '') ? payload : null;
}

function lateFillHasAppliedLinkedArtifactRebind(lateFill = null) {
    const normalized = sanitizeMessageLateFill(lateFill);
    if (!normalized) return false;
    const status = String(normalized.linkedArtifactRebindStatus || '').trim().toLowerCase();
    return status === 'applied' || normalized.linkedArtifactRebinds.some((item) => String(item.status || '').trim().toLowerCase() === 'applied');
}

function lateFillRepairStateIsAutoExecutable(state = null) {
    if (!state || typeof state !== 'object') return false;
    const status = String(state.status || '').trim().toLowerCase();
    const runnableStatus = !status || ['promoted', 'pending', 'queued', 'running', 'scheduled', 'accepted', 'planned'].includes(status);
    if (!runnableStatus) return false;
    if (state.materializationBlocked === true || state.needsExternalInput === true) return false;
    if (state.autoExecute !== true) return false;
    if (state.repairWorkAvailable === false) return false;
    return true;
}

function lateFillRepairContractIsAutoExecutable(contract = {}, inheritedAutoExecute = false) {
    if (!contract || typeof contract !== 'object') return false;
    const status = String(contract.status || '').trim().toLowerCase();
    const runnableStatus = !status || ['promoted', 'pending', 'queued', 'running', 'scheduled', 'accepted', 'planned'].includes(status);
    if (!runnableStatus) return false;
    if (contract.materializationBlocked === true || contract.needsExternalInput === true) return false;
    if (contract.autoExecute !== true && inheritedAutoExecute !== true) return false;
    if (contract.repairWorkAvailable === false) return false;
    return true;
}

function lateFillHasAutoExecutableRepairWork(lateFill = null) {
    const normalized = sanitizeMessageLateFill(lateFill);
    if (!normalized) return false;
    const activeLateFill = lateFillStatusIsActive(normalized)
        || normalized.lifecycleState === 'late_fill_running'
        || normalized.lifecycleState === 'late_fill_pending';
    if (!activeLateFill) return false;
    const repairLoopAuto = lateFillRepairStateIsAutoExecutable(normalized.repairLoop);
    const reconsiderationAuto = lateFillRepairStateIsAutoExecutable(normalized.reconsiderationRebuild);
    if (repairLoopAuto || reconsiderationAuto) return true;
    const inheritedAutoExecute = normalized.repairLoop?.autoExecute === true
        || normalized.reconsiderationRebuild?.autoExecute === true;
    return (normalized.repairRebuildContracts || []).some((contract) => (
        lateFillRepairContractIsAutoExecutable(contract, inheritedAutoExecute)
    ));
}

function formatLinkedArtifactRebindStatusLine(lateFill = null) {
    if (!lateFillHasAppliedLinkedArtifactRebind(lateFill)) return '';
    const normalized = sanitizeMessageLateFill(lateFill);
    const count = Number(normalized?.linkedArtifactRebindChangeCount || 0);
    if (Number.isFinite(count) && count > 1) {
        return `Unresolved bindings resolved: ${count} links.`;
    }
    return 'Unresolved bindings resolved.';
}

function lateFillNeedsRepairAttention(lateFill = null) {
    const normalized = sanitizeMessageLateFill(lateFill);
    if (!normalized) return false;
    const status = String(normalized.status || '').trim().toLowerCase();
    const lifecycleState = String(normalized.lifecycleState || '').trim().toLowerCase();
    const contractStatus = String(normalized.finalMaterializationContractStatus || '').trim().toLowerCase();
    const autoExecutableRepairWork = lateFillHasAutoExecutableRepairWork(normalized);
    if (['blocked', 'failed', 'partial_failed', 'repair_needed'].includes(status)) return true;
    if (['blocked', 'failed', 'partial_failed', 'repair_needed'].includes(lifecycleState)) return true;
    if (contractStatus === 'unmet' || normalized.materializationContractUnmet === true) return true;
    if (Number(normalized.materializationContractOpenCheckCount || 0) > 0) return true;
    if ((normalized.materializationContractOpenChecks || []).length > 0) return true;
    const attentionBranchStatus = new Set(['blocked', 'failed', 'partial_failed', 'repair_needed']);
    return [
        ...(normalized.pendingBranches || []),
        ...(normalized.failedBranches || []),
        ...(normalized.activeBranches || []),
    ].some((branch) => {
        const branchStatus = String(branch.status || '').trim().toLowerCase();
        if (branchStatus === 'repair_needed' && autoExecutableRepairWork) return false;
        return attentionBranchStatus.has(branchStatus);
    });
}

function formatLateFillRepairNeededStatusLine(lateFill = null) {
    const normalized = sanitizeMessageLateFill(lateFill);
    if (!lateFillNeedsRepairAttention(normalized)) return '';
    const openCheckCount = Number(normalized?.materializationContractOpenCheckCount || 0);
    const checks = normalized?.materializationContractOpenChecks || [];
    const hasSyntaxCheck = checks.some((check) => (
        String(check.checkKind || '').trim() === 'text_artifact_syntax_sanity'
        || String(check.evidence || '').trim() === 'text_artifact_syntax_issue'
        || String(check.role || '').trim() === 'text_artifact_syntax_repair'
    ));
    if (hasSyntaxCheck) {
        return openCheckCount > 1
            ? `Repair needed: syntax checks open (${openCheckCount}).`
            : 'Repair needed: syntax check open.';
    }
    if (openCheckCount > 0) {
        return openCheckCount > 1
            ? `Repair needed: ${openCheckCount} open checks.`
            : 'Repair needed: open check.';
    }
    const status = String(normalized?.status || normalized?.lifecycleState || '').trim().toLowerCase();
    if (status === 'partial_failed') return 'Repair needed: partial failure.';
    if (status === 'failed') return 'Repair needed: failed work.';
    if (status === 'blocked') return 'Repair needed: blocked work.';
    return 'Repair needed.';
}

function isResolvedLinkedArtifactBindingOutput(output = {}, lateFill = null) {
    if (!lateFillHasAppliedLinkedArtifactRebind(lateFill)) return false;
    const value = String(output?.value || output?.content_payload || output?.contentPayload || '').trim().toLowerCase();
    if (!value) return false;
    return value.includes('unresolved linked artifact binding')
        && value.includes('resolved runtime artifacts')
        && value.includes('update only the target text artifact');
}

function assistantOutputHasArtifactRecord(output = {}) {
    if (!output || typeof output !== 'object') return false;
    if (String(output.artifact_ref || output.artifactRef || output.ref || '').trim()) return true;
    return sanitizeResponseArtifacts(output.artifacts).length > 0;
}

function normalizeAssistantOutputTextForComparison(value = '') {
    return String(value || '').replace(/\r\n?/g, '\n').trim();
}

function assistantOutputTextDuplicatesArtifactOutput(value = '', outputs = []) {
    const target = normalizeAssistantOutputTextForComparison(value);
    if (!target) return false;
    return sanitizeResponseOutputs(outputs).some((candidate) => {
        const candidateType = String(candidate.type || '').trim().toLowerCase();
        if (!['text', 'document'].includes(candidateType)) return false;
        if (!assistantOutputHasArtifactRecord(candidate)) return false;
        const candidateValue = normalizeAssistantOutputTextForComparison(
            candidate.value
            || candidate.content_payload
            || candidate.contentPayload
            || ''
        );
        return Boolean(candidateValue) && candidateValue === target;
    });
}

function assistantOutputTextHasInternalMarker(value = '') {
    const normalized = String(value || '').trim().toLowerCase().replace(/\r\n/g, '\n');
    if (!normalized) return false;
    const firstLine = String(normalized.split('\n').find((line) => String(line || '').trim()) || '')
        .trim()
        .replace(/^#{1,6}\s+/, '')
        .replace(/^[*_`~\s]+|[*_`~\s]+$/g, '');
    if (firstLine.startsWith('image generation prompts')) return true;
    return normalized.includes('original user request for bounded intent context')
        || normalized.includes('target text artifact:')
        || normalized.includes('deterministic syntax sanity issues')
        || (
            (normalized.includes('```html') || normalized.includes('```css'))
            && (normalized.includes('index.html') || normalized.includes('styles.css'))
        )
        || (
            normalized.includes('unresolved linked artifact binding')
            && normalized.includes('resolved runtime artifacts')
        );
}

function shouldRenderAssistantOutputText(output = {}, outputs = [], lateFill = null) {
    const sourceOutput = output && typeof output === 'object' ? output : {};
    const normalizedOutput = sanitizeResponseOutputs([sourceOutput])[0] || {};
    const outputType = String(normalizedOutput.type || sourceOutput.type || '').trim().toLowerCase();
    if (!['text', 'document'].includes(outputType)) return false;
    const value = String(
        normalizedOutput.value
        || sourceOutput.value
        || sourceOutput.content_payload
        || sourceOutput.contentPayload
        || ''
    ).trim();
    if (!value) return false;
    if (assistantOutputHasArtifactRecord(normalizedOutput)) return false;
    if (isResolvedLinkedArtifactBindingOutput(normalizedOutput, lateFill)) return false;
    const normalizedOutputs = sanitizeResponseOutputs(outputs);
    const hasArtifactContext = normalizedOutputs.some((candidate) => assistantOutputHasArtifactRecord(candidate));
    if (hasArtifactContext && assistantOutputTextHasInternalMarker(value)) return false;
    if (hasArtifactContext && assistantOutputTextDuplicatesArtifactOutput(value, normalizedOutputs)) return false;
    return true;
}

function sanitizeOptionalBoolean(value) {
    if (value === true || value === false) return value;
    const normalized = String(value ?? '').trim().toLowerCase();
    if (normalized === 'true' || normalized === '1' || normalized === 'yes') return true;
    if (normalized === 'false' || normalized === '0' || normalized === 'no') return false;
    return null;
}

function sanitizeResponseStatusSemantics(value) {
    if (!value || typeof value !== 'object') return null;
    const payload = {
        canonicalLifecycleState: String(value.canonical_lifecycle_state || value.canonicalLifecycleState || '').trim().toLowerCase() || null,
        canonicalStatusField: String(value.canonical_status_field || value.canonicalStatusField || '').trim() || null,
        compatibilityStatus: String(value.compatibility_status || value.compatibilityStatus || '').trim().toLowerCase() || null,
        statusCompatibility: sanitizeOptionalBoolean(value.status_compatibility ?? value.statusCompatibility),
        hasOpenContinuation: sanitizeOptionalBoolean(value.has_open_continuation ?? value.hasOpenContinuation),
        hasActionableRepair: sanitizeOptionalBoolean(value.has_actionable_repair ?? value.hasActionableRepair),
        isTerminal: sanitizeOptionalBoolean(value.is_terminal ?? value.isTerminal ?? value.terminal),
    };
    return Object.values(payload).some((item) => item !== null && item !== '') ? payload : null;
}

function extractResponseStatusSemantics(payload = {}) {
    return sanitizeResponseStatusSemantics(payload.status_semantics || payload.statusSemantics);
}

function extractResponseLifecycleState(payload = {}, statusSemantics = null) {
    return String(
        payload.lifecycle_state
        || payload.lifecycleState
        || statusSemantics?.canonicalLifecycleState
        || ''
    ).trim().toLowerCase() || null;
}

function coerceResponseFrameSequence(value) {
    const numeric = Number(value);
    return Number.isFinite(numeric) && numeric >= 0 ? numeric : null;
}

function extractResponseFrameSequence(payload = {}) {
    const responseFrame = payload?.response_frame && typeof payload.response_frame === 'object'
        ? payload.response_frame
        : {};
    const statusLookup = payload?.status_lookup && typeof payload.status_lookup === 'object'
        ? payload.status_lookup
        : {};
    return coerceResponseFrameSequence(
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
}

function extractResponseFrameId(payload = {}) {
    const responseFrame = payload?.response_frame && typeof payload.response_frame === 'object'
        ? payload.response_frame
        : {};
    const statusLookup = payload?.status_lookup && typeof payload.status_lookup === 'object'
        ? payload.status_lookup
        : {};
    return String(
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
    ).trim() || null;
}

function extractResponseStateVersion(payload = {}) {
    const statusLookup = payload?.status_lookup && typeof payload.status_lookup === 'object'
        ? payload.status_lookup
        : {};
    return String(
        payload.state_version
        || payload.stateVersion
        || statusLookup.state_version
        || statusLookup.stateVersion
        || ''
    ).trim() || null;
}

function extractResponseTruthMetadata(payload = {}) {
    return {
        responseStateVersion: extractResponseStateVersion(payload),
        responseFrameSequence: extractResponseFrameSequence(payload),
        responseFrameId: extractResponseFrameId(payload),
    };
}

function messageHasNewerResponseFrame(message = {}, payload = {}) {
    const currentSequence = coerceResponseFrameSequence(message.responseFrameSequence ?? message.response_frame_sequence);
    const nextSequence = extractResponseFrameSequence(payload);
    return currentSequence !== null && nextSequence !== null && nextSequence < currentSequence;
}

function applyResponseTruthMetadata(message = {}, metadata = {}) {
    if (metadata.responseStateVersion !== null && metadata.responseStateVersion !== undefined) {
        message.responseStateVersion = metadata.responseStateVersion;
    }
    if (metadata.responseFrameSequence !== null && metadata.responseFrameSequence !== undefined) {
        message.responseFrameSequence = metadata.responseFrameSequence;
    }
    if (metadata.responseFrameId !== null && metadata.responseFrameId !== undefined) {
        message.responseFrameId = metadata.responseFrameId;
    }
}

function sanitizeSurfaceStateItem(value) {
    if (!value || typeof value !== 'object') return null;
    const payload = {
        category: String(value.category || '').trim().toLowerCase() || null,
        status: String(value.status || '').trim().toLowerCase() || null,
        checkKind: String(value.check_kind || value.checkKind || '').trim() || null,
        obligationId: String(value.obligation_id || value.obligationId || '').trim() || null,
        taskId: String(value.task_id || value.taskId || value.workload_task_id || value.workloadTaskId || '').trim() || null,
        phaseId: String(value.phase_id || value.phaseId || '').trim() || null,
        branchId: String(value.branch_id || value.branchId || '').trim() || null,
        candidateId: String(value.candidate_id || value.candidateId || '').trim() || null,
        capability: normalizeCapability(value.capability || '') || null,
        outputType: String(value.output_type || value.outputType || '').trim().toLowerCase() || null,
        action: String(value.action || value.recommended_action || value.recommendedAction || '').trim() || null,
        reviewType: String(value.review_type || value.reviewType || '').trim() || null,
        priority: String(value.priority || '').trim() || null,
        reason: String(value.reason || '').trim() || null,
    };
    return Object.values(payload).some((item) => item !== null && item !== '') ? payload : null;
}

function sanitizeSurfaceState(value) {
    if (!value || typeof value !== 'object') return null;
    const rawCounts = value.category_counts || value.categoryCounts || {};
    const categoryCounts = {};
    if (rawCounts && typeof rawCounts === 'object' && !Array.isArray(rawCounts)) {
        Object.entries(rawCounts).forEach(([key, count]) => {
            const normalizedKey = String(key || '').trim().toLowerCase();
            const normalizedCount = Number(count);
            if (normalizedKey && Number.isFinite(normalizedCount)) {
                categoryCounts[normalizedKey] = normalizedCount;
            }
        });
    }
    const activeCategories = Array.isArray(value.active_categories || value.activeCategories)
        ? (value.active_categories || value.activeCategories)
            .map((item) => String(item || '').trim().toLowerCase())
            .filter(Boolean)
        : Object.entries(categoryCounts)
            .filter(([, count]) => Number(count) > 0)
            .map(([category]) => category);
    const items = Array.isArray(value.items)
        ? value.items.map(sanitizeSurfaceStateItem).filter(Boolean)
        : [];
    const payload = {
        status: String(value.status || '').trim().toLowerCase() || null,
        reason: String(value.reason || '').trim() || null,
        authority: String(value.authority || '').trim() || null,
        lateFillStatus: String(value.late_fill_status || value.lateFillStatus || '').trim().toLowerCase() || null,
        categoryCounts,
        activeCategories,
        items,
    };
    return Object.entries(payload).some(([, item]) => {
        if (Array.isArray(item)) return item.length > 0;
        if (item && typeof item === 'object') return Object.keys(item).length > 0;
        return item !== null && item !== '';
    }) ? payload : null;
}

function extractSurfaceState(payload = {}, lateFill = null) {
    const runtime = payload?.runtime && typeof payload.runtime === 'object' ? payload.runtime : {};
    const closureReview = runtime.graph_closure_review || runtime.graphClosureReview || {};
    const runtimeLateFill = runtime.late_fill || runtime.lateFill || {};
    return sanitizeSurfaceState(
        payload.surface_state
        || payload.surfaceState
        || closureReview.surface_state
        || closureReview.surfaceState
        || runtimeLateFill.surface_state
        || runtimeLateFill.surfaceState
        || lateFill?.surfaceState
    );
}

function buildAssistantMessageProvenance(message = {}, conversationId = '') {
    const explicitModel = String(message.responseModel || '').trim();
    const explicitBackendRaw = String(message.responseBackend || '').trim();
    const explicitBackend = explicitBackendRaw ? normalizeBackend(explicitBackendRaw) : '';
    const explicitInstanceId = String(message.responseInstanceId || '').trim();
    const explicitRouteSource = String(message.routeSource || '').trim().toLowerCase();
    const targetMeta = explicitInstanceId
        ? getInstanceMeta(explicitInstanceId)
        : (!isResponsesWorkbenchConversationId(conversationId) ? getConversationInstanceMeta(conversationId) : null);
    const explicitRouterInstanceId = String(message.routeRouterInstanceId || '').trim() || null;
    const routerMeta = explicitRouterInstanceId ? getInstanceMeta(explicitRouterInstanceId) : null;
    return {
        model: explicitModel || String(targetMeta?.model || targetMeta?.modelName || explicitInstanceId || '').trim() || null,
        backend: explicitBackend || normalizeBackend(targetMeta?.backend || '') || null,
        instanceId: explicitInstanceId || String(targetMeta?.instance_id || '').trim() || null,
        routeSource: explicitRouteSource || null,
        routeReason: String(message.routeReason || '').trim() || null,
        routeRouterInstanceId: explicitRouterInstanceId,
        routeRouterModel: String(message.routeRouterModel || '').trim() || String(routerMeta?.model || routerMeta?.modelName || '').trim() || null,
        routeArtifactRef: String(message.routeArtifactRef || '').trim() || null,
        routeArtifactPath: String(message.routeArtifactPath || '').trim() || null,
        routeReuseLastArtifact: message.routeReuseLastArtifact === null || message.routeReuseLastArtifact === undefined
            ? null
            : Boolean(message.routeReuseLastArtifact),
        referenceImageCount: Number.isFinite(Number(message.referenceImageCount))
            ? Number(message.referenceImageCount)
            : null,
        referenceImageKind: String(message.referenceImageKind || '').trim() || null,
        contextMode: String(message.contextMode || '').trim().toLowerCase() || null,
        contextReason: String(message.contextReason || '').trim() || null,
    };
}

function isExternalChatGPTProvenance(provenance = {}) {
    const model = String(provenance.model || '').trim().toLowerCase();
    const backend = normalizeBackend(provenance.backend || '');
    const instanceId = String(provenance.instanceId || '').trim().toLowerCase();
    return (
        model === 'codex:auto'
        || backend === 'codex_cli'
        || instanceId === 'external:codex'
    );
}

function formatAssistantProvenanceText(message = {}, conversationId = '') {
    const provenance = buildAssistantMessageProvenance(message, conversationId);
    if (!provenance.model && !provenance.backend) {
        return '';
    }
    const externalChatGPT = isExternalChatGPTProvenance(provenance);
    const parts = [];
    if (externalChatGPT) {
        parts.push('Answered by ChatGPT through Ollmo (via Codex)');
        parts.push('· automatic model');
    } else {
        if (provenance.model) {
            parts.push(`Answered by ${formatModelDisplayName(provenance.model)}`);
        } else {
            parts.push('Answered locally');
        }
        if (provenance.backend) {
            parts.push(`(${formatBackendLabel(provenance.backend)})`);
        }
    }
    if (provenance.routeSource && !externalChatGPT) {
        if (provenance.routeSource === 'heuristic') {
            if (provenance.routeRouterModel) {
                parts.push(`via heuristic after Ollmo attempt (${formatModelDisplayName(provenance.routeRouterModel)})`);
            } else {
                parts.push('via heuristic');
            }
        } else if (provenance.routeSource === 'self_heal') {
            if (provenance.routeRouterModel) {
                parts.push(`via self-heal after Ollmo attempt (${formatModelDisplayName(provenance.routeRouterModel)})`);
            } else {
                parts.push('via self-heal');
            }
        } else if (provenance.routeSource === 'embedding_tiebreak') {
            if (provenance.routeRouterModel) {
                parts.push(`via embedding tie-break after Ollmo attempt (${formatModelDisplayName(provenance.routeRouterModel)})`);
            } else {
                parts.push('via embedding tie-break');
            }
        } else if (provenance.routeRouterModel) {
            parts.push(`via Ollmo route (${formatModelDisplayName(provenance.routeRouterModel)})`);
        } else {
            parts.push('via Ollmo route');
        }
    }
    if (provenance.contextMode) {
        const contextLabel = provenance.contextMode === 'compressed_history'
            ? 'compressed history'
            : provenance.contextMode === 'bounded_file_context'
                ? 'file context'
                : provenance.contextMode === 'raw_history'
                    ? 'raw history'
                    : provenance.contextMode.replace(/_/g, ' ');
        parts.push(`using ${contextLabel}`);
    }
    return parts.join(' ');
}

function formatAssistantArtifactProvenanceText(message = {}) {
    const lateFill = sanitizeMessageLateFill(message.lateFill || message.late_fill);
    if (!lateFill || lateFill.status !== 'completed') {
        return '';
    }
    let artifactType = lateFill.missingArtifactType;
    if (!artifactType) {
        if (lateFill.expectedCapability === 'text_to_speech') artifactType = 'audio';
        else if (lateFill.expectedCapability === 'image_generation') artifactType = 'image';
        else if (lateFill.expectedCapability === 'speech_to_text') artifactType = 'text';
    }
    if (!artifactType) {
        return '';
    }
    const label = artifactType === 'audio'
        ? 'Audio by'
        : artifactType === 'image'
            ? 'Image by'
            : artifactType === 'text'
                ? 'Text by'
                : 'Artifact by';
    const parts = [];
    const modelLabel = lateFill.fillModel || lateFill.fillInstanceId || '';
    if (modelLabel) {
        parts.push(`${label} ${formatModelDisplayName(modelLabel)}`);
    } else {
        parts.push(`${label} local continuation`);
    }
    if (lateFill.fillBackend) {
        parts.push(`(${formatBackendLabel(lateFill.fillBackend)})`);
    }
    if (lateFill.routeSource === 'late_fill') {
        parts.push('via late fill');
    } else if (lateFill.routeSource) {
        parts.push(`via ${lateFill.routeSource.replace(/_/g, ' ')}`);
    }
    return parts.join(' ');
}

function normalizeCurrentWorkItemType(item = {}) {
    const explicitType = String(item.type || '').trim().toLowerCase();
    if (explicitType === 'image' || explicitType === 'audio' || explicitType === 'text' || explicitType === 'document') {
        return explicitType;
    }
    const capability = normalizeCapability(item.capability || item.follow_up_capability || item.followUpCapability || item.expectedCapability || item.expected_capability || '');
    if (capability === 'image_generation') return 'image';
    if (capability === 'text_to_speech') return 'audio';
    if (capability === 'speech_to_text') return 'text';
    return explicitType || 'artifact';
}

function normalizeCurrentWorkItemPhase(item = {}) {
    const status = String(item.status || item.lifecycle || '').trim().toLowerCase();
    if (status === 'running' || status === 'in_progress' || status === 'processing') return 'running';
    if (status === 'pending' || status === 'queued' || status === 'scheduled' || status === 'accepted' || status === 'planned') return 'queued';
    if (status === 'failed' || status === 'error') return 'failed';
    if (status === 'blocked') return 'blocked';
    if (status === 'cancelled' || status === 'waived' || status === 'superseded') return status;
    return '';
}

function formatCurrentWorkTypeLabel(type = '', count = 1) {
    const normalized = normalizeCurrentWorkItemType({ type });
    const labels = {
        image: ['Image', 'images'],
        audio: ['Audio', 'audio outputs'],
        text: ['Text', 'text outputs'],
        document: ['Document', 'documents'],
        artifact: ['Artifact', 'artifacts'],
    };
    const [singular, plural] = labels[normalized] || ['Artifact', 'artifacts'];
    return count > 1 ? `${count} ${plural}` : singular;
}

function inferLateFillBranchDisplayIndex(branch = {}, fallbackIndex = 0) {
    const direct = Number(branch.batch_index ?? branch.batchIndex ?? branch.queue_index ?? branch.queueIndex);
    if (Number.isFinite(direct) && direct > 0) return direct;
    const identity = String(branch.branch_id || branch.branchId || branch.phase_id || branch.phaseId || '').trim();
    const match = identity.match(/(?:^|[-_])(\d+)(?:$|[-_])/);
    if (match) {
        const parsed = Number(match[1]);
        if (Number.isFinite(parsed) && parsed > 0) return parsed;
    }
    const fallback = Number(fallbackIndex);
    return Number.isFinite(fallback) && fallback > 0 ? fallback : null;
}

function compactBranchControlText(value = '', maxLength = 92) {
    const normalized = String(value || '').replace(/\s+/g, ' ').trim();
    if (!normalized) return '';
    const limit = Number(maxLength);
    if (!Number.isFinite(limit) || limit <= 0 || normalized.length <= limit) return normalized;
    return `${normalized.slice(0, limit - 1).trimEnd()}…`;
}

function branchControlWorkSummary(branch = {}) {
    const candidates = [
        branch.phase_summary,
        branch.phaseSummary,
        branch.objective,
        branch.deliverable,
        branch.semantic_intent,
        branch.semanticIntent,
        branch.artifact_prompt,
        branch.artifactPrompt,
        branch.content_payload,
        branch.contentPayload,
        branch.review_criteria,
        branch.reviewCriteria,
    ];
    for (const candidate of candidates) {
        if (Array.isArray(candidate)) {
            const joined = candidate.map((item) => String(item || '').trim()).filter(Boolean).join(', ');
            if (joined) return compactBranchControlText(joined);
            continue;
        }
        if (candidate && typeof candidate === 'object') {
            const text = String(candidate.summary || candidate.description || candidate.intent || candidate.kind || '').trim();
            if (text) return compactBranchControlText(text);
            continue;
        }
        const text = compactBranchControlText(candidate);
        if (text) return text;
    }
    return '';
}

function branchControlActionDescription(actionLabel, typeLabel, index, branch = {}) {
    const capability = normalizeCapability(branch.capability || branch.follow_up_capability || branch.followUpCapability || '') || '';
    const status = String(branch.status || branch.lifecycle || '').trim().toLowerCase();
    const verb = String(actionLabel || 'Cancel').trim() || 'Cancel';
    const numberedType = `${typeLabel.toLowerCase()}${index ? ` #${index}` : ''}`;
    const summary = branchControlWorkSummary(branch);
    const actionNoun = capability === 'vision_analysis'
        ? 'visual analysis'
        : capability === 'speech_to_text'
            ? 'audio transcription'
            : capability === 'text_to_speech'
                ? 'audio generation'
                : capability === 'image_generation'
                    ? 'image generation'
                    : numberedType;
    const state = status === 'running' ? 'currently running' : status ? status : 'queued';
    if (summary) {
        return `${verb} ${actionNoun}${index ? ` #${index}` : ''}: ${summary} (${state})`;
    }
    return `${verb} ${actionNoun}${index ? ` #${index}` : ''} (${state})`;
}

function describeLateFillBranchControlTarget(branch = {}, options = {}) {
    const actionLabel = String(options.actionLabel || options.action || 'Cancel').trim() || 'Cancel';
    const fallbackIndex = Number(options.fallbackIndex || 0);
    const type = normalizeCurrentWorkItemType(branch);
    const typeLabel = formatCurrentWorkTypeLabel(type, 1);
    const index = inferLateFillBranchDisplayIndex(branch, fallbackIndex);
    const visibleLabel = `${actionLabel} ${typeLabel}${index ? ` #${index}` : ''}`;
    const branchId = String(branch.branch_id || branch.branchId || '').trim();
    const phaseId = String(branch.phase_id || branch.phaseId || '').trim();
    const status = String(branch.status || branch.lifecycle || '').trim().toLowerCase();
    const capability = normalizeCapability(branch.capability || branch.follow_up_capability || branch.followUpCapability || '') || '';
    const dependsOn = Array.isArray(branch.depends_on || branch.dependsOn)
        ? (branch.depends_on || branch.dependsOn).map((item) => String(item || '').trim()).filter(Boolean)
        : [];
    const source = String(branch.content_payload_source || branch.contentPayloadSource || branch.source_name || branch.sourceName || branch.source || '').trim();
    const primary = branchControlActionDescription(actionLabel, typeLabel, index, branch);
    const details = [];
    if (dependsOn.length) details.push(`Waits for: ${dependsOn.join(', ')}`);
    if (branchId) details.push(`Branch: ${branchId}`);
    if (phaseId && phaseId !== branchId) details.push(`Phase: ${phaseId}`);
    if (capability) details.push(`Capability: ${capability.replace(/_/g, ' ')}`);
    if (source) details.push(`Source: ${source.replace(/_/g, ' ')}`);
    const title = details.length ? `${primary}\n${details.join('\n')}` : primary;
    return {
        label: visibleLabel,
        title,
        ariaLabel: title,
    };
}

function lateFillStatusIsActive(lateFill = null) {
    const status = String(lateFill?.status || '').trim().toLowerCase();
    return status === 'pending' || status === 'queued' || status === 'running' || status === 'scheduled' || status === 'accepted';
}

function sanitizeLateFillBranchList(value, status = '', options = {}) {
    if (!Array.isArray(value)) return [];
    const normalizedStatus = String(status || '').trim().toLowerCase();
    const forceStatus = Boolean(options.forceStatus && normalizedStatus);
    return sanitizeResponseOutputBranches(
        value
            .filter((item) => item && typeof item === 'object')
            .map((item) => ({
                ...item,
                status: forceStatus
                    ? normalizedStatus
                    : String(item.status || normalizedStatus || '').trim().toLowerCase() || undefined,
            }))
    );
}

function currentWorkItemIdentity(item = {}) {
    const branchId = String(item.branch_id || item.branchId || '').trim();
    const phaseId = String(item.phase_id || item.phaseId || '').trim();
    const slotId = String(item.slot_id || item.slotId || '').trim();
    const type = normalizeCurrentWorkItemType(item);
    if (branchId) return `branch:${branchId}:${type}`;
    if (phaseId) return `phase:${phaseId}:${type}`;
    if (slotId) return `slot:${slotId}:${type}`;
    return '';
}

function responseWorkItemIsInternalProjection(item = {}) {
    if (!item || typeof item !== 'object') return false;
    const ids = [
        item.slot_id,
        item.slotId,
        item.branch_id,
        item.branchId,
        item.phase_id,
        item.phaseId,
    ].map((value) => String(value || '').trim().toLowerCase()).filter(Boolean);
    if (ids.some((value) => value === 'repair-chat' || value === 'output-repair-chat')) {
        return true;
    }
    const source = String(item.source || item.source_name || item.sourceName || '').trim().toLowerCase();
    if (source.includes('closure_repair') || source.includes('closure-review')) {
        return true;
    }
    const value = String(item.value || item.content_payload || item.contentPayload || '').trim();
    return Boolean(value && assistantOutputTextHasInternalMarker(value) && !assistantOutputHasArtifactRecord(item));
}

function filterUserVisibleResponseWorkItems(items = [], options = {}) {
    const normalizedItems = Array.isArray(items) ? items : [];
    const keepActionableInternal = Boolean(options.keepActionableInternal);
    return normalizedItems.filter((item) => {
        if (!responseWorkItemIsInternalProjection(item)) return true;
        return keepActionableInternal;
    });
}

function lateFillBranchWorkItems(lateFill = null) {
    if (!lateFill || typeof lateFill !== 'object') return [];
    const includeOpenBranches = lateFillStatusIsActive(lateFill);
    return [
        ...(includeOpenBranches ? (lateFill.pendingBranches || []) : []),
        ...(includeOpenBranches ? (lateFill.activeBranches || []) : []),
        ...(lateFill.branchProgress || []),
        ...(lateFill.completedBranches || []),
        ...(lateFill.failedBranches || []),
        ...(lateFill.cancelledBranches || []),
        ...(lateFill.fillResults || []),
    ];
}

function filterStaleWorkItemsByLateFill(items = [], lateFill = null, options = {}) {
    const normalizedItems = Array.isArray(items) ? items : [];
    const branchItems = lateFillBranchWorkItems(lateFill);
    if (!branchItems.length) {
        return suppressOpenWorkItemsForTerminalTruth(normalizedItems, options);
    }
    const closedIdentities = new Set(
        branchItems
            .filter((item) => ['fulfilled', 'completed', 'blocked', 'failed', 'cancelled', 'waived', 'superseded'].includes(String(item.status || '').trim().toLowerCase()))
            .map((item) => currentWorkItemIdentity(item))
            .filter(Boolean)
    );
    if (!closedIdentities.size) {
        return suppressOpenWorkItemsForTerminalTruth(normalizedItems, options);
    }
    const filteredItems = normalizedItems.filter((item) => {
        const identity = currentWorkItemIdentity(item);
        if (!identity || !closedIdentities.has(identity)) return true;
        const phase = normalizeCurrentWorkItemPhase(item);
        return phase !== 'queued' && phase !== 'running';
    });
    return suppressOpenWorkItemsForTerminalTruth(filteredItems, options);
}

function mergeCurrentWorkItemsWithLateFill(items = [], lateFill = null, options = {}) {
    const merged = [];
    const seen = new Set();
    const add = (item) => {
        if (!item || typeof item !== 'object') return;
        const identity = currentWorkItemIdentity(item);
        if (identity && seen.has(identity)) return;
        if (identity) seen.add(identity);
        merged.push(item);
    };
    suppressOpenWorkItemsForTerminalTruth(lateFillBranchWorkItems(lateFill), options).forEach(add);
    filterStaleWorkItemsByLateFill(items, lateFill, options).forEach(add);
    return merged;
}

function formatOpenOutputStatusLine(type = '', count = 1) {
    const label = formatCurrentWorkTypeLabel(type, count);
    if (count > 1) {
        return `${label} still open.`;
    }
    if (type === 'text' || type === 'document') {
        return `${label} output still open.`;
    }
    return `${label} output still open.`;
}

function formatCurrentWorkStatusLinesFromItems(items = [], options = {}) {
    const activeLateFill = Boolean(options.activeLateFill);
    const autoExecutableRepairWork = Boolean(options.autoExecutableRepairWork);
    const groups = new Map();
    items
        .filter((item) => item && typeof item === 'object')
        .forEach((item) => {
            const status = String(item.status || item.lifecycle || '').trim().toLowerCase();
            let phase = normalizeCurrentWorkItemPhase(item);
            if (!phase && activeLateFill && autoExecutableRepairWork && status === 'repair_needed') {
                phase = 'queued';
            }
            if (!phase) return;
            const type = normalizeCurrentWorkItemType(item);
            const detail = phase === 'failed'
                ? String(item.error || item.error_ref?.code || '').trim()
                : phase === 'blocked'
                    ? String(item.blocked_reason || item.blockedReason || '').trim()
                    : phase === 'cancelled'
                        ? String(item.cancel_reason || item.cancelReason || '').trim()
                        : phase === 'waived'
                            ? String(item.waiver_reason || item.waiverReason || '').trim()
                            : phase === 'superseded'
                                ? String(item.supersession_reason || item.supersessionReason || '').trim()
                    : '';
            const identity = String(item.slot_id || item.slotId || item.branch_id || item.branchId || item.phase_id || item.phaseId || '').trim();
            const key = `${phase}:${type}:${detail}`;
            const existing = groups.get(key) || {
                phase,
                type,
                detail,
                count: 0,
                identities: new Set(),
            };
            if (!identity || !existing.identities.has(identity)) {
                existing.count += 1;
                if (identity) existing.identities.add(identity);
            }
            groups.set(key, existing);
        });
    return Array.from(groups.values()).map((group) => {
        const label = formatCurrentWorkTypeLabel(group.type, group.count);
        if (group.phase === 'running') return `${label} still generating...`;
        if (group.phase === 'queued') {
            return activeLateFill
                ? `${label} queued for late fill...`
                : formatOpenOutputStatusLine(group.type, group.count);
        }
        if (group.phase === 'blocked') {
            return group.detail ? `${label} blocked: ${group.detail}` : `${label} blocked.`;
        }
        if (group.phase === 'failed') {
            return group.detail ? `${label} late fill failed: ${group.detail}` : `${label} late fill failed.`;
        }
        if (group.phase === 'cancelled') {
            return group.detail ? `${label} cancelled: ${group.detail}` : `${label} cancelled.`;
        }
        if (group.phase === 'waived') {
            return group.detail ? `${label} waived: ${group.detail}` : `${label} waived.`;
        }
        if (group.phase === 'superseded') {
            return group.detail ? `${label} superseded: ${group.detail}` : `${label} superseded.`;
        }
        return '';
    }).filter(Boolean);
}

function formatLateFillStatusText(lateFill = null) {
    if (!lateFill) return '';
    let artifactType = lateFill.missingArtifactType;
    if (!artifactType) {
        if (lateFill.expectedCapability === 'text_to_speech') artifactType = 'audio';
        else if (lateFill.expectedCapability === 'image_generation') artifactType = 'image';
        else if (lateFill.expectedCapability === 'speech_to_text') artifactType = 'text';
    }
    const artifactLabel = artifactType
        ? `${artifactType.charAt(0).toUpperCase()}${artifactType.slice(1)}`
        : 'Follow-up';
    if (lateFill.status === 'pending') {
        return `${artifactLabel} queued for late fill...`;
    }
    if (lateFill.status === 'running') {
        return `${artifactLabel} still generating...`;
    }
    if (lateFill.status === 'failed') {
        if (lateFill.error) {
            return `${artifactLabel} late fill failed: ${lateFill.error}`;
        }
        return `${artifactLabel} late fill failed.`;
    }
    return '';
}

function formatSurfaceStateCountLine(category = '', count = 0) {
    if (!Number.isFinite(Number(count)) || Number(count) <= 0) return '';
    const normalized = String(category || '').trim().toLowerCase();
    const value = Number(count);
    const labels = {
        blocked: ['Blocked work', 'Blocked work'],
        repair_pending: ['Repair pending', 'Repairs pending'],
        semantic_review_pending: ['Semantic review pending', 'Semantic reviews pending'],
        controlled_attention_advisory: ['Attention signal', 'Attention signals'],
        aspiration_advisory: ['Aspiration signal', 'Aspiration signals'],
        commitment_advisory: ['Commitment signal', 'Commitment signals'],
        repair_advisory: ['Repair signal', 'Repair signals'],
        semantic_review_advisory: ['Semantic review signal', 'Semantic review signals'],
        open: ['Open work', 'Open work'],
        reconsiderable: ['Reconsiderable option', 'Reconsiderable options'],
        waived: ['Waived item', 'Waived items'],
        superseded: ['Superseded item', 'Superseded items'],
        cancelled: ['Cancelled item', 'Cancelled items'],
        late_fill_pending: ['Late-fill branch pending', 'Late-fill branches pending'],
        completed: ['Completed item', 'Completed items'],
    };
    const [singular, plural] = labels[normalized] || [normalized.replace(/_/g, ' '), `${normalized.replace(/_/g, ' ')} items`];
    return `${value > 1 ? plural : singular}: ${value}.`;
}

function formatSurfaceStateLines(surfaceState = null) {
    const statePayload = sanitizeSurfaceState(surfaceState);
    if (!statePayload) return [];
    const counts = statePayload.categoryCounts || {};
    const preferredCategories = [
        'blocked',
        'repair_pending',
        'semantic_review_pending',
        'open',
        'waived',
        'superseded',
        'cancelled',
        'late_fill_pending',
    ];
    const lines = preferredCategories
        .map((category) => formatSurfaceStateCountLine(category, counts[category]))
        .filter(Boolean);
    if (!lines.length && Number(counts.completed || 0) > 0) {
        lines.push(formatSurfaceStateCountLine('completed', counts.completed));
    }
    return Array.from(new Set(lines));
}

function currentWorkItemsAreTerminalFulfilled(items = [], lateFill = null) {
    const normalizedItems = Array.isArray(items) ? items : [];
    const relevantItems = normalizedItems.filter((item) => item && typeof item === 'object');
    const activeLateFill = lateFillStatusIsActive(lateFill);
    if (activeLateFill) return false;
    const lateFillStatus = String(lateFill?.status || '').trim().toLowerCase();
    const terminalLateFill = !lateFill || !lateFillStatus || ['completed', 'skipped', 'cancelled'].includes(lateFillStatus);
    if (!terminalLateFill) return false;
    if (!relevantItems.length) return false;
    return relevantItems.every((item) => {
        const status = String(item.status || item.lifecycle || '').trim().toLowerCase();
        return ['fulfilled', 'completed', 'materialized_output', 'waived', 'superseded', 'cancelled'].includes(status);
    });
}

function responseOutputItemsIndicateCleanResolution(items = []) {
    const normalizedItems = Array.isArray(items) ? items : [];
    const relevantItems = normalizedItems.filter((item) => item && typeof item === 'object');
    if (!relevantItems.length) return false;
    return relevantItems.every((item) => {
        const status = String(item.status || item.lifecycle || '').trim().toLowerCase();
        return ['fulfilled', 'completed', 'materialized_output', 'waived', 'superseded', 'cancelled'].includes(status);
    });
}

function responseMessageOutputContractIndicatesCleanResolution(message = {}) {
    const outputSlots = sanitizeResponseOutputSlots(message.outputSlots || message.output_slots);
    if (outputSlots.length) {
        return responseOutputItemsIndicateCleanResolution(outputSlots);
    }
    const outputs = sanitizeResponseOutputs(message.outputs || message.canonical_outputs || message.canonicalOutputs);
    if (outputs.length) {
        return responseOutputItemsIndicateCleanResolution(outputs);
    }
    const outputBranches = sanitizeResponseOutputBranches(message.outputBranches || message.output_branches);
    return responseOutputItemsIndicateCleanResolution(outputBranches);
}

function responseTruthIndicatesNoOpenWork(lateFill = null, statusSemantics = null, lifecycleState = '') {
    if (statusSemantics?.hasOpenContinuation === true) return false;
    if (lateFillStatusIsActive(lateFill)) return false;
    const normalizedLifecycle = String(
        lifecycleState
        || statusSemantics?.canonicalLifecycleState
        || ''
    ).trim().toLowerCase();
    const terminalLifecycle = ['completed', 'cancelled', 'failed', 'incomplete'].includes(normalizedLifecycle);
    const lateFillStatus = String(lateFill?.status || '').trim().toLowerCase();
    const terminalLateFill = Boolean(lateFillStatus) && ['completed', 'skipped', 'cancelled'].includes(lateFillStatus);
    if (statusSemantics?.isTerminal === true) return true;
    if (statusSemantics?.hasOpenContinuation === false && terminalLifecycle) return true;
    if (terminalLifecycle && statusSemantics?.hasOpenContinuation !== true) return true;
    return terminalLateFill;
}

function responseTruthIndicatesCleanResolution(lateFill = null, statusSemantics = null, lifecycleState = '') {
    if (!responseTruthIndicatesNoOpenWork(lateFill, statusSemantics, lifecycleState)) return false;
    const normalizedLifecycle = String(
        lifecycleState
        || statusSemantics?.canonicalLifecycleState
        || statusSemantics?.compatibilityStatus
        || ''
    ).trim().toLowerCase();
    const compatibilityStatus = String(statusSemantics?.compatibilityStatus || '').trim().toLowerCase();
    if (normalizedLifecycle !== 'completed' && compatibilityStatus !== 'completed') return false;
    const lateFillStatus = String(lateFill?.status || '').trim().toLowerCase();
    if (lateFillStatus && !['completed', 'skipped'].includes(lateFillStatus)) return false;
    if (Array.isArray(lateFill?.failedBranches) && lateFill.failedBranches.length) return false;
    return true;
}

function suppressOpenWorkItemsForTerminalTruth(items = [], options = {}) {
    const normalizedItems = Array.isArray(items) ? items : [];
    if (!options.terminalResponseTruth || options.actionableRepair) {
        return normalizedItems;
    }
    const suppressedPhases = new Set(options.cleanTerminalResolution
        ? ['queued', 'running', 'blocked', 'failed']
        : ['queued', 'running']);
    return normalizedItems.filter((item) => {
        const phase = normalizeCurrentWorkItemPhase(item);
        return !suppressedPhases.has(phase);
    });
}

function formatAssistantCurrentWorkStatusLines(message = {}) {
    const outputSlots = sanitizeResponseOutputSlots(message.outputSlots || message.output_slots);
    const outputs = sanitizeResponseOutputs(message.outputs || message.canonical_outputs || message.canonicalOutputs);
    const outputBranches = sanitizeResponseOutputBranches(message.outputBranches || message.output_branches);
    const lateFill = sanitizeMessageLateFill(message.lateFill || message.late_fill);
    const statusSemantics = sanitizeResponseStatusSemantics(message.statusSemantics || message.status_semantics);
    const lifecycleState = String(
        message.lifecycleState
        || message.lifecycle_state
        || statusSemantics?.canonicalLifecycleState
        || ''
    ).trim().toLowerCase();
    const liveStatusLines = sanitizeAssistantLiveStatusLines(message.liveStatusLines || message.live_status_lines);
    if (
        liveStatusLines.length
        && !responseTruthIndicatesCleanResolution(lateFill, statusSemantics, lifecycleState)
    ) {
        return liveStatusLines;
    }
    const outputContractCleanResolution = responseMessageOutputContractIndicatesCleanResolution({
        outputSlots,
        output_slots: outputSlots,
        outputs,
        outputBranches,
        output_branches: outputBranches,
    });
    const explicitOpenContinuation = statusSemantics?.hasOpenContinuation === true;
    const explicitActionableRepair = statusSemantics?.hasActionableRepair === true;
    const outputTruthSuppressesStaleWork = outputContractCleanResolution
        && !explicitOpenContinuation
        && !explicitActionableRepair;
    const effectiveLateFill = outputTruthSuppressesStaleWork ? null : lateFill;
    const activeCanonicalWork = !outputTruthSuppressesStaleWork && (
        lateFillStatusIsActive(effectiveLateFill)
        || explicitOpenContinuation
        || lifecycleState === 'late_fill_running'
        || lifecycleState === 'late_fill_pending'
    );
    const autoExecutableRepairWork = lateFillHasAutoExecutableRepairWork(lateFill);
    const linkedArtifactRebindLine = outputTruthSuppressesStaleWork ? '' : formatLinkedArtifactRebindStatusLine(lateFill);
    const surfaceState = sanitizeSurfaceState(message.surfaceState || message.surface_state || lateFill?.surfaceState);
    const repairNeededLine = outputTruthSuppressesStaleWork ? '' : formatLateFillRepairNeededStatusLine(lateFill);
    const actionableRepair = (
        explicitActionableRepair
        && !autoExecutableRepairWork
    ) || Boolean(repairNeededLine);
    const terminalResponseTruth = responseTruthIndicatesNoOpenWork(effectiveLateFill, statusSemantics, lifecycleState)
        || outputTruthSuppressesStaleWork;
    const cleanTerminalResolution = responseTruthIndicatesCleanResolution(effectiveLateFill, statusSemantics, lifecycleState)
        || outputTruthSuppressesStaleWork;
    const staleWorkOptions = {
        terminalResponseTruth,
        cleanTerminalResolution,
        actionableRepair,
    };
    const workItems = mergeCurrentWorkItemsWithLateFill(
        [...outputBranches, ...outputSlots, ...outputs],
        effectiveLateFill,
        staleWorkOptions
    );
    const lines = formatCurrentWorkStatusLinesFromItems(
        workItems,
        {
            activeLateFill: activeCanonicalWork,
            autoExecutableRepairWork,
        }
    );
    const hasSpecificRepairLine = lines.some((line) => {
        const normalized = String(line || '').trim().toLowerCase();
        return normalized.includes('blocked')
            || normalized.includes('failed')
            || normalized.includes('partial failure')
            || normalized.includes('syntax')
            || normalized.includes('unmet')
            || normalized.includes('open check');
    });
    const displayedRepairNeededLine = repairNeededLine && !hasSpecificRepairLine
        ? repairNeededLine
        : '';
    const suppressSurfaceLines = (activeCanonicalWork || terminalResponseTruth) && !actionableRepair;
    const surfaceLines = suppressSurfaceLines
        ? []
        : currentWorkItemsAreTerminalFulfilled(workItems, lateFill) && !displayedRepairNeededLine
        ? []
        : formatSurfaceStateLines(surfaceState);
    if (lines.length || displayedRepairNeededLine || surfaceLines.length || linkedArtifactRebindLine) {
        return Array.from(new Set([
            ...lines,
            ...(displayedRepairNeededLine ? [displayedRepairNeededLine] : []),
            ...surfaceLines,
            ...(linkedArtifactRebindLine ? [linkedArtifactRebindLine] : []),
        ]));
    }
    const fallback = formatLateFillStatusText(effectiveLateFill);
    return fallback ? [fallback] : [];
}

function formatAssistantLateFillStatusText(message = {}) {
    return formatAssistantCurrentWorkStatusLines(message).join(' · ');
}

function inferAssistantMessageCapability(message = {}, conversationId = '') {
    const outputs = sanitizeResponseOutputs(message.outputs || message.canonical_outputs || message.canonicalOutputs);
    const imageOutputs = outputs.filter((output) => output.type === 'image');
    const audioOutputs = outputs.filter((output) => output.type === 'audio');
    const textOutputs = outputs.filter((output) => output.type === 'text' || output.type === 'document');
    const outputSlots = sanitizeResponseOutputSlots(message.outputSlots || message.output_slots);
    const imageOutputSlots = outputSlots.filter((slot) => slot.type === 'image');
    const audioOutputSlots = outputSlots.filter((slot) => slot.type === 'audio');
    const textOutputSlots = outputSlots.filter((slot) => slot.type === 'text' || slot.type === 'document');
    const explicitCapability = normalizeCapability(message.responseCapability || '');
    if (explicitCapability) {
        return explicitCapability;
    }
    if (imageOutputs.length) {
        return 'image_generation';
    }
    if (audioOutputs.length) {
        return 'text_to_speech';
    }
    if (textOutputs.length) {
        return 'chat';
    }
    if (imageOutputSlots.length) {
        return 'image_generation';
    }
    if (audioOutputSlots.length) {
        return 'text_to_speech';
    }
    if (textOutputSlots.length) {
        return 'chat';
    }
    const instanceCapability = normalizeCapability(getInstanceMeta(message.responseInstanceId || '')?.capability || '');
    if (instanceCapability) {
        return instanceCapability;
    }
    const artifacts = sanitizeResponseArtifacts(message.artifacts);
    if (message.imageDataUrl || message.savedImagePath || artifacts.some((artifact) => artifact.type === 'image')) {
        return 'image_generation';
    }
    if (message.savedAudioPath || artifacts.some((artifact) => artifact.type === 'audio')) {
        return 'text_to_speech';
    }
    if (message.savedTextPath || artifacts.some((artifact) => artifact.type === 'text')) {
        return 'vision_analysis';
    }
    return normalizeCapability(getConversationHistoryMetadata(conversationId)?.capability || '') || null;
}

function formatAssistantImageDebugText(message = {}, conversationId = '') {
    const capability = inferAssistantMessageCapability(message, conversationId);
    const hasImageArtifact = Boolean(
        message.imageDataUrl
        || message.savedImagePath
        || sanitizeResponseArtifacts(message.artifacts).some((artifact) => artifact.type === 'image')
    );
    if (!hasImageArtifact && capability !== 'image_generation') {
        return '';
    }
    const provenance = buildAssistantMessageProvenance(message, conversationId);
    const parts = [];
    if (provenance.referenceImageCount !== null) {
        const kind = provenance.referenceImageKind ? ` (${provenance.referenceImageKind})` : '';
        parts.push(`reference_count ${provenance.referenceImageCount}${kind}`);
    }
    if (provenance.routeReuseLastArtifact !== null) {
        parts.push(`reuse_last_artifact ${provenance.routeReuseLastArtifact ? 'yes' : 'no'}`);
    }
    if (provenance.routeArtifactRef) {
        parts.push(`artifact ${provenance.routeArtifactRef}`);
    } else if (provenance.routeArtifactPath) {
        parts.push(`artifact ${formatAssistantArtifactPathLabel(provenance.routeArtifactPath)}`);
    }
    return parts.length ? `Image context: ${parts.join(' | ')}` : '';
}

function formatAssistantArtifactPathLabel(pathValue = '') {
    const raw = String(pathValue || '').trim();
    if (!raw) return '';
    const normalized = raw.replace(/\\/g, '/');
    const anchoredMatch = normalized.match(/\/(images|audio|ocr|transcripts|generated)\/.+$/i);
    if (anchoredMatch) {
        return anchoredMatch[0];
    }
    return basenameFromPath(normalized) || normalized;
}


function finalizeLoadingAssistantResponse(instanceId, clientMessageId, payload = {}, preferredText = '', requestSnapshot = null) {
    const provenance = buildAssistantResponseProvenance(payload, getRequestExecutionInstance(getConversationInstanceMeta(instanceId) || {}));
    const outputSlots = extractResponseOutputSlots(payload);
    const outputBranches = extractResponseOutputBranches(payload, { outputSlots });
    const outputs = extractResponseOutputs(payload, {
        artifacts: sanitizeResponseArtifacts(payload.artifacts),
        outputSlots,
    });
    const lateFill = sanitizeMessageLateFill(payload.late_fill || payload.lateFill || payload?.runtime?.late_fill || payload?.runtime?.lateFill);
    const surfaceState = extractSurfaceState(payload, lateFill);
    const statusSemantics = extractResponseStatusSemantics(payload);
    const lifecycleState = extractResponseLifecycleState(payload, statusSemantics);
    const responseTruthMetadata = extractResponseTruthMetadata(payload);
    const finalRequestSnapshot = mergeRequestSnapshotInputArtifacts(requestSnapshot, payload.input_artifacts);
    const requestSelectedReferences = sanitizeSelectedReferenceArtifacts(
        finalRequestSnapshot?.reference_artifacts
        ?? finalRequestSnapshot?.selected_reference_artifacts
        ?? finalRequestSnapshot?.selectedReferenceArtifacts
        ?? finalRequestSnapshot?.selectedReferenceArtifact
    );
    updateLoadingAssistantMessage(
        instanceId,
        getAssistantDisplayContent(payload, preferredText),
        {
            clientMessageId,
            imageDataUrl: payload.image_data_url || payload.imageDataUrl || null,
            savedImagePath: payload.saved_image_path || payload.savedImagePath || null,
            savedAudioPath: payload.saved_audio_path || payload.savedAudioPath || null,
            savedTextPath: payload.saved_text_path || payload.savedTextPath || null,
            artifacts: sanitizeResponseArtifacts(payload.artifacts),
            responseId: payload.id || payload.response_id || null,
            ...responseTruthMetadata,
            responseCapability: normalizeCapability(payload.capability || payload.mode || '') || null,
            responseModel: provenance.model,
            responseBackend: provenance.backend,
            responseInstanceId: provenance.instanceId,
            routeSource: provenance.routeSource,
            routeReason: provenance.routeReason,
            routeRouterInstanceId: provenance.routeRouterInstanceId,
            routeRouterModel: provenance.routeRouterModel,
            routeArtifactRef: provenance.routeArtifactRef,
            routeArtifactPath: provenance.routeArtifactPath,
            routeReuseLastArtifact: provenance.routeReuseLastArtifact,
            referenceImageCount: provenance.referenceImageCount,
            referenceImageKind: provenance.referenceImageKind,
            contextMode: provenance.contextMode,
            contextReason: provenance.contextReason,
            lifecycleState,
            statusSemantics,
            lateFill,
            surfaceState,
            liveStatusLines: [],
            outputs,
            outputSlots,
            outputBranches,
            artifactBundles: Array.isArray(payload.artifact_bundles)
                ? payload.artifact_bundles
                : (Array.isArray(payload.artifactBundles) ? payload.artifactBundles : []),
            requestSnapshot: finalRequestSnapshot,
        },
        true
    );
    if (state.modelSettingsAutoOpened) {
        state.modelSettingsOpen = false;
        state.modelSettingsAutoOpened = false;
        renderModelSettingsPanel();
    }
    if (
        requestSelectedReferences.length
        && areSelectedReferenceArtifactsEqual(getSelectedReferenceArtifacts(instanceId), requestSelectedReferences)
    ) {
        clearSelectedReferenceArtifact({ quiet: true, conversationId: instanceId });
    }
}


function updateAssistantResponseByResponseId(instanceId, responseId, payload = {}, requestSnapshot = null) {
    const targetResponseId = String(responseId || payload.id || payload.response_id || '').trim();
    if (!targetResponseId) {
        return false;
    }
    const conversation = state.conversations[instanceId] || [];
    const provenance = buildAssistantResponseProvenance(payload, getRequestExecutionInstance(getConversationInstanceMeta(instanceId) || {}));
    const outputSlots = extractResponseOutputSlots(payload);
    const outputBranches = extractResponseOutputBranches(payload, { outputSlots });
    const outputs = extractResponseOutputs(payload, {
        artifacts: sanitizeResponseArtifacts(payload.artifacts),
        outputSlots,
    });
    const lateFill = sanitizeMessageLateFill(payload.late_fill || payload.lateFill || payload?.runtime?.late_fill || payload?.runtime?.lateFill);
    const surfaceState = extractSurfaceState(payload, lateFill);
    const statusSemantics = extractResponseStatusSemantics(payload);
    const lifecycleState = extractResponseLifecycleState(payload, statusSemantics);
    const responseTruthMetadata = extractResponseTruthMetadata(payload);
    const finalRequestSnapshot = mergeRequestSnapshotInputArtifacts(requestSnapshot, payload.input_artifacts);
    let changed = false;
    let renderChanged = false;
    let previewChanged = false;
    let persistedChanged = false;
    conversation.forEach((message) => {
        if (String(message?.role || '').trim().toLowerCase() !== 'assistant') return;
        if (String(message?.responseId || message?.response_id || '').trim() !== targetResponseId) return;
        if (messageHasNewerResponseFrame(message, payload)) return;
        const previousContent = String(message.content || '');
        const previousRenderableFingerprint = fingerprintAssistantMessageState(message);
        const previousPersistedFingerprint = fingerprintAssistantMessageState(message, { includeRequestSnapshot: true });
        message.content = getAssistantDisplayContent(payload, message.content || '');
        message.trustedHtml = false;
        message.imageDataUrl = payload.image_data_url || payload.imageDataUrl || null;
        message.savedImagePath = payload.saved_image_path || payload.savedImagePath || null;
        message.savedAudioPath = payload.saved_audio_path || payload.savedAudioPath || null;
        message.savedTextPath = payload.saved_text_path || payload.savedTextPath || null;
        message.artifacts = sanitizeResponseArtifacts(payload.artifacts);
        message.responseId = targetResponseId;
        applyResponseTruthMetadata(message, responseTruthMetadata);
        message.responseCapability = normalizeCapability(payload.capability || payload.mode || '') || null;
        message.responseModel = provenance.model;
        message.responseBackend = provenance.backend;
        message.responseInstanceId = provenance.instanceId;
        message.routeSource = provenance.routeSource;
        message.routeReason = provenance.routeReason;
        message.routeRouterInstanceId = provenance.routeRouterInstanceId;
        message.routeRouterModel = provenance.routeRouterModel;
        message.routeArtifactRef = provenance.routeArtifactRef;
        message.routeArtifactPath = provenance.routeArtifactPath;
        message.routeReuseLastArtifact = provenance.routeReuseLastArtifact;
        message.referenceImageCount = provenance.referenceImageCount;
        message.referenceImageKind = provenance.referenceImageKind;
        message.contextMode = provenance.contextMode;
        message.contextReason = provenance.contextReason;
        message.lifecycleState = lifecycleState;
        message.statusSemantics = statusSemantics;
        message.lateFill = lateFill;
        message.surfaceState = surfaceState;
        message.outputs = outputs;
        message.outputSlots = outputSlots;
        message.outputBranches = outputBranches;
        if (Array.isArray(payload.artifact_bundles) || Array.isArray(payload.artifactBundles)) {
            message.artifactBundles = Array.isArray(payload.artifact_bundles)
                ? payload.artifact_bundles
                : payload.artifactBundles;
            message.artifact_bundles = message.artifactBundles;
            message.artifactBundle = message.artifactBundles[message.artifactBundles.length - 1] || null;
        }
        syncMessageArtifactLedger(message, { requestSnapshot: finalRequestSnapshot });
        const nextRenderableFingerprint = fingerprintAssistantMessageState(message);
        const nextPersistedFingerprint = fingerprintAssistantMessageState(message, { includeRequestSnapshot: true });
        if (nextRenderableFingerprint !== previousRenderableFingerprint) {
            renderChanged = true;
        }
        if (String(message.content || '') !== previousContent) {
            previewChanged = true;
        }
        if (nextPersistedFingerprint !== previousPersistedFingerprint) {
            persistedChanged = true;
        }
        changed = true;
    });
    if (!changed) {
        return false;
    }
    if (!persistedChanged) {
        return false;
    }
    if (previewChanged) {
        renderConversationHistoryList();
    }
    if (renderChanged) {
        if (state.arena.enabled) {
            renderArenaConversations();
        } else if (isConversationVisible(instanceId)) {
            renderConversation(instanceId);
        }
    }
    persistChatHistory(instanceId);
    return true;
}

function updateLoadingAssistantMessageFromResponsePayload(
    instanceId,
    responseId,
    payload = {},
    requestSnapshot = null,
    options = {}
) {
    const targetResponseId = String(responseId || payload.id || payload.response_id || '').trim();
    if (!targetResponseId) {
        return false;
    }
    const provenance = buildAssistantResponseProvenance(
        payload,
        getRequestExecutionInstance(getConversationInstanceMeta(instanceId) || {})
    );
    const outputSlots = extractResponseOutputSlots(payload);
    const outputBranches = extractResponseOutputBranches(payload, { outputSlots });
    const artifacts = sanitizeResponseArtifacts(payload.artifacts);
    const outputs = extractResponseOutputs(payload, { artifacts, outputSlots });
    const lateFill = sanitizeMessageLateFill(payload.late_fill || payload.lateFill || payload?.runtime?.late_fill || payload?.runtime?.lateFill);
    const surfaceState = extractSurfaceState(payload, lateFill);
    const statusSemantics = extractResponseStatusSemantics(payload);
    const lifecycleState = extractResponseLifecycleState(payload, statusSemantics);
    const responseTruthMetadata = extractResponseTruthMetadata(payload);
    const content = options.content !== undefined
        ? String(options.content || '')
        : getAssistantDisplayContent(payload);
    const liveStatusLines = options.liveStatusLines !== undefined || options.live_status_lines !== undefined
        ? sanitizeAssistantLiveStatusLines(options.liveStatusLines || options.live_status_lines)
        : formatAssistantCurrentWorkStatusLines({
            role: 'assistant',
            responseId: targetResponseId,
            lifecycleState,
            statusSemantics,
            lateFill,
            surfaceState,
            outputs,
            outputSlots,
            outputBranches,
        });
    updateLoadingAssistantMessage(
        instanceId,
        content,
        {
            clientMessageId: options.clientMessageId || undefined,
            trustedHtml: Boolean(options.trustedHtml),
            imageDataUrl: payload.image_data_url || payload.imageDataUrl || null,
            savedImagePath: payload.saved_image_path || payload.savedImagePath || null,
            savedAudioPath: payload.saved_audio_path || payload.savedAudioPath || null,
            savedTextPath: payload.saved_text_path || payload.savedTextPath || null,
            artifacts,
            responseId: targetResponseId,
            ...responseTruthMetadata,
            responseCapability: normalizeCapability(payload.capability || payload.mode || '') || null,
            responseModel: provenance.model,
            responseBackend: provenance.backend,
            responseInstanceId: provenance.instanceId,
            routeSource: provenance.routeSource,
            routeReason: provenance.routeReason,
            routeRouterInstanceId: provenance.routeRouterInstanceId,
            routeRouterModel: provenance.routeRouterModel,
            routeArtifactRef: provenance.routeArtifactRef,
            routeArtifactPath: provenance.routeArtifactPath,
            routeReuseLastArtifact: provenance.routeReuseLastArtifact,
            referenceImageCount: provenance.referenceImageCount,
            referenceImageKind: provenance.referenceImageKind,
            contextMode: provenance.contextMode,
            contextReason: provenance.contextReason,
            lifecycleState,
            statusSemantics,
            lateFill,
            surfaceState,
            liveStatusLines,
            outputs,
            outputSlots,
            outputBranches,
            artifactBundles: Array.isArray(payload.artifact_bundles)
                ? payload.artifact_bundles
                : (Array.isArray(payload.artifactBundles) ? payload.artifactBundles : undefined),
            requestSnapshot,
            forceRender: options.forceRender !== false,
        }
    );
    return true;
}


function addAssistantResponse(instanceId, payload = {}) {
    const imageDataUrl = payload.image_data_url || null;
    const savedImagePath = payload.saved_image_path || null;
    const savedAudioPath = payload.saved_audio_path || null;
    const savedTextPath = payload.saved_text_path || null;
    const artifacts = sanitizeResponseArtifacts(payload.artifacts);
    const content = getAssistantDisplayContent(payload);
    const provenance = buildAssistantResponseProvenance(payload, getRequestExecutionInstance(getConversationInstanceMeta(instanceId) || {}));
    const outputSlots = extractResponseOutputSlots(payload);
    const outputBranches = extractResponseOutputBranches(payload, { outputSlots });
    const outputs = extractResponseOutputs(payload, { artifacts, outputSlots });
    const lateFill = sanitizeMessageLateFill(payload.late_fill || payload.lateFill || payload?.runtime?.late_fill || payload?.runtime?.lateFill);
    const surfaceState = extractSurfaceState(payload, lateFill);
    const statusSemantics = extractResponseStatusSemantics(payload);
    const lifecycleState = extractResponseLifecycleState(payload, statusSemantics);
    const responseTruthMetadata = extractResponseTruthMetadata(payload);
    addMessageToConversation(
        instanceId,
        'assistant',
        content,
        false,
        {
            imageDataUrl,
            savedImagePath,
            savedAudioPath,
            savedTextPath,
            artifacts,
            responseId: payload.id || payload.response_id || null,
            ...responseTruthMetadata,
            responseModel: provenance.model,
            responseBackend: provenance.backend,
            responseInstanceId: provenance.instanceId,
            routeSource: provenance.routeSource,
            routeReason: provenance.routeReason,
            routeRouterInstanceId: provenance.routeRouterInstanceId,
            routeRouterModel: provenance.routeRouterModel,
            routeArtifactRef: provenance.routeArtifactRef,
            routeArtifactPath: provenance.routeArtifactPath,
            routeReuseLastArtifact: provenance.routeReuseLastArtifact,
            referenceImageCount: provenance.referenceImageCount,
            referenceImageKind: provenance.referenceImageKind,
            contextMode: provenance.contextMode,
            contextReason: provenance.contextReason,
            lifecycleState,
            statusSemantics,
            lateFill,
            surfaceState,
            outputs,
            outputSlots,
            outputBranches,
            artifactBundles: Array.isArray(payload.artifact_bundles)
                ? payload.artifact_bundles
                : (Array.isArray(payload.artifactBundles) ? payload.artifactBundles : []),
        }
    );
}

// Request send lifecycle helpers extracted to static/ui/request-lifecycle.js.

function addMessageToConversation(instanceId, role, content, isLoading = false, extras = {}) {
    if (!state.conversations[instanceId]) {
        state.conversations[instanceId] = [];
    }
    const clientMessageId = extras.clientMessageId || `${Date.now()}-${Math.random().toString(16).slice(2)}`;
    const timestamp = String(extras.timestamp || '').trim() || new Date().toISOString();
    state.conversations[instanceId].push({
        clientMessageId,
        role, content,
        timestamp,
        isLoading,
        trustedHtml: Boolean(extras.trustedHtml),
        ephemeralUiNotice: Boolean(extras.ephemeralUiNotice),
        imageDataUrl: extras.imageDataUrl || null,
        savedImagePath: extras.savedImagePath || null,
        savedAudioPath: extras.savedAudioPath || null,
        savedTextPath: extras.savedTextPath || null,
        responseId: extras.responseId || null,
        responseStateVersion: extras.responseStateVersion || extras.response_state_version || null,
        responseFrameSequence: coerceResponseFrameSequence(extras.responseFrameSequence ?? extras.response_frame_sequence),
        responseFrameId: extras.responseFrameId || extras.response_frame_id || null,
        responseCapability: extras.responseCapability || null,
        responseModel: extras.responseModel || null,
        responseBackend: extras.responseBackend || null,
        responseInstanceId: extras.responseInstanceId || null,
        routeSource: extras.routeSource || null,
        routeReason: extras.routeReason || null,
        routeRouterInstanceId: extras.routeRouterInstanceId || null,
        routeRouterModel: extras.routeRouterModel || null,
        routeArtifactRef: extras.routeArtifactRef || null,
        routeArtifactPath: extras.routeArtifactPath || null,
        routeReuseLastArtifact: extras.routeReuseLastArtifact !== undefined ? Boolean(extras.routeReuseLastArtifact) : null,
        referenceImageCount: Number.isFinite(Number(extras.referenceImageCount)) ? Number(extras.referenceImageCount) : null,
        referenceImageKind: extras.referenceImageKind || null,
        contextMode: extras.contextMode || null,
        contextReason: extras.contextReason || null,
        lifecycleState: String(extras.lifecycleState || extras.lifecycle_state || '').trim().toLowerCase() || null,
        statusSemantics: sanitizeResponseStatusSemantics(extras.statusSemantics || extras.status_semantics),
        lateFill: sanitizeMessageLateFill(extras.lateFill),
        surfaceState: sanitizeSurfaceState(extras.surfaceState || extras.surface_state),
        liveStatusLines: sanitizeAssistantLiveStatusLines(extras.liveStatusLines || extras.live_status_lines),
        live_status_lines: sanitizeAssistantLiveStatusLines(extras.liveStatusLines || extras.live_status_lines),
        artifacts: sanitizeResponseArtifacts(extras.artifacts),
        outputs: sanitizeResponseOutputs(extras.outputs),
        outputSlots: sanitizeResponseOutputSlots(extras.outputSlots || extras.output_slots),
        outputBranches: sanitizeResponseOutputBranches(extras.outputBranches || extras.output_branches),
        artifactBundle: extras.artifactBundle || null,
        artifactBundles: Array.isArray(extras.artifactBundles)
            ? extras.artifactBundles
            : (Array.isArray(extras.artifact_bundles) ? extras.artifact_bundles : []),
        artifact_bundles: Array.isArray(extras.artifact_bundles)
            ? extras.artifact_bundles
            : (Array.isArray(extras.artifactBundles) ? extras.artifactBundles : []),
        requestSnapshot: sanitizeRequestSnapshot(extras.requestSnapshot),
    });
    syncMessageArtifactLedger(state.conversations[instanceId][state.conversations[instanceId].length - 1]);
    renderConversationHistoryList();
    if (state.arena.enabled) {
        renderArenaConversations();
        return;
    }
    if (isConversationVisible(instanceId)) {
        activateConversationBottomAnchor(instanceId, isLoading ? 2200 : 1400);
        renderConversation(instanceId);
    }
    if (!isLoading && !extras.ephemeralUiNotice) {
        persistChatHistory(instanceId);
    }
    return clientMessageId;
}

function updateMessageRequestSnapshot(instanceId, clientMessageId, requestSnapshot) {
    const conversation = state.conversations[instanceId] || [];
    const targetId = String(clientMessageId || '').trim();
    if (!targetId) return;
    const nextSnapshot = sanitizeRequestSnapshot(requestSnapshot);
    let changed = false;
    conversation.forEach((message) => {
        if (String(message?.clientMessageId || '').trim() !== targetId) return;
        syncMessageArtifactLedger(message, { requestSnapshot: nextSnapshot });
        changed = true;
    });
    if (!changed) return;
    if (isConversationVisible(instanceId)) {
        renderConversation(instanceId);
    }
    persistChatHistory(instanceId);
}
