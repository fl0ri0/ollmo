function requestTransportTargetIsExternal(instance) {
    return Boolean(
        instance
        && (
            String(instance.target_kind || '').trim().toLowerCase() === 'external'
            || String(instance.instance_id || '').trim() === 'external:codex'
        )
    );
}

function getExternalResponseAttachments(requestContext = null, attachment = null) {
    const attachments = Array.isArray(requestContext?.externalAttachments)
        ? requestContext.externalAttachments.filter(Boolean)
        : [];
    if (!attachments.length && attachment) {
        attachments.push(attachment);
    }
    return attachments;
}

function getExternalResponseLocalPaths(requestContext = null, localPath = '') {
    const localPaths = Array.isArray(requestContext?.externalLocalPaths)
        ? requestContext.externalLocalPaths
            .map((item) => String(item || '').trim())
            .filter(Boolean)
        : [];
    const fallbackPath = String(localPath || '').trim();
    if (!localPaths.length && fallbackPath) {
        localPaths.push(fallbackPath);
    }
    return localPaths;
}

function buildUserPromptPreview(message, attachment, capability = 'chat', localPath = '') {
    const text = String(message || '').trim();
    if (!attachment && !localPath) {
        return text;
    }
    const displayName = attachment?.name || basenameFromPath(localPath) || localPath;
    const kind = inferFileKindFromName(displayName);
    let suffix = localPath ? `[Local file: ${displayName}]` : `[Attachment: ${displayName}]`;
    if (normalizeCapability(capability) === 'speech_to_text') {
        suffix = `[Audio: ${displayName}]`;
    } else if (normalizeCapability(capability) === 'image_generation' && kind === 'image') {
        suffix = `[Reference image: ${displayName}]`;
    } else if (kind === 'image') {
        suffix = `[Image: ${displayName}]`;
    } else if (kind === 'text') {
        suffix = `[Text file: ${displayName}]`;
    } else if (kind === 'pdf') {
        suffix = `[PDF: ${displayName}]`;
    }
    return text ? `${text}\n${suffix}` : suffix;
}

function getInferTimeoutForRequest(instance, attachment = null, localPath = '') {
    const requestInstance = getRequestExecutionInstance(instance);
    const capability = normalizeCapability(requestInstance?.capability || 'chat');
    const modelName = String(requestInstance?.model || requestInstance?.modelName || '').toLowerCase();
    const sourceName = attachment?.name || localPath || '';
    const fileKind = inferFileKindFromName(sourceName);
    if (fileKind === 'pdf') {
        return Number(state.inference.inferTimeoutMs || (60 * 60 * 1000));
    }
    if (
        capability === 'vision_analysis' &&
        modelName.includes('deepseek-ocr') &&
        (fileKind === 'image' || fileKind === 'binary' || !fileKind)
    ) {
        return Number(state.inference.deepseekImageTimeoutMs || (3 * 60 * 1000));
    }
    return Number(state.inference.inferTimeoutMs || (60 * 60 * 1000));
}

function buildSessionControlRequestFields(instance) {
    const schema = getSessionControlsSchema(instance);
    const schemaFields = schema?.fields || {};
    const fields = {};
    Object.entries(SESSION_CONTROL_BINDINGS).forEach(([fieldKey, binding]) => {
        const field = schemaFields[fieldKey];
        if (!fieldIsVisible(field) || !binding.requestKey) {
            return;
        }
        const rawValue = state.settings[binding.stateKey];
        let transformed = typeof binding.transform === 'function'
            ? binding.transform(rawValue, instance, schemaFields[fieldKey])
            : rawValue;
        const options = Array.isArray(field?.options)
            ? field.options.map((item) => String(item || '').trim()).filter(Boolean)
            : [];
        const normalizedTransformed = typeof transformed === 'string'
            ? transformed.trim()
            : transformed;
        const transformedMatchesOption = typeof normalizedTransformed === 'string'
            ? options.some((item) => item === normalizedTransformed)
            : false;
        if (fieldKey === 'reasoning_effort') {
            const explicit = typeof hasExplicitReasoningEffortPreference === 'function'
                ? hasExplicitReasoningEffortPreference(state.settings)
                : Object.prototype.hasOwnProperty.call(state.settings || {}, 'reasoningEffortExplicit')
                    ? state.settings.reasoningEffortExplicit === true
                    : Object.prototype.hasOwnProperty.call(state.settings || {}, 'reasoningEffort');
            if (!explicit) {
                transformed = typeof getReasoningEffortDefaultValue === 'function'
                    ? getReasoningEffortDefaultValue(field, options)
                    : (options.includes('medium') ? 'medium' : (options.find((item) => item !== 'off') || 'off'));
            } else if (!transformedMatchesOption) {
                return;
            }
            const resolvedReasoningEffort = typeof transformed === 'string'
                ? transformed.trim()
                : transformed;
            if (!options.includes(resolvedReasoningEffort)) {
                return;
            }
        }
        if (
            field?.default_first_option &&
            options.length &&
            (
                transformed === null
                || typeof transformed === 'undefined'
                || transformed === ''
                || (typeof normalizedTransformed === 'string' && normalizedTransformed && !transformedMatchesOption)
            )
        ) {
            transformed = options[0];
        }
        if (transformed === null || typeof transformed === 'undefined' || transformed === '') {
            return;
        }
        fields[binding.requestKey] = typeof binding.serialize === 'function'
            ? binding.serialize(transformed, instance, schemaFields[fieldKey])
            : transformed;
    });
    if (
        fieldIsVisible(schemaFields.pdf_max_pages)
        || fieldIsVisible(schemaFields.pdf_dpi)
        || fieldIsVisible(schemaFields.pdf_page_timeout_sec)
        || fieldIsVisible(schemaFields.pdf_synthesize)
    ) {
        fields.reuse_cached = 'false';
    }
    return fields;
}

async function sendViaResponsesTransport(
    instance,
    instanceId,
    conversationId,
    message,
    attachment,
    localPath = '',
    clientMessageId = '',
    requestContext = null,
    responseId = '',
    pendingRequestId = ''
) {
    const isGhostAuto = Boolean(instance?.ghostAuto);
    const requestInstance = requestContext?.requestInstance || getRequestExecutionInstance(instance);
    const capability = normalizeCapability(requestInstance?.capability || 'chat');
    const isExternalTarget = requestTransportTargetIsExternal(requestInstance);
    const usesExternalPluralInputs = isExternalTarget && !isGhostAuto;
    const externalAttachments = usesExternalPluralInputs
        ? getExternalResponseAttachments(requestContext, attachment)
        : [];
    const externalLocalPaths = usesExternalPluralInputs
        ? getExternalResponseLocalPaths(requestContext, localPath)
        : [];
    const hasExternalFileContext = externalAttachments.length > 0 || externalLocalPaths.length > 0;
    const hasFileContext = usesExternalPluralInputs
        ? hasExternalFileContext
        : Boolean(attachment || localPath);
    const batchPrompts = !hasFileContext && (isGhostAuto || capability === 'image_generation')
        ? parseExplicitBatchPrompts(message)
        : [];
    if (capability === 'chat' && !hasFileContext && !batchPrompts.length) {
        const payload = await sendViaResponsesStream(
            instance,
            instanceId,
            conversationId,
            clientMessageId,
            requestContext,
            message,
            responseId,
            pendingRequestId
        );
        return { payload, streamed: true };
    }

    const timeoutMs = Math.max(
        30_000,
        getInferTimeoutForRequest(
            instance,
            externalAttachments[0] || attachment,
            externalLocalPaths[0] || localPath
        )
    );
    const timeoutSec = Math.max(30, Math.ceil(timeoutMs / 1000));
    const clientTimeoutMs = timeoutMs + 15_000;
    const ghostMessages = isGhostAuto
        ? buildGhostRoutingConversationSnapshot(conversationId)
        : [];
    const ghostPreview = isGhostAuto
        ? (requestContext?.ghostPreview || buildGhostExecutionPreviewPayload(undefined, undefined, conversationId))
        : null;
    const ghostPreferences = isGhostAuto ? getResponsesGhostPreferencesPayload() : null;
    const requestMeta = isGhostAuto ? getResponsesGhostRequestMetaPayload() : null;
    const selectedReferenceArtifact = buildSelectedReferenceArtifactPayload(conversationId);
    const sessionFields = requestContext?.requestControlFields || buildSessionControlRequestFields(requestInstance);
    const fields = {
        ...(!isGhostAuto ? { instance_id: instanceId } : {}),
        ...(batchPrompts.length ? {} : (message ? { prompt: message } : {})),
        ...(!isGhostAuto && requestInstance?.model ? { model: requestInstance.model } : {}),
        ...(!isGhostAuto && requestInstance?.backend ? { backend: normalizeBackend(requestInstance.backend) } : {}),
        ...(!isGhostAuto && requestInstance?.capability ? { capability: normalizeCapability(requestInstance.capability) } : {}),
        ...(isGhostAuto ? { ghost_route: 'true', conversation_id: conversationId } : {}),
        ...(responseId ? { response_id: responseId } : {}),
        ...sessionFields,
        infer_timeout_sec: String(timeoutSec),
    };

    if (usesExternalPluralInputs && hasExternalFileContext) {
        const formData = new FormData();
        Object.entries(fields).forEach(([key, value]) => {
            if (value !== undefined && value !== null && value !== '') {
                formData.append(key, value);
            }
        });
        if (selectedReferenceArtifact) {
            formData.append('reference_artifacts', JSON.stringify(selectedReferenceArtifact));
        }
        externalAttachments.forEach((file) => {
            formData.append('files', file, file.name);
        });
        if (externalLocalPaths.length) {
            formData.append('file_paths_json', JSON.stringify(externalLocalPaths));
        }
        const response = await axios.post(
            `${state.flaskServerUrl}/api/responses`,
            formData,
            { timeout: clientTimeoutMs }
        );
        return { payload: response.data || {}, streamed: false };
    }

    if (attachment) {
        const formData = new FormData();
        Object.entries(fields).forEach(([key, value]) => {
            if (value !== undefined && value !== null && value !== '') {
                formData.append(key, value);
            }
        });
        if (isGhostAuto) {
            formData.append('ghost_messages_json', JSON.stringify(ghostMessages));
            if (ghostPreview) {
                formData.append('ghost_preview', JSON.stringify(ghostPreview));
            }
            if (ghostPreferences) {
                formData.append('ghost_preferences', JSON.stringify(ghostPreferences));
            }
            if (requestMeta) {
                formData.append('request_meta', JSON.stringify(requestMeta));
            }
        }
        if (selectedReferenceArtifact) {
            formData.append('reference_artifacts', JSON.stringify(selectedReferenceArtifact));
        }
        if (batchPrompts.length) {
            formData.append('batch_prompts', JSON.stringify(batchPrompts));
        }
        formData.append('file', attachment, attachment.name);
        if (localPath) {
            formData.append('file_path', localPath);
        }
        const response = await axios.post(
            `${state.flaskServerUrl}/api/responses`,
            formData,
            { timeout: clientTimeoutMs }
        );
        return { payload: response.data || {}, streamed: false };
    }

    const payload = {
        ...fields,
        ...(batchPrompts.length ? { batch_prompts: batchPrompts } : {}),
        ...(localPath ? { file_path: localPath } : {}),
        ...(isGhostAuto ? { ghost_messages: ghostMessages } : {}),
        ...(ghostPreview ? { ghost_preview: ghostPreview } : {}),
        ...(ghostPreferences ? { ghost_preferences: ghostPreferences } : {}),
        ...(requestMeta ? { request_meta: requestMeta } : {}),
        ...(selectedReferenceArtifact ? { reference_artifacts: selectedReferenceArtifact } : {}),
    };
    const response = await axios.post(
        `${state.flaskServerUrl}/api/responses`,
        payload,
        { timeout: clientTimeoutMs }
    );
    return { payload: response.data || {}, streamed: false };
}
