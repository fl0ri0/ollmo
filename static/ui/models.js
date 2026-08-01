function updateModelMaintenanceHints() {
    const backend = elements.pullModelBackend?.value || 'ollama';
    if (!elements.pullModelInput) return;
    if (backend === 'mlx') {
        elements.pullModelInput.placeholder = 'org/model-name';
        if (elements.pullModelHint) {
            elements.pullModelHint.textContent = 'enter any Hugging Face repo id to cache and classify for local use (e.g., mlx-community/Qwen3.5-27B-4bit or apple/starflow)';
        }
    } else if (normalizeBackend(backend) === 'llama_cpp') {
        elements.pullModelInput.placeholder = 'org/model-GGUF or /path/to/model.gguf';
        if (elements.pullModelHint) {
            elements.pullModelHint.textContent = 'enter a GGUF Hugging Face repo or a local .gguf path for llama.cpp (e.g., ggml-org/gemma-4-26B-A4B-it-GGUF or ~/Models/llama.cpp/model.gguf)';
        }
    } else {
        elements.pullModelInput.placeholder = 'model:tag';
        if (elements.pullModelHint) {
            elements.pullModelHint.textContent = 'paste names from ollama.com/search (e.g., llama3.1:8b)';
        }
    }
}

async function fetchRunningInstances() {
    try {
        const response = await axios.get(`${state.flaskServerUrl}/api/running_instances`);
        state.runningInstances = response.data || [];
        syncGhostResolvedTargetWithRunningInstances();
        renderActiveInstancesList();
        renderModelTabs();
        renderAvailableModelsList();
        refreshArenaOptions();
        ensureResponsesWorkbenchTarget();
        renderResponsesWorkbenchTargetOptions();

        if (state.activeWorkspace === 'instance') {
            if (!state.currentInstanceId) {
                state.activeWorkspace = 'responses';
            } else if (
                !getInstanceMeta(state.currentInstanceId)
                || !isUserFacingInstance(getInstanceMeta(state.currentInstanceId))
            ) {
                state.currentInstanceId = getDefaultInteractiveInstance()?.instance_id || null;
                if (!state.currentInstanceId) {
                    state.activeWorkspace = 'responses';
                }
            }
        }
        if (isResponsesWorkbenchActive()) {
            const activeConversationId = String(getActiveConversationId() || '').trim();
            const target = ensureResponsesWorkbenchTarget();
            const settingsOwner = getCurrentSettingsOwnerInstance() || target;
            if (settingsOwner) {
                loadSettingsForInstance(settingsOwner);
                refreshTtsSettingOptions(settingsOwner);
            }
            updateActiveTabHighlight();
            updateSessionControlMode();
            updateActiveModelToolbar();
            renderConversationHistoryList();
            if (!activeConversationId || (!state.chatHistoryLoaded[activeConversationId] && getPersistedConversationMessageCount(activeConversationId) === 0)) {
                await switchToResponsesWorkbench({ focusInput: false });
            } else {
                renderConversation(activeConversationId);
                updatePromptPlaceholder();
            }
        } else if (state.currentInstanceId && getInstanceMeta(state.currentInstanceId)) {
            const instance = getInstanceMeta(state.currentInstanceId);
            const activeConversationId = String(getActiveConversationId() || '').trim();
            if (!instance || !activeConversationId || (!state.chatHistoryLoaded[activeConversationId] && getPersistedConversationMessageCount(activeConversationId) === 0)) {
                await switchToInstance(state.currentInstanceId, { focusInput: false });
            } else {
                loadSettingsForInstance(instance);
                updateChatControls();
                updateSessionControlMode();
                refreshTtsSettingOptions(instance);
                updateActiveTabHighlight();
                updateActiveModelToolbar();
                renderResponsesWorkbenchTargetOptions();
                renderConversation(activeConversationId);
                renderConversationHistoryList();
                updatePromptPlaceholder();
            }
        } else if (getDefaultInteractiveInstance()) {
            await switchToInstance(getDefaultInteractiveInstance().instance_id, { focusInput: false });
        } else {
            state.activeWorkspace = 'responses';
            state.responsesWorkbench.targetInstanceId = RESPONSES_GHOST_AUTO_ID;
            clearGhostResolvedTarget();
            state.currentInstanceId = null;
            renderNoModelSelected();
        }
        clearStaleLoadingMessages();
        renderVoiceInputPreferences();
        renderModelSettingsPanel();
        renderConversationHistoryList();
        updateVoiceInputButtonState();
        updateSendButtonState();
    } catch (error) {
        console.error('Error fetching running instances:', error);
        const activeExternalTarget = (
            state.activeWorkspace === 'instance'
            && isExternalConversationTarget(getInstanceMeta(state.currentInstanceId))
        )
            ? getInstanceMeta(state.currentInstanceId)
            : null;
        state.runningInstances = [];
        renderActiveInstancesList();
        renderModelTabs();
        renderAvailableModelsList();
        updateSidebarBadge();
        refreshArenaOptions();
        renderResponsesWorkbenchTargetOptions();
        if (activeExternalTarget) {
            state.activeWorkspace = 'instance';
            state.currentInstanceId = activeExternalTarget.instance_id;
            updateActiveTabHighlight();
            updateChatControls();
            updateSessionControlMode();
            updateActiveModelToolbar();
            const activeConversationId = String(getActiveConversationId() || '').trim();
            if (activeConversationId) {
                renderConversation(activeConversationId);
            }
            updatePromptPlaceholder();
            updateGlobalModelStatus('Local model status is unavailable. ChatGPT remains ready through Codex.');
        } else {
            state.activeWorkspace = 'responses';
            state.responsesWorkbench.targetInstanceId = RESPONSES_GHOST_AUTO_ID;
            clearGhostResolvedTarget();
            state.currentInstanceId = null;
            renderNoModelSelected();
        }
        renderVoiceInputPreferences();
        renderModelSettingsPanel();
        renderConversationHistoryList();
        updateVoiceInputButtonState();
        updateSendButtonState();
    }
}

async function fetchAvailableModels() {
    elements.availableModelsList.innerHTML = '<div class="ledger-placeholder"><span class="spinner"></span> Loading catalog</div>';
    try {
        const response = await axios.get(`${state.flaskServerUrl}/api/available_models`);
        state.availableModels = response.data?.models || [];
        renderAvailableModelsList();
        renderVoiceInputPreferences();
        renderModelSettingsPanel();
        updateVoiceInputButtonState();
    } catch (error) {
        console.error('Error fetching available models:', error);
        const message = error.response?.data?.error || error.message || 'Unknown error';
        elements.availableModelsList.innerHTML = `<div class="ledger-placeholder ledger-error"><i class="fas fa-exclamation-triangle"></i> Failed: ${message}</div>`;
        state.availableModels = [];
        populateRemoveModelSelect();
        renderVoiceInputPreferences();
        renderModelSettingsPanel();
        updateVoiceInputButtonState();
    }
}

function getCatalogCardRunningInstances(modelName, backend) {
    return state.runningInstances.filter((instance) => (
        normalizeBackend(instance.backend) === backend
        && instance.model === modelName
    ));
}

function getCatalogRuntimeHealth(instance) {
    if (!instance || typeof instance !== 'object') {
        return null;
    }
    const runtimeStatus = instance.runtime_status && typeof instance.runtime_status === 'object'
        ? instance.runtime_status
        : {};
    const readiness = String(instance.readiness || runtimeStatus.readiness || '').trim().toLowerCase();
    const port = instance.port || runtimeStatus.port || '';
    const portListening = instance.port_listening ?? runtimeStatus.port_listening;
    const processAlive = instance.process_alive ?? runtimeStatus.process_alive;
    const offlineReadiness = new Set(['failed', 'error', 'unreachable', 'stopped', 'stopping']);
    const hasLiveEvidence = portListening === true || processAlive === true;
    const hardReadinessWithoutLiveEvidence = offlineReadiness.has(readiness) && !hasLiveEvidence;
    if (hardReadinessWithoutLiveEvidence || portListening === false || processAlive === false) {
        return {
            state: 'offline',
            label: port ? `Port ${port} offline` : 'Unavailable',
            meta: port ? `Port ${port} offline` : 'Unavailable',
            icon: '<i class="fas fa-exclamation-triangle"></i>',
        };
    }
    if (readiness === 'degraded') {
        return {
            state: 'advisory',
            className: 'status-pill--advisory',
            label: port ? `Port ${port}` : 'Running',
            meta: port ? `Port ${port} • cached warning` : 'Running • cached warning',
            icon: '<i class="fas fa-info-circle"></i>',
        };
    }
    return null;
}

function getCatalogRuntimePill(instance) {
    const instanceId = instance?.instance_id || '';
    const isStopping = Boolean(instanceId) && state.modelOperations.stopping.has(instanceId);
    const isHelperOnly = !isUserFacingInstance(instance);
    const runtimeHealth = getCatalogRuntimeHealth(instance);
    if (isStopping) {
        return {
            className: 'status-pill--pending',
            icon: '<i class="fas fa-spinner fa-spin"></i>',
            label: 'Stopping',
        };
    }
    if (isHelperOnly) {
        return {
            className: 'status-pill--active',
            icon: '<i class="fas fa-wand-magic-sparkles"></i>',
            label: 'Helper',
        };
    }
    if (runtimeHealth) {
        return {
            className: runtimeHealth.className || 'status-pill--pending',
            icon: runtimeHealth.icon,
            label: runtimeHealth.label,
        };
    }
    if (instance?.port) {
        return {
            className: 'status-pill--active',
            icon: '<i class="fas fa-check"></i>',
            label: `Port ${instance.port}`,
        };
    }
    return {
        className: 'status-pill--active',
        icon: '<i class="fas fa-check"></i>',
        label: 'Running',
    };
}

function getCatalogRuntimeMeta(instance) {
    const runtimeTruth = getInstanceRuntimeTruthSummary(instance);
    const runtimeHealth = getCatalogRuntimeHealth(instance);
    const sessionAlias = formatModelDisplayName(instance?.instance_id) || instance?.instance_id || 'session';
    if (runtimeHealth) {
        const runtimeSuffix = runtimeTruth ? ` • ${runtimeTruth}` : '';
        return `Session ${sessionAlias} • ${runtimeHealth.meta}${runtimeSuffix}`;
    }
    if (runtimeTruth) {
        return runtimeTruth;
    }
    if (!isUserFacingInstance(instance)) {
        return `Session ${sessionAlias} • Helper runtime`;
    }
    if (instance?.port) {
        return `Session ${sessionAlias} • Port ${instance.port}`;
    }
    return `Session ${sessionAlias}`;
}

function buildRuntimeStopButtonMarkup({ instanceId = '', isStopping = false, hasOtherStops = false, extraClass = '' } = {}) {
    const stopTitle = hasOtherStops ? 'Finish current action before stopping' : `Stop ${instanceId}`;
    const classes = [
        'icon-button',
        'icon-button--stop',
        'icon-button--small',
        'model-runtime-stop',
        extraClass,
        isStopping ? 'is-loading' : '',
        hasOtherStops ? 'is-locked' : '',
    ].filter(Boolean).join(' ');
    return `
        <button
            class="${classes}"
            type="button"
            data-instance-id="${instanceId}"
            ${isStopping ? 'disabled' : ''}
            data-locked="${hasOtherStops ? 'true' : 'false'}"
            title="${stopTitle}"
        >
            ${isStopping ? '<span class="spinner spinner--danger"></span>' : '<span class="model-runtime-stop__glyph" aria-hidden="true"></span>'}
        </button>
    `;
}

function buildCatalogRuntimeRowMarkup(instance) {
    const instanceId = instance?.instance_id || '';
    const isStopping = Boolean(instanceId) && state.modelOperations.stopping.has(instanceId);
    const hasOtherStops = state.modelOperations.stopping.size > 0 && !isStopping;
    const sessionAlias = formatModelDisplayName(instanceId) || instanceId || 'Session';
    const helperBadge = !isUserFacingInstance(instance)
        ? '<span class="badge-soft badge-inline model-instance-badge">Helper</span>'
        : '';
    const runtimeHealth = getCatalogRuntimeHealth(instance);
    const runtimeTruth = getInstanceRuntimeTruthSummary(instance);
    const inlineMeta = !isUserFacingInstance(instance)
        ? 'Helper runtime'
        : runtimeHealth
            ? runtimeHealth.meta
        : instance?.port
            ? `Port ${instance.port}`
            : (runtimeTruth || 'Running');

    return `
        <div class="model-instance-row" data-instance-id="${instanceId}">
            <div class="model-instance-main">
                <div class="model-instance-title">
                    <span class="truncate">${sessionAlias}</span>
                    ${helperBadge}
                    <span class="model-instance-inline-meta">• ${inlineMeta}</span>
                </div>
            </div>
            ${buildRuntimeStopButtonMarkup({
                instanceId,
                isStopping,
                hasOtherStops,
                extraClass: 'model-instance-stop',
            })}
        </div>
    `;
}

function renderAvailableModelsList() {
    if (typeof state.availableModels === 'undefined') {
        return;
    }
    if (!state.availableModels || state.availableModels.length === 0) {
        updateAvailableModelsToggle(0);
        elements.availableModelsList.innerHTML = '<div class="ledger-placeholder">No local models found</div>';
        populateRemoveModelSelect();
        return;
    }
    const normalizedEntries = state.availableModels.map((modelEntry) => (
        typeof modelEntry === 'string' ? { name: modelEntry } : (modelEntry || {})
    ));
    const hiddenEntries = normalizedEntries.filter((entry) => (
        entry.runnable === false
        && getCatalogCardRunningInstances(entry.name || '', normalizeBackend(entry.backend)).length === 0
    ));
    updateAvailableModelsToggle(hiddenEntries.length);
    const visibleEntries = state.showKnownNonRunnableModels
        ? normalizedEntries
        : normalizedEntries.filter((entry) => (
            entry.runnable !== false
            || getCatalogCardRunningInstances(entry.name || '', normalizeBackend(entry.backend)).length > 0
        ));
    elements.availableModelsList.innerHTML = '';
    if (!visibleEntries.length) {
        const hiddenCount = hiddenEntries.length;
        const hiddenText = hiddenCount > 0
            ? ` ${hiddenCount} known cache entr${hiddenCount === 1 ? 'y is' : 'ies are'} hidden behind the broken-link toggle.`
            : '';
        elements.availableModelsList.innerHTML = `<div class="ledger-placeholder">No runnable local checkpoints.${hiddenText}</div>`;
        populateRemoveModelSelect();
        return;
    }
    visibleEntries.forEach((entry) => {
        const modelName = entry.name || 'Unknown model';
        const displayName = formatModelDisplayName(modelName);
        const backend = normalizeBackend(entry.backend);
        const runnable = entry.runnable !== false;
        const opKey = makeModelKey(modelName, backend);
        const isStarting = state.modelOperations.starting.has(opKey);
        const backendBadgeClass = 'badge-soft badge-muted';
        const backendBadgeInline = `${backendBadgeClass} badge-inline`;
        const backendLabel = formatBackendLabel(backend);
        const sourceLabel = getAvailableModelSourceLabel(entry);
        const truthSummary = getAvailableModelTruthSummary(entry);
        const secondLine = entry.disabled_reason || truthSummary || entry.description || `Source: ${sourceLabel}`;
        const runningInstances = getCatalogCardRunningInstances(modelName, backend);
        const runningCount = runningInstances.length;
        const singleRunningInstance = runningCount === 1 ? runningInstances[0] : null;
        const hasMultipleRunning = runningCount > 1;
        const hasOtherStarts = state.modelOperations.starting.size > 0 && !isStarting;
        const startDisabled = isStarting || !runnable;
        const startIcon = isStarting ? 'fa-spinner fa-spin' : 'fa-play';
        const buttonTitle = !runnable
            ? (entry.disabled_reason || `Model ${displayName} is not runnable via the current ${backendLabel} path.`)
            : hasOtherStarts
                ? 'Finish current action before launching'
                : `Start new ${displayName} instance`;
        let statusLabel = 'Idle';
        let statusIcon = '<i class="fas fa-circle"></i>';
        let statusClass = 'status-pill--pending';
        let runtimeSummary = '';
        if (singleRunningInstance) {
            const runtimePill = getCatalogRuntimePill(singleRunningInstance);
            statusLabel = runtimePill.label;
            statusIcon = runtimePill.icon;
            statusClass = runtimePill.className;
            runtimeSummary = getCatalogRuntimeMeta(singleRunningInstance);
        } else if (hasMultipleRunning) {
            statusLabel = `${runningCount} running`;
            statusIcon = '<i class="fas fa-layer-group"></i>';
            statusClass = 'status-pill--active';
            runtimeSummary = `${runningCount} live sessions listed below.`;
        }
        const singleIsStopping = Boolean(singleRunningInstance?.instance_id) && state.modelOperations.stopping.has(singleRunningInstance.instance_id);
        const singleHasOtherStops = state.modelOperations.stopping.size > 0 && !singleIsStopping;
        const item = document.createElement('div');
        item.className = 'model-card';
        item.innerHTML = `
            <div class="model-row">
                <div class="model-meta" title="${modelName}">
                    <div class="model-meta-title">
                        <span class="truncate">${displayName}</span>
                    </div>
                    <div class="model-meta-sub model-meta-sub--inline">
                        <span class="${backendBadgeInline}">${backendLabel}</span>
                        <span class="model-meta-note">${secondLine}</span>
                    </div>
                    ${runtimeSummary ? `<div class="model-runtime-summary">${runtimeSummary}</div>` : ''}
                </div>
                <div class="model-actions">
                    <span class="status-pill ${statusClass}">
                        ${statusIcon} ${statusLabel}
                    </span>
                    ${singleRunningInstance ? `
                        ${buildRuntimeStopButtonMarkup({
                            instanceId: singleRunningInstance.instance_id,
                            isStopping: singleIsStopping,
                            hasOtherStops: singleHasOtherStops,
                            extraClass: 'model-card__stop-single',
                        })}
                    ` : ''}
                    <button
                        class="icon-button icon-button--launch icon-button--small ${isStarting ? 'is-loading' : ''} ${hasOtherStarts ? 'is-locked' : ''}"
                        type="button"
                        data-model="${modelName}"
                        data-backend="${backend}"
                        ${startDisabled ? 'disabled' : ''}
                        data-locked="${hasOtherStarts ? 'true' : 'false'}"
                        title="${buttonTitle}"
                    >
                        <i class="fas ${startIcon}"></i>
                    </button>
                </div>
            </div>
            ${hasMultipleRunning ? `
                <div class="model-instance-list">
                    ${runningInstances.map(instance => buildCatalogRuntimeRowMarkup(instance)).join('')}
                </div>
            ` : ''}
        `;

        const startBtn = item.querySelector('.icon-button--launch');
        startBtn.addEventListener('click', (event) => {
            event.stopPropagation();
            if (!runnable) {
                updateGlobalModelStatus(entry.disabled_reason || `Model ${displayName} is known but not startable via the current backend path.`);
                return;
            }
            if (startBtn.dataset.locked === 'true') {
                updateGlobalModelStatus('Finish the current start before launching another.');
                return;
            }
            startModel(modelName, backend, {
                capability: entry.capability || null,
                forceStart: runningCount > 0,
                modelName: entry.modelName || modelName,
                modelPath: entry.model_path || null,
                hfFile: entry.hf_file || null,
                launchDefaults: entry.launch_defaults || null,
                preferredPort: entry.preferred_port || null,
            });
        });
        item.querySelectorAll('.model-runtime-stop').forEach((stopBtn) => {
            stopBtn.addEventListener('click', (event) => {
                event.stopPropagation();
                if (stopBtn.dataset.locked === 'true') {
                    updateGlobalModelStatus('Finish the current stop before issuing another.');
                    return;
                }
                stopModel(stopBtn.dataset.instanceId);
            });
        });
        elements.availableModelsList.appendChild(item);
    });
    populateRemoveModelSelect();
}

function updateAvailableModelsToggle(hiddenCount) {
    if (!elements.availableModelsToggle || !elements.availableModelsToggleLabel) {
        return;
    }
    if (!hiddenCount) {
        elements.availableModelsToggle.hidden = true;
        elements.availableModelsToggle.setAttribute('aria-pressed', 'false');
        elements.availableModelsToggleLabel.textContent = '0';
        state.showKnownNonRunnableModels = false;
        return;
    }
    elements.availableModelsToggle.hidden = false;
    elements.availableModelsToggle.setAttribute('aria-pressed', state.showKnownNonRunnableModels ? 'true' : 'false');
    elements.availableModelsToggle.setAttribute(
        'aria-label',
        state.showKnownNonRunnableModels
            ? `Hide ${hiddenCount} known but not runnable cache entr${hiddenCount === 1 ? 'y' : 'ies'}`
            : `Show ${hiddenCount} known but not runnable cache entr${hiddenCount === 1 ? 'y' : 'ies'}`
    );
    elements.availableModelsToggle.title = state.showKnownNonRunnableModels
        ? `Hide ${hiddenCount} known but not runnable cache entr${hiddenCount === 1 ? 'y' : 'ies'}`
        : `Show ${hiddenCount} known but not runnable cache entr${hiddenCount === 1 ? 'y' : 'ies'}`;
    elements.availableModelsToggleLabel.textContent = String(hiddenCount);
}

function populateRemoveModelSelect() {
    if (!elements.removeModelSelect) return;
    const select = elements.removeModelSelect;
    select.innerHTML = '';
    const removableModels = (state.availableModels || []).map(modelEntry => (
        typeof modelEntry === 'string' ? { name: modelEntry, backend: 'ollama' } : (modelEntry || {})
    ));
    if (removableModels.length === 0) {
        const option = document.createElement('option');
        option.value = '';
        option.textContent = 'No removable models';
        select.appendChild(option);
        select.disabled = true;
        return;
    }
    select.disabled = false;
    const placeholder = document.createElement('option');
    placeholder.value = '';
    placeholder.textContent = 'Choose model';
    placeholder.disabled = true;
    placeholder.selected = true;
    select.appendChild(placeholder);
    const seenKeys = new Set();
    removableModels.forEach(modelEntry => {
        const entry = modelEntry || {};
        const modelName = entry.name || 'unknown';
        const backend = normalizeBackend(entry.backend || 'ollama');
        const removalKey = [
            backend,
            String(entry.model_source || ''),
            String(entry.hf_repo || ''),
            String(entry.hf_file || ''),
            String(entry.model_path || ''),
            String(modelName),
        ].join('::');
        if (seenKeys.has(removalKey)) {
            return;
        }
        seenKeys.add(removalKey);
        const source = backend === 'mlx'
            ? (entry.model_source || 'huggingface')
            : backend === 'llama_cpp'
                ? (entry.model_source || 'llama.cpp')
                : 'ollama';
        const runnable = entry.runnable !== false;
        const option = document.createElement('option');
        option.value = modelName;
        option.dataset.backend = backend;
        option.dataset.modelSource = String(entry.model_source || '').trim();
        option.dataset.modelPath = String(entry.model_path || '').trim();
        option.dataset.hfRepo = String(entry.hf_repo || '').trim();
        option.dataset.hfFile = String(entry.hf_file || '').trim();
        option.textContent = backend === 'mlx'
            ? `${modelName} • ${runnable ? 'HF/MLX' : 'HF cache'}`
            : backend === 'llama_cpp'
                ? `${modelName} • llama.cpp`
                : `${modelName} • Ollama`;
        option.title = entry.disabled_reason
            ? `${modelName} (${source}) — ${entry.disabled_reason}`
            : `${modelName} (${source})`;
        select.appendChild(option);
    });
}

function renderActiveInstancesList() {
    updateSidebarBadge();
    updateStopAllButtonState();
}

function renderModelTabs() {
    elements.modelTabs.innerHTML = '';
    createResponsesWorkbenchTab();
    getUserFacingRunningInstances().forEach(instance => {
        const status = state.modelOperations.stopping.has(instance.instance_id) ? 'pending' : 'active';
        createModelTab(instance.instance_id, instance.model, instance.backend, status);
    });
    getSelectableExternalTargets().forEach((target) => {
        createExternalTargetTab(target);
    });
    updateActiveTabHighlight();
    updateChatControls();
    renderResponsesWorkbenchTargetOptions();
}

async function startModel(modelName, backend = 'ollama', options = {}) {
    const normalizedBackend = normalizeBackend(backend);
    const opKey = makeModelKey(modelName, normalizedBackend);
    const backendLabel = formatBackendLabel(normalizedBackend);
    if (state.modelOperations.starting.has(opKey)) {
        return null;
    }
    if (state.modelOperations.starting.size > 0) {
        updateGlobalModelStatus('Waiting for the current start to finish.');
        return null;
    }
    const previousInstanceIds = new Set((state.runningInstances || []).map((instance) => instance.instance_id).filter(Boolean));
    state.modelOperations.starting.add(opKey);
    renderAvailableModelsList();
    updateGlobalModelStatus(`Starting ${modelName} (${backendLabel})...`);
    try {
        const payload = {
            model: modelName,
            backend: normalizedBackend,
            start_source: options.startSource || 'frontend_button',
        };
        if (options.forceStart) payload.force_start = true;
        if (options.modelName) payload.modelName = options.modelName;
        if (options.capability) payload.capability = options.capability;
        if (normalizedBackend === 'mlx' || normalizedBackend === 'llama_cpp') {
            if (options.modelPath) payload.model_path = options.modelPath;
            if (options.preferredPort) payload.preferred_port = options.preferredPort;
            if (options.hfFile) payload.hf_file = options.hfFile;
            if (options.launchDefaults) payload.launch_defaults = options.launchDefaults;
        }
        const response = await axios.post(`${state.flaskServerUrl}/api/start_model`, payload);
        await fetchRunningInstances();
        const normalizedCapability = normalizeCapability(options.capability || '');
        const startedInstance = (
            state.runningInstances.find((instance) => (
                instance?.instance_id
                && !previousInstanceIds.has(instance.instance_id)
                && normalizeBackend(instance.backend) === normalizedBackend
                && String(instance.model || instance.modelName || '').trim() === String(modelName || '').trim()
                && (!normalizedCapability || frontendInstanceSupportsCapability(instance, normalizedCapability))
            ))
            || state.runningInstances.find((instance) => (
                normalizeBackend(instance?.backend) === normalizedBackend
                && String(instance?.model || instance?.modelName || '').trim() === String(modelName || '').trim()
                && (!normalizedCapability || frontendInstanceSupportsCapability(instance, normalizedCapability))
            ))
            || response?.data?.instance
            || null
        );
        updateGlobalModelStatus(`Started ${modelName} (${backendLabel})`);
        setTimeout(() => updateGlobalModelStatus(''), 2500);
        return startedInstance;
    } catch (error) {
        console.error('Error starting model:', error);
        const message = error.response?.data?.error || error.message || 'Unknown error';
        updateGlobalModelStatus(`Failed to start ${modelName} (${backendLabel}): ${message}`);
        if (!options.suppressAlert) {
            alert(`Failed to start ${modelName} (${backendLabel}): ${message}`);
        }
        return null;
    } finally {
        state.modelOperations.starting.delete(opKey);
        renderAvailableModelsList();
        try {
            await fetchAvailableModels();
        } catch (err) {
            console.error('Error refreshing available models after start:', err);
        }
    }
}

async function stopModel(instanceId) {
    if (state.modelOperations.stopping.has(instanceId)) {
        return;
    }
    state.modelOperations.stopping.add(instanceId);
    renderAvailableModelsList();
    renderActiveInstancesList();
    renderModelTabs();
    updateGlobalModelStatus(`Stopping ${instanceId}...`);
    try {
        const response = await axios.post(`${state.flaskServerUrl}/api/stop_model`, { instance_id: instanceId });
        const stopState = response?.data?.status || 'stopped';
        const stopMessage = response?.data?.message || `Stop request sent for ${instanceId}`;
        const stopDetails = summarizeStopDetails(response?.data?.details);
        const initialStatus = stopDetails ? `${stopMessage} (${stopDetails})` : stopMessage;
        updateGlobalModelStatus(initialStatus);

        const removed = await waitForInstanceRemoval(instanceId, { timeoutMs: 25000, intervalMs: 1200 });
        if (state.currentInstanceId === instanceId && removed) {
            state.currentInstanceId = null;
        }
        if (!removed) {
            await fetchRunningInstances();
        }
        if (removed) {
            updateGlobalModelStatus(`Stopped ${instanceId}`);
        } else if (stopState === 'stopping') {
            updateGlobalModelStatus(`${stopMessage} (still waiting on shutdown)`);
        } else {
            updateGlobalModelStatus(`Stop request accepted but ${instanceId} is still running. Run ./stop_multi_models.sh if it persists.`);
        }
        setTimeout(() => updateGlobalModelStatus(''), 3000);
    } catch (error) {
        console.error('Error stopping model:', error);
        const serverMessage = error.response?.data?.message || error.response?.data?.error || error.message || 'Unknown error';
        const stopDetails = summarizeStopDetails(error.response?.data?.details);
        const combinedMessage = stopDetails ? `${serverMessage} (${stopDetails})` : serverMessage;
        const removed = await waitForInstanceRemoval(instanceId, { timeoutMs: 30000, intervalMs: 1500 });
        if (removed) {
            updateGlobalModelStatus(`Stopped ${instanceId}`);
            setTimeout(() => updateGlobalModelStatus(''), 2500);
        } else {
            updateGlobalModelStatus(`Failed to stop ${instanceId}: ${combinedMessage}`);
        }
    } finally {
        state.modelOperations.stopping.delete(instanceId);
        renderAvailableModelsList();
        renderActiveInstancesList();
        renderModelTabs();
        updateActiveTabHighlight();
        updateChatControls();
        updateStopAllButtonState();
        try {
            await fetchAvailableModels();
        } catch (err) {
            console.error('Error refreshing available models after stop:', err);
        }
        if (!state.currentInstanceId && state.runningInstances.length === 0) {
            renderNoModelSelected();
        }
        updateSendButtonState();
    }
}

async function stopAllModels() {
    if (state.stopAllInProgress) {
        return;
    }
    const targets = [...new Set(state.runningInstances.map(inst => inst.instance_id).filter(Boolean))];
    if (targets.length === 0) {
        updateGlobalModelStatus('No running models to stop.');
        setTimeout(() => updateGlobalModelStatus(''), 2000);
        return;
    }
    state.stopAllInProgress = true;
    updateStopAllButtonState();
    try {
        for (const instanceId of targets) {
            if (!instanceId) continue;
            if (!state.runningInstances.some(inst => inst.instance_id === instanceId)) {
                continue;
            }
            await stopModel(instanceId);
        }
        state.stopAllAwaitingGhostSettle = true;
        updateGlobalModelStatus('All models stopped.');
        setTimeout(() => updateGlobalModelStatus(''), 2500);
    } finally {
        state.stopAllInProgress = false;
        updateStopAllButtonState();
        updateActiveTabHighlight();
    }
}

async function stopCurrentModel() {
    const instanceId = state.currentInstanceId;
    if (!instanceId) {
        updateGlobalModelStatus('Select a model before stopping.');
        setTimeout(() => updateGlobalModelStatus(''), 2000);
        return;
    }
    if (isExternalConversationTarget(getInstanceMeta(instanceId))) {
        updateGlobalModelStatus('ChatGPT has no local process to stop. Turn off the cloud connection instead.');
        return;
    }
    await stopModel(instanceId);
}

function createResponsesWorkbenchTab() {
    const tab = document.createElement('button');
    tab.type = 'button';
    tab.className = 'model-tab model-tab--responses';
    tab.dataset.workspace = 'responses';
    tab.innerHTML = `
        <span class="model-tab__indicator"></span>
        <span class="model-tab__label" title="Route requests through Ollmo">
            Ollmo
        </span>
    `;
    if (isResponsesWorkbenchActive()) {
        tab.classList.add('model-tab--active');
    }
    elements.modelTabs.appendChild(tab);
    tab.addEventListener('click', () => {
        void switchToResponsesWorkbench();
    });
}

function createModelTab(instanceId, modelName, backend = 'ollama', status = 'active') {
    const tab = document.createElement('button');
    tab.type = 'button';
    tab.className = 'model-tab';
    tab.dataset.instanceId = instanceId;
    const displayName = formatTabLabel(instanceId, modelName);
    tab.innerHTML = `
        <span class="model-tab__indicator ${status === 'pending' ? 'model-tab__indicator--pending' : ''}"></span>
        <span class="model-tab__label" title="${modelName || instanceId}">
            ${displayName}
        </span>
    `;
    if (instanceId === state.currentInstanceId) {
        tab.classList.add('model-tab--active');
    }
    elements.modelTabs.appendChild(tab);
    tab.addEventListener('click', () => {
        void switchToInstance(instanceId);
    });
}

function createExternalTargetTab(target) {
    const instanceId = String(target?.instance_id || '').trim();
    if (!instanceId) return;
    const tab = document.createElement('button');
    tab.type = 'button';
    tab.className = 'model-tab model-tab--external';
    tab.dataset.instanceId = instanceId;
    tab.dataset.targetKind = 'external';
    tab.innerHTML = `
        <span class="model-tab__indicator"></span>
        <span class="model-tab__label" title="Optional ChatGPT provider via Codex; prompts and explicitly selected files can be sent; each turn is independent; exact model variant is not exposed">
            ChatGPT
        </span>
        <span class="model-tab__kind">Cloud</span>
    `;
    if (!isResponsesWorkbenchActive() && instanceId === state.currentInstanceId) {
        tab.classList.add('model-tab--active');
    }
    elements.modelTabs.appendChild(tab);
    tab.addEventListener('click', () => {
        void switchToInstance(instanceId);
    });
}

async function switchToResponsesWorkbench({
    focusInput = true,
    historyConversationId = '',
} = {}) {
    persistSettingsForCurrentInstance();
    if (state.arena.enabled) {
        await setArenaEnabled(false);
    }
    state.activeWorkspace = 'responses';
    const target = ensureResponsesWorkbenchTarget();
    const settingsOwner = getCurrentSettingsOwnerInstance() || target;
    if (settingsOwner) {
        loadSettingsForInstance(settingsOwner);
        refreshTtsSettingOptions(settingsOwner);
    } else {
        loadSettingsForInstance(null);
        refreshTtsSettingOptions(null);
    }
    clearConversationViewOverride('responses', 'responses-workbench');
    const conversationId = setResponsesWorkbenchConversationId(getResponsesWorkbenchConversationId(), {
        workspace: 'responses',
        slot_id: 'responses-workbench',
        label: 'responses-workbench',
    });
    if (!state.conversations[conversationId]) {
        state.conversations[conversationId] = [];
    }
    updateActiveTabHighlight();
    updateChatControls();
    updateSessionControlMode();
    updateActiveModelToolbar();
    renderResponsesWorkbenchTargetOptions();
    showConversationLoadingPlaceholder('Ollmo');
    updatePromptPlaceholder();
    updateVoiceInputButtonState();
    updateSendButtonState();
    const hasLocalMessages = getPersistedConversationMessageCount(conversationId) > 0;
    const recoveredConversationId = await recoverResponsesWorkbenchConversationSlot({
        expectedConversationId: conversationId,
        preservePopulatedLocalState: hasLocalMessages,
        suppressVisibleRender: true,
    });
    let conversationIdToRender = getResponsesWorkbenchConversationId();
    const requestedHistoryConversationId = String(historyConversationId || '').trim();
    if (requestedHistoryConversationId) {
        setConversationViewOverride('responses', 'responses-workbench', requestedHistoryConversationId);
        if (!state.chatHistoryLoaded[requestedHistoryConversationId]) {
            await fetchChatHistory(requestedHistoryConversationId, { suppressVisibleRender: true });
        }
        conversationIdToRender = requestedHistoryConversationId;
    } else {
        clearConversationViewOverride('responses', 'responses-workbench');
        if (!state.chatHistoryLoaded[conversationIdToRender] && recoveredConversationId === conversationIdToRender) {
            await fetchChatHistory(conversationIdToRender, { suppressVisibleRender: true });
        }
    }
    renderConversation(conversationIdToRender);
    renderConversationHistoryList();
    if (isResponsesWorkbenchAutoTarget()) {
        updateGlobalModelStatus(buildResponsesAutoStatusText());
    } else if (target) {
        const backendLabel = formatBackendLabel(target.backend);
        const displayModel = target.model || target.instance_id;
        updateGlobalModelStatus(`Ollmo ready for ${displayModel} (${backendLabel})`);
    } else {
        updateGlobalModelStatus('Ollmo needs a running target model.');
    }
    if (focusInput) {
        elements.userInput.focus();
    }
}

async function switchToInstance(instanceId, {
    focusInput = true,
    historyConversationId = '',
} = {}) {
    const instance = getInstanceMeta(instanceId);
    if (!instance || !isUserFacingInstance(instance)) return;
    if (state.arena.enabled) {
        if (historyConversationId || isExternalConversationTarget(instance)) {
            await setArenaEnabled(false);
        } else {
            restartArenaWithForcedInstance(instanceId).catch((error) => {
                console.warn('Unable to restart arena with forced instance:', error?.message || error);
            });
            return;
        }
    }

    persistSettingsForCurrentInstance();
    state.activeWorkspace = 'instance';
    state.currentInstanceId = instanceId;
    loadSettingsForInstance(instance);
    updateChatControls();
    updateSessionControlMode();
    refreshTtsSettingOptions(instance);

    const slotId = buildInstanceConversationSlotId(instanceId);
    clearConversationViewOverride('instance', slotId);
    const conversationId = setInstanceConversationId(instanceId, getInstanceConversationId(instanceId), {
        workspace: 'instance',
        slot_id: slotId,
        source_instance_id: instanceId,
        label: instanceId,
    });

    if (!state.conversations[conversationId]) {
        state.conversations[conversationId] = [];
    }

    updateActiveTabHighlight();
    const backendLabel = formatBackendLabel(instance.backend);
    const displayModel = isExternalConversationTarget(instance)
        ? (instance.label || 'ChatGPT')
        : (instance.model || instanceId);
    updateGlobalModelStatus(
        isExternalConversationTarget(instance)
            ? `Ready with ${displayModel} (automatic model via ${formatCodexSourceLabel(instance)}; each turn is independent)`
            : `Ready with ${displayModel} (${backendLabel})`
    );
    updateActiveModelToolbar();
    renderResponsesWorkbenchTargetOptions();
    updatePromptPlaceholder();
    updateVoiceInputButtonState();
    updateSendButtonState();
    showConversationLoadingPlaceholder(
        isExternalConversationTarget(instance)
            ? 'ChatGPT'
            : formatModelDisplayName(displayModel)
    );
    const hasLocalMessages = getPersistedConversationMessageCount(conversationId) > 0;
    const recoveredConversationId = await recoverInstanceConversationSlot(instanceId, {
        expectedConversationId: conversationId,
        preservePopulatedLocalState: hasLocalMessages,
        suppressVisibleRender: true,
    });
    let conversationIdToRender = getInstanceConversationId(instanceId);
    const requestedHistoryConversationId = String(historyConversationId || '').trim();
    if (requestedHistoryConversationId) {
        setConversationViewOverride('instance', slotId, requestedHistoryConversationId);
        if (!state.chatHistoryLoaded[requestedHistoryConversationId]) {
            await fetchChatHistory(requestedHistoryConversationId, { suppressVisibleRender: true });
        }
        conversationIdToRender = requestedHistoryConversationId;
    } else {
        clearConversationViewOverride('instance', slotId);
        if (!state.chatHistoryLoaded[conversationIdToRender] && recoveredConversationId === conversationIdToRender) {
            await fetchChatHistory(conversationIdToRender, { suppressVisibleRender: true });
        }
    }
    renderConversation(conversationIdToRender);
    renderConversationHistoryList();
    if (focusInput) {
        elements.userInput.focus();
    }
}
