function getSessionControlsSchema(instance) {
    if (!instance || typeof instance !== 'object') return null;
    const schema = instance.session_controls;
    if (!schema || typeof schema !== 'object') return null;
    const fields = (schema.fields && typeof schema.fields === 'object') ? schema.fields : {};
    return {
        enabled: Boolean(schema.enabled),
        hint: String(schema.hint || '').trim(),
        fields,
    };
}

function getSessionControlField(instance, key) {
    const schema = getSessionControlsSchema(instance);
    if (!schema) return null;
    const field = schema.fields?.[key];
    if (!field || typeof field !== 'object') return null;
    return field;
}

function fieldIsVisible(field) {
    return Boolean(field && field.visible !== false);
}

function hasActualSessionControls(instance) {
    const schema = getSessionControlsSchema(instance);
    if (!schema || !schema.enabled) return false;
    return Object.values(schema.fields || {}).some((field) => fieldIsVisible(field));
}

function getSessionControlsHint(instance) {
    const schema = getSessionControlsSchema(instance);
    const fields = schema?.fields || {};
    if (fields.tts_voice || fields.tts_language || fields.tts_response_format) {
        return 'Voice controls for the current model.';
    }
    if (fields.image_aspect_ratio || fields.image_width || fields.image_height || fields.image_count) {
        return 'Image controls for the current model.';
    }
    if (fields.ocr_mode || fields.pdf_max_pages || fields.pdf_dpi) {
        return 'OCR controls for the current model.';
    }
    if (fields.stt_language || fields.stt_task) {
        return 'Speech input controls for the current model.';
    }
    if (fields.reasoning_effort) {
        return 'Reasoning controls for the current model.';
    }
    return 'Per-chat controls for the current model.';
}

function getSessionControlDefaultMeta(fieldKey) {
    const key = String(fieldKey || '').trim();
    if (!key) return '';
    const defaults = {
        temperature: 'Leave blank for the model default. Lower values stay closer to the prompt.',
        top_p: 'Leave blank for the model default. Limits generation to the most likely probability mass.',
        chat_meta: 'Only this chat.',
        stt_meta: 'Speech input controls.',
        image_meta: 'Size plus optional edit/reference image.',
        ocr_meta: 'PDF OCR controls.',
        tts_meta: 'Named speaker or described voice style.',
    };
    return defaults[key] || '';
}

function getSessionControlDefaultValue(instance, fieldKey) {
    const field = getSessionControlField(instance, fieldKey);
    if (!field || typeof field !== 'object' || field.default_value === undefined || field.default_value === null || field.default_value === '') {
        return null;
    }
    return field.default_value;
}

function applySessionControlFieldPresentation(node, field) {
    if (!node || !field || typeof field !== 'object') return;
    const fieldKey = String(node.dataset.settingKey || '').trim();
    const labelNode = node.querySelector('.setting-label');
    if (labelNode) {
        if (!node.dataset.defaultLabel) {
            node.dataset.defaultLabel = labelNode.textContent || '';
        }
        const baseLabel = field.label ? String(field.label) : String(node.dataset.defaultLabel || '');
        labelNode.textContent = field.required ? `${baseLabel} *` : baseLabel;
    }
    const metaNode = node.querySelector('.setting-meta');
    if (metaNode) {
        if (!node.dataset.defaultMeta) {
            node.dataset.defaultMeta = metaNode.textContent || '';
        }
        const fallbackMeta = getSessionControlDefaultMeta(fieldKey) || String(node.dataset.defaultMeta || '');
        metaNode.textContent = fallbackMeta;
    }
    const inputNode = node.querySelector('input, select, textarea');
    if (inputNode) {
        if ('required' in inputNode) {
            inputNode.required = Boolean(field.required);
        }
        if (field.required) {
            inputNode.setAttribute('aria-required', 'true');
        } else {
            inputNode.removeAttribute('aria-required');
        }
    }
}

function getSessionControlFieldNode(fieldKey) {
    if (!fieldKey) return null;
    return document.querySelector(`[data-setting-key="${String(fieldKey).trim()}"]`);
}

function getSessionControlFieldLabel(fieldKey, field) {
    const explicitLabel = String(field?.label || '').trim();
    if (explicitLabel) {
        return explicitLabel;
    }
    const node = getSessionControlFieldNode(fieldKey);
    const labelText = String(node?.dataset?.defaultLabel || node?.querySelector('.setting-label')?.textContent || '').trim();
    return labelText.replace(/\s+\*$/, '') || String(fieldKey || '').trim();
}

function validateRequiredSessionControls(instance) {
    const schema = getSessionControlsSchema(instance);
    if (!schema?.enabled) return null;
    const schemaFields = schema.fields || {};
    for (const [fieldKey, field] of Object.entries(schemaFields)) {
        if (!fieldIsVisible(field) || !field?.required) {
            continue;
        }
        const binding = SESSION_CONTROL_BINDINGS[fieldKey];
        if (!binding) {
            continue;
        }
        const rawValue = state.settings[binding.stateKey];
        const transformed = typeof binding.transform === 'function'
            ? binding.transform(rawValue, instance, field)
            : rawValue;
        if (transformed === null || typeof transformed === 'undefined' || String(transformed).trim() === '') {
            return {
                fieldKey,
                message: String(field.required_message || '').trim()
                    || `${getSessionControlFieldLabel(fieldKey, field)} is required for this model.`,
            };
        }
    }
    return null;
}

function focusSessionControlField(fieldKey) {
    const fieldNode = getSessionControlFieldNode(fieldKey);
    if (!fieldNode) return;
    state.modelSettingsOpen = true;
    state.modelSettingsAutoOpened = true;
    renderModelSettingsPanel();
    const inputNode = fieldNode.querySelector('input, select, textarea');
    if (inputNode && typeof inputNode.focus === 'function') {
        inputNode.focus();
    }
}

function updateSessionControlMode() {
    const currentInstance = getCurrentSettingsOwnerInstance();
    document.querySelectorAll('[data-setting-key]').forEach((node) => {
        const key = String(node.dataset.settingKey || '').trim();
        const field = getSessionControlField(currentInstance, key);
        node.hidden = !fieldIsVisible(field);
        if (fieldIsVisible(field)) {
            applySessionControlFieldPresentation(node, field);
        }
    });
    refreshSessionControlSelectOptions(currentInstance);
    renderModelSettingsPanel();
}

function applyImageAspectPreset(preset) {
    const token = String(preset || '').trim();
    if (!token || token === 'auto' || token === 'custom') {
        state.settings.imageAspectRatio = token || 'auto';
        return;
    }
    const dims = IMAGE_ASPECT_PRESET_DIMENSIONS[token];
    if (!dims) return;
    state.settings.imageAspectRatio = token;
    state.settings.imageWidth = dims.width;
    state.settings.imageHeight = dims.height;
    if (elements.settingImageWidth) elements.settingImageWidth.value = dims.width;
    if (elements.settingImageHeight) elements.settingImageHeight.value = dims.height;
}

function sanitizeSettingsObject(raw = {}) {
    const rawObject = raw && typeof raw === 'object' ? raw : {};
    const hasOwn = (key) => Object.prototype.hasOwnProperty.call(rawObject, key);
    const source = { ...settingsDefaults, ...rawObject };
    const reasoningEffortToken = String(source.reasoningEffort ?? settingsDefaults.reasoningEffort ?? 'off')
        .trim()
        .toLowerCase();
    const reasoningEffortExplicit = hasOwn('reasoningEffortExplicit')
        ? rawObject.reasoningEffortExplicit === true
        : hasOwn('reasoningEffort') && ['low', 'medium', 'xhigh'].includes(reasoningEffortToken);
    const reasoningEffort = ['off', 'low', 'medium', 'xhigh'].includes(reasoningEffortToken)
        ? reasoningEffortToken
        : 'off';
    const imageWidth = source.imageWidth === '' || source.imageWidth === null || typeof source.imageWidth === 'undefined'
        ? null
        : (() => {
            const numeric = parseInt(source.imageWidth, 10);
            return Number.isFinite(numeric) ? Math.min(4096, Math.max(64, numeric)) : null;
        })();
    const imageHeight = source.imageHeight === '' || source.imageHeight === null || typeof source.imageHeight === 'undefined'
        ? null
        : (() => {
            const numeric = parseInt(source.imageHeight, 10);
            return Number.isFinite(numeric) ? Math.min(4096, Math.max(64, numeric)) : null;
        })();
    const imageAspectRatio = (() => {
        const explicit = String(source.imageAspectRatio || '').trim();
        if (explicit && explicit !== 'custom') {
            return explicit;
        }
        return inferImageAspectPreset(imageWidth, imageHeight);
    })();
    return {
        temperature: source.temperature === '' || source.temperature === null || typeof source.temperature === 'undefined'
            ? null
            : Math.min(2, Math.max(0, Number(source.temperature))),
        topP: source.topP === '' || source.topP === null || typeof source.topP === 'undefined'
            ? null
            : Math.min(1, Math.max(0, Number(source.topP))),
        reasoningEffort,
        reasoningEffortExplicit,
        sttLanguage: String(source.sttLanguage || '').trim(),
        sttTask: String(source.sttTask || settingsDefaults.sttTask).trim() || settingsDefaults.sttTask,
        imageAspectRatio,
        imageWidth,
        imageHeight,
        imageCount: Math.min(8, Math.max(1, parseInt(source.imageCount, 10) || settingsDefaults.imageCount)),
        ocrMode: String(source.ocrMode || settingsDefaults.ocrMode).trim() || settingsDefaults.ocrMode,
        pdfMaxPages: source.pdfMaxPages === '' || source.pdfMaxPages === null || typeof source.pdfMaxPages === 'undefined'
            ? null
            : (() => {
                const numeric = parseInt(source.pdfMaxPages, 10);
                return Number.isFinite(numeric) ? Math.min(500, Math.max(1, numeric)) : null;
            })(),
        pdfDpi: Math.min(600, Math.max(300, parseInt(source.pdfDpi, 10) || settingsDefaults.pdfDpi)),
        pdfPageTimeoutSec: Math.min(1800, Math.max(45, parseInt(source.pdfPageTimeoutSec, 10) || settingsDefaults.pdfPageTimeoutSec)),
        pdfSynthesize: Boolean(source.pdfSynthesize),
        ttsVoice: String(source.ttsVoice || '').trim(),
        ttsLanguage: String(source.ttsLanguage || settingsDefaults.ttsLanguage).trim() || settingsDefaults.ttsLanguage,
        ttsResponseFormat: String(source.ttsResponseFormat || '').trim().toLowerCase(),
        ttsInstruct: String(source.ttsInstruct || '').trim(),
        ttsSpeed: source.ttsSpeed === '' || source.ttsSpeed === null || typeof source.ttsSpeed === 'undefined'
            ? null
            : Math.min(2, Math.max(0.5, Number(source.ttsSpeed))),
        ttsPitch: source.ttsPitch === '' || source.ttsPitch === null || typeof source.ttsPitch === 'undefined'
            ? null
            : Math.min(2, Math.max(0.5, Number(source.ttsPitch))),
    };
}

function populateSelectOptions(select, values, selectedValue, {
    fallbackLabel = 'Auto / not set',
    includeEmptyOption = true,
    labelForValue = null,
} = {}) {
    if (!select) return;
    const options = Array.isArray(values) ? values.filter(Boolean) : [];
    const desiredValue = String(selectedValue || '').trim();
    select.innerHTML = '';

    if (includeEmptyOption) {
        const autoOption = document.createElement('option');
        autoOption.value = '';
        autoOption.textContent = fallbackLabel;
        select.appendChild(autoOption);
    }

    options.forEach((value) => {
        const token = String(value || '').trim();
        if (!token) return;
        const option = document.createElement('option');
        option.value = token;
        option.textContent = typeof labelForValue === 'function' ? labelForValue(token) : token;
        select.appendChild(option);
    });

    if (desiredValue && !options.some((item) => String(item).trim() === desiredValue)) {
        const currentOption = document.createElement('option');
        currentOption.value = desiredValue;
        const currentLabel = typeof labelForValue === 'function' ? labelForValue(desiredValue) : desiredValue;
        currentOption.textContent = `${currentLabel} (current)`;
        select.appendChild(currentOption);
    }

    select.value = desiredValue;
}

function getSessionControlOptionLabel(fieldKey, field, value) {
    const token = String(value || '').trim();
    const optionLabels = field?.option_labels && typeof field.option_labels === 'object'
        ? field.option_labels
        : {};
    if (optionLabels[token]) {
        return String(optionLabels[token]);
    }
    if (fieldKey === 'reasoning_effort') {
        return ({ off: 'Off', low: 'Low', medium: 'Medium', xhigh: 'XHigh' })[token] || token;
    }
    return token;
}

function hasExplicitReasoningEffortPreference(settings = state.settings) {
    if (!settings || typeof settings !== 'object') return false;
    if (Object.prototype.hasOwnProperty.call(settings, 'reasoningEffortExplicit')) {
        return settings.reasoningEffortExplicit === true;
    }
    return Object.prototype.hasOwnProperty.call(settings, 'reasoningEffort');
}

function getReasoningEffortDefaultValue(field, options = []) {
    const normalizedOptions = (Array.isArray(options) ? options : [])
        .map((item) => String(item || '').trim().toLowerCase())
        .filter(Boolean);
    const declaredDefault = String(field?.default_value || '').trim().toLowerCase();
    if (declaredDefault && normalizedOptions.includes(declaredDefault)) {
        return declaredDefault;
    }
    if (normalizedOptions.includes('medium')) return 'medium';
    return normalizedOptions.find((item) => item !== 'off') || 'off';
}

function refreshSessionControlSelectOptions(instance = getCurrentSettingsOwnerInstance()) {
    const sttLanguageField = getSessionControlField(instance, 'stt_language');
    const sttTaskField = getSessionControlField(instance, 'stt_task');
    const reasoningEffortField = getSessionControlField(instance, 'reasoning_effort');
    const imageAspectRatioField = getSessionControlField(instance, 'image_aspect_ratio');
    const ocrModeField = getSessionControlField(instance, 'ocr_mode');
    const ttsVoiceField = getSessionControlField(instance, 'tts_voice');
    const ttsLanguageField = getSessionControlField(instance, 'tts_language');
    const ttsResponseFormatField = getSessionControlField(instance, 'tts_response_format');

    const reasoningEffortOptions = Array.isArray(reasoningEffortField?.options)
        ? reasoningEffortField.options.map((item) => String(item || '').trim()).filter(Boolean)
        : [];
    const currentReasoningEffort = String(state.settings.reasoningEffort || '').trim();
    const reasoningEffortDefault = getReasoningEffortDefaultValue(reasoningEffortField, reasoningEffortOptions);
    const desiredReasoningEffort = hasExplicitReasoningEffortPreference()
        && reasoningEffortOptions.includes(currentReasoningEffort)
        ? currentReasoningEffort
        : reasoningEffortDefault;
    if (
        reasoningEffortField
        && reasoningEffortOptions.length
        && desiredReasoningEffort !== currentReasoningEffort
    ) {
        state.settings.reasoningEffort = desiredReasoningEffort;
        saveSettings();
    }

    populateSelectOptions(
        elements.settingSttLanguage,
        Array.isArray(sttLanguageField?.options) ? sttLanguageField.options : [],
        state.settings.sttLanguage,
        { fallbackLabel: 'Auto detect', includeEmptyOption: true }
    );
    populateSelectOptions(
        elements.settingSttTask,
        Array.isArray(sttTaskField?.options) ? sttTaskField.options : ['transcribe', 'translate'],
        state.settings.sttTask,
        { includeEmptyOption: false }
    );
    populateSelectOptions(
        elements.settingReasoningEffort,
        reasoningEffortOptions,
        reasoningEffortField ? desiredReasoningEffort : state.settings.reasoningEffort,
        {
            includeEmptyOption: false,
            labelForValue: (value) => getSessionControlOptionLabel('reasoning_effort', reasoningEffortField, value),
        }
    );
    populateSelectOptions(
        elements.settingImageAspectRatio,
        Array.isArray(imageAspectRatioField?.options) ? imageAspectRatioField.options : ['auto', '1:1', '4:3', '3:4', '3:2', '2:3', '16:9', '9:16', 'custom'],
        state.settings.imageAspectRatio,
        { includeEmptyOption: false }
    );
    populateSelectOptions(
        elements.settingOcrMode,
        Array.isArray(ocrModeField?.options) ? ocrModeField.options : ['auto'],
        state.settings.ocrMode,
        { includeEmptyOption: false }
    );

    const ttsVoiceOptions = Array.isArray(ttsVoiceField?.options) ? ttsVoiceField.options : [];
    const currentTtsVoice = String(state.settings.ttsVoice || '').trim();
    const currentTtsVoiceValid = currentTtsVoice
        ? ttsVoiceOptions.some((item) => String(item || '').trim() === currentTtsVoice)
        : false;
    const desiredTtsVoice = (
        ttsVoiceField?.default_first_option && ttsVoiceOptions.length
    )
        ? (currentTtsVoiceValid ? currentTtsVoice : String(ttsVoiceOptions[0] || '').trim())
        : currentTtsVoice;
    if (
        desiredTtsVoice &&
        desiredTtsVoice !== currentTtsVoice
    ) {
        state.settings.ttsVoice = desiredTtsVoice;
        saveSettings();
    }
    populateSelectOptions(
        elements.settingTtsVoice,
        ttsVoiceOptions,
        state.settings.ttsVoice,
        { includeEmptyOption: true }
    );
    populateSelectOptions(
        elements.settingTtsLanguage,
        Array.isArray(ttsLanguageField?.options) ? ttsLanguageField.options : [],
        state.settings.ttsLanguage,
        { fallbackLabel: 'Auto detect', includeEmptyOption: true }
    );
    populateSelectOptions(
        elements.settingTtsResponseFormat,
        Array.isArray(ttsResponseFormatField?.options) ? ttsResponseFormatField.options : [],
        state.settings.ttsResponseFormat,
        { fallbackLabel: 'Backend default', includeEmptyOption: true }
    );
}

function refreshTtsSettingOptions(instance = getCurrentSettingsOwnerInstance()) {
    refreshSessionControlSelectOptions(instance);
}

function getInstanceSettingsKey(instance) {
    if (!instance) return null;
    const modelName = String(instance.model || instance.modelName || instance.name || instance.instance_id || '').trim();
    if (!modelName) return null;
    return makeModelKey(modelName, instance.backend || 'ollama');
}

function persistSettingsForCurrentInstance() {
    const sanitized = sanitizeSettingsObject(state.settings);
    const currentInstance = getCurrentSettingsOwnerInstance();
    if (!currentInstance) {
        state.settingsGlobal = { ...sanitized };
        state.settings = { ...sanitized };
        return;
    }
    const key = getInstanceSettingsKey(currentInstance);
    if (!key) return;
    state.settingsByModel[key] = { ...sanitized };
    state.settings = { ...sanitized };
}

function loadSettingsForInstance(instance) {
    const key = getInstanceSettingsKey(instance);
    if (!key) {
        state.settings = sanitizeSettingsObject(state.settingsGlobal);
        applySettings(instance);
        return;
    }
    const saved = state.settingsByModel[key];
    state.settings = sanitizeSettingsObject(saved || state.settingsGlobal);
    applySettings(instance);
}

function getSessionControlBindingByRequestKey(requestKey) {
    const normalized = String(requestKey || '').trim();
    if (!normalized) return null;
    for (const [fieldKey, binding] of Object.entries(SESSION_CONTROL_BINDINGS)) {
        if (String(binding?.requestKey || '').trim() === normalized) {
            return { fieldKey, binding };
        }
    }
    return null;
}

function applyRuntimeControlHintsToLiveSettings(runtime, instance) {
    const hints = runtime && typeof runtime === 'object' && runtime.control_hints && typeof runtime.control_hints === 'object'
        ? runtime.control_hints
        : null;
    if (!hints || !Object.keys(hints).length) {
        return [];
    }
    const draft = { ...state.settings };
    const schemaFields = getSessionControlsSchema(instance)?.fields || {};
    const applied = [];
    Object.entries(hints).forEach(([requestKey, hintedValue]) => {
        const resolved = getSessionControlBindingByRequestKey(requestKey);
        if (!resolved?.binding?.stateKey) {
            return;
        }
        if (hintedValue === null || typeof hintedValue === 'undefined' || hintedValue === '') {
            return;
        }
        const { fieldKey, binding } = resolved;
        const normalizedValue = typeof binding.transform === 'function'
            ? binding.transform(hintedValue, instance, schemaFields[fieldKey])
            : hintedValue;
        if (normalizedValue === null || typeof normalizedValue === 'undefined' || normalizedValue === '') {
            return;
        }
        draft[binding.stateKey] = normalizedValue;
        applied.push({
            requestKey: String(requestKey || '').trim(),
            fieldKey,
            label: getSessionControlFieldLabel(fieldKey, schemaFields[fieldKey]),
        });
    });
    if (!applied.length) {
        return [];
    }
    state.settings = sanitizeSettingsObject(draft);
    applySettings(instance);
    return applied;
}

function summarizeAppliedControlHintLabels(appliedHints = []) {
    const labels = [];
    const requestKeys = new Set(appliedHints.map((item) => String(item?.requestKey || '').trim()).filter(Boolean));
    if (requestKeys.has('width') && requestKeys.has('height')) {
        labels.push('Image size');
    } else {
        if (requestKeys.has('width')) labels.push('Image width');
        if (requestKeys.has('height')) labels.push('Image height');
    }
    if (requestKeys.has('lang_code')) labels.push('Language');
    if (requestKeys.has('response_format')) labels.push('Output format');
    if (requestKeys.has('instruct')) labels.push('Style / Instruct');
    if (requestKeys.has('voice')) labels.push('Voice');
    if (requestKeys.has('speed')) labels.push('Speed');
    if (requestKeys.has('pitch')) labels.push('Pitch');
    appliedHints.forEach((item) => {
        const label = String(item?.label || '').trim();
        if (label && !labels.includes(label)) {
            labels.push(label);
        }
    });
    return labels;
}

function formatNaturalLanguageList(items = []) {
    const tokens = Array.from(new Set((items || []).map((item) => String(item || '').trim()).filter(Boolean)));
    if (!tokens.length) return '';
    if (tokens.length === 1) return tokens[0];
    if (tokens.length === 2) return `${tokens[0]} and ${tokens[1]}`;
    return `${tokens.slice(0, -1).join(', ')}, and ${tokens[tokens.length - 1]}`;
}

function buildMissingSessionControlsAssistantMessage(missingSessionControls, appliedHints = [], overrideMissingFields = null) {
    const missingFields = Array.isArray(overrideMissingFields)
        ? overrideMissingFields
        : (
            Array.isArray(missingSessionControls?.missing_fields)
                ? missingSessionControls.missing_fields
                : []
        );
    const schemaFields = getSessionControlsSchema(missingSessionControls?.instance)?.fields || {};
    const missingLabels = missingFields
        .map((field) => getSessionControlFieldLabel(field?.field_key, schemaFields[field?.field_key]))
        .filter(Boolean);
    const missingSummary = formatNaturalLanguageList(missingLabels) || 'additional model input';
    const appliedSummary = formatNaturalLanguageList(summarizeAppliedControlHintLabels(appliedHints));
    let message = `Need ${missingLabels.length > 1 ? 'more model inputs' : 'one more model input'} before I can continue: ${missingSummary}.`;
    if (appliedSummary) {
        message += ` I prefilled ${appliedSummary}.`;
    }
    message += ' Open Session Controls and send again.';
    return message;
}

function initializeSettings() {
    loadSettings();
    applySettings();
    attachSettingsListeners();
    updateSessionControlMode();
}

function loadSettings() {
    try {
        const saved = localStorage.getItem('ollmo_settings');
        if (saved) {
            const parsed = JSON.parse(saved);
            if (parsed && typeof parsed === 'object' && (parsed.global || parsed.byModel)) {
                state.settingsGlobal = sanitizeSettingsObject(parsed.global || settingsDefaults);
                state.settingsByModel = {};
                const byModel = parsed.byModel && typeof parsed.byModel === 'object' ? parsed.byModel : {};
                Object.entries(byModel).forEach(([key, value]) => {
                    if (!key) return;
                    state.settingsByModel[key] = sanitizeSettingsObject(value);
                });
            } else if (parsed && typeof parsed === 'object') {
                // Legacy format: one global settings object
                state.settingsGlobal = sanitizeSettingsObject(parsed);
                state.settingsByModel = {};
            }
        }
        state.settings = sanitizeSettingsObject(state.settingsGlobal);
    } catch (error) {
        console.warn('Could not load persisted settings:', error);
        state.settingsGlobal = { ...settingsDefaults };
        state.settingsByModel = {};
        state.settings = { ...settingsDefaults };
    }
}

function saveSettings() {
    try {
        persistSettingsForCurrentInstance();
        localStorage.setItem('ollmo_settings', JSON.stringify({
            global: sanitizeSettingsObject(state.settingsGlobal),
            byModel: state.settingsByModel,
        }));
    } catch (error) {
        console.warn('Could not persist settings:', error);
    }
}

function applySettings(instance = getCurrentSettingsOwnerInstance()) {
    const {
        temperature,
        topP,
        reasoningEffort,
        sttLanguage,
        sttTask,
        imageAspectRatio,
        imageWidth,
        imageHeight,
        imageCount,
        ocrMode,
        pdfMaxPages,
        pdfDpi,
        pdfPageTimeoutSec,
        pdfSynthesize,
        ttsVoice,
        ttsLanguage,
        ttsResponseFormat,
        ttsInstruct,
        ttsSpeed,
        ttsPitch,
    } = state.settings;
    refreshSessionControlSelectOptions(instance);
    if (elements.settingTemperature) {
        elements.settingTemperature.value = temperature ?? '';
    }
    if (elements.settingTopP) {
        elements.settingTopP.value = topP ?? '';
    }
    if (elements.settingReasoningEffort) {
        elements.settingReasoningEffort.value = state.settings.reasoningEffort;
    }
    if (elements.settingSttLanguage) {
        elements.settingSttLanguage.value = sttLanguage;
    }
    if (elements.settingSttTask) {
        elements.settingSttTask.value = sttTask;
    }
    if (elements.settingImageAspectRatio) {
        elements.settingImageAspectRatio.value = imageAspectRatio;
    }
    if (elements.settingImageWidth) {
        elements.settingImageWidth.value = imageWidth ?? '';
    }
    if (elements.settingImageHeight) {
        elements.settingImageHeight.value = imageHeight ?? '';
    }
    if (elements.settingImageCount) {
        elements.settingImageCount.value = imageCount;
    }
    if (elements.settingOcrMode) {
        elements.settingOcrMode.value = ocrMode;
    }
    if (elements.settingPdfMaxPages) {
        elements.settingPdfMaxPages.value = pdfMaxPages ?? '';
    }
    if (elements.settingPdfDpi) {
        elements.settingPdfDpi.value = pdfDpi;
    }
    if (elements.settingPdfPageTimeout) {
        elements.settingPdfPageTimeout.value = pdfPageTimeoutSec;
    }
    if (elements.settingPdfSynthesize) {
        elements.settingPdfSynthesize.checked = Boolean(pdfSynthesize);
    }
    if (elements.settingTtsVoice) {
        refreshSessionControlSelectOptions();
        elements.settingTtsVoice.value = ttsVoice;
    }
    if (elements.settingTtsLanguage) {
        elements.settingTtsLanguage.value = ttsLanguage === 'auto' ? '' : ttsLanguage;
    }
    if (elements.settingTtsResponseFormat) {
        elements.settingTtsResponseFormat.value = ttsResponseFormat || '';
    }
    if (elements.settingTtsInstruct) {
        elements.settingTtsInstruct.value = ttsInstruct;
    }
    if (elements.settingTtsSpeed) {
        elements.settingTtsSpeed.value = ttsSpeed ?? '';
    }
    if (elements.settingTtsPitch) {
        elements.settingTtsPitch.value = ttsPitch ?? '';
    }
}

function attachSettingsListeners() {
    if (!elements.settingsPanel) return;

    elements.settingsPanel.classList.remove('open');

    function seedNumericStepperDefault(input, fieldKey, event) {
        if (!input || String(input.value || '').trim() !== '') return;
        const pointerEvent = event instanceof PointerEvent ? event : null;
        if (pointerEvent) {
            const rect = input.getBoundingClientRect();
            const spinnerZoneWidth = Math.min(28, rect.width * 0.22);
            if (pointerEvent.clientX < rect.right - spinnerZoneWidth) {
                return;
            }
        }
        const defaultValue = getSessionControlDefaultValue(getCurrentSettingsOwnerInstance(), fieldKey);
        if (defaultValue === null || typeof defaultValue === 'undefined') return;
        input.value = String(defaultValue);
    }

    function primeNumericStepperOnKeydown(input, fieldKey, event) {
        if (!input || String(input.value || '').trim() !== '') return false;
        if (event.key !== 'ArrowUp' && event.key !== 'ArrowDown') return false;
        const defaultValue = getSessionControlDefaultValue(getCurrentSettingsOwnerInstance(), fieldKey);
        if (defaultValue === null || typeof defaultValue === 'undefined') return false;
        event.preventDefault();
        input.value = String(defaultValue);
        if (event.key === 'ArrowUp') {
            input.stepUp();
        } else {
            input.stepDown();
        }
        input.dispatchEvent(new Event('input', { bubbles: true }));
        return true;
    }

    function bindNumberSetting(input, fieldKey, applyValue) {
        if (!input) return;
        const commit = (event) => {
            applyValue(event);
            saveSettings();
        };
        input.addEventListener('pointerdown', (event) => {
            seedNumericStepperDefault(input, fieldKey, event);
        });
        input.addEventListener('keydown', (event) => {
            primeNumericStepperOnKeydown(input, fieldKey, event);
        });
        input.addEventListener('input', commit);
        input.addEventListener('change', commit);
    }

    if (elements.settingTemperature) {
        bindNumberSetting(elements.settingTemperature, 'temperature', (event) => {
            const raw = String(event.target.value || '').trim();
            state.settings.temperature = raw === '' ? null : Math.min(2, Math.max(0, parseFloat(raw) || 0));
        });
    }

    if (elements.settingTopP) {
        bindNumberSetting(elements.settingTopP, 'top_p', (event) => {
            const raw = String(event.target.value || '').trim();
            state.settings.topP = raw === '' ? null : Math.min(1, Math.max(0, parseFloat(raw) || 0));
        });
    }

    if (elements.settingReasoningEffort) {
        elements.settingReasoningEffort.addEventListener('change', (event) => {
            state.settings.reasoningEffort = sanitizeSettingsObject({
                ...state.settings,
                reasoningEffort: event.target.value,
                reasoningEffortExplicit: true,
            }).reasoningEffort;
            state.settings.reasoningEffortExplicit = true;
            elements.settingReasoningEffort.value = state.settings.reasoningEffort;
            saveSettings();
        });
    }

    if (elements.settingSttLanguage) {
        elements.settingSttLanguage.addEventListener('change', (event) => {
            state.settings.sttLanguage = String(event.target.value || '').trim();
            saveSettings();
        });
    }

    if (elements.settingSttTask) {
        elements.settingSttTask.addEventListener('change', (event) => {
            const value = String(event.target.value || '').trim() || settingsDefaults.sttTask;
            state.settings.sttTask = value;
            elements.settingSttTask.value = value;
            saveSettings();
        });
    }
    if (elements.settingImageAspectRatio) {
        elements.settingImageAspectRatio.addEventListener('change', (event) => {
            const value = String(event.target.value || '').trim() || 'auto';
            applyImageAspectPreset(value);
            if (elements.settingImageAspectRatio) {
                elements.settingImageAspectRatio.value = state.settings.imageAspectRatio;
            }
            saveSettings();
        });
    }
    if (elements.settingImageWidth) {
        bindNumberSetting(elements.settingImageWidth, 'image_width', (event) => {
            const raw = String(event.target.value || '').trim();
            const parsed = parseInt(raw, 10);
            const value = raw === '' || !Number.isFinite(parsed)
                ? null
                : Math.min(4096, Math.max(64, parsed));
            state.settings.imageWidth = value;
            state.settings.imageAspectRatio = inferImageAspectPreset(state.settings.imageWidth, state.settings.imageHeight);
            elements.settingImageWidth.value = value ?? '';
            if (elements.settingImageAspectRatio) {
                elements.settingImageAspectRatio.value = state.settings.imageAspectRatio;
            }
        });
    }
    if (elements.settingImageHeight) {
        bindNumberSetting(elements.settingImageHeight, 'image_height', (event) => {
            const raw = String(event.target.value || '').trim();
            const parsed = parseInt(raw, 10);
            const value = raw === '' || !Number.isFinite(parsed)
                ? null
                : Math.min(4096, Math.max(64, parsed));
            state.settings.imageHeight = value;
            state.settings.imageAspectRatio = inferImageAspectPreset(state.settings.imageWidth, state.settings.imageHeight);
            elements.settingImageHeight.value = value ?? '';
            if (elements.settingImageAspectRatio) {
                elements.settingImageAspectRatio.value = state.settings.imageAspectRatio;
            }
        });
    }
    if (elements.settingImageCount) {
        bindNumberSetting(elements.settingImageCount, 'image_count', (event) => {
            const value = Math.min(8, Math.max(1, parseInt(event.target.value, 10) || settingsDefaults.imageCount));
            state.settings.imageCount = value;
            elements.settingImageCount.value = value;
        });
    }
    if (elements.settingOcrMode) {
        elements.settingOcrMode.addEventListener('change', (event) => {
            const value = String(event.target.value || '').trim() || settingsDefaults.ocrMode;
            state.settings.ocrMode = value;
            elements.settingOcrMode.value = value;
            saveSettings();
        });
    }

    if (elements.settingPdfMaxPages) {
        bindNumberSetting(elements.settingPdfMaxPages, 'pdf_max_pages', (event) => {
            const raw = String(event.target.value || '').trim();
            const parsed = parseInt(raw, 10);
            const value = raw === '' || !Number.isFinite(parsed)
                ? null
                : Math.min(500, Math.max(1, parsed));
            state.settings.pdfMaxPages = value;
            elements.settingPdfMaxPages.value = value ?? '';
        });
    }

    if (elements.settingPdfDpi) {
        bindNumberSetting(elements.settingPdfDpi, 'pdf_dpi', (event) => {
            const value = Math.min(600, Math.max(300, parseInt(event.target.value, 10) || settingsDefaults.pdfDpi));
            state.settings.pdfDpi = value;
            elements.settingPdfDpi.value = value;
        });
    }

    if (elements.settingPdfPageTimeout) {
        bindNumberSetting(elements.settingPdfPageTimeout, 'pdf_page_timeout_sec', (event) => {
            const value = Math.min(1800, Math.max(45, parseInt(event.target.value, 10) || settingsDefaults.pdfPageTimeoutSec));
            state.settings.pdfPageTimeoutSec = value;
            elements.settingPdfPageTimeout.value = value;
        });
    }

    if (elements.settingPdfSynthesize) {
        elements.settingPdfSynthesize.addEventListener('change', (event) => {
            state.settings.pdfSynthesize = Boolean(event.target.checked);
            saveSettings();
        });
    }

    if (elements.settingTtsVoice) {
        elements.settingTtsVoice.addEventListener('change', (event) => {
            state.settings.ttsVoice = String(event.target.value || '').trim();
            elements.settingTtsVoice.value = state.settings.ttsVoice;
            saveSettings();
        });
    }

    if (elements.settingTtsLanguage) {
        elements.settingTtsLanguage.addEventListener('change', (event) => {
            state.settings.ttsLanguage = String(event.target.value || '').trim() || 'auto';
            elements.settingTtsLanguage.value = state.settings.ttsLanguage === 'auto' ? '' : state.settings.ttsLanguage;
            saveSettings();
        });
    }

    if (elements.settingTtsResponseFormat) {
        elements.settingTtsResponseFormat.addEventListener('change', (event) => {
            state.settings.ttsResponseFormat = String(event.target.value || '').trim().toLowerCase();
            elements.settingTtsResponseFormat.value = state.settings.ttsResponseFormat;
            saveSettings();
        });
    }

    if (elements.settingTtsInstruct) {
        elements.settingTtsInstruct.addEventListener('change', (event) => {
            state.settings.ttsInstruct = String(event.target.value || '').trim();
            elements.settingTtsInstruct.value = state.settings.ttsInstruct;
            saveSettings();
        });
    }

    if (elements.settingTtsSpeed) {
        bindNumberSetting(elements.settingTtsSpeed, 'tts_speed', (event) => {
            const raw = String(event.target.value || '').trim();
            const value = raw === '' ? null : Math.min(2, Math.max(0.5, parseFloat(raw) || 0.5));
            state.settings.ttsSpeed = value;
            elements.settingTtsSpeed.value = value ?? '';
        });
    }

    if (elements.settingTtsPitch) {
        bindNumberSetting(elements.settingTtsPitch, 'tts_pitch', (event) => {
            const raw = String(event.target.value || '').trim();
            const value = raw === '' ? null : Math.min(2, Math.max(0.5, parseFloat(raw) || 0.5));
            state.settings.ttsPitch = value;
            elements.settingTtsPitch.value = value ?? '';
        });
    }
}

function getPrimaryHistoryArtifactByType(artifacts = [], type = '') {
    const normalizedType = String(type || '').trim().toLowerCase();
    return (Array.isArray(artifacts) ? artifacts : []).find((artifact) => {
        if (!artifact || String(artifact.type || '').trim().toLowerCase() !== normalizedType) {
            return false;
        }
        if (normalizedType === 'image') {
            return Boolean(artifact.path || artifact.image_data_url);
        }
        return Boolean(artifact.path);
    }) || null;
}

function buildDurableHistoryArtifacts(artifacts = []) {
    return sanitizeResponseArtifacts(artifacts).map((artifact) => {
        const { image_data_url: _ignoredImageDataUrl, ...rest } = artifact;
        return rest;
    });
}

function buildDurableHistoryFallbackContent({
    content = '',
    responseCapability = '',
    savedImagePath = null,
    savedAudioPath = null,
    savedTextPath = null,
    artifacts = [],
} = {}) {
    const explicitContent = String(content || '');
    if (explicitContent.trim()) {
        return explicitContent;
    }
    const capability = normalizeCapability(responseCapability || '');
    const hasImage = Boolean(
        savedImagePath
        || (Array.isArray(artifacts) && artifacts.some((artifact) => String(artifact?.type || '').trim().toLowerCase() === 'image'))
    );
    const imageCount = Array.isArray(artifacts)
        ? artifacts.filter((artifact) => String(artifact?.type || '').trim().toLowerCase() === 'image').length
        : 0;
    const hasAudio = Boolean(savedAudioPath || getPrimaryHistoryArtifactByType(artifacts, 'audio'));
    const hasText = Boolean(savedTextPath || getPrimaryHistoryArtifactByType(artifacts, 'text'));
    if (capability === 'speech_to_text' || (hasText && /\/transcripts?\//i.test(String(savedTextPath || getPrimaryHistoryArtifactByType(artifacts, 'text')?.path || '')))) {
        return 'Transcript saved.';
    }
    if (hasText) {
        return 'Text artifact saved.';
    }
    if (capability === 'text_to_speech' || hasAudio) {
        return 'Audio generated.';
    }
    if (imageCount > 1) {
        return `Generated ${imageCount} images.`;
    }
    if (hasImage) {
        return 'Image generated.';
    }
    return explicitContent;
}

function promoteHistoryMessageArtifacts(message = {}) {
    const source = message && typeof message === 'object' ? message : {};
    const role = String(source.role || '').trim().toLowerCase();
    const isUserMessage = role === 'user';
    const normalizedRequestSnapshot = sanitizeRequestSnapshot(source.request_snapshot || source.requestSnapshot);
    const normalizedArtifacts = buildCanonicalMessageArtifacts({
        ...source,
        requestSnapshot: normalizedRequestSnapshot,
        request_snapshot: normalizedRequestSnapshot,
    });
    const durableArtifacts = buildDurableHistoryArtifacts(normalizedArtifacts);
    const primaryImage = getPrimaryHistoryArtifactByType(durableArtifacts, 'image');
    const primaryAudio = getPrimaryHistoryArtifactByType(durableArtifacts, 'audio');
    const primaryText = getPrimaryHistoryArtifactByType(durableArtifacts, 'text');
    const savedImagePath = isUserMessage
        ? null
        : (source.saved_image_path || source.savedImagePath || primaryImage?.path || null);
    const savedAudioPath = isUserMessage
        ? null
        : (source.saved_audio_path || source.savedAudioPath || primaryAudio?.path || null);
    const savedTextPath = isUserMessage
        ? null
        : (source.saved_text_path || source.savedTextPath || primaryText?.path || null);
    const responseCapability = source.response_capability || source.responseCapability || source.capability || null;
    const mirroredRequestSnapshot = mergeRequestSnapshotInputArtifacts(normalizedRequestSnapshot);
    const hasRouteReuseLastArtifact = 'route_reuse_last_artifact' in source || 'routeReuseLastArtifact' in source;
    const outputs = sanitizeResponseOutputs(source.outputs || source.canonical_outputs || source.canonicalOutputs);
    const outputSlots = sanitizeResponseOutputSlots(source.output_slots || source.outputSlots);
    const outputBranches = sanitizeResponseOutputBranches(source.output_branches || source.outputBranches);
    const artifactBundles = Array.isArray(source.artifact_bundles)
        ? source.artifact_bundles
        : (Array.isArray(source.artifactBundles) ? source.artifactBundles : []);
    const statusSemantics = sanitizeResponseStatusSemantics(source.status_semantics || source.statusSemantics);
    const lateFill = sanitizeMessageLateFill(source.late_fill || source.lateFill);
    const surfaceState = sanitizeSurfaceState(source.surface_state || source.surfaceState || lateFill?.surfaceState);
    return {
        ...source,
        message_id: source.message_id || source.messageId || source.clientMessageId || null,
        content: buildDurableHistoryFallbackContent({
            content: source.content,
            responseCapability,
            savedImagePath,
            savedAudioPath,
            savedTextPath,
            artifacts: durableArtifacts,
        }),
        saved_image_path: savedImagePath,
        saved_audio_path: savedAudioPath,
        saved_text_path: savedTextPath,
        response_id: source.response_id || source.responseId || null,
        response_state_version: source.response_state_version || source.responseStateVersion || null,
        response_frame_sequence: Number.isFinite(Number(source.response_frame_sequence ?? source.responseFrameSequence))
            ? Number(source.response_frame_sequence ?? source.responseFrameSequence)
            : null,
        response_frame_id: source.response_frame_id || source.responseFrameId || null,
        response_capability: responseCapability,
        response_model: source.response_model || source.responseModel || null,
        response_backend: source.response_backend || source.responseBackend || null,
        response_instance_id: source.response_instance_id || source.responseInstanceId || null,
        route_source: source.route_source || source.routeSource || null,
        route_reason: source.route_reason || source.routeReason || null,
        route_router_instance_id: source.route_router_instance_id || source.routeRouterInstanceId || null,
        route_router_model: source.route_router_model || source.routeRouterModel || null,
        route_artifact_ref: source.route_artifact_ref || source.routeArtifactRef || null,
        route_artifact_path: source.route_artifact_path || source.routeArtifactPath || null,
        route_reuse_last_artifact: hasRouteReuseLastArtifact
            ? Boolean(source.route_reuse_last_artifact ?? source.routeReuseLastArtifact)
            : null,
        reference_image_count: Number.isFinite(Number(source.reference_image_count ?? source.referenceImageCount))
            ? Number(source.reference_image_count ?? source.referenceImageCount)
            : null,
        reference_image_kind: source.reference_image_kind || source.referenceImageKind || null,
        context_mode: source.context_mode || source.contextMode || null,
        context_reason: source.context_reason || source.contextReason || null,
        lifecycle_state: String(source.lifecycle_state || source.lifecycleState || statusSemantics?.canonicalLifecycleState || '').trim().toLowerCase() || null,
        status_semantics: statusSemantics,
        late_fill: lateFill,
        surface_state: surfaceState,
        request_snapshot: mirroredRequestSnapshot,
        artifacts: durableArtifacts,
        outputs,
        output_slots: outputSlots,
        output_branches: outputBranches,
        artifact_bundles: artifactBundles,
    };
}

function serializeConversationForHistory(instanceId) {
    const conversation = state.conversations[instanceId] || [];
    return conversation
        .filter(message => message && !message.isLoading && !message.ephemeralUiNotice)
        .map((message) => {
            const normalized = promoteHistoryMessageArtifacts(message);
            return {
                message_id: normalized.message_id || null,
                role: normalized.role,
                content: normalized.content,
                timestamp: normalized.timestamp || message.timestamp,
                saved_image_path: normalized.saved_image_path || null,
                saved_audio_path: normalized.saved_audio_path || null,
                saved_text_path: normalized.saved_text_path || null,
                response_id: normalized.response_id || null,
                response_state_version: normalized.response_state_version || null,
                response_frame_sequence: Number.isFinite(Number(normalized.response_frame_sequence))
                    ? Number(normalized.response_frame_sequence)
                    : null,
                response_frame_id: normalized.response_frame_id || null,
                response_capability: normalized.response_capability || null,
                response_model: normalized.response_model || null,
                response_backend: normalized.response_backend || null,
                response_instance_id: normalized.response_instance_id || null,
                route_source: normalized.route_source || null,
                route_reason: normalized.route_reason || null,
                route_router_instance_id: normalized.route_router_instance_id || null,
                route_router_model: normalized.route_router_model || null,
                route_artifact_ref: normalized.route_artifact_ref || null,
                route_artifact_path: normalized.route_artifact_path || null,
                route_reuse_last_artifact: normalized.route_reuse_last_artifact === null || normalized.route_reuse_last_artifact === undefined
                    ? null
                    : Boolean(normalized.route_reuse_last_artifact),
                reference_image_count: Number.isFinite(Number(normalized.reference_image_count)) ? Number(normalized.reference_image_count) : null,
                reference_image_kind: normalized.reference_image_kind || null,
                context_mode: normalized.context_mode || null,
                context_reason: normalized.context_reason || null,
                lifecycle_state: normalized.lifecycle_state || null,
                status_semantics: normalized.status_semantics || null,
                late_fill: normalized.late_fill || null,
                surface_state: normalized.surface_state || null,
                artifacts: normalized.artifacts,
                outputs: normalized.outputs || [],
                output_slots: normalized.output_slots || [],
                output_branches: normalized.output_branches || [],
                artifact_bundles: normalized.artifact_bundles || [],
                request_snapshot: sanitizeRequestSnapshot(normalized.request_snapshot || normalized.requestSnapshot),
            };
        });
}

function deserializeHistoryMessage(message = {}) {
    const normalized = promoteHistoryMessageArtifacts(message);
    return {
        clientMessageId: normalized.message_id || null,
        role: normalized.role,
        content: normalized.content,
        timestamp: normalized.timestamp || new Date().toISOString(),
        isLoading: false,
        imageDataUrl: String(message.image_data_url || message.imageDataUrl || '').trim() || null,
        savedImagePath: normalized.saved_image_path || null,
        savedAudioPath: normalized.saved_audio_path || null,
        savedTextPath: normalized.saved_text_path || null,
        responseId: normalized.response_id || null,
        responseStateVersion: normalized.response_state_version || null,
        responseFrameSequence: Number.isFinite(Number(normalized.response_frame_sequence))
            ? Number(normalized.response_frame_sequence)
            : null,
        responseFrameId: normalized.response_frame_id || null,
        responseCapability: normalized.response_capability || null,
        responseModel: normalized.response_model || null,
        responseBackend: normalized.response_backend || null,
        responseInstanceId: normalized.response_instance_id || null,
        routeSource: normalized.route_source || null,
        routeReason: normalized.route_reason || null,
        routeRouterInstanceId: normalized.route_router_instance_id || null,
        routeRouterModel: normalized.route_router_model || null,
        routeArtifactRef: normalized.route_artifact_ref || null,
        routeArtifactPath: normalized.route_artifact_path || null,
        routeReuseLastArtifact: normalized.route_reuse_last_artifact === null || normalized.route_reuse_last_artifact === undefined
            ? null
            : Boolean(normalized.route_reuse_last_artifact),
        referenceImageCount: Number.isFinite(Number(normalized.reference_image_count))
            ? Number(normalized.reference_image_count)
            : null,
        referenceImageKind: normalized.reference_image_kind || null,
        contextMode: normalized.context_mode || null,
        contextReason: normalized.context_reason || null,
        lifecycleState: normalized.lifecycle_state || null,
        statusSemantics: normalized.status_semantics || null,
        lateFill: normalized.late_fill || null,
        surfaceState: normalized.surface_state || null,
        artifacts: normalized.artifacts,
        outputs: normalized.outputs || [],
        outputSlots: normalized.output_slots || [],
        outputBranches: normalized.output_branches || [],
        artifactBundles: normalized.artifact_bundles || [],
        artifact_bundles: normalized.artifact_bundles || [],
        artifactBundle: Array.isArray(normalized.artifact_bundles) && normalized.artifact_bundles.length
            ? normalized.artifact_bundles[normalized.artifact_bundles.length - 1]
            : null,
        requestSnapshot: sanitizeRequestSnapshot(normalized.request_snapshot || normalized.requestSnapshot),
    };
}

function getHydrationMessageResponseId(message = {}) {
    return String(message?.responseId || message?.response_id || '').trim();
}

function getHydrationMessageClientId(message = {}) {
    return String(message?.clientMessageId || message?.client_message_id || message?.message_id || '').trim();
}

function hydrationBoolean(value) {
    if (value === true || value === false) return value;
    const normalized = String(value ?? '').trim().toLowerCase();
    if (normalized === 'true' || normalized === '1' || normalized === 'yes') return true;
    if (normalized === 'false' || normalized === '0' || normalized === 'no') return false;
    return null;
}

function hydrationMessageIsTerminalResponse(message = {}) {
    const statusSemantics = message?.statusSemantics || message?.status_semantics || {};
    const lifecycleState = String(
        message?.lifecycleState
        || message?.lifecycle_state
        || statusSemantics?.canonicalLifecycleState
        || statusSemantics?.canonical_lifecycle_state
        || ''
    ).trim().toLowerCase();
    const hasOpenContinuation = hydrationBoolean(statusSemantics?.has_open_continuation ?? statusSemantics?.hasOpenContinuation);
    if (hasOpenContinuation === true) return false;
    const terminalFlag = hydrationBoolean(statusSemantics?.is_terminal ?? statusSemantics?.isTerminal ?? statusSemantics?.terminal);
    if (terminalFlag === true) return true;
    return ['completed', 'cancelled', 'failed', 'incomplete'].includes(lifecycleState);
}

function pendingRequestMatchesHydrationMessage(request = {}, conversationId = '', message = {}) {
    if (!request || request.phase !== 'running') return false;
    const targetConversationId = String(conversationId || '').trim();
    if (targetConversationId && String(request.conversationId || '').trim() !== targetConversationId) {
        return false;
    }
    const messageResponseId = getHydrationMessageResponseId(message);
    const requestResponseId = String(request.responseId || '').trim();
    if (messageResponseId && requestResponseId && messageResponseId === requestResponseId) {
        return true;
    }
    const messageClientId = getHydrationMessageClientId(message);
    const requestLoadingMessageId = String(request.loadingMessageId || '').trim();
    return Boolean(messageClientId && requestLoadingMessageId && messageClientId === requestLoadingMessageId);
}

function hydrationLoadingMessageHasActiveOwner(conversationId = '', message = {}) {
    if (!message || !message.isLoading) return false;
    const responseId = getHydrationMessageResponseId(message);
    if (
        responseId
        && typeof responseHasActiveSseStream === 'function'
        && responseHasActiveSseStream(responseId)
    ) {
        return true;
    }
    const pendingRequests = typeof listPendingRequests === 'function' ? listPendingRequests() : [];
    return pendingRequests.some((request) => pendingRequestMatchesHydrationMessage(request, conversationId, message));
}

function getLiveLoadingMessagesForHistoryHydration(conversationId = '') {
    const currentConversation = Array.isArray(state.conversations?.[conversationId])
        ? state.conversations[conversationId]
        : [];
    return currentConversation.filter((message) => hydrationLoadingMessageHasActiveOwner(conversationId, message));
}

function mergeHydratedConversationWithLiveLoadingMessages(conversationId = '', hydratedMessages = [], liveLoadingMessages = []) {
    const merged = Array.isArray(hydratedMessages) ? [...hydratedMessages] : [];
    (Array.isArray(liveLoadingMessages) ? liveLoadingMessages : []).forEach((liveMessage) => {
        if (!hydrationLoadingMessageHasActiveOwner(conversationId, liveMessage)) return;
        const responseId = getHydrationMessageResponseId(liveMessage);
        const clientMessageId = getHydrationMessageClientId(liveMessage);
        const matchingIndex = merged.findIndex((message) => {
            const candidateResponseId = getHydrationMessageResponseId(message);
            if (responseId && candidateResponseId && responseId === candidateResponseId) {
                return true;
            }
            const candidateClientId = getHydrationMessageClientId(message);
            return Boolean(clientMessageId && candidateClientId && clientMessageId === candidateClientId);
        });
        if (matchingIndex >= 0) {
            if (!hydrationMessageIsTerminalResponse(merged[matchingIndex])) {
                merged[matchingIndex] = liveMessage;
            }
            return;
        }
        merged.push(liveMessage);
    });
    return merged;
}

function hydrateConversationFromHistoryPayload(instanceId, payload = {}) {
    const messages = Array.isArray(payload.messages) ? payload.messages : [];
    const liveLoadingMessages = getLiveLoadingMessagesForHistoryHydration(instanceId);
    const hydratedMessages = messages.map((message) => deserializeHistoryMessage(message));
    state.conversations[instanceId] = mergeHydratedConversationWithLiveLoadingMessages(
        instanceId,
        hydratedMessages,
        liveLoadingMessages
    );
    const ledgerMetadata = buildConversationLedgerMetadata(instanceId);
    const metadata = registerConversationMetadata(instanceId, {
        ...(payload.conversation_metadata || {}),
        ...ledgerMetadata,
    });
    if (metadata) {
        appendConversationToSlotHistory(instanceId, metadata);
        const slotHistoryIds = Array.isArray(payload.slot_history_ids) ? payload.slot_history_ids : null;
        if (slotHistoryIds && metadata.workspace && metadata.slotId) {
            setSlotHistoryConversationIds(metadata.workspace, metadata.slotId, slotHistoryIds);
        }
    }
    state.chatHistoryLoaded[instanceId] = true;
    if (typeof resumeHydratedLateFillResponses === 'function') {
        resumeHydratedLateFillResponses(instanceId);
    }
    saveConversationSlots();
    renderConversationHistoryList();
    return state.conversations[instanceId];
}

function queueConversationLedgerBackfill(instanceId) {
    const key = String(instanceId || '').trim();
    if (!key) return;
    state.conversationHistoryLedgerBackfillInFlight = state.conversationHistoryLedgerBackfillInFlight || {};
    if (state.conversationHistoryLedgerBackfillInFlight[key]) return;
    state.conversationHistoryLedgerBackfillInFlight[key] = true;
    queueMicrotask(async () => {
        try {
            await persistChatHistory(key);
        } finally {
            if (state.conversationHistoryLedgerBackfillInFlight?.[key]) {
                delete state.conversationHistoryLedgerBackfillInFlight[key];
            }
        }
    });
}

async function fetchChatHistory(instanceId, { force = false, suppressVisibleRender = false } = {}) {
    if (!instanceId) return [];
    state.chatHistoryRequestsInFlight = state.chatHistoryRequestsInFlight || {};
    if (state.chatHistoryRequestsInFlight[instanceId]) {
        return state.chatHistoryRequestsInFlight[instanceId];
    }
    const requestPromise = (async () => {
        if (!isConversationEligibleForDurableHistory(instanceId)) {
            if (!state.conversations[instanceId]) {
                state.conversations[instanceId] = [];
            }
            state.chatHistoryLoaded[instanceId] = true;
            return state.conversations[instanceId];
        }
        const metadata = getConversationMetadata(instanceId);
        const slotHistoryKey = getConversationSlotHistoryKey(instanceId);
        const hasLocalSlotHistory = normalizeConversationHistoryIds(state.slotHistoryConversationIdsByKey?.[slotHistoryKey]).length > 0;
        const needsLineageRecovery = Boolean(metadata?.parentConversationId) && !hasLocalSlotHistory;
        if (!force && state.chatHistoryLoaded[instanceId] && Array.isArray(state.conversations[instanceId]) && !needsLineageRecovery) {
            appendInterruptedPendingRequestNotices(instanceId);
            return state.conversations[instanceId] || [];
        }
        const initialConversationSnapshot = buildConversationHistorySnapshot(instanceId);
        try {
            const response = await axios.get(`${state.flaskServerUrl}/api/chat_history`, {
                params: { instance_id: instanceId }
            });
            const currentConversation = Array.isArray(state.conversations[instanceId]) ? state.conversations[instanceId] : [];
            const conversationMutatedLocally = buildConversationHistorySnapshot(instanceId) !== initialConversationSnapshot;
            const payloadHasNewState = historyPayloadHasAnyNewState(instanceId, response.data || {});
            if (conversationMutatedLocally && !payloadHasNewState) {
                state.chatHistoryLoaded[instanceId] = true;
                syncConversationSlotHistoryFromPayload(instanceId, response.data || {});
                appendInterruptedPendingRequestNotices(instanceId);
                if (!suppressVisibleRender && isConversationVisible(instanceId)) {
                    activateConversationBottomAnchor(instanceId);
                    renderConversation(instanceId);
                }
                return currentConversation;
            }
            hydrateConversationFromHistoryPayload(instanceId, response.data || {});
            if (conversationLedgerMetadataNeedsBackfill(instanceId, response.data?.conversation_metadata || null)) {
                queueConversationLedgerBackfill(instanceId);
            }
            appendInterruptedPendingRequestNotices(instanceId);
            if (!suppressVisibleRender && isConversationVisible(instanceId)) {
                activateConversationBottomAnchor(instanceId);
                renderConversation(instanceId);
            }
            return state.conversations[instanceId];
        } catch (error) {
            console.warn('Unable to load chat history:', error?.message || error);
            if (!state.conversations[instanceId]) {
                state.conversations[instanceId] = [];
            }
            state.chatHistoryLoaded[instanceId] = true;
            appendInterruptedPendingRequestNotices(instanceId);
            renderConversationHistoryList();
            if (!suppressVisibleRender && isConversationVisible(instanceId)) {
                renderConversation(instanceId);
            }
            return state.conversations[instanceId];
        }
    })();
    state.chatHistoryRequestsInFlight[instanceId] = requestPromise;
    try {
        return await requestPromise;
    } finally {
        if (state.chatHistoryRequestsInFlight?.[instanceId] === requestPromise) {
            delete state.chatHistoryRequestsInFlight[instanceId];
        }
    }
}

async function persistChatHistory(instanceId) {
    if (!instanceId) return;
    const ledgerMetadata = buildConversationLedgerMetadata(instanceId);
    const mergedConversationMetadata = registerConversationMetadata(instanceId, ledgerMetadata);
    saveConversationSlots();
    const metadata = getConversationHistoryMetadata(instanceId);
    if (!metadata) return;
    const payload = {
        instance_id: instanceId,
        model: metadata.model,
        backend: metadata.backend,
        capability: metadata.capability,
        conversation_metadata: serializeConversationMetadata(mergedConversationMetadata || metadata.conversationMetadata),
        messages: serializeConversationForHistory(instanceId),
    };
    let serializedPayload = '';
    try {
        serializedPayload = JSON.stringify(payload);
    } catch (_error) {
        serializedPayload = '';
    }
    state.chatHistoryPersistedSnapshotsById = state.chatHistoryPersistedSnapshotsById || {};
    state.chatHistoryPersistRequestsInFlight = state.chatHistoryPersistRequestsInFlight || {};
    if (serializedPayload && state.chatHistoryPersistedSnapshotsById[instanceId] === serializedPayload) {
        return;
    }
    const inFlight = state.chatHistoryPersistRequestsInFlight[instanceId];
    if (serializedPayload && inFlight?.serializedPayload === serializedPayload && inFlight.promise) {
        return inFlight.promise;
    }
    const persistPromise = axios.post(`${state.flaskServerUrl}/api/chat_history`, payload);
    if (serializedPayload) {
        state.chatHistoryPersistRequestsInFlight[instanceId] = {
            serializedPayload,
            promise: persistPromise,
        };
    }
    try {
        await persistPromise;
        if (serializedPayload) {
            state.chatHistoryPersistedSnapshotsById[instanceId] = serializedPayload;
        }
    } catch (error) {
        console.warn('Unable to persist chat history:', error?.message || error);
    } finally {
        if (state.chatHistoryPersistRequestsInFlight?.[instanceId]?.promise === persistPromise) {
            delete state.chatHistoryPersistRequestsInFlight[instanceId];
        }
    }
}

async function deleteChatHistory(instanceId) {
    if (!instanceId) return;
    if (!isConversationEligibleForDurableHistory(instanceId)) return;
    try {
        await axios.delete(`${state.flaskServerUrl}/api/chat_history`, {
            params: { instance_id: instanceId }
        });
    } catch (error) {
        console.warn('Unable to delete chat history:', error?.message || error);
    }
}
