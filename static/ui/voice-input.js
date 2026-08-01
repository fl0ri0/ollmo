const VOICE_INPUT_PREFERENCES_STORAGE_KEY = 'ollmo_voice_input_preferences_v1';

function sanitizeVoiceInputPreferenceTarget(rawValue) {
    if (!rawValue || typeof rawValue !== 'object') return null;
    const model = String(rawValue.model || '').trim();
    const backend = normalizeBackend(rawValue.backend || '');
    const capability = normalizeCapability(rawValue.capability || 'speech_to_text');
    if (!model && !backend) return null;
    const payload = {};
    if (model) payload.model = model;
    if (backend) payload.backend = backend;
    if (capability) payload.capability = capability;
    return Object.keys(payload).length ? payload : null;
}

function cloneVoiceInputPreferences(rawValue) {
    const source = rawValue && typeof rawValue === 'object' ? rawValue : {};
    return {
        primaryTarget: sanitizeVoiceInputPreferenceTarget(source.primaryTarget || source.primary_target),
        fallbackTarget: sanitizeVoiceInputPreferenceTarget(source.fallbackTarget || source.fallback_target),
    };
}

function serializeVoiceInputPreferenceTarget(target) {
    const normalized = sanitizeVoiceInputPreferenceTarget(target);
    return normalized ? JSON.stringify(normalized) : '';
}

function parseVoiceInputPreferenceTarget(value) {
    const token = String(value || '').trim();
    if (!token) return null;
    try {
        return sanitizeVoiceInputPreferenceTarget(JSON.parse(token));
    } catch (_error) {
        return null;
    }
}

function saveVoiceInputPreferences() {
    const payload = {
        preferences: cloneVoiceInputPreferences(state.voice.preferences),
        expanded: Boolean(state.voice.preferencesExpanded),
    };
    try {
        localStorage.setItem(VOICE_INPUT_PREFERENCES_STORAGE_KEY, JSON.stringify(payload));
    } catch (error) {
        console.warn('Could not persist voice input preferences:', error);
    }
    return payload;
}

function initializeVoiceInputPreferences() {
    state.voice.preferences = cloneVoiceInputPreferences(null);
    state.voice.preferencesExpanded = false;
    try {
        const raw = localStorage.getItem(VOICE_INPUT_PREFERENCES_STORAGE_KEY);
        if (!raw) return;
        const parsed = JSON.parse(raw);
        if (!parsed || typeof parsed !== 'object') return;
        state.voice.preferences = cloneVoiceInputPreferences(parsed.preferences);
        state.voice.preferencesExpanded = Boolean(parsed.expanded);
    } catch (error) {
        console.warn('Could not load voice input preferences:', error);
    }
}

function buildVoiceInputPreferenceKey(modelName, backend) {
    return `${normalizeBackend(backend)}::${String(modelName || '').trim()}`;
}

function isUsableVoiceInputSpeechInstance(instance) {
    if (!instance || !instance.instance_id) return false;
    if (state.modelOperations.stopping.has(instance.instance_id)) return false;
    if (!frontendInstanceSupportsCapability(instance, 'speech_to_text')) return false;
    const readiness = String(instance.readiness || '').trim().toLowerCase();
    if (['failed', 'error', 'unreachable', 'stopped', 'stopping'].includes(readiness)) {
        return false;
    }
    if (instance.port_listening === false) {
        return false;
    }
    return true;
}

function getVoiceInputRunningInstances() {
    return (state.runningInstances || []).filter((instance) => isUsableVoiceInputSpeechInstance(instance));
}

function scoreVoiceInputSpeechInstance(instance) {
    const readiness = String(instance?.readiness || '').trim().toLowerCase();
    const activity = String(instance?.activity || '').trim().toLowerCase();
    const readinessRank = readiness === 'ready' ? 2 : readiness === 'started' || readiness === 'idle' ? 1 : 0;
    const activityRank = activity === 'idle' || activity === 'ready' ? 1 : 0;
    return [readinessRank, activityRank, String(instance?.instance_id || '')];
}

function pickBestVoiceInputSpeechInstance(instances) {
    const ranked = Array.isArray(instances) ? [...instances] : [];
    ranked.sort((left, right) => {
        const leftScore = scoreVoiceInputSpeechInstance(left);
        const rightScore = scoreVoiceInputSpeechInstance(right);
        if (leftScore[0] !== rightScore[0]) return rightScore[0] - leftScore[0];
        if (leftScore[1] !== rightScore[1]) return rightScore[1] - leftScore[1];
        return String(rightScore[2]).localeCompare(String(leftScore[2]));
    });
    return ranked[0] || null;
}

function getVoiceInputPreferenceCandidates() {
    const candidatesByKey = new Map();
    const runningInstances = getVoiceInputRunningInstances();
    const runningByKey = new Map();

    runningInstances.forEach((instance) => {
        const modelName = String(instance.model || instance.modelName || '').trim();
        const backend = normalizeBackend(instance.backend || 'ollama');
        if (!modelName) return;
        const key = buildVoiceInputPreferenceKey(modelName, backend);
        const items = runningByKey.get(key) || [];
        items.push(instance);
        runningByKey.set(key, items);
    });

    (state.availableModels || []).forEach((entry) => {
        if (!entry || !frontendInstanceSupportsCapability(entry, 'speech_to_text')) return;
        const modelName = String(entry.name || entry.model || entry.modelName || '').trim();
        const backend = normalizeBackend(entry.backend || 'ollama');
        if (!modelName) return;
        const key = buildVoiceInputPreferenceKey(modelName, backend);
        candidatesByKey.set(key, {
            key,
            model: modelName,
            backend,
            availableEntry: entry,
            launchable: entry.runnable !== false,
            runningInstances: [...(runningByKey.get(key) || [])],
        });
    });

    runningByKey.forEach((instances, key) => {
        if (candidatesByKey.has(key)) return;
        const sample = instances[0];
        candidatesByKey.set(key, {
            key,
            model: String(sample?.model || sample?.modelName || '').trim(),
            backend: normalizeBackend(sample?.backend || 'ollama'),
            availableEntry: null,
            launchable: false,
            runningInstances: [...instances],
        });
    });

    return Array.from(candidatesByKey.values()).sort((left, right) => {
        const leftRunning = left.runningInstances.length > 0 ? 1 : 0;
        const rightRunning = right.runningInstances.length > 0 ? 1 : 0;
        if (leftRunning !== rightRunning) return rightRunning - leftRunning;
        const leftLaunchable = left.launchable ? 1 : 0;
        const rightLaunchable = right.launchable ? 1 : 0;
        if (leftLaunchable !== rightLaunchable) return rightLaunchable - leftLaunchable;
        const backendCompare = formatBackendLabel(left.backend).localeCompare(formatBackendLabel(right.backend));
        if (backendCompare !== 0) return backendCompare;
        return formatModelDisplayName(left.model).localeCompare(formatModelDisplayName(right.model));
    });
}

function buildVoiceInputPreferenceCandidateLabel(candidate) {
    const base = `${formatModelDisplayName(candidate?.model || 'speech helper')} (${formatBackendLabel(candidate?.backend || 'ollama')})`;
    const runningCount = Array.isArray(candidate?.runningInstances) ? candidate.runningInstances.length : 0;
    if (runningCount > 1) {
        return `${base} • ${runningCount} running`;
    }
    if (runningCount === 1) {
        return `${base} • running`;
    }
    if (candidate?.launchable) {
        return `${base} • auto-start`;
    }
    return `${base} • unavailable`;
}

function populateVoiceInputPreferenceSelect(select, candidates, selectedTarget, emptyLabel = 'Ollmo default speech helper') {
    if (!select) return;
    const normalizedSelected = sanitizeVoiceInputPreferenceTarget(selectedTarget);
    const selectedValue = serializeVoiceInputPreferenceTarget(normalizedSelected);
    const seenValues = new Set();
    select.innerHTML = '';

    const emptyOption = document.createElement('option');
    emptyOption.value = '';
    emptyOption.textContent = emptyLabel;
    select.appendChild(emptyOption);

    candidates.forEach((candidate) => {
        const value = serializeVoiceInputPreferenceTarget({
            model: candidate.model,
            backend: candidate.backend,
            capability: 'speech_to_text',
        });
        if (!value || seenValues.has(value)) return;
        seenValues.add(value);
        const option = document.createElement('option');
        option.value = value;
        option.textContent = buildVoiceInputPreferenceCandidateLabel(candidate);
        select.appendChild(option);
    });

    if (selectedValue && !seenValues.has(selectedValue) && normalizedSelected) {
        const option = document.createElement('option');
        option.value = selectedValue;
        option.textContent = `${formatModelDisplayName(normalizedSelected.model || 'speech helper')} (${formatBackendLabel(normalizedSelected.backend || 'runtime')} • unavailable)`;
        select.appendChild(option);
    }

    select.value = selectedValue;
}

function hasVoiceInputPreferenceSurface() {
    const preferences = cloneVoiceInputPreferences(state.voice.preferences);
    return Boolean(
        getVoiceInputPreferenceCandidates().length
        || preferences.primaryTarget
        || preferences.fallbackTarget
    );
}

function renderVoiceInputPreferences() {
    if (
        !elements.voiceInputPreferences
        || !elements.voiceInputPrefsToggle
        || !elements.voiceInputPrefsBody
        || !elements.voiceInputPrimaryTarget
        || !elements.voiceInputFallbackTarget
    ) {
        return;
    }
    const preferences = cloneVoiceInputPreferences(state.voice.preferences);
    const show = isResponsesWorkbenchActive() && !state.arena.enabled && hasVoiceInputPreferenceSurface();
    elements.voiceInputPreferences.hidden = !show;
    if (!show) return;

    const candidates = getVoiceInputPreferenceCandidates();
    elements.voiceInputPrefsBody.hidden = !state.voice.preferencesExpanded;
    elements.voiceInputPrefsToggle.setAttribute('aria-expanded', state.voice.preferencesExpanded ? 'true' : 'false');
    populateVoiceInputPreferenceSelect(
        elements.voiceInputPrimaryTarget,
        candidates,
        preferences.primaryTarget,
        'Ollmo default speech helper'
    );
    populateVoiceInputPreferenceSelect(
        elements.voiceInputFallbackTarget,
        candidates,
        preferences.fallbackTarget,
        'No fallback speech helper'
    );
}

function applyVoiceInputPreferencesFromUi() {
    state.voice.preferences = cloneVoiceInputPreferences({
        primaryTarget: parseVoiceInputPreferenceTarget(elements.voiceInputPrimaryTarget?.value || ''),
        fallbackTarget: parseVoiceInputPreferenceTarget(elements.voiceInputFallbackTarget?.value || ''),
    });
    saveVoiceInputPreferences();
    renderVoiceInputPreferences();
    updateVoiceInputButtonState();
}

function voiceInputTargetMatchesInstance(instance, target) {
    const normalizedTarget = sanitizeVoiceInputPreferenceTarget(target);
    if (!instance || !normalizedTarget) return false;
    const modelName = String(instance.model || instance.modelName || '').trim();
    if (normalizedTarget.model && modelName !== normalizedTarget.model) {
        return false;
    }
    if (normalizedTarget.backend && normalizeBackend(instance.backend) !== normalizedTarget.backend) {
        return false;
    }
    return frontendInstanceSupportsCapability(instance, 'speech_to_text');
}

function voiceInputTargetMatchesCandidate(candidate, target) {
    const normalizedTarget = sanitizeVoiceInputPreferenceTarget(target);
    if (!candidate || !normalizedTarget) return false;
    if (normalizedTarget.model && String(candidate.model || '').trim() !== normalizedTarget.model) {
        return false;
    }
    if (normalizedTarget.backend && normalizeBackend(candidate.backend) !== normalizedTarget.backend) {
        return false;
    }
    return true;
}

function findVoiceInputRunningInstanceByTarget(target) {
    if (!target) return null;
    return pickBestVoiceInputSpeechInstance(
        getVoiceInputRunningInstances().filter((instance) => voiceInputTargetMatchesInstance(instance, target))
    );
}

function findVoiceInputLaunchCandidateByTarget(target, excludedKeys = null) {
    if (!target) return null;
    return getVoiceInputPreferenceCandidates().find((candidate) => (
        !(excludedKeys instanceof Set && excludedKeys.has(candidate.key))
        && candidate.launchable
        && voiceInputTargetMatchesCandidate(candidate, target)
    )) || null;
}

function pickDefaultVoiceInputLaunchCandidate(excludedKeys = null) {
    return getVoiceInputPreferenceCandidates().find((candidate) => (
        !(excludedKeys instanceof Set && excludedKeys.has(candidate.key))
        && candidate.launchable
    )) || null;
}

function resolveVoiceInputButtonPlan(options = {}) {
    const excludedLaunchKeys = options.excludedLaunchKeys instanceof Set
        ? options.excludedLaunchKeys
        : null;
    const preferences = cloneVoiceInputPreferences(state.voice.preferences);
    const primaryRunning = findVoiceInputRunningInstanceByTarget(preferences.primaryTarget);
    if (primaryRunning) {
        return { instance: primaryRunning, candidate: null, source: 'primary_running' };
    }
    const primaryLaunch = findVoiceInputLaunchCandidateByTarget(preferences.primaryTarget, excludedLaunchKeys);
    if (primaryLaunch) {
        return { instance: null, candidate: primaryLaunch, source: 'primary_launch' };
    }
    const fallbackRunning = findVoiceInputRunningInstanceByTarget(preferences.fallbackTarget);
    if (fallbackRunning) {
        return { instance: fallbackRunning, candidate: null, source: 'fallback_running' };
    }
    const fallbackLaunch = findVoiceInputLaunchCandidateByTarget(preferences.fallbackTarget, excludedLaunchKeys);
    if (fallbackLaunch) {
        return { instance: null, candidate: fallbackLaunch, source: 'fallback_launch' };
    }
    const defaultRunning = pickBestVoiceInputSpeechInstance(getVoiceInputRunningInstances());
    if (defaultRunning) {
        return { instance: defaultRunning, candidate: null, source: 'default_running' };
    }
    const defaultLaunch = pickDefaultVoiceInputLaunchCandidate(excludedLaunchKeys);
    if (defaultLaunch) {
        return { instance: null, candidate: defaultLaunch, source: 'default_launch' };
    }
    return { instance: null, candidate: null, source: '' };
}

function getVoiceInputLaunchSequence() {
    const preferences = cloneVoiceInputPreferences(state.voice.preferences);
    const sequence = [];
    const seen = new Set();

    const pushCandidate = (candidate) => {
        if (!candidate?.key || seen.has(candidate.key)) return;
        seen.add(candidate.key);
        sequence.push(candidate);
    };

    pushCandidate(findVoiceInputLaunchCandidateByTarget(preferences.primaryTarget));
    pushCandidate(findVoiceInputLaunchCandidateByTarget(preferences.fallbackTarget));
    pushCandidate(pickDefaultVoiceInputLaunchCandidate());
    return sequence;
}

function hasVoiceCaptureSupport() {
    return Boolean(
        navigator?.mediaDevices?.getUserMedia &&
        typeof MediaRecorder !== 'undefined'
    );
}

function selectVoiceMimeType() {
    if (typeof MediaRecorder === 'undefined' || typeof MediaRecorder.isTypeSupported !== 'function') {
        return '';
    }
    const choices = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4'];
    return choices.find((item) => MediaRecorder.isTypeSupported(item)) || '';
}

function stopVoiceStreamTracks() {
    if (!state.voice.stream) return;
    try {
        state.voice.stream.getTracks().forEach((track) => track.stop());
    } catch (error) {
        console.warn('Failed to stop voice stream tracks:', error);
    }
    state.voice.stream = null;
}

function clearVoiceTimers() {
    if (state.voice.silenceRafId) {
        cancelAnimationFrame(state.voice.silenceRafId);
        state.voice.silenceRafId = null;
    }
    if (state.voice.maxDurationTimer) {
        clearTimeout(state.voice.maxDurationTimer);
        state.voice.maxDurationTimer = null;
    }
    state.voice.silenceStartedAt = null;
}

function stopVoiceMonitoring() {
    clearVoiceTimers();
    if (state.voice.sourceNode) {
        try {
            state.voice.sourceNode.disconnect();
        } catch (error) {
            console.warn('Failed to disconnect voice source node:', error);
        }
    }
    if (state.voice.analyser) {
        try {
            state.voice.analyser.disconnect();
        } catch (error) {
            console.warn('Failed to disconnect voice analyser node:', error);
        }
    }
    state.voice.sourceNode = null;
    state.voice.analyser = null;
    if (state.voice.audioContext) {
        const ctx = state.voice.audioContext;
        state.voice.audioContext = null;
        try {
            if (ctx.state !== 'closed') {
                ctx.close();
            }
        } catch (error) {
            console.warn('Failed to close voice audio context:', error);
        }
    }
}

function startVoiceMonitoring(stream, recorder) {
    stopVoiceMonitoring();
    const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextCtor) {
        return;
    }
    try {
        const ctx = new AudioContextCtor();
        const source = ctx.createMediaStreamSource(stream);
        const analyser = ctx.createAnalyser();
        analyser.fftSize = 1024;
        analyser.smoothingTimeConstant = 0.2;
        source.connect(analyser);

        state.voice.audioContext = ctx;
        state.voice.sourceNode = source;
        state.voice.analyser = analyser;
        state.voice.silenceStartedAt = null;

        const data = new Uint8Array(analyser.fftSize);
        const threshold = Number(state.voice.silenceThreshold || 0.018);
        const silenceDurationMs = Number(state.voice.silenceDurationMs || 2000);

        const monitor = () => {
            if (!state.voice.recording || !state.voice.analyser || recorder.state !== 'recording') {
                state.voice.silenceRafId = null;
                return;
            }
            state.voice.analyser.getByteTimeDomainData(data);
            let sum = 0;
            for (let i = 0; i < data.length; i += 1) {
                const normalized = (data[i] - 128) / 128;
                sum += normalized * normalized;
            }
            const rms = Math.sqrt(sum / data.length);
            const now = Date.now();
            if (rms < threshold) {
                if (!state.voice.silenceStartedAt) {
                    state.voice.silenceStartedAt = now;
                } else if ((now - state.voice.silenceStartedAt) >= silenceDurationMs) {
                    updateGlobalModelStatus('Silence detected. Finishing recording...');
                    stopVoiceCapture();
                    state.voice.silenceRafId = null;
                    return;
                }
            } else {
                state.voice.silenceStartedAt = null;
            }
            state.voice.silenceRafId = requestAnimationFrame(monitor);
        };
        state.voice.silenceRafId = requestAnimationFrame(monitor);
    } catch (error) {
        console.warn('Voice silence monitor unavailable:', error);
    }

    const maxDurationMs = Number(state.voice.maxDurationMs || 14000);
    state.voice.maxDurationTimer = setTimeout(() => {
        if (state.voice.recording && recorder.state === 'recording') {
            updateGlobalModelStatus('Max recording time reached. Transcribing...');
            stopVoiceCapture();
        }
    }, maxDurationMs);
}

function formatVoiceInputSpeechLabel(target) {
    if (!target) return 'speech helper';
    const modelName = String(target.model || target.instance_id || '').trim();
    if (!modelName) return 'speech helper';
    return `${formatModelDisplayName(modelName)} (${formatBackendLabel(target.backend || 'ollama')})`;
}

async function ensureVoiceInputSpeechInstance({ startIfNeeded = true } = {}) {
    const failedLaunchKeys = new Set();
    const plan = resolveVoiceInputButtonPlan({ excludedLaunchKeys: failedLaunchKeys });
    if (plan.instance) {
        return plan.instance;
    }
    if (!startIfNeeded) {
        return null;
    }
    const launchSequence = getVoiceInputLaunchSequence();
    if (!launchSequence.length) {
        updateGlobalModelStatus('No launchable speech helper is available for voice input.');
        updateVoiceInputButtonState();
        return null;
    }
    state.voice.preparing = true;
    updateVoiceInputButtonState();
    try {
        while (true) {
            const nextPlan = resolveVoiceInputButtonPlan({ excludedLaunchKeys: failedLaunchKeys });
            if (nextPlan.instance) {
                return nextPlan.instance;
            }
            const candidate = nextPlan.candidate;
            if (!candidate) {
                break;
            }
            const entry = candidate.availableEntry || {};
            const label = buildVoiceInputPreferenceCandidateLabel(candidate);
            updateGlobalModelStatus(`Starting voice helper ${label}...`);
            const startedInstance = await startModel(candidate.model, candidate.backend, {
                capability: 'speech_to_text',
                modelName: entry.modelName || entry.model || candidate.model,
                modelPath: entry.model_path || null,
                hfFile: entry.hf_file || null,
                launchDefaults: entry.launch_defaults || null,
                suppressAlert: true,
            });
            if (startedInstance && isUsableVoiceInputSpeechInstance(startedInstance)) {
                return startedInstance;
            }
            failedLaunchKeys.add(candidate.key);
        }
        updateGlobalModelStatus('No speech helper could be started for voice input.');
        return null;
    } finally {
        state.voice.preparing = false;
        updateVoiceInputButtonState();
    }
}

function updateVoiceInputButtonState() {
    if (!elements.voiceInputBtn) return;
    const support = hasVoiceCaptureSupport();
    const plan = resolveVoiceInputButtonPlan();
    const isRecording = Boolean(state.voice.recording);
    const isPreparing = Boolean(state.voice.preparing);
    const isTranscribing = Boolean(state.voice.transcribing);
    const disabled = !support || isPreparing || isTranscribing || (!isRecording && !plan.instance && !plan.candidate);

    elements.voiceInputBtn.disabled = disabled;
    elements.voiceInputBtn.classList.toggle('is-recording', isRecording);
    elements.voiceInputBtn.classList.toggle('is-loading', isPreparing || isTranscribing);

    if (isRecording) {
        elements.voiceInputBtn.innerHTML = '<i class="fas fa-circle"></i> Listening...';
        elements.voiceInputBtn.title = 'Auto-stop on silence. Click to stop now.';
        return;
    }
    if (isPreparing) {
        elements.voiceInputBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Starting';
        elements.voiceInputBtn.title = 'Starting a speech helper for voice input';
        return;
    }
    if (isTranscribing) {
        elements.voiceInputBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Transcribing';
        elements.voiceInputBtn.title = 'Transcribing audio via speech helper';
        return;
    }

    elements.voiceInputBtn.innerHTML = '<i class="fas fa-microphone"></i> Voice Input';
    if (!support) {
        elements.voiceInputBtn.title = 'Browser has no microphone recording support.';
    } else if (plan.instance) {
        elements.voiceInputBtn.title = `Record voice and transcribe with ${formatVoiceInputSpeechLabel(plan.instance)}.`;
    } else if (plan.candidate) {
        elements.voiceInputBtn.title = `Record voice input. Ollmo will start ${buildVoiceInputPreferenceCandidateLabel(plan.candidate)} automatically.`;
    } else {
        elements.voiceInputBtn.title = 'No runnable speech_to_text helper is available for voice input.';
    }
}

async function transcribeVoiceToPrompt(file) {
    const speechInstance = await ensureVoiceInputSpeechInstance({ startIfNeeded: true });
    if (!speechInstance) {
        updateGlobalModelStatus('No speech helper is available for voice input.');
        updateVoiceInputButtonState();
        return;
    }
    state.voice.transcribing = true;
    updateVoiceInputButtonState();
    updateGlobalModelStatus(`Transcribing voice input with ${formatVoiceInputSpeechLabel(speechInstance)}...`);
    try {
        const formData = new FormData();
        formData.append('instance_id', speechInstance.instance_id);
        formData.append('capability', 'speech_to_text');
        formData.append('file', file, file.name || 'voice-input.webm');

        const configuredLanguage = String(state.settings?.sttLanguage || '').trim();
        if (configuredLanguage) {
            formData.append('language', configuredLanguage);
        } else {
            const languageTag = String(navigator.language || '').trim();
            if (languageTag) {
                const language = languageTag.split('-')[0];
                if (language) {
                    formData.append('language', language);
                }
            }
        }

        const configuredTask = String(state.settings?.sttTask || 'transcribe').trim() || 'transcribe';
        if (configuredTask) {
            formData.append('task', configuredTask);
        }

        const timeoutMs = Math.max(30_000, getInferTimeoutForRequest(speechInstance, file, ''));
        const timeoutSec = Math.max(30, Math.ceil(timeoutMs / 1000));
        formData.append('infer_timeout_sec', String(timeoutSec));
        const response = await axios.post(
            `${state.flaskServerUrl}/api/responses`,
            formData,
            { timeout: timeoutMs + 15_000 }
        );
        const transcript = String(response.data?.output_text || response.data?.content || '').trim();
        if (!transcript) {
            updateGlobalModelStatus('Voice captured, but transcription was empty.');
            return;
        }
        const existing = String(elements.userInput?.value || '').trim();
        elements.userInput.value = existing ? `${existing}\n${transcript}` : transcript;
        if (typeof syncUserInputComposer === 'function') {
            syncUserInputComposer();
        }
        updateSendButtonState();
        elements.userInput.focus();
        if (canSendPrompt()) {
            updateGlobalModelStatus('Voice transcription ready. Sending prompt...');
            await sendMessage();
            return;
        }
        updateGlobalModelStatus('Voice transcription ready.');
        setTimeout(() => {
            if (elements.modelStatus?.textContent?.includes('Voice transcription ready')) {
                updateGlobalModelStatus('');
            }
        }, 1800);
    } catch (error) {
        console.error('Voice transcription failed:', error);
        const message = error.response?.data?.error || error.message || 'Unknown error';
        updateGlobalModelStatus(`Voice transcription failed: ${message}`);
    } finally {
        state.voice.transcribing = false;
        updateVoiceInputButtonState();
    }
}

async function startVoiceCapture() {
    if (state.voice.recording || state.voice.preparing || state.voice.transcribing) return;
    if (!hasVoiceCaptureSupport()) {
        updateGlobalModelStatus('Microphone capture is not supported in this browser.');
        updateVoiceInputButtonState();
        return;
    }

    const speechInstance = await ensureVoiceInputSpeechInstance({ startIfNeeded: true });
    if (!speechInstance) {
        updateGlobalModelStatus('No speech helper is available for voice input.');
        updateVoiceInputButtonState();
        return;
    }

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const mimeType = selectVoiceMimeType();
        const recorder = mimeType
            ? new MediaRecorder(stream, { mimeType })
            : new MediaRecorder(stream);
        state.voice.stream = stream;
        state.voice.recorder = recorder;
        state.voice.chunks = [];
        state.voice.mimeType = mimeType || recorder.mimeType || 'audio/webm';
        state.voice.recording = true;

        recorder.ondataavailable = (event) => {
            if (event.data && event.data.size > 0) {
                state.voice.chunks.push(event.data);
            }
        };
        recorder.onerror = (event) => {
            console.error('Voice recorder error:', event);
            updateGlobalModelStatus('Voice recording failed.');
        };
        recorder.onstop = async () => {
            const chunks = [...(state.voice.chunks || [])];
            const finalMime = state.voice.mimeType || recorder.mimeType || 'audio/webm';
            state.voice.recording = false;
            state.voice.recorder = null;
            state.voice.chunks = [];
            stopVoiceMonitoring();
            stopVoiceStreamTracks();
            updateVoiceInputButtonState();

            if (!chunks.length) {
                updateGlobalModelStatus('No audio captured.');
                return;
            }
            const extension = finalMime.includes('mp4') ? 'm4a' : 'webm';
            const blob = new Blob(chunks, { type: finalMime });
            const file = new File([blob], `voice-${Date.now()}.${extension}`, { type: finalMime });
            await transcribeVoiceToPrompt(file);
        };

        recorder.start();
        startVoiceMonitoring(stream, recorder);
        updateGlobalModelStatus(`Recording with ${formatVoiceInputSpeechLabel(speechInstance)}... auto-stop on silence.`);
        updateVoiceInputButtonState();
    } catch (error) {
        console.error('Unable to start voice capture:', error);
        const message = error?.name === 'NotAllowedError'
            ? 'Microphone permission denied.'
            : (error?.message || 'Unknown microphone error');
        updateGlobalModelStatus(`Voice capture unavailable: ${message}`);
        state.voice.recording = false;
        state.voice.recorder = null;
        state.voice.chunks = [];
        stopVoiceMonitoring();
        stopVoiceStreamTracks();
        updateVoiceInputButtonState();
    }
}

function stopVoiceCapture() {
    if (!state.voice.recording || !state.voice.recorder) return;
    try {
        if (state.voice.recorder.state !== 'recording') {
            return;
        }
        state.voice.recorder.stop();
    } catch (error) {
        console.error('Unable to stop voice recorder:', error);
        state.voice.recording = false;
        state.voice.recorder = null;
        state.voice.chunks = [];
        stopVoiceMonitoring();
        stopVoiceStreamTracks();
        updateVoiceInputButtonState();
    }
}

function toggleVoiceCapture() {
    if (state.voice.preparing || state.voice.transcribing) return;
    if (state.voice.recording) {
        stopVoiceCapture();
    } else {
        startVoiceCapture();
    }
}
