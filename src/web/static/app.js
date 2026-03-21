// Global state
let currentTaskId = null;
let viewerIframe = null;
let pendingTaskId = null;
let mediaRecorder = null;
let mediaStream = null;
let capturedChunks = [];
let transcriptionQueue = [];
let isProcessingChunk = false;
let liveTranscriptText = '';
let isRecording = false;
let stopRecordingTimer = null;
let transcribeController = null;
let recordingSessionId = 0;
let selectedMicDeviceId = '';
let isPresentationMode = false;
let hasRequestedDeviceLabels = false;
let ttsTarget = 'web';
let ttsVoice = 'alloy';
let autoSpeak = false;
let currentAudio = null;
let isSpeaking = false;

const MAX_RECORDING_MS = 20000;
const LIVE_CHUNK_MS = 1400;
const MIC_DEVICE_STORAGE_KEY = 'speechMicDeviceId';
const UI_MODE_STORAGE_KEY = 'uiMode';
const TTS_TARGET_STORAGE_KEY = 'ttsTarget';
const TTS_VOICE_STORAGE_KEY = 'ttsVoice';
const AUTO_SPEAK_STORAGE_KEY = 'autoSpeak';
const PRESENTATION_LINE_MAX = 3;

const PRESENTATION_STEP_LINES = {
  llm: [
    'Understanding user request',
    'Refining language into motion prompt',
    'Preparing the command',
  ],
  kimodo: [
    'Generating movement',
    'Building trajectory',
    'Finalizing motion',
  ],
  zmq: [
    'Streaming motion to the robot',
    'Publishing commands to robot',
    'Robot is alive!',
  ],
};

document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('prompt-form');
  const input = document.getElementById('prompt-input');
  const statusMsg = document.getElementById('status-message');
  const refinedText = document.getElementById('refined-prompt-text');
  const spinner = document.getElementById('loading-spinner');
  const viewerContainer = document.getElementById('viewer-container');
  const actionBar = document.getElementById('action-bar');
  const deployStatus = document.getElementById('deploy-status');
  const deployBtn = document.getElementById('deploy-btn');
  const micBtn = document.getElementById('mic-btn');
  const settingsBtn = document.getElementById('settings-btn');
  const settingsModal = document.getElementById('settings-modal');
  const settingsCloseBtn = document.getElementById('settings-close-btn');
  const micSelect = document.getElementById('mic-select');
  const presentationModeToggle = document.getElementById('presentation-mode-toggle');
  const modeDescription = document.getElementById('mode-description');
  const presentationThoughtsEl = document.getElementById('presentation-thoughts');
  const presTextArea = document.getElementById('pres-text-area');
  const presStepsEl = document.getElementById('pres-steps');
  const STEP_ORDER = ['llm', 'kimodo', 'zmq'];
  const agentReply = document.getElementById('agent-reply');
  const agentReplyText = document.getElementById('agent-reply-text');
  const speakBtn = document.getElementById('speak-btn');
  const speakIconPlay = document.getElementById('speak-icon-play');
  const speakIconStop = document.getElementById('speak-icon-stop');
  const ttsTargetSelect = document.getElementById('tts-target-select');
  const ttsVoiceSelect = document.getElementById('tts-voice-select');
  const autoSpeakToggle = document.getElementById('auto-speak-toggle');
  let presentationLines = [];
  let presentationLineId = 0;
  const lastPresentationLineByStep = { llm: '', kimodo: '', zmq: '' };

  function readStorage(key, fallback = '') {
    try {
      const value = window.localStorage.getItem(key);
      return value == null ? fallback : value;
    } catch (err) {
      return fallback;
    }
  }

  function writeStorage(key, value) {
    try {
      window.localStorage.setItem(key, value);
    } catch (err) {
      // Ignore storage write failures.
    }
  }

  function updateModeDescription() {
    modeDescription.textContent = isPresentationMode
      ? 'Presentation mode hides the animation preview.'
      : 'Developer mode shows the animation preview.';
  }

  function showStatusMessage(text) {
    if (isPresentationMode) return;
    statusMsg.classList.remove('hidden');
    refinedText.textContent = text;
  }

  function showAgentReply(text) {
    if (!text) return;
    agentReply.classList.remove('hidden');
    agentReplyText.textContent = text;
  }

  function hideAgentReply() {
    agentReply.classList.add('hidden');
    agentReplyText.textContent = '';
    stopSpeaking();
  }

  function setSpeakingState(speaking) {
    isSpeaking = speaking;
    speakBtn.classList.toggle('is-speaking', speaking);
    speakIconPlay.classList.toggle('hidden', speaking);
    speakIconStop.classList.toggle('hidden', !speaking);
    speakBtn.title = speaking ? 'Stop playback' : 'Play reply aloud';
  }

  function stopSpeaking() {
    if (currentAudio) {
      currentAudio.pause();
      currentAudio = null;
    }
    setSpeakingState(false);
  }

  async function speakText(text) {
    if (!text) return;
    stopSpeaking();
    setSpeakingState(true);

    if (ttsTarget === 'robot') {
      try {
        const res = await fetch('/api/speak', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text, target: 'robot' }),
        });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          showStatusMessage('Robot TTS: ' + (err.detail || 'failed'));
        }
      } catch (err) {
        console.error(err);
        showStatusMessage('Robot TTS request failed.');
      }
      setSpeakingState(false);
      return;
    }

    try {
      const res = await fetch('/api/speak', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, target: 'web', voice: ttsVoice }),
      });
      if (!res.ok) throw new Error('TTS request failed');
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      currentAudio = new Audio(url);
      currentAudio.addEventListener('ended', () => {
        URL.revokeObjectURL(url);
        setSpeakingState(false);
        currentAudio = null;
      });
      currentAudio.addEventListener('error', () => {
        URL.revokeObjectURL(url);
        setSpeakingState(false);
        currentAudio = null;
      });
      await currentAudio.play();
    } catch (err) {
      console.error(err);
      setSpeakingState(false);
      showStatusMessage('Web TTS failed.');
    }
  }

  function setMicState(state, tooltip) {
    micBtn.classList.remove('is-recording', 'is-transcribing');
    if (state === 'recording') {
      micBtn.classList.add('is-recording');
    } else if (state === 'transcribing') {
      micBtn.classList.add('is-transcribing');
    }
    micBtn.title = tooltip;
    micBtn.setAttribute('aria-label', tooltip);
  }

  function stopMicTracks() {
    if (!mediaStream) return;
    for (const track of mediaStream.getTracks()) {
      track.stop();
    }
    mediaStream = null;
  }

  function clearRecordingTimer() {
    if (stopRecordingTimer) {
      clearTimeout(stopRecordingTimer);
      stopRecordingTimer = null;
    }
  }

  function applyViewerMode() {
    if (isPresentationMode) {
      viewerContainer.classList.add('presentation-hidden');
      statusMsg.classList.add('hidden');
      agentReply.classList.add('hidden');
      presentationThoughtsEl.classList.remove('hidden');
    } else {
      viewerContainer.classList.remove('presentation-hidden');
      presentationThoughtsEl.classList.add('hidden');
      clearPresentationLines();
      if (!viewerIframe) {
        ensureViewerFrame(currentTaskId || '__bootstrap__');
      } else if (currentTaskId) {
        ensureViewerFrame(currentTaskId);
      }
    }
    updateModeDescription();
  }

  function pickPresentationLine(stepKey) {
    const lines = PRESENTATION_STEP_LINES[stepKey] || [];
    if (!lines.length) return '';
    if (lines.length === 1) {
      lastPresentationLineByStep[stepKey] = lines[0];
      return lines[0];
    }
    const lastLine = lastPresentationLineByStep[stepKey];
    const candidates = lines.filter((line) => line !== lastLine);
    const pool = candidates.length ? candidates : lines;
    const picked = pool[Math.floor(Math.random() * pool.length)];
    lastPresentationLineByStep[stepKey] = picked;
    return picked;
  }

  function clearPresentationLines() {
    presentationLines = [];
    if (presTextArea) presTextArea.innerHTML = '';
    resetStepIndicators();
  }

  function resetStepIndicators() {
    if (!presStepsEl) return;
    for (const step of presStepsEl.querySelectorAll('.pres-step')) {
      step.classList.remove('active', 'completed');
    }
    for (const line of presStepsEl.querySelectorAll('.pres-step-line')) {
      line.classList.remove('filled');
    }
  }

  function activateStep(stepKey) {
    if (!presStepsEl) return;
    const idx = STEP_ORDER.indexOf(stepKey);
    if (idx < 0) return;

    const steps = presStepsEl.querySelectorAll('.pres-step');
    const lines = presStepsEl.querySelectorAll('.pres-step-line');

    steps.forEach((el, i) => {
      el.classList.remove('active', 'completed');
      if (i < idx) el.classList.add('completed');
      else if (i === idx) el.classList.add('active');
    });

    lines.forEach((el, i) => {
      el.classList.toggle('filled', i < idx);
    });
  }

  function renderPresentationLines() {
    if (!presTextArea) return;

    const activeIds = new Set(presentationLines.map((entry) => String(entry.id)));
    const nodes = presTextArea.querySelectorAll('.presentation-thought');
    for (const node of nodes) {
      if (!activeIds.has(node.dataset.lineId || '')) {
        node.remove();
      }
    }

    presentationLines.forEach((entry, index) => {
      const node = presTextArea.querySelector(`[data-line-id="${entry.id}"]`);
      if (!node) return;
      const y = -index * 78;
      const opacity = Math.max(0.16, 0.92 - (index * 0.24));
      node.style.transform = `translate(-50%, ${y}px)`;
      node.style.opacity = `${opacity}`;
    });
  }

  function pushPresentationLine(stepKey) {
    if (!isPresentationMode || !presTextArea) return;
    const text = pickPresentationLine(stepKey);
    if (!text) return;

    activateStep(stepKey);

    presentationLineId += 1;
    const entry = { id: presentationLineId, text };
    presentationLines.unshift(entry);
    if (presentationLines.length > PRESENTATION_LINE_MAX) {
      const removed = presentationLines.pop();
      if (removed) {
        const staleNode = presTextArea.querySelector(`[data-line-id="${removed.id}"]`);
        if (staleNode) {
          staleNode.style.opacity = '0';
          staleNode.style.transform = 'translate(-50%, -160px)';
          setTimeout(() => staleNode.remove(), 600);
        }
      }
    }

    const node = document.createElement('div');
    node.className = 'presentation-thought';
    node.dataset.lineId = String(entry.id);
    node.textContent = entry.text;
    node.style.opacity = '0';
    node.style.transform = 'translate(-50%, 30px)';
    presTextArea.appendChild(node);

    requestAnimationFrame(() => {
      renderPresentationLines();
    });
  }

  function openSettingsModal() {
    settingsModal.classList.remove('hidden');
    if (!hasRequestedDeviceLabels) {
      hasRequestedDeviceLabels = true;
      void refreshMicrophoneList(true);
    } else {
      void refreshMicrophoneList(false);
    }
  }

  function closeSettingsModal() {
    settingsModal.classList.add('hidden');
  }

  function buildMicrophoneLabel(device, index) {
    const label = (device.label || '').trim();
    return label || `Microphone ${index + 1}`;
  }

  async function refreshMicrophoneList(requestPermission) {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
      micSelect.innerHTML = '<option value="">Microphone unavailable</option>';
      micSelect.disabled = true;
      return;
    }

    let permissionStream = null;
    if (requestPermission) {
      try {
        permissionStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      } catch (err) {
        // Ignore permission failures here.
      } finally {
        if (permissionStream) {
          for (const track of permissionStream.getTracks()) {
            track.stop();
          }
        }
      }
    }

    let devices = [];
    try {
      devices = await navigator.mediaDevices.enumerateDevices();
    } catch (err) {
      micSelect.innerHTML = '<option value="">Unable to list microphones</option>';
      micSelect.disabled = true;
      return;
    }

    const microphones = devices.filter((device) => device.kind === 'audioinput');
    micSelect.innerHTML = '';

    const defaultOption = document.createElement('option');
    defaultOption.value = '';
    defaultOption.textContent = 'System default microphone';
    micSelect.appendChild(defaultOption);

    if (!microphones.length) {
      micSelect.disabled = true;
      defaultOption.textContent = 'No microphones detected';
      return;
    }

    micSelect.disabled = false;
    microphones.forEach((device, index) => {
      const option = document.createElement('option');
      option.value = device.deviceId;
      option.textContent = buildMicrophoneLabel(device, index);
      micSelect.appendChild(option);
    });

    if (selectedMicDeviceId && !microphones.some((device) => device.deviceId === selectedMicDeviceId)) {
      selectedMicDeviceId = '';
      writeStorage(MIC_DEVICE_STORAGE_KEY, '');
      showStatusMessage('Previously selected microphone is unavailable. Using system default.');
    }
    micSelect.value = selectedMicDeviceId;
  }

  function pickRecordingMimeType() {
    if (!window.MediaRecorder || !window.MediaRecorder.isTypeSupported) return '';
    const candidates = [
      'audio/webm;codecs=opus',
      'audio/webm',
      'audio/mp4',
      'audio/ogg;codecs=opus',
      'audio/ogg',
    ];
    for (const mime of candidates) {
      if (window.MediaRecorder.isTypeSupported(mime)) return mime;
    }
    return '';
  }

  function arrayBufferToBase64(buffer) {
    let binary = '';
    const bytes = new Uint8Array(buffer);
    const chunkSize = 0x8000;
    for (let i = 0; i < bytes.length; i += chunkSize) {
      binary += String.fromCharCode(...bytes.subarray(i, i + chunkSize));
    }
    return btoa(binary);
  }

  function resetTranscriptionState() {
    capturedChunks = [];
    transcriptionQueue = [];
    isProcessingChunk = false;
    liveTranscriptText = '';
    if (transcribeController) {
      transcribeController.abort();
    }
    transcribeController = new AbortController();
  }

  function setLiveTranscript(text) {
    const clean = (text || '').trim();
    if (!clean) return;
    liveTranscriptText = clean;
    input.value = liveTranscriptText;
    input.focus();
  }

  function finalizeTranscriptionUi() {
    spinner.classList.add('hidden');
    micBtn.disabled = false;
    setMicState('idle', 'Use microphone');
    if (liveTranscriptText.trim()) {
      showStatusMessage('Live speech recognized. You can edit and press enter.');
    } else {
      showStatusMessage('No speech recognized.');
    }
  }

  async function transcribeChunk(blob, sessionId) {
    if (!transcribeController || sessionId !== recordingSessionId) {
      return '';
    }

    const audioBuffer = await blob.arrayBuffer();
    if (!audioBuffer.byteLength) return '';

    const audioBase64 = arrayBufferToBase64(audioBuffer);
    const response = await fetch('/api/transcribe', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        audio_base64: audioBase64,
        mime_type: blob.type || 'audio/webm',
      }),
      signal: transcribeController.signal,
    });

    if (!response.ok) {
      let detail = 'Speech transcription failed';
      try {
        const payload = await response.json();
        if (payload && payload.detail) detail = payload.detail;
      } catch (err) {
        // Keep default detail.
      }
      throw new Error(detail);
    }

    const data = await response.json();
    return (data.text || '').trim();
  }

  function enqueueTranscriptionChunk(blob, sessionId) {
    if (!blob || blob.size <= 0) return;
    transcriptionQueue = [{ blob, sessionId }];
    if (!isProcessingChunk) {
      void processTranscriptionQueue();
    }
  }

  async function processTranscriptionQueue() {
    if (isProcessingChunk) return;
    isProcessingChunk = true;

    try {
      while (transcriptionQueue.length > 0) {
        const item = transcriptionQueue.shift();
        if (!item) continue;
        if (item.sessionId !== recordingSessionId) continue;

        try {
          const text = await transcribeChunk(item.blob, item.sessionId);
          if (text) {
            setLiveTranscript(text);
            showStatusMessage('Listening... live transcript updating.');
          }
        } catch (err) {
          if (err && err.name === 'AbortError') return;
          console.error(err);
        }
      }
    } finally {
      isProcessingChunk = false;
      if (!isRecording && transcriptionQueue.length === 0) {
        finalizeTranscriptionUi();
      }
    }
  }

  async function startRecording() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showStatusMessage('Microphone not supported in this browser.');
      return;
    }
    if (!window.MediaRecorder) {
      showStatusMessage('MediaRecorder API is not available in this browser.');
      return;
    }

    recordingSessionId += 1;
    resetTranscriptionState();
    clearRecordingTimer();

    const audioConstraints = {
      channelCount: { ideal: 1 },
      echoCancellation: { ideal: true },
      noiseSuppression: { ideal: true },
      autoGainControl: { ideal: true },
    };

    try {
      if (selectedMicDeviceId) {
        try {
          mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: {
              ...audioConstraints,
              deviceId: { exact: selectedMicDeviceId },
            },
          });
        } catch (err) {
          const recoverable = err && (
            err.name === 'NotFoundError' ||
            err.name === 'OverconstrainedError'
          );
          if (!recoverable) {
            throw err;
          }
          selectedMicDeviceId = '';
          writeStorage(MIC_DEVICE_STORAGE_KEY, '');
          await refreshMicrophoneList(false);
          showStatusMessage('Selected microphone unavailable. Falling back to system default.');
          mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: audioConstraints,
          });
        }
      } else {
        mediaStream = await navigator.mediaDevices.getUserMedia({
          audio: audioConstraints,
        });
      }
    } catch (err) {
      const msg = err && err.name === 'NotAllowedError'
        ? 'Microphone permission denied.'
        : 'Could not access microphone.';
      showStatusMessage(msg);
      console.error(err);
      return;
    }

    const mimeType = pickRecordingMimeType();
    try {
      mediaRecorder = mimeType
        ? new MediaRecorder(mediaStream, { mimeType })
        : new MediaRecorder(mediaStream);
    } catch (err) {
      stopMicTracks();
      showStatusMessage('Failed to initialize audio recorder.');
      console.error(err);
      return;
    }

    liveTranscriptText = input.value.trim();
    const sessionId = recordingSessionId;
    mediaRecorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) {
        capturedChunks.push(event.data);
        const mime = (mediaRecorder && mediaRecorder.mimeType)
          ? mediaRecorder.mimeType
          : (event.data.type || 'audio/webm');
        const snapshotBlob = new Blob(capturedChunks, { type: mime });
        enqueueTranscriptionChunk(snapshotBlob, sessionId);
      }
    };

    mediaRecorder.onerror = (event) => {
      console.error(event);
      showStatusMessage('Microphone recording error.');
      isRecording = false;
      clearRecordingTimer();
      stopMicTracks();
      micBtn.disabled = false;
      setMicState('idle', 'Use microphone');
      spinner.classList.add('hidden');
    };

    mediaRecorder.onstop = async () => {
      clearRecordingTimer();
      stopMicTracks();
      isRecording = false;
      mediaRecorder = null;

      if (transcriptionQueue.length > 0 || isProcessingChunk) {
        micBtn.disabled = true;
        setMicState('transcribing', 'Finalizing transcription...');
        spinner.classList.remove('hidden');
        showStatusMessage('Finalizing transcription...');
        if (!isProcessingChunk) {
          void processTranscriptionQueue();
        }
      } else {
        finalizeTranscriptionUi();
      }
    };

    mediaRecorder.start(LIVE_CHUNK_MS);
    isRecording = true;
    setMicState('recording', 'Listening... click again to stop');
    micBtn.disabled = false;
    spinner.classList.add('hidden');
    showStatusMessage('Listening... speak now.');

    stopRecordingTimer = setTimeout(() => {
      if (isRecording && mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
      }
    }, MAX_RECORDING_MS);
  }

  function stopRecording() {
    if (!isRecording || !mediaRecorder) return;
    clearRecordingTimer();
    try {
      if (mediaRecorder.state === 'recording') {
        mediaRecorder.requestData();
      }
    } catch (err) {
      // Ignore requestData failures, stop still works.
    }
    if (mediaRecorder.state !== 'inactive') {
      mediaRecorder.stop();
    }
  }

  function ensureViewerFrame(srcTaskId) {
    if (!viewerIframe) {
      viewerContainer.innerHTML = `<iframe src="/viewer/${srcTaskId}" width="100%" height="100%" style="border:none;" loading="eager"></iframe>`;
      viewerIframe = viewerContainer.querySelector('iframe');
      viewerIframe.addEventListener('load', () => {
        if (pendingTaskId) {
          viewerIframe.contentWindow.postMessage({ type: 'loadMotion', taskId: pendingTaskId }, window.location.origin);
          pendingTaskId = null;
        }
      });
      return;
    }
    pendingTaskId = srcTaskId;
    if (viewerIframe.contentWindow) {
      viewerIframe.contentWindow.postMessage({ type: 'loadMotion', taskId: srcTaskId }, window.location.origin);
      setTimeout(() => {
        if (pendingTaskId === srcTaskId) {
          pendingTaskId = null;
        }
      }, 250);
    }
  }

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const prompt = input.value.trim();
    if (!prompt) return;
    if (isRecording) stopRecording();

    // Reset UI
    hideAgentReply();
    if (!isPresentationMode) {
      statusMsg.classList.remove('hidden');
      refinedText.textContent = "Generating motion...";
      spinner.classList.remove('hidden');
    } else {
      spinner.classList.add('hidden');
      pushPresentationLine('llm');
    }
    actionBar.classList.add('hidden');
    deployStatus.classList.add('hidden');

    let kimodoNarrated = false;
    let kimodoTimer = null;
    const narrateKimodo = () => {
      if (kimodoNarrated || !isPresentationMode) return;
      kimodoNarrated = true;
      pushPresentationLine('kimodo');
    };
    if (isPresentationMode) {
      kimodoTimer = setTimeout(() => {
        narrateKimodo();
      }, 900);
    }

    try {
      const response = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt })
      });

      if (!response.ok) throw new Error('API Error');
      const data = await response.json();

      spinner.classList.add('hidden');
      if (!isPresentationMode) {
        if (data.prompts && data.prompts.length > 0) {
          refinedText.textContent = data.prompts[0];
        } else {
          refinedText.textContent = prompt;
        }
      } else {
        narrateKimodo();
      }

      // Show agent text reply
      if (data.text_reply) {
        showAgentReply(data.text_reply);
        if (autoSpeak) {
          void speakText(data.text_reply);
        }
      }

      currentTaskId = data.task_id;

      if (data.viewer_url) {
        if (!isPresentationMode) {
          ensureViewerFrame(data.task_id);
        }
        actionBar.classList.remove('hidden');
      }

    } catch (err) {
      spinner.classList.add('hidden');
      if (!isPresentationMode) {
        refinedText.textContent = "Error generating motion.";
      }
      console.error(err);
    } finally {
      if (kimodoTimer) {
        clearTimeout(kimodoTimer);
      }
    }
  });

  micBtn.addEventListener('click', async () => {
    if (micBtn.disabled) return;
    if (isRecording) {
      stopRecording();
      return;
    }
    await startRecording();
  });

  settingsBtn.addEventListener('click', () => {
    openSettingsModal();
  });

  settingsCloseBtn.addEventListener('click', () => {
    closeSettingsModal();
  });

  settingsModal.addEventListener('click', (event) => {
    if (event.target === settingsModal) {
      closeSettingsModal();
    }
  });

  micSelect.addEventListener('change', () => {
    selectedMicDeviceId = micSelect.value;
    writeStorage(MIC_DEVICE_STORAGE_KEY, selectedMicDeviceId);
  });

  presentationModeToggle.addEventListener('change', () => {
    isPresentationMode = presentationModeToggle.checked;
    writeStorage(UI_MODE_STORAGE_KEY, isPresentationMode ? 'presentation' : 'developer');
    applyViewerMode();
  });

  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && !settingsModal.classList.contains('hidden')) {
      closeSettingsModal();
    }
  });

  deployBtn.addEventListener('click', async () => {
    if (!currentTaskId) return;

    if (isPresentationMode) {
      pushPresentationLine('zmq');
    }

    deployBtn.disabled = true;
    deployBtn.textContent = "Deploying...";

    try {
      const response = await fetch('/api/deploy', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task_id: currentTaskId })
      });

      if (!response.ok) throw new Error('Deploy Error');

      deployStatus.classList.remove('hidden');
      deployBtn.textContent = "Deploy to Robot";
      deployBtn.disabled = false;

      setTimeout(() => {
        deployStatus.classList.add('hidden');
      }, 3000);

    } catch (err) {
      deployBtn.textContent = "Deploy to Robot";
      deployBtn.disabled = false;
      alert("Failed to deploy");
      console.error(err);
    }
  });

  speakBtn.addEventListener('click', () => {
    if (isSpeaking) {
      stopSpeaking();
    } else {
      const text = agentReplyText.textContent;
      if (text) void speakText(text);
    }
  });

  ttsTargetSelect.addEventListener('change', () => {
    ttsTarget = ttsTargetSelect.value;
    writeStorage(TTS_TARGET_STORAGE_KEY, ttsTarget);
  });

  ttsVoiceSelect.addEventListener('change', () => {
    ttsVoice = ttsVoiceSelect.value;
    writeStorage(TTS_VOICE_STORAGE_KEY, ttsVoice);
  });

  autoSpeakToggle.addEventListener('change', () => {
    autoSpeak = autoSpeakToggle.checked;
    writeStorage(AUTO_SPEAK_STORAGE_KEY, autoSpeak ? 'true' : 'false');
  });

  selectedMicDeviceId = readStorage(MIC_DEVICE_STORAGE_KEY, '');
  ttsTarget = readStorage(TTS_TARGET_STORAGE_KEY, 'web');
  ttsVoice = readStorage(TTS_VOICE_STORAGE_KEY, 'alloy');
  autoSpeak = readStorage(AUTO_SPEAK_STORAGE_KEY, 'false') === 'true';
  ttsTargetSelect.value = ttsTarget;
  ttsVoiceSelect.value = ttsVoice;
  autoSpeakToggle.checked = autoSpeak;
  isPresentationMode = readStorage(UI_MODE_STORAGE_KEY, 'developer') === 'presentation';
  presentationModeToggle.checked = isPresentationMode;
  applyViewerMode();
  void refreshMicrophoneList(false);

  if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
    navigator.mediaDevices.addEventListener('devicechange', () => {
      void refreshMicrophoneList(false);
    });
  }

  window.addEventListener('beforeunload', () => {
    clearRecordingTimer();
    if (isRecording) stopRecording();
    stopMicTracks();
    if (transcribeController) transcribeController.abort();
  });

  document.addEventListener('visibilitychange', () => {
    if (document.hidden && isRecording) {
      stopRecording();
    }
  });
});
