// simulator/frontend/js/main.js
// AETHEL123 Real-Time Live Audio Console Frontend

const API_BASE = window.location.origin.includes('http') ? window.location.origin : 'http://127.0.0.1:8000';
let ws = null;
let pollInterval = null;
let currentDevices = { input_devices: [], output_devices: [] };
let isStreaming = false;
let isPaused = false;
let isPassthrough = false;

// Tooltip dictionary
const TOOLTIPS = {
  'RTF': 'Real-Time Factor. Values below 1.0 indicate processing completes faster than incoming audio.',
  'GCC-PHAT': 'Generalized Cross-Correlation with Phase Transform used to estimate relative acoustic delay.',
  'Kalman': 'Kalman-filtered delay estimation between primary and reference microphones.',
  'NLMS': 'Normalized Least Mean Squares adaptive noise cancellation filter.',
  'VAD': 'Voice Activity Detection indicating whether active human speech is present.',
  'STOI': 'Short-Time Objective Intelligibility measure relative to a clean reference signal.',
  'PESQ': 'Perceptual Evaluation of Speech Quality (WB 16kHz) relative to a clean reference signal.',
  'SNR': 'Signal-to-Noise Ratio improvement in decibels (dB).'
};

// Rolling waveform history (256 samples)
const WAVEFORM_WINDOW = 256;
let m1History = new Array(WAVEFORM_WINDOW).fill(0);
let m2History = new Array(WAVEFORM_WINDOW).fill(0);
let outHistory = new Array(WAVEFORM_WINDOW).fill(0);

// Spectrogram history (40 columns x 64 frequency bins)
const SPEC_COLUMNS = 40;
let spectrogramHistory = [];

// DOM Element Selectors
const getEl = id => document.getElementById(id);

// Setup Tab Switching
function setupTabs() {
  const tabBtns = document.querySelectorAll('.tab-btn');
  const tabContents = document.querySelectorAll('.tab-content');

  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetId = btn.getAttribute('data-tab');
      activateTab(targetId);
    });
  });

  const linkViewDetails = getEl('linkViewDetails');
  if (linkViewDetails) {
    linkViewDetails.addEventListener('click', (e) => {
      e.preventDefault();
      activateTab('tab-ai-dsp');
    });
  }
}

function activateTab(tabId) {
  const tabBtns = document.querySelectorAll('.tab-btn');
  const tabContents = document.querySelectorAll('.tab-content');

  tabBtns.forEach(btn => {
    if (btn.getAttribute('data-tab') === tabId) {
      btn.classList.add('active');
    } else {
      btn.classList.remove('active');
    }
  });

  tabContents.forEach(content => {
    if (content.id === tabId) {
      content.classList.add('active');
    } else {
      content.classList.remove('active');
    }
  });

  // Re-trigger canvas resize if tab became visible
  if (tabId === 'tab-live-audio') {
    drawWaveform('canvasM1', m1History, '#22D3EE');
    drawWaveform('canvasM2', m2History, '#14B8A6');
    drawWaveform('canvasOut', outHistory, '#8B5CF6');
    drawSpectrogram(null);
  }
}

// Setup Tooltips
function setupTooltips() {
  const tooltipPopup = getEl('tooltipPopup');
  if (!tooltipPopup) return;

  const icons = document.querySelectorAll('.info-icon');
  icons.forEach(icon => {
    const key = icon.getAttribute('data-tooltip');
    const text = TOOLTIPS[key] || 'Technical specification parameter.';

    icon.addEventListener('mouseenter', (e) => {
      tooltipPopup.textContent = text;
      tooltipPopup.classList.add('visible');
      const rect = icon.getBoundingClientRect();
      tooltipPopup.style.left = `${rect.left + window.scrollX - 10}px`;
      tooltipPopup.style.top = `${rect.bottom + window.scrollY + 6}px`;
    });

    icon.addEventListener('mouseleave', () => {
      tooltipPopup.classList.remove('visible');
    });
  });
}

// Log to UI console
function log(msg, level = 'info') {
  const consoleBody = getEl('consoleBody');
  if (!consoleBody) return;
  const entry = document.createElement('div');
  entry.className = `log-entry ${level}`;
  const t = new Date().toLocaleTimeString('en-US', { hour12: false });
  entry.textContent = `[${t}] ${msg}`;
  consoleBody.appendChild(entry);
  consoleBody.scrollTop = consoleBody.scrollHeight;
}

const btnClearLog = getEl('btnClearLog');
if (btnClearLog) {
  btnClearLog.addEventListener('click', () => {
    const consoleBody = getEl('consoleBody');
    if (consoleBody) consoleBody.innerHTML = '';
  });
}

// Fetch Quality Metrics Endpoint
async function loadQualityMetrics() {
  try {
    const res = await fetch(`${API_BASE}/metrics/quality`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    if (data.available && data.stoi) {
      // Overview summary
      if (getEl('ovSTOI')) getEl('ovSTOI').textContent = data.stoi.enhanced.toFixed(3);
      if (getEl('ovPESQ')) getEl('ovPESQ').textContent = data.pesq.enhanced.toFixed(2);
      if (getEl('ovSNR')) getEl('ovSNR').textContent = `${data.snr_db.enhanced >= 0 ? '+' : ''}${data.snr_db.enhanced.toFixed(2)} dB`;

      // AI & DSP Detailed Table
      if (getEl('tblStoiNoisy')) getEl('tblStoiNoisy').textContent = data.stoi.noisy.toFixed(3);
      if (getEl('tblStoiEnhanced')) getEl('tblStoiEnhanced').textContent = data.stoi.enhanced.toFixed(3);
      if (getEl('tblStoiDelta')) {
        const delta = data.stoi.delta;
        getEl('tblStoiDelta').textContent = `${delta >= 0 ? '+' : ''}${delta.toFixed(3)}`;
        getEl('tblStoiDelta').className = delta >= 0 ? 'text-green font-mono' : 'text-amber font-mono';
      }
      if (getEl('tblStoiTarget')) getEl('tblStoiTarget').textContent = `> ${data.stoi.target}`;
      if (getEl('tblStoiStatus')) {
        const met = data.stoi.enhanced >= data.stoi.target;
        getEl('tblStoiStatus').textContent = met ? 'TARGET MET' : 'TARGET NOT MET';
        getEl('tblStoiStatus').className = met ? 'text-green font-mono' : 'text-amber font-mono';
      }

      if (getEl('tblPesqNoisy')) getEl('tblPesqNoisy').textContent = data.pesq.noisy.toFixed(2);
      if (getEl('tblPesqEnhanced')) getEl('tblPesqEnhanced').textContent = data.pesq.enhanced.toFixed(2);
      if (getEl('tblPesqDelta')) {
        const delta = data.pesq.delta;
        getEl('tblPesqDelta').textContent = `${delta >= 0 ? '+' : ''}${delta.toFixed(2)}`;
        getEl('tblPesqDelta').className = delta >= 0 ? 'text-green font-mono' : 'text-amber font-mono';
      }
      if (getEl('tblPesqTarget')) getEl('tblPesqTarget').textContent = `> ${data.pesq.target}`;
      if (getEl('tblPesqStatus')) {
        const met = data.pesq.enhanced >= data.pesq.target;
        getEl('tblPesqStatus').textContent = met ? 'TARGET MET' : 'TARGET NOT MET';
        getEl('tblPesqStatus').className = met ? 'text-green font-mono' : 'text-amber font-mono';
      }

      if (getEl('tblSnrNoisy')) getEl('tblSnrNoisy').textContent = `${data.snr_db.noisy >= 0 ? '+' : ''}${data.snr_db.noisy.toFixed(2)} dB`;
      if (getEl('tblSnrEnhanced')) getEl('tblSnrEnhanced').textContent = `${data.snr_db.enhanced >= 0 ? '+' : ''}${data.snr_db.enhanced.toFixed(2)} dB`;
      if (getEl('tblSnrDelta')) {
        const delta = data.snr_db.delta;
        getEl('tblSnrDelta').textContent = `${delta >= 0 ? '+' : ''}${delta.toFixed(2)} dB`;
        getEl('tblSnrDelta').className = delta >= 0 ? 'text-green font-mono' : 'text-amber font-mono';
      }
      if (getEl('tblSnrTarget')) getEl('tblSnrTarget').textContent = `> +${data.snr_db.target} dB`;
      if (getEl('tblSnrStatus')) {
        const met = data.snr_db.enhanced >= data.snr_db.target;
        getEl('tblSnrStatus').textContent = met ? 'TARGET MET' : 'TARGET NOT MET';
        getEl('tblSnrStatus').className = met ? 'text-green font-mono' : 'text-amber font-mono';
      }

      if (getEl('lblEvalSamples') && data.samples) getEl('lblEvalSamples').textContent = `${data.samples} test samples`;
      if (getEl('lblEvalAcc') && data.classification_accuracy) getEl('lblEvalAcc').textContent = `${(data.classification_accuracy * 100).toFixed(2)}% accuracy`;
      if (getEl('lblEvalTimestamp') && data.timestamp) getEl('lblEvalTimestamp').textContent = `Validated: ${data.timestamp}`;

      log('Speech quality metrics loaded from authoritative evaluation artifact.', 'info');
    } else {
      setQualityUnavailable();
    }
  } catch (e) {
    setQualityUnavailable();
  }
}

function setQualityUnavailable() {
  if (getEl('ovSTOI')) getEl('ovSTOI').textContent = 'N/A';
  if (getEl('ovPESQ')) getEl('ovPESQ').textContent = 'N/A';
  if (getEl('ovSNR')) getEl('ovSNR').textContent = 'N/A';
}

// Device Loading & Binding
async function loadDevices() {
  try {
    const res = await fetch(`${API_BASE}/api/devices`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    currentDevices = data;
    renderDeviceDropdowns(data);
    updateDeviceDetails(data);
    log(`Discovered ${data.input_devices.length} input and ${data.output_devices.length} output physical audio devices.`, 'info');
  } catch (e) {
    log(`Device query failed: ${e.message}. Is server running?`, 'error');
  }
}

function renderDeviceDropdowns(data) {
  const selectInputDevice = getEl('selectInputDevice');
  const selectOutputDevice = getEl('selectOutputDevice');
  if (!selectInputDevice || !selectOutputDevice) return;

  selectInputDevice.innerHTML = '';
  data.input_devices.forEach(dev => {
    const opt = document.createElement('option');
    opt.value = dev.id;
    const defTag = dev.is_default_input ? ' ★ System Default' : '';
    opt.textContent = `[#${dev.id}] ${dev.name} (${dev.hostapi_name || dev.host_api}, ${dev.max_input_channels}ch, ${dev.default_samplerate || dev.default_sample_rate}Hz)${defTag}`;
    if (dev.is_default_input) opt.selected = true;
    selectInputDevice.appendChild(opt);
  });

  selectOutputDevice.innerHTML = '';
  data.output_devices.forEach(dev => {
    const opt = document.createElement('option');
    opt.value = dev.id;
    const defTag = dev.is_default_output ? ' ★ System Default' : '';
    opt.textContent = `[#${dev.id}] ${dev.name} (${dev.hostapi_name || dev.host_api}, ${dev.max_output_channels}ch, ${dev.default_samplerate || dev.default_sample_rate}Hz)${defTag}`;
    if (dev.is_default_output) opt.selected = true;
    selectOutputDevice.appendChild(opt);
  });

  updateChannelOptions();
}

function updateChannelOptions() {
  const selectInputDevice = getEl('selectInputDevice');
  const selectPrimaryChan = getEl('selectPrimaryChan');
  const selectRefChan = getEl('selectRefChan');
  const singleMicAlert = getEl('singleMicAlert');
  if (!selectInputDevice || !selectPrimaryChan || !selectRefChan) return;

  const inId = parseInt(selectInputDevice.value);
  const dev = currentDevices.input_devices.find(d => d.id === inId);

  selectPrimaryChan.innerHTML = '';
  selectRefChan.innerHTML = '';

  if (!dev) return;

  const nCh = dev.max_input_channels;
  for (let c = 0; c < nCh; c++) {
    const opt1 = document.createElement('option');
    opt1.value = c;
    opt1.textContent = `Channel ${c} ${c === 0 ? '(Primary / Mic 1)' : ''}`;
    selectPrimaryChan.appendChild(opt1);

    const opt2 = document.createElement('option');
    opt2.value = c;
    opt2.textContent = `Channel ${c} ${c === 1 ? '(Reference / Mic 2)' : ''}`;
    if (c === 1) opt2.selected = true;
    selectRefChan.appendChild(opt2);
  }

  if (nCh < 2) {
    if (singleMicAlert) singleMicAlert.style.display = 'block';
    const optRef = document.createElement('option');
    optRef.value = -1;
    optRef.textContent = 'None (Single-Mic)';
    optRef.selected = true;
    selectRefChan.appendChild(optRef);
  } else {
    if (singleMicAlert) singleMicAlert.style.display = 'none';
  }

  updateDeviceDetails(currentDevices);
}

function updateDeviceDetails(data) {
  const selectInputDevice = getEl('selectInputDevice');
  const selectOutputDevice = getEl('selectOutputDevice');

  if (selectInputDevice && data.input_devices) {
    const inId = parseInt(selectInputDevice.value);
    const inDev = data.input_devices.find(d => d.id === inId);
    if (inDev) {
      if (getEl('devInId')) getEl('devInId').textContent = inDev.id;
      if (getEl('devInHost')) getEl('devInHost').textContent = inDev.hostapi_name || inDev.host_api || 'MME';
      if (getEl('devInCh')) getEl('devInCh').textContent = `${inDev.max_input_channels} Channels`;
      if (getEl('devInSR')) getEl('devInSR').textContent = `${inDev.default_samplerate || inDev.default_sample_rate || 16000} Hz`;
      if (getEl('ovInputDev')) getEl('ovInputDev').textContent = inDev.name;
    }
  }

  if (selectOutputDevice && data.output_devices) {
    const outId = parseInt(selectOutputDevice.value);
    const outDev = data.output_devices.find(d => d.id === outId);
    if (outDev) {
      if (getEl('devOutId')) getEl('devOutId').textContent = outDev.id;
      if (getEl('devOutHost')) getEl('devOutHost').textContent = outDev.hostapi_name || outDev.host_api || 'MME';
      if (getEl('devOutCh')) getEl('devOutCh').textContent = `${outDev.max_output_channels} Channels`;
      if (getEl('devOutSR')) getEl('devOutSR').textContent = `${outDev.default_samplerate || outDev.default_sample_rate || 16000} Hz`;
      if (getEl('ovOutputDev')) getEl('ovOutputDev').textContent = outDev.name;
    }
  }
}

const selectInputDevice = getEl('selectInputDevice');
if (selectInputDevice) {
  selectInputDevice.addEventListener('change', updateChannelOptions);
}
const selectOutputDevice = getEl('selectOutputDevice');
if (selectOutputDevice) {
  selectOutputDevice.addEventListener('change', () => updateDeviceDetails(currentDevices));
}

const btnRefreshDevices = getEl('btnRefreshDevices');
if (btnRefreshDevices) {
  btnRefreshDevices.addEventListener('click', async () => {
    log('Scanning hardware for audio device changes...', 'info');
    try {
      const res = await fetch(`${API_BASE}/api/devices/refresh`, { method: 'POST' });
      const data = await res.json();
      currentDevices = data;
      renderDeviceDropdowns(data);
      log('Audio device list updated from hardware.', 'success');
    } catch (e) {
      log(`Device refresh error: ${e.message}`, 'error');
    }
  });
}

// Master Gain Slider
const masterGainSlider = getEl('masterGainSlider');
const gainDisplay = getEl('gainDisplay');
if (masterGainSlider && gainDisplay) {
  masterGainSlider.addEventListener('input', () => {
    gainDisplay.textContent = `${masterGainSlider.value}%`;
    if (isStreaming) {
      fetch(`${API_BASE}/api/stream/gain`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ gain: parseFloat(masterGainSlider.value) / 100.0 }),
      }).catch(err => console.error('Gain error:', err));
    }
  });
}

// Stream Control Buttons
const btnStartStream = getEl('btnStartStream');
const btnStopStream = getEl('btnStopStream');
const btnPauseStream = getEl('btnPauseStream');
const btnPassthrough = getEl('btnPassthrough');

if (btnStartStream) {
  btnStartStream.addEventListener('click', async () => {
    const inId = parseInt(getEl('selectInputDevice').value);
    const outId = parseInt(getEl('selectOutputDevice').value);
    const pCh = parseInt(getEl('selectPrimaryChan').value);
    const rCh = parseInt(getEl('selectRefChan').value);
    const bSize = parseInt(getEl('selectBlockSize').value);
    const gain = parseFloat(getEl('masterGainSlider').value) / 100.0;

    log(`Opening physical audio stream on in=${inId}, out=${outId}, block=${bSize}...`, 'info');
    btnStartStream.disabled = true;

    try {
      const res = await fetch(`${API_BASE}/api/stream/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          input_device_id: inId,
          output_device_id: outId,
          primary_channel: pCh,
          ref_channel: rCh,
          sample_rate: 16000,
          block_size: bSize,
          master_gain: gain,
        }),
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Failed to start stream.');
      }

      isStreaming = true;
      isPaused = false;
      if (btnStopStream) btnStopStream.disabled = false;
      if (btnPauseStream) {
        btnPauseStream.disabled = false;
        btnPauseStream.textContent = '🔇 Mute';
      }

      setStreamHeaderStatus('LIVE STREAMING', true);
      setPipelineActive(true);

      // Hide waiting overlays
      ['spectrogramWaiting', 'm1Waiting', 'm2Waiting', 'outWaiting'].forEach(id => {
        const el = getEl(id);
        if (el) el.style.display = 'none';
      });

      log(`[LIVE] ${data.message}`, 'success');
      connectWebSocket();

    } catch (e) {
      btnStartStream.disabled = false;
      log(`[ERR] Failed to start real-time stream: ${e.message}`, 'error');
      alert(`Could not start stream: ${e.message}`);
    }
  });
}

if (btnStopStream) {
  btnStopStream.addEventListener('click', async () => {
    log('Stopping physical audio streams...', 'info');
    btnStopStream.disabled = true;

    try {
      await fetch(`${API_BASE}/api/stream/stop`, { method: 'POST' });
    } catch (e) {
      console.error('Stop error:', e);
    } finally {
      isStreaming = false;
      if (btnStartStream) btnStartStream.disabled = false;
      if (btnPauseStream) btnPauseStream.disabled = true;

      setStreamHeaderStatus('STREAM STOPPED', false);
      setPipelineActive(false);

      ['spectrogramWaiting', 'm1Waiting', 'm2Waiting', 'outWaiting'].forEach(id => {
        const el = getEl(id);
        if (el) el.style.display = 'flex';
      });

      if (ws) {
        ws.close();
        ws = null;
      }
      if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
      }

      m1History.fill(0);
      m2History.fill(0);
      outHistory.fill(0);
      drawWaveform('canvasM1', m1History, '#22D3EE');
      drawWaveform('canvasM2', m2History, '#14B8A6');
      drawWaveform('canvasOut', outHistory, '#8B5CF6');
      drawSpectrogram(null);

      log('[SYS] Live audio stream stopped.', 'info');
    }
  });
}

if (btnPauseStream) {
  btnPauseStream.addEventListener('click', async () => {
    isPaused = !isPaused;
    btnPauseStream.textContent = isPaused ? '▶ Resume' : '🔇 Mute';
    try {
      await fetch(`${API_BASE}/api/stream/pause`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ paused: isPaused }),
      });
      log(isPaused ? 'Audio stream paused/muted.' : 'Audio stream resumed.', 'info');
    } catch (e) {
      console.error('Pause error:', e);
    }
  });
}

if (btnPassthrough) {
  btnPassthrough.addEventListener('click', async () => {
    isPassthrough = !isPassthrough;
    try {
      const res = await fetch(`${API_BASE}/api/stream/passthrough`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled: isPassthrough }),
      });
      const data = await res.json();
      log(`Output Path Test Mode: ${data.passthrough_mode ? 'PASSTHROUGH ACTIVE (Mic 1 Direct to Headphones)' : 'NORMAL AI PROCESSING ACTIVE'}`, 'info');
    } catch (e) {
      log(`Passthrough toggle error: ${e.message}`, 'error');
    }
  });
}

function setStreamHeaderStatus(text, active) {
  const hdrStreamStatus = getEl('hdrStreamStatus');
  if (!hdrStreamStatus) return;
  const dot = hdrStreamStatus.querySelector('.status-dot');
  const txt = hdrStreamStatus.querySelector('.status-text');
  if (dot) dot.className = `status-dot ${active ? 'active' : 'inactive'}`;
  if (txt) txt.textContent = text;
}

function setPipelineActive(active) {
  ['ovInputStatus', 'ovProcStatus', 'ovOutputStatus', 'pipePreprocessStatus', 'pipeAiStatus', 'pipeDspStatus', 'pipeOutputStatus'].forEach(id => {
    const el = getEl(id);
    if (el) {
      const dot = el.querySelector('.status-dot');
      if (dot) dot.className = `status-dot ${active ? 'active' : 'inactive'}`;
    }
  });
}

// WebSocket Live Telemetry Connection
function connectWebSocket() {
  const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsHost = window.location.host || '127.0.0.1:8000';
  const url = `${wsProtocol}//${wsHost}/ws/live`;

  log(`Connecting live telemetry via WebSocket (${url})...`, 'info');

  try {
    ws = new WebSocket(url);

    ws.onopen = () => {
      log('WebSocket live telemetry connected.', 'success');
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        handleTelemetry(data);
      } catch (err) {
        console.error('Parse telemetry error:', err);
      }
    };

    ws.onerror = (err) => {
      console.warn('WebSocket error, falling back to HTTP polling:', err);
      startPolling();
    };

    ws.onclose = () => {
      if (isStreaming) {
        startPolling();
      }
    };
  } catch (err) {
    startPolling();
  }
}

function startPolling() {
  if (pollInterval) return;
  pollInterval = setInterval(async () => {
    if (!isStreaming) {
      clearInterval(pollInterval);
      pollInterval = null;
      return;
    }
    try {
      const res = await fetch(`${API_BASE}/api/stream/telemetry`);
      if (res.ok) {
        const data = await res.json();
        handleTelemetry(data);
      }
    } catch (e) {
      // ignore
    }
  }, 50);
}

function toDBFS(amp) {
  if (!amp || amp <= 1e-6) return -96.0;
  return Math.max(-96.0, 20.0 * Math.log10(amp));
}

// Update UI with Genuine Telemetry
function handleTelemetry(t) {
  if (!t) return;

  const dbM1 = toDBFS(t.levels.input_rms_m1);
  const dbM2 = toDBFS(t.levels.input_rms_m2);
  const dbOut = toDBFS(t.levels.output_rms);
  const latency = t.performance.last_processing_ms || 0.0;
  const rtf = t.performance.rtf || 0.0;

  // Overview Tab
  if (getEl('ovM1RMS')) getEl('ovM1RMS').textContent = `${dbM1.toFixed(1)} dBFS`;
  if (getEl('ovOutRMS')) getEl('ovOutRMS').textContent = `${dbOut.toFixed(1)} dBFS`;
  if (getEl('ovLatency')) getEl('ovLatency').textContent = `${latency.toFixed(1)} ms`;
  if (getEl('ovRTF')) {
    getEl('ovRTF').textContent = `${rtf.toFixed(2)}×`;
    getEl('ovRTF').style.color = rtf > 1.0 ? 'var(--accent-red)' : 'var(--accent-cyan)';
  }

  if (getEl('ovNoiseClass')) getEl('ovNoiseClass').textContent = t.dsp.noise_class || 'Stationary';
  if (getEl('ovNoiseConf')) getEl('ovNoiseConf').textContent = `${((t.dsp.confidence || 0) * 100).toFixed(1)}% confidence`;
  if (getEl('ovVADState')) {
    const isSpeech = t.dsp.is_speech;
    getEl('ovVADState').textContent = isSpeech ? 'Speech Active' : 'Non-Speech';
    getEl('ovVADState').style.color = isSpeech ? 'var(--accent-green)' : 'var(--text-secondary)';
  }

  // Live Audio Tab Labels
  if (getEl('lblM1RMS')) getEl('lblM1RMS').textContent = `${dbM1.toFixed(1)} dBFS`;
  if (getEl('lblM1Peak')) getEl('lblM1Peak').textContent = `${toDBFS(t.levels.input_peak_m1).toFixed(1)} dBFS`;
  if (getEl('lblM1Chan')) getEl('lblM1Chan').textContent = t.primary_channel !== undefined ? t.primary_channel : 0;

  if (getEl('lblM2RMS')) getEl('lblM2RMS').textContent = `${dbM2.toFixed(1)} dBFS`;
  if (getEl('lblM2Peak')) getEl('lblM2Peak').textContent = `${toDBFS(t.levels.input_peak_m2).toFixed(1)} dBFS`;
  if (getEl('lblM2Chan')) getEl('lblM2Chan').textContent = t.ref_channel !== undefined ? t.ref_channel : 1;

  if (getEl('lblOutRMS')) getEl('lblOutRMS').textContent = `${dbOut.toFixed(1)} dBFS`;
  if (getEl('lblOutPeak')) getEl('lblOutPeak').textContent = `${toDBFS(t.levels.output_peak).toFixed(1)} dBFS`;
  if (getEl('lblOutLatency')) getEl('lblOutLatency').textContent = `${latency.toFixed(1)} ms`;

  // AI & DSP Tab
  const clsId = t.dsp.noise_class_id;
  const conf = t.dsp.confidence || 0.33;
  if (getEl('barStationary')) getEl('barStationary').style.width = clsId === 0 ? `${(conf * 100).toFixed(0)}%` : '10%';
  if (getEl('pctStationary')) getEl('pctStationary').textContent = clsId === 0 ? `${(conf * 100).toFixed(0)}%` : '10%';
  if (getEl('barNonStationary')) getEl('barNonStationary').style.width = clsId === 1 ? `${(conf * 100).toFixed(0)}%` : '10%';
  if (getEl('pctNonStationary')) getEl('pctNonStationary').textContent = clsId === 1 ? `${(conf * 100).toFixed(0)}%` : '10%';
  if (getEl('barImpulsive')) getEl('barImpulsive').style.width = clsId === 2 ? `${(conf * 100).toFixed(0)}%` : '10%';
  if (getEl('pctImpulsive')) getEl('pctImpulsive').textContent = clsId === 2 ? `${(conf * 100).toFixed(0)}%` : '10%';

  if (getEl('dispGCCDelay')) getEl('dispGCCDelay').textContent = `${(t.dsp.gcc_delay || 0.0).toFixed(1)} samples`;
  if (getEl('dispKalmanDelay')) getEl('dispKalmanDelay').textContent = `${(t.dsp.kalman_delay || 0.0).toFixed(1)} samples`;
  if (getEl('dispNLMS')) getEl('dispNLMS').textContent = t.dsp.nlms_active !== false ? 'ACTIVE' : 'BYPASSED';
  if (getEl('dispVAD')) getEl('dispVAD').textContent = t.dsp.is_speech ? 'SPEECH' : 'NON-SPEECH';
  if (getEl('dispFusion')) getEl('dispFusion').textContent = t.dsp.fusion_active !== false ? 'AI + DSP BLENDED' : 'AI ONLY';
  if (getEl('dispLimiter')) getEl('dispLimiter').textContent = t.dsp.limiter_active ? 'ATTENUATING' : 'NORMAL';
  if (getEl('dispClipCount')) getEl('dispClipCount').textContent = t.dsp.clipping_count || 0;
  if (getEl('dispInferenceTime')) getEl('dispInferenceTime').textContent = `${latency.toFixed(2)} ms`;

  // Diagnostics Tab
  if (t.audio && t.audio.input) {
    if (getEl('diagInCB')) getEl('diagInCB').textContent = t.audio.input.callback_count || 0;
    if (getEl('diagInFrames')) getEl('diagInFrames').textContent = (t.audio.input.frames_received || 0).toLocaleString();
    if (getEl('diagOutCB')) getEl('diagOutCB').textContent = t.audio.output.callback_count || 0;
    if (getEl('diagOutFrames')) getEl('diagOutFrames').textContent = (t.audio.output.frames_sent || 0).toLocaleString();
    if (getEl('diagProcBlocks')) getEl('diagProcBlocks').textContent = (t.audio.processing.blocks || 0).toLocaleString();
    if (getEl('diagProcErrs')) getEl('diagProcErrs').textContent = t.audio.processing.errors || 0;
    if (getEl('diagCBErrs')) getEl('diagCBErrs').textContent = (t.audio.input.errors || 0) + (t.audio.output.errors || 0);
  }

  if (getEl('diagQueueDepth')) getEl('diagQueueDepth').textContent = `${t.performance.queue_depth || 0} blocks`;
  if (getEl('diagOverflows')) getEl('diagOverflows').textContent = t.performance.input_overflows || 0;
  if (getEl('diagUnderflows')) getEl('diagUnderflows').textContent = t.performance.output_underflows || 0;
  if (getEl('diagDrops')) getEl('diagDrops').textContent = t.performance.dropped_blocks || 0;

  if (getEl('diagMeanProc')) getEl('diagMeanProc').textContent = `${(t.performance.mean_processing_ms || 0).toFixed(2)} ms`;
  if (getEl('diagP95')) getEl('diagP95').textContent = `${(t.performance.p95_processing_ms || 0).toFixed(2)} ms`;
  if (getEl('diagP99')) getEl('diagP99').textContent = `${(t.performance.p99_processing_ms || 0).toFixed(2)} ms`;
  if (getEl('diagMaxProc')) getEl('diagMaxProc').textContent = `${(t.performance.max_processing_ms || 0).toFixed(2)} ms`;
  if (getEl('diagRTF')) getEl('diagRTF').textContent = `${rtf.toFixed(3)}×`;

  if (getEl('diagTotalAudio')) getEl('diagTotalAudio').textContent = `${(t.performance.total_audio_s || 0).toFixed(1)} s`;
  if (getEl('diagM1RMS')) getEl('diagM1RMS').textContent = `${dbM1.toFixed(1)} dBFS`;
  if (getEl('diagM1Peak')) getEl('diagM1Peak').textContent = `${toDBFS(t.levels.input_peak_m1).toFixed(1)} dBFS`;
  if (getEl('diagM2RMS')) getEl('diagM2RMS').textContent = `${dbM2.toFixed(1)} dBFS`;
  if (getEl('diagM2Peak')) getEl('diagM2Peak').textContent = `${toDBFS(t.levels.input_peak_m2).toFixed(1)} dBFS`;
  if (getEl('diagOutRMS')) getEl('diagOutRMS').textContent = `${dbOut.toFixed(1)} dBFS`;
  if (getEl('diagOutPeak')) getEl('diagOutPeak').textContent = `${toDBFS(t.levels.output_peak).toFixed(1)} dBFS`;

  // Waveforms
  if (t.waveforms && t.waveforms.m1 && t.waveforms.m1.length > 0) {
    const k = t.waveforms.m1.length;
    m1History = m1History.slice(k).concat(t.waveforms.m1);
    m2History = m2History.slice(k).concat(t.waveforms.m2);
    outHistory = outHistory.slice(k).concat(t.waveforms.output);

    ['m1Waiting', 'm2Waiting', 'outWaiting'].forEach(id => {
      const el = getEl(id);
      if (el) el.style.display = 'none';
    });

    drawWaveform('canvasM1', m1History, '#22D3EE');
    drawWaveform('canvasM2', m2History, '#14B8A6');
    drawWaveform('canvasOut', outHistory, '#8B5CF6');
  }

  // Spectrogram
  if (t.spectrogram_slice && t.spectrogram_slice.length > 0) {
    const el = getEl('spectrogramWaiting');
    if (el) el.style.display = 'none';
    drawSpectrogram(t.spectrogram_slice);
  }
}

// Canvas Waveform Drawer
function drawWaveform(canvasId, samples, color) {
  const canvas = getEl(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.offsetWidth;
  const H = canvas.offsetHeight;
  if (W === 0 || H === 0) return;

  if (canvas.width !== W || canvas.height !== H) {
    canvas.width = W;
    canvas.height = H;
  }

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#070B14';
  ctx.fillRect(0, 0, W, H);

  // Subtle center line
  ctx.strokeStyle = '#111B2E';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, H / 2);
  ctx.lineTo(W, H / 2);
  ctx.stroke();

  if (!samples || samples.length === 0) return;

  let peak = 0.0001;
  for (let i = 0; i < samples.length; i++) {
    const a = Math.abs(samples[i]);
    if (a > peak) peak = a;
  }
  const autoscale = Math.min(200.0, Math.max(1.0, 0.45 / peak));
  const midY = H / 2;

  ctx.beginPath();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;

  for (let i = 0; i < samples.length; i++) {
    const x = (i / (samples.length - 1)) * W;
    const y = midY - Math.max(-1.0, Math.min(1.0, samples[i] * (peak > 0.05 ? 1.0 : (autoscale * 0.5)))) * (H * 0.42);
    if (i === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  }
  ctx.stroke();
}

// Live STFT Spectrogram Heatmap Drawer
function drawSpectrogram(slice) {
  const canvas = getEl('canvasSpectrogram');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.offsetWidth;
  const H = canvas.offsetHeight;
  if (W === 0 || H === 0) return;

  if (canvas.width !== W || canvas.height !== H) {
    canvas.width = W;
    canvas.height = H;
  }

  if (slice && slice.length > 0) {
    spectrogramHistory.push(slice);
    if (spectrogramHistory.length > SPEC_COLUMNS) {
      spectrogramHistory.shift();
    }
  }

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#070B14';
  ctx.fillRect(0, 0, W, H);

  if (spectrogramHistory.length === 0) return;

  const numCols = spectrogramHistory.length;
  const colWidth = W / SPEC_COLUMNS;
  const numBins = spectrogramHistory[0].length;
  const binHeight = H / numBins;

  for (let colIdx = 0; colIdx < numCols; colIdx++) {
    const colData = spectrogramHistory[colIdx];
    const x = (SPEC_COLUMNS - numCols + colIdx) * colWidth;

    for (let binIdx = 0; binIdx < numBins; binIdx++) {
      const y = H - (binIdx + 1) * binHeight;
      const val = colData[binIdx];
      ctx.fillStyle = getSpectrogramColor(val);
      ctx.fillRect(x, y, Math.ceil(colWidth), Math.ceil(binHeight));
    }
  }

  // Grid lines
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
  ctx.lineWidth = 1;
  [0.25, 0.5, 0.75].forEach(frac => {
    const gy = H * (1.0 - frac);
    ctx.beginPath();
    ctx.moveTo(0, gy);
    ctx.lineTo(W, gy);
    ctx.stroke();
  });
}

// Color map for STFT Spectrogram: Deep navy -> blue -> cyan -> green -> yellow -> orange
function getSpectrogramColor(v) {
  const t = Math.max(0.0, Math.min(1.0, v));
  if (t < 0.15) {
    // Deep navy
    const u = t / 0.15;
    return `rgb(${Math.floor(7 + u * 6)}, ${Math.floor(11 + u * 9)}, ${Math.floor(20 + u * 14)})`;
  } else if (t < 0.35) {
    // Blue
    const u = (t - 0.15) / 0.2;
    return `rgb(${Math.floor(13 + u * 15)}, ${Math.floor(20 + u * 80)}, ${Math.floor(34 + u * 150)})`;
  } else if (t < 0.55) {
    // Cyan (#22D3EE)
    const u = (t - 0.35) / 0.2;
    return `rgb(${Math.floor(28 + u * 6)}, ${Math.floor(100 + u * 111)}, ${Math.floor(184 + u * 54)})`;
  } else if (t < 0.75) {
    // Green (#22C55E)
    const u = (t - 0.55) / 0.2;
    return `rgb(${Math.floor(34 + u * 0)}, ${Math.floor(211 - u * 14)}, ${Math.floor(238 - u * 144)})`;
  } else if (t < 0.90) {
    // Yellow (#EAB308)
    const u = (t - 0.75) / 0.15;
    return `rgb(${Math.floor(34 + u * 200)}, ${Math.floor(197 - u * 18)}, ${Math.floor(94 - u * 86)})`;
  } else {
    // Orange (#F59E0B)
    const u = (t - 0.90) / 0.1;
    return `rgb(${Math.floor(234 + u * 11)}, ${Math.floor(179 - u * 21)}, ${Math.floor(8 + u * 3)})`;
  }
}

// System Status Check
async function checkStatus() {
  try {
    const res = await fetch(`${API_BASE}/status`);
    if (res.ok) {
      const data = await res.json();
      const hdrHwStatus = getEl('hdrHwStatus');
      if (hdrHwStatus) {
        const dot = hdrHwStatus.querySelector('.status-dot');
        if (dot) dot.className = data.running ? 'status-dot active' : 'status-dot active';
      }
      setStreamHeaderStatus(data.running ? 'LIVE STREAMING' : 'STREAM STOPPED', data.running);
      setPipelineActive(data.running);
      return data;
    }
  } catch (err) {
    setStreamHeaderStatus('AUDIO ERROR', false);
    setPipelineActive(false);
  }
}

// Initial Boot
window.addEventListener('DOMContentLoaded', () => {
  log('Initializing AETHEL123 Real-Time Audio Console...', 'info');
  setupTabs();
  setupTooltips();
  checkStatus();
  loadDevices();
  loadQualityMetrics();

  drawWaveform('canvasM1', m1History, '#22D3EE');
  drawWaveform('canvasM2', m2History, '#14B8A6');
  drawWaveform('canvasOut', outHistory, '#8B5CF6');
  drawSpectrogram(null);
});

