/**
 * simulator/frontend/js/audio.js
 * ==============================
 * Global Audio Engine and State Management Service for AETHEL123.
 *
 * Implements a unified application-level audio service that:
 *  - Manages physical microphone capture and playback controls (start, stop, mute, gain).
 *  - Enforces immutable processing block: 256 samples (16 ms @ 16 kHz).
 *  - Maintains persistent WebSocket / telemetry stream without interruption during page navigation.
 *  - Retains hardware device selection and audio history across all pages.
 *  - Dispatches events to page subscribers (Overview, Hardware, Monitoring, Performance).
 */

class AudioEngineService {
  constructor() {
    this.apiBase = window.location.origin.includes('http') ? window.location.origin : 'http://127.0.0.1:8000';
    
    // Immutable Engine Constants
    this.SAMPLE_RATE = 16000;
    this.BLOCK_SIZE = 256;
    this.BLOCK_DURATION_MS = 16.0;
    this.HOP_SIZE = 128;
    this.N_FFT = 512;
    this.CONFIG_INDICATOR = "256 samples | 16 ms | 16 kHz";

    // Engine State
    this.isStreaming = false;
    this.isPaused = false;
    this.masterGain = 0.5; // 0.0 to 1.0
    this.selectedInputId = null;
    this.selectedOutputId = null;
    this.primaryChannel = 0;
    this.refChannel = 1;

    // Discovered Devices
    this.devices = { input_devices: [], output_devices: [] };

    // Rolling Audio Buffers for Visualizers (persisted across pages)
    this.WAVEFORM_CAPACITY = 512;
    this.m1History = new Array(this.WAVEFORM_CAPACITY).fill(0);
    this.m2History = new Array(this.WAVEFORM_CAPACITY).fill(0);
    this.outHistory = new Array(this.WAVEFORM_CAPACITY).fill(0);

    // Spectrogram History (persisted)
    this.SPEC_COLUMNS = 60;
    this.spectrogramHistory = [];

    // Performance History (for real-time charts)
    this.HISTORY_POINTS = 60;
    this.latencyHistory = [];
    this.rtfHistory = [];

    // Latest Telemetry Snapshot
    this.latestTelemetry = null;
    this.latestMetrics = null;

    // WebSocket & Polling handles
    this.ws = null;
    this.pollInterval = null;
    this.listeners = new Map();

    // Load persisted device choices
    this.loadPersistedSettings();
  }

  // Event Subscription
  subscribe(event, callback) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event).add(callback);
    return () => this.listeners.get(event).delete(callback);
  }

  emit(event, data) {
    if (this.listeners.has(event)) {
      for (const cb of this.listeners.get(event)) {
        try {
          cb(data);
        } catch (err) {
          console.error(`Error in listener for ${event}:`, err);
        }
      }
    }
  }

  // Persistence helpers
  loadPersistedSettings() {
    try {
      const savedInput = localStorage.getItem('aethel_input_device');
      if (savedInput !== null) this.selectedInputId = parseInt(savedInput, 10);
      const savedOutput = localStorage.getItem('aethel_output_device');
      if (savedOutput !== null) this.selectedOutputId = parseInt(savedOutput, 10);
      const savedPCh = localStorage.getItem('aethel_primary_channel');
      if (savedPCh !== null) this.primaryChannel = parseInt(savedPCh, 10);
      const savedRCh = localStorage.getItem('aethel_ref_channel');
      if (savedRCh !== null) this.refChannel = parseInt(savedRCh, 10);
      const savedGain = localStorage.getItem('aethel_master_gain');
      if (savedGain !== null) this.masterGain = parseFloat(savedGain);
    } catch (e) {
      // localStorage unavailable or restricted
    }
  }

  savePersistedSettings() {
    try {
      if (this.selectedInputId !== null) localStorage.setItem('aethel_input_device', this.selectedInputId);
      if (this.selectedOutputId !== null) localStorage.setItem('aethel_output_device', this.selectedOutputId);
      localStorage.setItem('aethel_primary_channel', this.primaryChannel);
      localStorage.setItem('aethel_ref_channel', this.refChannel);
      localStorage.setItem('aethel_master_gain', this.masterGain);
    } catch (e) {
      // ignore
    }
  }

  // Hardware Device Management
  async refreshDevices() {
    try {
      const res = await fetch(`${this.apiBase}/api/devices/refresh`, { method: 'POST' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      this.devices = data;
      this.validateDeviceSelection();
      this.emit('devices', this.devices);
      return data;
    } catch (e) {
      console.error('Failed to refresh devices:', e);
      throw e;
    }
  }

  async loadDevices() {
    try {
      const res = await fetch(`${this.apiBase}/api/devices`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      this.devices = data;
      this.validateDeviceSelection();
      this.emit('devices', this.devices);
      return data;
    } catch (e) {
      console.error('Failed to load devices:', e);
      throw e;
    }
  }

  validateDeviceSelection() {
    const inDevs = this.devices.input_devices || [];
    const outDevs = this.devices.output_devices || [];

    if (inDevs.length > 0) {
      const exists = inDevs.some(d => d.id === this.selectedInputId);
      if (!exists) {
        const def = inDevs.find(d => d.is_default_input) || inDevs[0];
        this.selectedInputId = def.id;
      }
    }

    if (outDevs.length > 0) {
      const exists = outDevs.some(d => d.id === this.selectedOutputId);
      if (!exists) {
        const def = outDevs.find(d => d.is_default_output) || outDevs[0];
        this.selectedOutputId = def.id;
      }
    }
    this.savePersistedSettings();
  }

  setInputDevice(id) {
    this.selectedInputId = parseInt(id, 10);
    this.savePersistedSettings();
    this.emit('configChange', { inputId: this.selectedInputId });
  }

  setOutputDevice(id) {
    this.selectedOutputId = parseInt(id, 10);
    this.savePersistedSettings();
    this.emit('configChange', { outputId: this.selectedOutputId });
  }

  setChannels(primary, ref) {
    this.primaryChannel = parseInt(primary, 10);
    this.refChannel = parseInt(ref, 10);
    this.savePersistedSettings();
    this.emit('configChange', { primaryChannel: this.primaryChannel, refChannel: this.refChannel });
  }

  // Stream Lifecycle Controls
  async startStream() {
    if (this.isStreaming) return;

    try {
      const res = await fetch(`${this.apiBase}/api/stream/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          input_device_id: this.selectedInputId,
          output_device_id: this.selectedOutputId,
          primary_channel: this.primaryChannel,
          ref_channel: this.refChannel,
          sample_rate: this.SAMPLE_RATE,
          block_size: this.BLOCK_SIZE, // Fixed 256 samples
          master_gain: this.masterGain,
        }),
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Failed to start stream.');

      this.isStreaming = true;
      this.isPaused = false;
      this.emit('streamState', { running: true, paused: false });

      this.connectWebSocket();
      return data;
    } catch (err) {
      console.error('Error starting stream:', err);
      this.emit('streamError', err.message);
      throw err;
    }
  }

  async stopStream() {
    if (!this.isStreaming) return;

    try {
      await fetch(`${this.apiBase}/api/stream/stop`, { method: 'POST' });
    } catch (e) {
      console.error('Stop stream error:', e);
    } finally {
      this.isStreaming = false;
      this.isPaused = false;
      this.disconnectWebSocket();
      this.emit('streamState', { running: false, paused: false });
    }
  }

  async setMuted(muted) {
    this.isPaused = !!muted;
    try {
      await fetch(`${this.apiBase}/api/stream/pause`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ paused: this.isPaused }),
      });
      this.emit('streamState', { running: this.isStreaming, paused: this.isPaused });
    } catch (e) {
      console.error('Mute error:', e);
    }
  }

  async setGain(gainFraction) {
    this.masterGain = Math.max(0.0, Math.min(1.0, parseFloat(gainFraction)));
    this.savePersistedSettings();
    this.emit('gainChange', this.masterGain);

    if (this.isStreaming) {
      try {
        await fetch(`${this.apiBase}/api/stream/gain`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ gain: this.masterGain }),
        });
      } catch (err) {
        console.error('Gain change error:', err);
      }
    }
  }

  // Authoritative Metrics
  async loadQualityMetrics() {
    try {
      const res = await fetch(`${this.apiBase}/api/metrics/quality`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      this.latestMetrics = data;
      this.emit('qualityMetrics', data);
      return data;
    } catch (e) {
      console.warn('Could not load authoritative metrics:', e.message);
      return null;
    }
  }

  async checkInitialStatus() {
    try {
      const res = await fetch(`${this.apiBase}/status`);
      if (res.ok) {
        const data = await res.json();
        this.isStreaming = !!data.running;
        this.emit('streamState', { running: this.isStreaming, paused: this.isPaused });
        if (this.isStreaming) {
          this.connectWebSocket();
        }
        return data;
      }
    } catch (e) {
      // server offline or booting
    }
    return null;
  }

  // WebSocket & Telemetry Stream
  connectWebSocket() {
    if (this.ws) return;

    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsHost = window.location.host || '127.0.0.1:8000';
    const url = `${wsProtocol}//${wsHost}/ws/live`;

    try {
      this.ws = new WebSocket(url);

      this.ws.onmessage = (event) => {
        try {
          const telemetry = JSON.parse(event.data);
          this.handleTelemetry(telemetry);
        } catch (err) {
          console.error('Telemetry parse error:', err);
        }
      };

      this.ws.onerror = () => {
        this.startPollingFallback();
      };

      this.ws.onclose = () => {
        this.ws = null;
        if (this.isStreaming) this.startPollingFallback();
      };
    } catch (err) {
      this.startPollingFallback();
    }
  }

  disconnectWebSocket() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    if (this.pollInterval) {
      clearInterval(this.pollInterval);
      this.pollInterval = null;
    }
  }

  startPollingFallback() {
    if (this.pollInterval) return;
    this.pollInterval = setInterval(async () => {
      if (!this.isStreaming) {
        clearInterval(this.pollInterval);
        this.pollInterval = null;
        return;
      }
      try {
        const res = await fetch(`${this.apiBase}/api/stream/telemetry`);
        if (res.ok) {
          const data = await res.json();
          this.handleTelemetry(data);
        }
      } catch (e) {
        // ignore
      }
    }, 40);
  }

  handleTelemetry(t) {
    if (!t) return;
    this.latestTelemetry = t;

    // Update Waveform History
    if (t.waveforms && t.waveforms.m1 && t.waveforms.m1.length > 0) {
      const k = t.waveforms.m1.length;
      this.m1History = this.m1History.slice(k).concat(t.waveforms.m1);
      this.m2History = this.m2History.slice(k).concat(t.waveforms.m2);
      this.outHistory = this.outHistory.slice(k).concat(t.waveforms.output);
    }

    // Update Spectrogram Slice
    if (t.spectrogram_slice && t.spectrogram_slice.length > 0) {
      this.spectrogramHistory.push(t.spectrogram_slice);
      if (this.spectrogramHistory.length > this.SPEC_COLUMNS) {
        this.spectrogramHistory.shift();
      }
    }

    // Update Rolling Latency and RTF History
    if (t.performance) {
      const lat = t.performance.last_processing_ms || 0.0;
      const rtf = t.performance.rtf || 0.0;
      this.latencyHistory.push(lat);
      if (this.latencyHistory.length > this.HISTORY_POINTS) this.latencyHistory.shift();

      this.rtfHistory.push(rtf);
      if (this.rtfHistory.length > this.HISTORY_POINTS) this.rtfHistory.shift();
    }

    // Broadcast to active pages
    this.emit('telemetry', t);
  }

  toDBFS(amp) {
    if (!amp || amp <= 1e-6) return -96.0;
    return Math.max(-96.0, 20.0 * Math.log10(amp));
  }
}

// Global Audio Engine Service Singleton
window.audioEngine = new AudioEngineService();
