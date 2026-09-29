/**
 * simulator/frontend/js/plots.js
 * ==============================
 * Page 3: Real-Time Audio Monitoring Visualizer Controller.
 *
 * Implements:
 *  - 3 Separate Oscilloscopes:
 *     1. Mic 1 — Primary Input (Speech + Noise)
 *     2. Mic 2 — Reference Input (Environmental Noise)
 *     3. Enhanced Speech Output (AI CRM + NLMS + Limiter)
 *  - Live STFT Spectrogram (512 FFT, 128 Hop, 0 - 8000 Hz)
 *  - Interactive visualization controls:
 *     * Amplitude zoom (0.5×, 1×, 2×, 4×, Autoscale)
 *     * Time window resolution (128, 256, 512 samples)
 *     * Spectrogram colormap selector (Cyberpunk, Inferno, Emerald)
 *     * Freeze/Pause display toggle
 *  - High-DPI canvas rendering with actual live audio buffers.
 */

class RealTimeMonitoringPage {
  constructor() {
    this.audio = window.audioEngine;
    this.initialized = false;
    this.isFrozen = false;
    this.ampScale = 1.0;
    this.autoScale = true;
    this.timeWindow = 256;
    this.colorScheme = 'cyberpunk'; // 'cyberpunk', 'inferno', 'emerald'
    this.animationId = null;
  }

  init() {
    if (this.initialized) return;
    this.initialized = true;

    // Controls
    const selAmp = document.getElementById('selAmpScale');
    if (selAmp) {
      selAmp.addEventListener('change', (e) => {
        const val = e.target.value;
        if (val === 'auto') {
          this.autoScale = true;
          this.ampScale = 1.0;
        } else {
          this.autoScale = false;
          this.ampScale = parseFloat(val);
        }
        this.renderAll();
      });
    }

    const selTime = document.getElementById('selTimeWindow');
    if (selTime) {
      selTime.addEventListener('change', (e) => {
        this.timeWindow = parseInt(e.target.value, 10);
        this.renderAll();
      });
    }

    const selColormap = document.getElementById('selSpectrogramColor');
    if (selColormap) {
      selColormap.addEventListener('change', (e) => {
        this.colorScheme = e.target.value;
        this.renderSpectrogram();
      });
    }

    const btnFreeze = document.getElementById('btnFreezeVisualizer');
    if (btnFreeze) {
      btnFreeze.addEventListener('click', () => {
        this.isFrozen = !this.isFrozen;
        btnFreeze.classList.toggle('active', this.isFrozen);
        btnFreeze.textContent = this.isFrozen ? '▶ Resume Display' : '⏸ Freeze Display';
      });
    }

    // Subscribe to telemetry
    this.audio.subscribe('telemetry', (t) => this.handleTelemetry(t));
    this.audio.subscribe('streamState', (s) => this.handleStreamState(s));

    // Window resize
    window.addEventListener('resize', () => {
      if (document.getElementById('pageMonitoring')?.style.display !== 'none') {
        this.renderAll();
      }
    });

    // Initial render
    this.renderAll();
  }

  handleStreamState(state) {
    const overlays = ['m1Waiting', 'm2Waiting', 'outWaiting', 'spectrogramWaiting'];
    overlays.forEach(id => {
      const el = document.getElementById(id);
      if (el) el.style.display = state.running ? 'none' : 'block';
    });
  }

  handleTelemetry(t) {
    if (this.isFrozen) return;

    // Level meters
    if (t.levels) {
      const dbM1 = this.audio.toDBFS(t.levels.input_rms_m1);
      const dbM2 = this.audio.toDBFS(t.levels.input_rms_m2);
      const dbOut = this.audio.toDBFS(t.levels.output_rms);
      const pkM1 = this.audio.toDBFS(t.levels.input_peak_m1);
      const pkM2 = this.audio.toDBFS(t.levels.input_peak_m2);
      const pkOut = this.audio.toDBFS(t.levels.output_peak);

      const elM1RMS = document.getElementById('lblM1RMS');
      const elM1Pk = document.getElementById('lblM1Peak');
      const elM2RMS = document.getElementById('lblM2RMS');
      const elM2Pk = document.getElementById('lblM2Peak');
      const elOutRMS = document.getElementById('lblOutRMS');
      const elOutPk = document.getElementById('lblOutPeak');

      if (elM1RMS) elM1RMS.textContent = `${dbM1.toFixed(1)} dBFS`;
      if (elM1Pk) elM1Pk.textContent = `${pkM1.toFixed(1)} dBFS`;
      if (elM2RMS) elM2RMS.textContent = `${dbM2.toFixed(1)} dBFS`;
      if (elM2Pk) elM2Pk.textContent = `${pkM2.toFixed(1)} dBFS`;
      if (elOutRMS) elOutRMS.textContent = `${dbOut.toFixed(1)} dBFS`;
      if (elOutPk) elOutPk.textContent = `${pkOut.toFixed(1)} dBFS`;
    }

    // Hide waiting banners when live
    if (this.audio.isStreaming) {
      ['m1Waiting', 'm2Waiting', 'outWaiting', 'spectrogramWaiting'].forEach(id => {
        const el = document.getElementById(id);
        if (el && el.style.display !== 'none') el.style.display = 'none';
      });
    }

    // Render Waveforms & Spectrogram
    this.renderAll();
  }

  renderAll() {
    const k = this.timeWindow;
    const m1Slice = this.audio.m1History.slice(-k);
    const m2Slice = this.audio.m2History.slice(-k);
    const outSlice = this.audio.outHistory.slice(-k);

    this.drawWaveform('canvasM1', m1Slice, '#22D3EE'); // Neon Cyan
    this.drawWaveform('canvasM2', m2Slice, '#14B8A6'); // Teal
    this.drawWaveform('canvasOut', outSlice, '#A78BFA'); // Purple
    this.renderSpectrogram();
  }

  drawWaveform(canvasId, samples, color) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.offsetWidth;
    const H = canvas.offsetHeight;
    if (W === 0 || H === 0) return;

    const dpr = window.devicePixelRatio || 1;
    const targetW = Math.round(W * dpr);
    const targetH = Math.round(H * dpr);
    if (canvas.width !== targetW || canvas.height !== targetH) {
      canvas.width = targetW;
      canvas.height = targetH;
    }

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    // Deep background
    ctx.fillStyle = '#060A14';
    ctx.fillRect(0, 0, W, H);

    // Subtle grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, H / 2);
    ctx.lineTo(W, H / 2);
    ctx.moveTo(0, H * 0.25);
    ctx.lineTo(W, H * 0.25);
    ctx.moveTo(0, H * 0.75);
    ctx.lineTo(W, H * 0.75);
    ctx.stroke();

    if (!samples || samples.length === 0) return;

    // Peak autoscale calculation
    let peak = 0.0001;
    for (let i = 0; i < samples.length; i++) {
      const a = Math.abs(samples[i]);
      if (a > peak) peak = a;
    }

    let multiplier = this.ampScale;
    if (this.autoScale) {
      multiplier = Math.min(100.0, Math.max(1.0, 0.5 / peak));
    }

    const midY = H / 2;
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.75;
    ctx.shadowColor = color;
    ctx.shadowBlur = 6;

    const len = samples.length;
    for (let i = 0; i < len; i++) {
      const x = (i / (len - 1)) * W;
      const s = Math.max(-1.0, Math.min(1.0, samples[i] * multiplier));
      const y = midY - s * (H * 0.42);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }

    ctx.stroke();
    ctx.shadowBlur = 0;
  }

  renderSpectrogram() {
    const canvas = document.getElementById('canvasSpectrogram');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.offsetWidth;
    const H = canvas.offsetHeight;
    if (W === 0 || H === 0) return;

    const dpr = window.devicePixelRatio || 1;
    const targetW = Math.round(W * dpr);
    const targetH = Math.round(H * dpr);
    if (canvas.width !== targetW || canvas.height !== targetH) {
      canvas.width = targetW;
      canvas.height = targetH;
    }

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#060A14';
    ctx.fillRect(0, 0, W, H);

    const history = this.audio.spectrogramHistory;
    if (history.length === 0) return;

    const numCols = history.length;
    const maxCols = this.audio.SPEC_COLUMNS;
    const colWidth = W / maxCols;
    const numBins = history[0].length;
    const binHeight = H / numBins;

    for (let colIdx = 0; colIdx < numCols; colIdx++) {
      const colData = history[colIdx];
      const x = (maxCols - numCols + colIdx) * colWidth;

      for (let binIdx = 0; binIdx < numBins; binIdx++) {
        const y = H - (binIdx + 1) * binHeight;
        const val = colData[binIdx];
        ctx.fillStyle = this.getColormapColor(val);
        ctx.fillRect(x, y, Math.ceil(colWidth), Math.ceil(binHeight));
      }
    }

    // Frequency guide lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
    ctx.lineWidth = 1;
    [0.25, 0.5, 0.75].forEach(frac => {
      const gy = H * (1.0 - frac);
      ctx.beginPath();
      ctx.moveTo(0, gy);
      ctx.lineTo(W, gy);
      ctx.stroke();
    });
  }

  getColormapColor(v) {
    const t = Math.max(0.0, Math.min(1.0, v));

    if (this.colorScheme === 'inferno') {
      // Thermal sunset palette
      if (t < 0.25) {
        const u = t / 0.25;
        return `rgb(${Math.floor(10 + u * 40)}, ${Math.floor(5 + u * 15)}, ${Math.floor(25 + u * 60)})`;
      } else if (t < 0.5) {
        const u = (t - 0.25) / 0.25;
        return `rgb(${Math.floor(50 + u * 130)}, ${Math.floor(20 + u * 30)}, ${Math.floor(85 - u * 30)})`;
      } else if (t < 0.75) {
        const u = (t - 0.5) / 0.25;
        return `rgb(${Math.floor(180 + u * 60)}, ${Math.floor(50 + u * 100)}, ${Math.floor(55 - u * 40)})`;
      } else {
        const u = (t - 0.75) / 0.25;
        return `rgb(${Math.floor(240 + u * 15)}, ${Math.floor(150 + u * 95)}, ${Math.floor(15 + u * 180)})`;
      }
    } else if (this.colorScheme === 'emerald') {
      // Matrix Emerald palette
      if (t < 0.3) {
        const u = t / 0.3;
        return `rgb(${Math.floor(5 + u * 5)}, ${Math.floor(15 + u * 45)}, ${Math.floor(10 + u * 20)})`;
      } else if (t < 0.7) {
        const u = (t - 0.3) / 0.4;
        return `rgb(${Math.floor(10 + u * 20)}, ${Math.floor(60 + u * 150)}, ${Math.floor(30 + u * 90)})`;
      } else {
        const u = (t - 0.7) / 0.3;
        return `rgb(${Math.floor(30 + u * 190)}, ${Math.floor(210 + u * 45)}, ${Math.floor(120 + u * 100)})`;
      }
    }

    // Default: Cyberpunk Cyan / Purple / Neon Yellow
    if (t < 0.15) {
      const u = t / 0.15;
      return `rgb(${Math.floor(7 + u * 6)}, ${Math.floor(11 + u * 9)}, ${Math.floor(20 + u * 14)})`;
    } else if (t < 0.35) {
      const u = (t - 0.15) / 0.2;
      return `rgb(${Math.floor(13 + u * 15)}, ${Math.floor(20 + u * 80)}, ${Math.floor(34 + u * 150)})`;
    } else if (t < 0.55) {
      const u = (t - 0.35) / 0.2;
      return `rgb(${Math.floor(28 + u * 6)}, ${Math.floor(100 + u * 111)}, ${Math.floor(184 + u * 54)})`;
    } else if (t < 0.75) {
      const u = (t - 0.55) / 0.2;
      return `rgb(${Math.floor(34 + u * 0)}, ${Math.floor(211 - u * 14)}, ${Math.floor(238 - u * 144)})`;
    } else if (t < 0.90) {
      const u = (t - 0.75) / 0.15;
      return `rgb(${Math.floor(34 + u * 200)}, ${Math.floor(197 - u * 18)}, ${Math.floor(94 - u * 86)})`;
    } else {
      const u = (t - 0.90) / 0.1;
      return `rgb(${Math.floor(234 + u * 21)}, ${Math.floor(179 - u * 21)}, ${Math.floor(8 + u * 3)})`;
    }
  }
}

window.monitoringPage = new RealTimeMonitoringPage();
