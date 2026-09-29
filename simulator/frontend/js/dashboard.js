/**
 * simulator/frontend/js/dashboard.js
 * ==================================
 * Page 1: Dashboard / Overview Page Controller.
 *
 * Displays:
 *  - AI model status (CNN+GRU, 70.8k parameters, epoch 29 checkpoint).
 *  - Live streaming indicators.
 *  - Key performance indicators: SNR, PESQ, STOI, Latency, and RTF in compact cards.
 *  - Quick navigation shortcuts to Hardware, Monitoring, and Performance pages.
 */

class DashboardPage {
  constructor() {
    this.audio = window.audioEngine;
    this.initialized = false;
    this.evaluationPollTimer = null;
  }

  init() {
    if (this.initialized) return;
    this.initialized = true;

    // Subscribe to global engine events
    this.audio.subscribe('telemetry', (t) => this.handleTelemetry(t));
    this.audio.subscribe('qualityMetrics', (m) => this.handleQualityMetrics(m));
    this.audio.subscribe('streamState', (s) => this.handleStreamState(s));
    this.audio.subscribe('devices', () => this.updateHardwareSummary());
    this.audio.subscribe('configChange', () => this.updateHardwareSummary());
    this.initEvaluationActions();

    // Navigation cards click listeners
    const navLinks = document.querySelectorAll('[data-nav-target]');
    navLinks.forEach(btn => {
      btn.addEventListener('click', (e) => {
        const target = btn.getAttribute('data-nav-target');
        if (target && window.router) {
          window.router.navigateTo(target);
        }
      });
    });

    // Populate initial state if available
    if (this.audio.latestMetrics) {
      this.handleQualityMetrics(this.audio.latestMetrics);
    }
    this.updateHardwareSummary();
  }

  initEvaluationActions() {
    document.querySelectorAll('[data-evaluation-task]').forEach(button => {
      button.addEventListener('click', () => {
        this.startEvaluation(button.dataset.evaluationTask);
      });
    });
    this.syncEvaluationJob();
  }

  async startEvaluation(task) {
    this.setEvaluationState(true, `Starting ${task}...`);
    try {
      const response = await fetch(`${this.audio.apiBase}/api/evaluation/jobs`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task }),
      });
      const job = await response.json();
      if (!response.ok) throw new Error(job.detail || 'Could not start evaluation');
      this.updateEvaluationJob(job);
    } catch (error) {
      this.setEvaluationState(false, `Could not start: ${error.message}`);
    }
  }

  async syncEvaluationJob() {
    try {
      const response = await fetch(`${this.audio.apiBase}/api/evaluation/jobs`);
      if (!response.ok) return;
      const data = await response.json();
      if (data.job) this.updateEvaluationJob(data.job);
    } catch (error) {
      // The backend may not be ready during page startup.
    }
  }

  updateEvaluationJob(job) {
    const taskName = job.task === 'ablation' ? 'Ablation study' : 'Evaluation metrics';
    const active = job.state === 'queued' || job.state === 'running';
    const messages = {
      queued: `${taskName} queued`,
      running: `${taskName} running...`,
      completed: `${taskName} complete`,
      failed: `${taskName} failed${job.error ? `: ${job.error}` : ''}`,
    };
    this.setEvaluationState(active, messages[job.state] || 'Ready');

    if (job.state === 'completed' && job.task === 'metrics') {
      this.audio.loadQualityMetrics();
    }

    if (this.evaluationPollTimer) {
      clearTimeout(this.evaluationPollTimer);
      this.evaluationPollTimer = null;
    }
    if (active) {
      this.evaluationPollTimer = setTimeout(() => this.syncEvaluationJob(), 1500);
    }
  }

  setEvaluationState(active, message) {
    document.querySelectorAll('[data-evaluation-task]').forEach(button => {
      button.disabled = active;
    });
    document.querySelectorAll('[data-job-status]').forEach(status => {
      status.textContent = message;
    });
  }

  handleStreamState(state) {
    const elModelLiveTag = document.getElementById('ovModelLiveTag');
    if (elModelLiveTag) {
      if (state.running) {
        elModelLiveTag.textContent = state.paused ? 'PAUSED / MUTED' : 'INFERENCE ACTIVE';
        elModelLiveTag.className = `badge-chip ${state.paused ? 'amber' : 'green'}`;
      } else {
        elModelLiveTag.textContent = 'STANDBY';
        elModelLiveTag.className = 'badge-chip gray';
      }
    }
  }

  handleQualityMetrics(data) {
    if (!data) return;
    const qualityStatus = document.getElementById('dashboardQualityStatus');
    if (!data.available) {
      if (qualityStatus) {
        qualityStatus.textContent = 'Current evaluation unavailable';
        qualityStatus.title = data.reason || '';
      }
      [
        'ovSNR', 'lblSnrNoisy', 'paramSnrDelta',
        'ovPESQ', 'lblPesqNoisy', 'paramPesqDelta',
        'ovSTOI', 'lblStoiNoisy', 'paramStoiDelta',
      ].forEach(id => {
        const element = document.getElementById(id);
        if (element) element.textContent = '--';
      });
      return;
    }
    if (qualityStatus) {
      qualityStatus.textContent = `${data.samples} samples | ${data.timestamp}`;
      qualityStatus.removeAttribute('title');
    }

    // SNR
    const elSNR = document.getElementById('ovSNR');
    const elSnrNoisy = document.getElementById('lblSnrNoisy');
    const elSnrDelta = document.getElementById('paramSnrDelta');
    if (data.snr_db) {
      if (elSNR) elSNR.textContent = `${data.snr_db.enhanced >= 0 ? '+' : ''}${data.snr_db.enhanced.toFixed(2)} dB`;
      if (elSnrNoisy) elSnrNoisy.textContent = `${data.snr_db.noisy >= 0 ? '+' : ''}${data.snr_db.noisy.toFixed(2)} dB`;
      if (elSnrDelta) {
        const d = data.snr_db.delta;
        elSnrDelta.textContent = `${d >= 0 ? '+' : ''}${d.toFixed(2)} dB`;
        elSnrDelta.className = `param-delta ${d >= 0 ? 'positive' : 'negative'}`;
      }
    }

    // PESQ
    const elPESQ = document.getElementById('ovPESQ');
    const elPesqNoisy = document.getElementById('lblPesqNoisy');
    const elPesqDelta = document.getElementById('paramPesqDelta');
    if (data.pesq) {
      if (elPESQ) elPESQ.textContent = data.pesq.enhanced.toFixed(2);
      if (elPesqNoisy) elPesqNoisy.textContent = data.pesq.noisy.toFixed(2);
      if (elPesqDelta) {
        const d = data.pesq.delta;
        elPesqDelta.textContent = `${d >= 0 ? '+' : ''}${d.toFixed(2)}`;
        elPesqDelta.className = `param-delta ${d >= 0 ? 'positive' : 'negative'}`;
      }
    }

    // STOI
    const elSTOI = document.getElementById('ovSTOI');
    const elStoiNoisy = document.getElementById('lblStoiNoisy');
    const elStoiDelta = document.getElementById('paramStoiDelta');
    if (data.stoi) {
      if (elSTOI) elSTOI.textContent = data.stoi.enhanced.toFixed(3);
      if (elStoiNoisy) elStoiNoisy.textContent = data.stoi.noisy.toFixed(3);
      if (elStoiDelta) {
        const d = data.stoi.delta;
        elStoiDelta.textContent = `${d >= 0 ? '+' : ''}${d.toFixed(3)}`;
        elStoiDelta.className = `param-delta ${d >= 0 ? 'positive' : 'negative'}`;
      }
    }

    // Model Info
    const elModelParams = document.getElementById('ovModelParams');
    if (elModelParams && data.model_info) {
      elModelParams.textContent = `${data.model_info.parameters.toLocaleString()} params (Epoch ${data.model_info.model_epoch})`;
    }
  }

  handleTelemetry(t) {
    if (!t) return;

    // Latency
    const elLatency = document.getElementById('ovLatency');
    const elLatencyE2E = document.getElementById('lblLatencyE2E');
    const elInferenceTime = document.getElementById('dispAIInferenceTime');
    if (t.performance) {
      const lat = t.performance.last_processing_ms || 0.0;
      const e2e = t.performance.total_end_to_end_latency_ms || 32.0;

      if (elLatency) {
        elLatency.textContent = `${lat.toFixed(1)} ms`;
        elLatency.className = `param-num ${lat > 8.0 ? 'amber' : 'green'}`;
      }
      if (elLatencyE2E) elLatencyE2E.textContent = `${e2e.toFixed(0)} ms E2E`;
      if (elInferenceTime) elInferenceTime.textContent = `~${lat.toFixed(1)} ms`;

      // RTF
      const elRTF = document.getElementById('ovRTF');
      const rtf = t.performance.rtf || 0.0;
      if (elRTF) {
        elRTF.textContent = `${rtf.toFixed(2)}×`;
        elRTF.className = `param-num ${rtf > 1.0 ? 'amber' : 'green'}`;
      }
    }

    // DSP & Speech status
    if (t.dsp) {
      const elNoiseClass = document.getElementById('ovNoiseClass');
      const elVAD = document.getElementById('ovVADState');
      if (elNoiseClass) elNoiseClass.textContent = t.dsp.noise_class || 'Stationary';
      if (elVAD) {
        const isSpeech = !!t.dsp.is_speech;
        elVAD.textContent = isSpeech ? 'Speech Active' : 'Non-Speech';
        elVAD.className = `vad-tag ${isSpeech ? 'active' : ''}`;
      }
    }

    // Mini Live Oscilloscope Indicators in the Overview Quick Card
    if (t.levels) {
      const dbM1 = this.audio.toDBFS(t.levels.input_rms_m1);
      const dbOut = this.audio.toDBFS(t.levels.output_rms);
      const elQuickM1 = document.getElementById('ovQuickM1RMS');
      const elQuickOut = document.getElementById('ovQuickOutRMS');
      if (elQuickM1) elQuickM1.textContent = `${dbM1.toFixed(1)} dBFS`;
      if (elQuickOut) elQuickOut.textContent = `${dbOut.toFixed(1)} dBFS`;
    }
  }

  updateHardwareSummary() {
    const elInSummary = document.getElementById('ovInputDeviceSummary');
    const elOutSummary = document.getElementById('ovOutputDeviceSummary');
    const inDevs = this.audio.devices.input_devices || [];
    const outDevs = this.audio.devices.output_devices || [];

    const activeIn = inDevs.find(d => d.id === this.audio.selectedInputId);
    const activeOut = outDevs.find(d => d.id === this.audio.selectedOutputId);

    if (elInSummary) {
      elInSummary.textContent = activeIn ? `[#${activeIn.id}] ${activeIn.name}` : 'Default System Input';
    }
    if (elOutSummary) {
      elOutSummary.textContent = activeOut ? `[#${activeOut.id}] ${activeOut.name}` : 'Default System Output';
    }
  }
}

window.dashboardPage = new DashboardPage();
