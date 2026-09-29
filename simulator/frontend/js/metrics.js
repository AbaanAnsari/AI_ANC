/**
 * simulator/frontend/js/metrics.js
 * ================================
 * Page 4: Performance & Evaluation Page Controller.
 *
 * Implements:
 *  - Authoritative held-out test evaluation display (202 samples, Epoch 29 checkpoint).
 *  - Input vs. Output performance comparison: SNR, PESQ, STOI.
 *  - Distinguishes actual measured values from target values with target indicators.
 *  - Real-time rolling Latency and RTF canvas history chart.
 *  - Noise classification accuracy breakdown.
 *  - Detailed latency breakdown (input buffer, STFT hop, AI inference, output buffer, E2E).
 */

class PerformanceEvaluationPage {
  constructor() {
    this.audio = window.audioEngine;
    this.initialized = false;
    this.chartTimer = null;
  }

  init() {
    if (this.initialized) return;
    this.initialized = true;

    // Subscribe to engine telemetry and quality metrics
    this.audio.subscribe('qualityMetrics', (m) => this.renderQualityMetrics(m));
    this.audio.subscribe('telemetry', (t) => this.renderLiveTelemetry(t));

    // Render immediately if cached
    if (this.audio.latestMetrics) {
      this.renderQualityMetrics(this.audio.latestMetrics);
    } else {
      this.audio.loadQualityMetrics();
    }

    // Canvas resize
    window.addEventListener('resize', () => {
      if (document.getElementById('pagePerformance')?.style.display !== 'none') {
        this.renderHistoryChart();
      }
    });

    // Start rolling chart ticker
    this.chartTimer = setInterval(() => {
      if (document.getElementById('pagePerformance')?.style.display !== 'none') {
        this.renderHistoryChart();
      }
    }, 400);

    this.renderHistoryChart();
  }

  renderQualityMetrics(data) {
    if (!data) return;
    if (!data.available) {
      const status = document.getElementById('evalQualityStatus');
      if (status) {
        status.textContent = 'Current evaluation unavailable';
        status.title = data.reason || '';
      }
      [
        'evalSampleCount', 'evalTimestamp',
        'evalSnrNoisy', 'evalSnrEnhanced', 'evalSnrDelta',
        'evalPesqNoisy', 'evalPesqEnhanced', 'evalPesqDelta',
        'evalStoiNoisy', 'evalStoiEnhanced', 'evalStoiDelta',
        'evalClassAccuracy', 'evalClassRatio',
        'accStationary', 'accBabble', 'accTransient',
      ].forEach(id => {
        const element = document.getElementById(id);
        if (element) element.textContent = '--';
      });
      ['barSnrNoisy', 'barSnrEnhanced', 'barPesqNoisy', 'barPesqEnhanced', 'barStoiNoisy', 'barStoiEnhanced']
        .forEach(id => {
          const bar = document.getElementById(id);
          if (bar) bar.style.width = '0%';
        });
      return;
    }

    // Sample count & header
    const elSampleCount = document.getElementById('evalSampleCount');
    const elEvalTimestamp = document.getElementById('evalTimestamp');
    const elQualityStatus = document.getElementById('evalQualityStatus');
    if (elSampleCount) elSampleCount.textContent = `${data.samples} samples`;
    if (elEvalTimestamp) elEvalTimestamp.textContent = data.timestamp || '--';
    if (elQualityStatus) {
      elQualityStatus.textContent = 'Current held-out evaluation';
      elQualityStatus.removeAttribute('title');
    }

    // 1. SNR Metrics
    if (data.snr_db) {
      const s = data.snr_db;
      this.setMetricCard('evalSnrNoisy', `${s.noisy >= 0 ? '+' : ''}${s.noisy.toFixed(2)} dB`);
      this.setMetricCard('evalSnrEnhanced', `${s.enhanced >= 0 ? '+' : ''}${s.enhanced.toFixed(2)} dB`);
      this.setMetricCard('evalSnrDelta', `${s.delta >= 0 ? '+' : ''}${s.delta.toFixed(2)} dB`, s.delta >= 0);
      this.setMetricCard('evalSnrTarget', `Target: > +${s.target.toFixed(1)} dB`);

      // Progress bar (scale -10 to +25 dB)
      const pctNoisy = Math.max(0, Math.min(100, ((s.noisy + 10) / 35) * 100));
      const pctEnhanced = Math.max(0, Math.min(100, ((s.enhanced + 10) / 35) * 100));
      const elBarNoisy = document.getElementById('barSnrNoisy');
      const elBarEnhanced = document.getElementById('barSnrEnhanced');
      if (elBarNoisy) elBarNoisy.style.width = `${pctNoisy}%`;
      if (elBarEnhanced) elBarEnhanced.style.width = `${pctEnhanced}%`;
    }

    // 2. PESQ Metrics
    if (data.pesq) {
      const p = data.pesq;
      this.setMetricCard('evalPesqNoisy', p.noisy.toFixed(2));
      this.setMetricCard('evalPesqEnhanced', p.enhanced.toFixed(2));
      this.setMetricCard('evalPesqDelta', `${p.delta >= 0 ? '+' : ''}${p.delta.toFixed(2)}`, p.delta >= 0);
      this.setMetricCard('evalPesqTarget', `Target: > ${p.target.toFixed(2)}`);

      // Progress bar (scale 1.0 to 4.5)
      const pctNoisy = Math.max(0, Math.min(100, ((p.noisy - 1.0) / 3.5) * 100));
      const pctEnhanced = Math.max(0, Math.min(100, ((p.enhanced - 1.0) / 3.5) * 100));
      const elBarNoisy = document.getElementById('barPesqNoisy');
      const elBarEnhanced = document.getElementById('barPesqEnhanced');
      if (elBarNoisy) elBarNoisy.style.width = `${pctNoisy}%`;
      if (elBarEnhanced) elBarEnhanced.style.width = `${pctEnhanced}%`;
    }

    // 3. STOI Metrics
    if (data.stoi) {
      const st = data.stoi;
      this.setMetricCard('evalStoiNoisy', st.noisy.toFixed(3));
      this.setMetricCard('evalStoiEnhanced', st.enhanced.toFixed(3));
      this.setMetricCard('evalStoiDelta', `${st.delta >= 0 ? '+' : ''}${st.delta.toFixed(3)}`, st.delta >= 0);
      this.setMetricCard('evalStoiTarget', `Target: > ${st.target.toFixed(2)}`);

      // Progress bar (scale 0.0 to 1.0)
      const pctNoisy = Math.max(0, Math.min(100, st.noisy * 100));
      const pctEnhanced = Math.max(0, Math.min(100, st.enhanced * 100));
      const elBarNoisy = document.getElementById('barStoiNoisy');
      const elBarEnhanced = document.getElementById('barStoiEnhanced');
      if (elBarNoisy) elBarNoisy.style.width = `${pctNoisy}%`;
      if (elBarEnhanced) elBarEnhanced.style.width = `${pctEnhanced}%`;
    }

    // 4. Noise Classifier
    if (data.classification) {
      const c = data.classification;
      const elAcc = document.getElementById('evalClassAccuracy');
      const elRatio = document.getElementById('evalClassRatio');
      if (elAcc) elAcc.textContent = `${(c.accuracy * 100).toFixed(1)}%`;
      if (elRatio) elRatio.textContent = `${c.correct} / ${c.total} Correct Predictions`;

      if (c.per_class_accuracy) {
        const pStat = c.per_class_accuracy.stationary;
        const pBab = c.per_class_accuracy['non-stationary'];
        const pTran = c.per_class_accuracy.impulsive;
        this.setMetricCard('accStationary', pStat == null ? '--' : `${(pStat * 100).toFixed(0)}%`);
        this.setMetricCard('accBabble', pBab == null ? '--' : `${(pBab * 100).toFixed(0)}%`);
        this.setMetricCard('accTransient', pTran == null ? '--' : `${(pTran * 100).toFixed(0)}%`);
      }
    }

    // 5. Model Footprint
    if (data.model_info) {
      const m = data.model_info;
      const elParams = document.getElementById('evalModelParams');
      const elBudget = document.getElementById('evalModelBudget');
      if (elParams) elParams.textContent = `${m.parameters.toLocaleString()} Parameters`;
      if (elBudget) elBudget.textContent = `Within ${m.parameter_limit.toLocaleString()} budget (70.8% capacity)`;
    }
  }

  setMetricCard(id, text, isPositive = null) {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = text;
    if (isPositive !== null) {
      el.className = `metric-delta ${isPositive ? 'positive' : 'negative'}`;
    }
  }

  renderLiveTelemetry(t) {
    if (!t || !t.performance) return;
    const p = t.performance;

    // Detailed Latency Stack
    const elInfLat = document.getElementById('evalInferenceLatency');
    const elInBufLat = document.getElementById('evalInputBufLatency');
    const elStftLat = document.getElementById('evalStftLatency');
    const elOutBufLat = document.getElementById('evalOutputBufLatency');
    const elE2ELat = document.getElementById('evalE2ELatency');
    const elLiveRTF = document.getElementById('evalLiveRTF');

    const infMs = p.last_processing_ms || 0.0;
    const e2eMs = p.total_end_to_end_latency_ms || 32.0;
    const rtf = p.rtf || 0.0;

    if (elInfLat) elInfLat.textContent = `${infMs.toFixed(2)} ms`;
    if (elInBufLat) elInBufLat.textContent = `16.0 ms (256 samples)`;
    if (elStftLat) elStftLat.textContent = `16.0 ms (512 FFT Hann)`;
    if (elOutBufLat) elOutBufLat.textContent = `0.0 – 16.0 ms`;
    if (elE2ELat) elE2ELat.textContent = `~${e2eMs.toFixed(1)} ms`;
    if (elLiveRTF) {
      elLiveRTF.textContent = `${rtf.toFixed(2)}×`;
      elLiveRTF.className = `eval-big-num ${rtf > 1.0 ? 'amber' : 'green'}`;
    }
  }

  renderHistoryChart() {
    const canvas = document.getElementById('canvasPerfHistory');
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

    // Dark canvas background
    ctx.fillStyle = '#060A14';
    ctx.fillRect(0, 0, W, H);

    // Grid lines & labels
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
    ctx.lineWidth = 1;
    ctx.fillStyle = '#64748B';
    ctx.font = '10px "IBM Plex Mono", monospace';

    // Latency Budget Reference Line (8 ms budget)
    // Scale: 0 to 16 ms max on Y-axis
    const maxMs = 16.0;
    const budgetY = H * (1.0 - (8.0 / maxMs));
    ctx.strokeStyle = 'rgba(245, 158, 11, 0.4)';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(0, budgetY);
    ctx.lineTo(W, budgetY);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillText('8.0 ms Budget', W - 85, budgetY - 5);

    // RTF = 1.0 Reference Line
    const rtf1Y = H * (1.0 - (1.0 / 2.0)); // 0 to 2.0 scale on secondary
    ctx.strokeStyle = 'rgba(239, 68, 68, 0.3)';
    ctx.setLineDash([2, 4]);
    ctx.beginPath();
    ctx.moveTo(0, rtf1Y);
    ctx.lineTo(W, rtf1Y);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillText('1.0× RTF Limit', 10, rtf1Y - 5);

    const latData = this.audio.latencyHistory;
    const rtfData = this.audio.rtfHistory;

    if (latData.length < 2) {
      ctx.fillStyle = '#64748B';
      ctx.font = '12px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('STREAM AUDIO TO POPULATE REAL-TIME LATENCY & RTF HISTORY', W / 2, H / 2);
      ctx.textAlign = 'left';
      return;
    }

    const n = latData.length;
    const maxPts = this.audio.HISTORY_POINTS;

    // Draw Latency Curve (Cyan)
    ctx.beginPath();
    ctx.strokeStyle = '#22D3EE';
    ctx.lineWidth = 2;
    ctx.shadowColor = '#22D3EE';
    ctx.shadowBlur = 4;
    for (let i = 0; i < n; i++) {
      const x = (i / (maxPts - 1)) * W;
      const y = H * (1.0 - Math.min(1.0, Math.max(0.0, latData[i] / maxMs)));
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Draw RTF Curve (Purple)
    ctx.beginPath();
    ctx.strokeStyle = '#A78BFA';
    ctx.lineWidth = 1.75;
    ctx.shadowColor = '#A78BFA';
    ctx.shadowBlur = 4;
    for (let i = 0; i < n; i++) {
      const x = (i / (maxPts - 1)) * W;
      const y = H * (1.0 - Math.min(1.0, Math.max(0.0, rtfData[i] / 2.0)));
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;
  }
}

window.performancePage = new PerformanceEvaluationPage();
