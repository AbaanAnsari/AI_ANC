/**
 * simulator/frontend/js/main.js
 * =============================
 * Main Application Orchestrator and SPA Router for AETHEL123.
 *
 * Coordinates:
 *  - Persistent Multi-Page Navigation without restarting the audio stream or resetting state.
 *  - Global Stream Controls (Start, Stop, Mute, Gain, Status Pill).
 *  - Page 2: Audio Hardware & Routing controller.
 *  - Page-level lifecycle hooks (Overview, Hardware, Monitoring, Performance).
 */

class AppRouter {
  constructor() {
    this.pages = ['overview', 'hardware', 'monitoring', 'performance'];
    this.currentPage = 'overview';
    this.navButtons = document.querySelectorAll('.nav-tab-btn');
    this.pageContainers = {};

    this.pages.forEach(p => {
      this.pageContainers[p] = document.getElementById(`page${this.capitalize(p)}`);
    });
  }

  capitalize(s) {
    return s.charAt(0).toUpperCase() + s.slice(1);
  }

  init() {
    // Tab clicks
    this.navButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        const target = btn.getAttribute('data-tab');
        if (target) this.navigateTo(target);
      });
    });

    // Hash change / popstate
    window.addEventListener('hashchange', () => this.handleHashChange());
    window.addEventListener('popstate', () => this.handleHashChange());

    // Resolve initial route
    this.handleHashChange();
  }

  handleHashChange() {
    let route = window.location.hash.replace('#', '').toLowerCase();
    if (!route || !this.pages.includes(route)) {
      const path = window.location.pathname.replace('/', '').toLowerCase();
      route = this.pages.includes(path) ? path : 'overview';
    }
    this.navigateTo(route, false);
  }

  navigateTo(page, updateHash = true) {
    if (!this.pages.includes(page)) page = 'overview';
    this.currentPage = page;

    if (updateHash) {
      window.location.hash = `#${page}`;
    }

    // Update active tab buttons
    this.navButtons.forEach(btn => {
      const isTarget = btn.getAttribute('data-tab') === page;
      btn.classList.toggle('active', isTarget);
    });

    // Toggle page visibility without tearing down DOM or stopping audio!
    this.pages.forEach(p => {
      const container = this.pageContainers[p];
      if (container) {
        if (p === page) {
          container.style.display = 'block';
          container.classList.add('page-active');
        } else {
          container.style.display = 'none';
          container.classList.remove('page-active');
        }
      }
    });

    // Page-specific activation triggers
    if (page === 'monitoring' && window.monitoringPage) {
      window.monitoringPage.renderAll();
    } else if (page === 'performance' && window.performancePage) {
      window.performancePage.renderHistoryChart();
    } else if (page === 'hardware') {
      updateHardwarePageControls();
    }
  }
}

// ===========================================================================
// PAGE 2: AUDIO HARDWARE & ROUTING CONTROLLER
// ===========================================================================

function initHardwarePage() {
  const audio = window.audioEngine;

  const selInput = document.getElementById('selectInputDevice');
  const selOutput = document.getElementById('selectOutputDevice');
  const selPrimary = document.getElementById('selectPrimaryChan');
  const selRef = document.getElementById('selectRefChan');
  const btnRefresh = document.getElementById('btnRefreshDevices');
  const chkPassthrough = document.getElementById('chkPassthrough');

  // Input Device changed
  if (selInput) {
    selInput.addEventListener('change', () => {
      audio.setInputDevice(selInput.value);
      populateChannelOptions();
    });
  }

  // Output Device changed
  if (selOutput) {
    selOutput.addEventListener('change', () => {
      audio.setOutputDevice(selOutput.value);
    });
  }

  // Channel selections changed
  if (selPrimary && selRef) {
    const handleChanChange = () => {
      audio.setChannels(selPrimary.value, selRef.value);
    };
    selPrimary.addEventListener('change', handleChanChange);
    selRef.addEventListener('change', handleChanChange);
  }

  // Rescan Devices
  if (btnRefresh) {
    btnRefresh.addEventListener('click', async () => {
      btnRefresh.disabled = true;
      btnRefresh.classList.add('loading');
      try {
        await audio.refreshDevices();
        updateHardwarePageControls();
        fetchAudioHealth();
      } catch (e) {
        alert(`Rescan failed: ${e.message}`);
      } finally {
        btnRefresh.disabled = false;
        btnRefresh.classList.remove('loading');
      }
    });
  }

  // Passthrough diagnostic mode toggle
  if (chkPassthrough) {
    chkPassthrough.addEventListener('change', async () => {
      try {
        await fetch(`${audio.apiBase}/api/stream/passthrough`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ enabled: chkPassthrough.checked }),
        });
      } catch (err) {
        console.error('Passthrough mode error:', err);
      }
    });
  }

  // Subscribe to device updates
  audio.subscribe('devices', () => updateHardwarePageControls());

  // Initial load
  updateHardwarePageControls();
  fetchAudioHealth();
}

function updateHardwarePageControls() {
  const audio = window.audioEngine;
  const selInput = document.getElementById('selectInputDevice');
  const selOutput = document.getElementById('selectOutputDevice');
  if (!selInput || !selOutput) return;

  const inDevs = audio.devices.input_devices || [];
  const outDevs = audio.devices.output_devices || [];

  // Populate Input Select
  selInput.innerHTML = '';
  if (inDevs.length > 0) {
    inDevs.forEach(dev => {
      const opt = document.createElement('option');
      opt.value = dev.id;
      const defTag = dev.is_default_input ? ' ★ Default' : '';
      const api = dev.hostapi_name || dev.host_api || 'Audio';
      opt.textContent = `[#${dev.id}] ${dev.name} (${api}, ${dev.max_input_channels} ch)${defTag}`;
      if (dev.id === audio.selectedInputId) opt.selected = true;
      selInput.appendChild(opt);
    });
  } else {
    selInput.innerHTML = '<option value="">No input devices detected</option>';
  }

  // Populate Output Select
  selOutput.innerHTML = '';
  if (outDevs.length > 0) {
    outDevs.forEach(dev => {
      const opt = document.createElement('option');
      opt.value = dev.id;
      const defTag = dev.is_default_output ? ' ★ Default' : '';
      const api = dev.hostapi_name || dev.host_api || 'Audio';
      opt.textContent = `[#${dev.id}] ${dev.name} (${api}, ${dev.max_output_channels} ch)${defTag}`;
      if (dev.id === audio.selectedOutputId) opt.selected = true;
      selOutput.appendChild(opt);
    });
  } else {
    selOutput.innerHTML = '<option value="">No output devices detected</option>';
  }

  // Update device count pills
  const elInCount = document.getElementById('hwInputCount');
  const elOutCount = document.getElementById('hwOutputCount');
  const elGlobalHw = document.getElementById('lblHardware');
  if (elInCount) elInCount.textContent = `${inDevs.length} Input Devices`;
  if (elOutCount) elOutCount.textContent = `${outDevs.length} Output Devices`;
  if (elGlobalHw) elGlobalHw.textContent = `${inDevs.length} In / ${outDevs.length} Out Devices`;

  populateChannelOptions();
}

function populateChannelOptions() {
  const audio = window.audioEngine;
  const selInput = document.getElementById('selectInputDevice');
  const selPrimary = document.getElementById('selectPrimaryChan');
  const selRef = document.getElementById('selectRefChan');
  const alertSingleMic = document.getElementById('singleMicAlert');
  if (!selInput || !selPrimary || !selRef) return;

  const inId = parseInt(selInput.value, 10);
  const inDevs = audio.devices.input_devices || [];
  const dev = inDevs.find(d => d.id === inId);

  selPrimary.innerHTML = '';
  selRef.innerHTML = '';

  if (!dev) return;

  const nCh = dev.max_input_channels || 1;
  for (let c = 0; c < nCh; c++) {
    const optP = document.createElement('option');
    optP.value = c;
    optP.textContent = `Channel ${c} ${c === 0 ? '(Speech+Noise)' : ''}`;
    if (c === audio.primaryChannel) optP.selected = true;
    selPrimary.appendChild(optP);

    const optR = document.createElement('option');
    optR.value = c;
    optR.textContent = `Channel ${c} ${c === 1 ? '(Environmental Noise)' : ''}`;
    if (c === audio.refChannel) optR.selected = true;
    selRef.appendChild(optR);
  }

  if (nCh < 2) {
    if (alertSingleMic) alertSingleMic.style.display = 'flex';
    const optNone = document.createElement('option');
    optNone.value = -1;
    optNone.textContent = 'None (Single-Mic Synthesis)';
    optNone.selected = true;
    selRef.appendChild(optNone);
  } else {
    if (alertSingleMic) alertSingleMic.style.display = 'none';
  }
}

async function fetchAudioHealth() {
  const audio = window.audioEngine;
  try {
    const res = await fetch(`${audio.apiBase}/api/audio/health`);
    if (res.ok) {
      const data = await res.json();
      const elHealthStatus = document.getElementById('hwHealthStatus');
      const elHealthNote = document.getElementById('hwHealthNote');
      if (elHealthStatus) {
        elHealthStatus.textContent = data.healthy ? 'HEALTHY / OPERATIONAL' : 'DEGRADED';
        elHealthStatus.className = `badge-chip ${data.healthy ? 'green' : 'amber'}`;
      }
      if (elHealthNote) {
        elHealthNote.textContent = data.details || 'Audio engine and PortAudio I/O ready.';
      }
    }
  } catch (e) {
    // health check optional
  }
}

// ===========================================================================
// GLOBAL HEADER CONTROLS (Always available, state-persistent across pages)
// ===========================================================================

function initGlobalHeaderControls() {
  const audio = window.audioEngine;

  const btnStart = document.getElementById('btnStartStream');
  const btnStop = document.getElementById('btnStopStream');
  const btnPause = document.getElementById('btnPauseStream');
  const sliderGain = document.getElementById('masterGainSlider');
  const lblGain = document.getElementById('gainDisplay');
  const dotStream = document.getElementById('dotStream');
  const lblStream = document.getElementById('lblStream');
  const footerStatus = document.getElementById('footerStatusText');

  // Start Stream
  if (btnStart) {
    btnStart.addEventListener('click', async () => {
      btnStart.disabled = true;
      try {
        await audio.startStream();
      } catch (err) {
        btnStart.disabled = false;
        alert(`Could not start stream: ${err.message}`);
      }
    });
  }

  // Stop Stream
  if (btnStop) {
    btnStop.addEventListener('click', async () => {
      btnStop.disabled = true;
      try {
        await audio.stopStream();
      } finally {
        btnStart.disabled = false;
      }
    });
  }

  // Mute Stream
  if (btnPause) {
    btnPause.addEventListener('click', async () => {
      await audio.setMuted(!audio.isPaused);
    });
  }

  // Master Gain Slider
  if (sliderGain && lblGain) {
    sliderGain.value = Math.round(audio.masterGain * 100);
    lblGain.textContent = `${sliderGain.value}%`;

    sliderGain.addEventListener('input', () => {
      lblGain.textContent = `${sliderGain.value}%`;
      audio.setGain(parseFloat(sliderGain.value) / 100.0);
    });
  }

  // Sync with engine stream state
  audio.subscribe('streamState', (state) => {
    if (btnStart) btnStart.disabled = state.running;
    if (btnStop) btnStop.disabled = !state.running;
    if (btnPause) {
      btnPause.disabled = !state.running;
      btnPause.textContent = state.paused ? '▶ Unmute' : '🔇 Mute';
      btnPause.classList.toggle('active', state.paused);
    }

    if (dotStream) {
      dotStream.className = `status-dot ${state.running ? (state.paused ? 'paused' : 'active') : 'offline'}`;
    }
    if (lblStream) {
      lblStream.textContent = state.running ? (state.paused ? 'STREAM MUTED' : 'LIVE STREAMING') : 'STREAM IDLE';
    }
    if (footerStatus) {
      footerStatus.textContent = state.running
        ? (state.paused ? 'Stream Active (Muted)' : '● Real-Time AI Stream Active')
        : 'Ready for Native Audio Stream';
    }
  });

  audio.subscribe('gainChange', (gain) => {
    if (sliderGain) sliderGain.value = Math.round(gain * 100);
    if (lblGain) lblGain.textContent = `${Math.round(gain * 100)}%`;
  });
}

// ===========================================================================
// APPLICATION BOOT
// ===========================================================================

window.addEventListener('DOMContentLoaded', async () => {
  // 1. Initialize Global Header & Controls
  initGlobalHeaderControls();

  // 2. Initialize Router
  window.router = new AppRouter();
  window.router.init();

  // 3. Initialize Pages
  if (window.dashboardPage) window.dashboardPage.init();
  initHardwarePage();
  if (window.monitoringPage) window.monitoringPage.init();
  if (window.performancePage) window.performancePage.init();

  // 4. Fetch initial backend status, devices, and metrics
  await window.audioEngine.checkInitialStatus();
  await window.audioEngine.loadDevices();
  await window.audioEngine.loadQualityMetrics();
});
