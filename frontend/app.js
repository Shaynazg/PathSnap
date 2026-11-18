/**
 * PathSnap — Evaluator-Facing Map & Route Reconstruction Controller
 */

// 1. Config & State
const DEFAULT_URL = window.location.port ? window.location.origin : 'http://localhost:8000';
let API_BASE_URL = localStorage.getItem('pathsnap_backend_url') || DEFAULT_URL;

let map = null;
let gpsLayer = L.layerGroup();
let routeLayer = L.layerGroup();
let markerLayer = L.layerGroup();
let currentPolylineBounds = null;

// DOM Elements
const statusIndicator = document.getElementById('backendStatus');
const statusText = statusIndicator.querySelector('.status-text');
const apiUrlInput = document.getElementById('apiUrlInput');
const saveUrlBtn = document.getElementById('saveUrlBtn');
const tripSelect = document.getElementById('tripSelect');
const reconstructBtn = document.getElementById('reconstructBtn');

const progressAlert = document.getElementById('progressAlert');
const progressTitle = document.getElementById('progressTitle');
const progressDesc = document.getElementById('progressDesc');
const errorAlert = document.getElementById('errorAlert');
const errorDesc = document.getElementById('errorDesc');

const resultsSection = document.getElementById('resultsSection');
const tripBadge = document.getElementById('tripBadge');
const metricGpsObs = document.getElementById('metricGpsObs');
const metricCandidates = document.getElementById('metricCandidates');
const metricDecodedArcs = document.getElementById('metricDecodedArcs');
const metricNoPath = document.getElementById('metricNoPath');

const evalComparisonCard = document.getElementById('evalComparisonCard');
const evalVitPrec = document.getElementById('evalVitPrec');
const evalVitRec = document.getElementById('evalVitRec');
const evalVitF1 = document.getElementById('evalVitF1');
const evalTop1Prec = document.getElementById('evalTop1Prec');
const evalTop1Rec = document.getElementById('evalTop1Rec');
const evalTop1F1 = document.getElementById('evalTop1F1');
const evalF1Gain = document.getElementById('evalF1Gain');

const totalTimeMs = document.getElementById('totalTimeMs');
const timeMlMs = document.getElementById('timeMlMs');
const timeHmmMs = document.getElementById('timeHmmMs');
const timeVitMs = document.getElementById('timeVitMs');
const arcSequenceText = document.getElementById('arcSequenceText');
const arcSeqCount = document.getElementById('arcSeqCount');

const toggleGpsBtn = document.getElementById('toggleGpsBtn');
const toggleRouteBtn = document.getElementById('toggleRouteBtn');
const fitBoundsBtn = document.getElementById('fitBoundsBtn');

// 2. Initialize Map
function initMap() {
  map = L.map('map', {
    center: [49.85, 8.65],
    zoom: 13,
    zoomControl: true,
  });

  // CartoDB Dark Matter tile layer for high-contrast visual display
  L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/attributions">CARTO</a>',
    subdomains: 'abcd',
    maxZoom: 19,
  }).addTo(map);

  gpsLayer.addTo(map);
  routeLayer.addTo(map);
  markerLayer.addTo(map);
}

// 3. API Communication
async function checkHealth() {
  statusIndicator.className = 'status-indicator';
  statusText.textContent = 'Connecting...';

  try {
    const res = await fetch(`${API_BASE_URL}/health`, { method: 'GET' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    statusIndicator.className = 'status-indicator connected';
    statusText.textContent = `Online: ${data.model_name}`;
    reconstructBtn.disabled = false;
    return true;
  } catch (err) {
    statusIndicator.className = 'status-indicator disconnected';
    statusText.textContent = `Offline (${err.message})`;
    reconstructBtn.disabled = true;
    showError(`Cannot reach backend at ${API_BASE_URL}. Check Laptop 1 IP and port 8000.`);
    return false;
  }
}

async function loadTrips() {
  try {
    const res = await fetch(`${API_BASE_URL}/trips`, { method: 'GET' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    tripSelect.innerHTML = '';
    data.trips.forEach((t) => {
      const opt = document.createElement('option');
      opt.value = t.trip_id;
      opt.textContent = `${t.label}`;
      tripSelect.appendChild(opt);
    });

    if (data.trips.length > 0) {
      tripSelect.selectedIndex = 0;
      reconstructBtn.disabled = false;
    }
  } catch (err) {
    console.error('Failed to load trips:', err);
    showError(`Failed to fetch trip list: ${err.message}`);
  }
}

// 4. Run Reconstruction
async function runReconstruction() {
  const selectedTrip = tripSelect.value;
  if (!selectedTrip) return;

  hideError();
  showProgress('Starting reconstruction...', `Processing trip ${selectedTrip} through ML & HMM pipeline`);
  reconstructBtn.disabled = true;
  reconstructBtn.querySelector('.btn-spinner').style.display = 'inline-block';
  reconstructBtn.querySelector('.btn-text').textContent = 'Reconstructing...';

  try {
    const res = await fetch(`${API_BASE_URL}/reconstruct`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ trip_id: selectedTrip, split: 'student_test' }),
    });

    if (!res.ok) {
      const errJson = await res.json().catch(() => ({}));
      throw new Error(errJson.detail || `Server error (${res.status})`);
    }

    const data = await res.json();
    displayResults(data);
    hideProgress();
  } catch (err) {
    console.error('Reconstruction failed:', err);
    hideProgress();
    showError(`Reconstruction failed: ${err.message}`);
  } finally {
    reconstructBtn.disabled = false;
    reconstructBtn.querySelector('.btn-spinner').style.display = 'none';
    reconstructBtn.querySelector('.btn-text').textContent = 'Reconstruct Route';
  }
}

// 5. Render Results and Map
function displayResults(data) {
  resultsSection.style.display = 'block';
  tripBadge.textContent = `Trip ${data.trip_id}`;
  metricGpsObs.textContent = data.total_observations.toLocaleString();
  metricCandidates.textContent = data.total_candidates.toLocaleString();
  metricDecodedArcs.textContent = data.total_decoded_arcs.toLocaleString();
  metricNoPath.textContent = data.no_path_segments.toString();

  if (data.no_path_segments > 0) {
    metricNoPath.style.color = '#f59e0b';
  } else {
    metricNoPath.style.color = '#10b981';
  }

  // Evaluation Card
  if (data.evaluation) {
    evalComparisonCard.style.display = 'block';
    const vit = data.evaluation.viterbi_reconstruction;
    const top1 = data.evaluation.classifier_top1_baseline;

    evalVitPrec.textContent = (vit.precision * 100).toFixed(1) + '%';
    evalVitRec.textContent = (vit.recall * 100).toFixed(1) + '%';
    evalVitF1.textContent = (vit.f1_score * 100).toFixed(1) + '%';

    evalTop1Prec.textContent = (top1.precision * 100).toFixed(1) + '%';
    evalTop1Rec.textContent = (top1.recall * 100).toFixed(1) + '%';
    evalTop1F1.textContent = (top1.f1_score * 100).toFixed(1) + '%';

    const gain = data.evaluation.hmm_f1_improvement * 100;
    evalF1Gain.textContent = `${gain >= 0 ? '+' : ''}${gain.toFixed(2)}%`;
    evalF1Gain.className = gain >= 0 ? 'positive-gain' : 'negative-gain';
  } else {
    evalComparisonCard.style.display = 'none';
  }

  // Timing
  if (data.timing_ms) {
    totalTimeMs.textContent = `${data.timing_ms.total_ms.toFixed(1)} ms`;
    timeMlMs.textContent = data.timing_ms.model_inference_ms.toFixed(1);
    timeHmmMs.textContent = data.timing_ms.hmm_scoring_ms.toFixed(1);
    timeVitMs.textContent = data.timing_ms.viterbi_decoding_ms.toFixed(1);
  }

  // Arc Sequence
  arcSeqCount.textContent = data.decoded_arc_sequence.length.toString();
  arcSequenceText.textContent = data.decoded_arc_sequence.join(' → ');

  // Clear previous layers
  gpsLayer.clearLayers();
  routeLayer.clearLayers();
  markerLayer.clearLayers();

  const allLatLngs = [];

  // Plot GPS Points
  if (data.gps_points && data.gps_points.length > 0) {
    data.gps_points.forEach((pt) => {
      const latlng = [pt.lat, pt.lon];
      allLatLngs.push(latlng);

      const circle = L.circleMarker(latlng, {
        radius: 4,
        fillColor: '#f97316',
        color: '#ffffff',
        weight: 1.5,
        opacity: 0.9,
        fillOpacity: 0.8,
      }).bindPopup(`<b>Observation #${pt.obs_idx}</b><br>Lat: ${pt.lat.toFixed(6)}<br>Lon: ${pt.lon.toFixed(6)}<br>Timestamp: ${pt.timestamp}`);

      gpsLayer.addLayer(circle);
    });
  }

  // Plot Reconstructed Route Polyline
  if (data.route_polyline && data.route_polyline.length > 0) {
    const polyline = L.polyline(data.route_polyline, {
      color: '#10b981',
      weight: 5,
      opacity: 0.9,
      lineCap: 'round',
      lineJoin: 'round',
    }).bindTooltip(`Reconstructed Route (${data.total_decoded_arcs} Arcs)`, { sticky: true });

    routeLayer.addLayer(polyline);

    // Start / End Markers
    const startPt = data.route_polyline[0];
    const endPt = data.route_polyline[data.route_polyline.length - 1];

    const startIcon = L.divIcon({
      className: 'start-marker-icon',
      html: '<div style="background:#22c55e;color:white;width:24px;height:24px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:bold;font-size:12px;border:2px solid white;box-shadow:0 2px 6px rgba(0,0,0,0.5)">S</div>',
      iconSize: [24, 24],
      iconAnchor: [12, 12],
    });

    const endIcon = L.divIcon({
      className: 'end-marker-icon',
      html: '<div style="background:#ef4444;color:white;width:24px;height:24px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:bold;font-size:12px;border:2px solid white;box-shadow:0 2px 6px rgba(0,0,0,0.5)">E</div>',
      iconSize: [24, 24],
      iconAnchor: [12, 12],
    });

    markerLayer.addLayer(L.marker(startPt, { icon: startIcon }).bindPopup('<b>Trip Start Point</b>'));
    markerLayer.addLayer(L.marker(endPt, { icon: endIcon }).bindPopup('<b>Trip End Point</b>'));

    currentPolylineBounds = polyline.getBounds();
    map.fitBounds(currentPolylineBounds, { padding: [40, 40] });
  } else if (allLatLngs.length > 0) {
    currentPolylineBounds = L.latLngBounds(allLatLngs);
    map.fitBounds(currentPolylineBounds, { padding: [40, 40] });
  }
}

// 6. Helpers & Alerts
function showProgress(title, desc) {
  progressTitle.textContent = title;
  progressDesc.textContent = desc;
  progressAlert.style.display = 'flex';
}

function hideProgress() {
  progressAlert.style.display = 'none';
}

function showError(msg) {
  errorDesc.textContent = msg;
  errorAlert.style.display = 'flex';
}

function hideError() {
  errorAlert.style.display = 'none';
}

// 7. Event Listeners
apiUrlInput.value = API_BASE_URL;

saveUrlBtn.addEventListener('click', async () => {
  let val = apiUrlInput.value.trim();
  if (!val.startsWith('http')) {
    val = `http://${val}`;
  }
  API_BASE_URL = val;
  localStorage.setItem('pathsnap_backend_url', API_BASE_URL);
  const ok = await checkHealth();
  if (ok) {
    loadTrips();
  }
});

reconstructBtn.addEventListener('click', runReconstruction);

tripSelect.addEventListener('change', () => {
  hideError();
});

toggleGpsBtn.addEventListener('click', () => {
  if (map.hasLayer(gpsLayer)) {
    map.removeLayer(gpsLayer);
    toggleGpsBtn.classList.remove('active');
  } else {
    map.addLayer(gpsLayer);
    toggleGpsBtn.classList.add('active');
  }
});

toggleRouteBtn.addEventListener('click', () => {
  if (map.hasLayer(routeLayer)) {
    map.removeLayer(routeLayer);
    toggleRouteBtn.classList.remove('active');
  } else {
    map.addLayer(routeLayer);
    toggleRouteBtn.classList.add('active');
  }
});

fitBoundsBtn.addEventListener('click', () => {
  if (currentPolylineBounds) {
    map.fitBounds(currentPolylineBounds, { padding: [40, 40] });
  }
});

// Auto-run on load
window.addEventListener('DOMContentLoaded', async () => {
  initMap();
  const ok = await checkHealth();
  if (ok) {
    await loadTrips();
  }
});
