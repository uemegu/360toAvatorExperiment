import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';
import { createLook } from './look.js';
import { createFaceTracker, findPerfectSync, applyPerfectSync, toPresets, boost } from './facetrack.js';
import * as Splats from '@mkkellogg/gaussian-splats-3d';
import { createMorph } from './morph.js?v=12';
import { findHairMesh, createHairFX } from './hairfx.js?v=14';
import { createMotion } from './motion.js?v=3';

const params = new URLSearchParams(location.search);
// default: the Japanese face with ARKit face units (perfect sync) and its Gaussian hair
const MODEL_URL = params.get('model') || './assets/bang-face-arkit.vrm';
const HAIR_URL = params.get('hair') ?? (params.has('model') ? '' : './assets/bang-hair-v2.ply');
const MORPH_URL = params.get('morph') || '';   // a second VRM with the same meshes to shape-shift into
const MOTION = params.get('motion') === 'walk' ? 'walk' : 'none';   // ?motion=walk
const BASE_BLINK = params.has('blink') ? +params.get('blink') : 0.1;   // lids a little lowered by default (?blink=0 opens them)
const SAMPLE_VOICE = './assets/sample-voice.mp3';
const MOUTH = ['aa', 'ih', 'ou', 'ee', 'oh'];
const EMOTIONS = [
  ['neutral', '通常'], ['happy', '笑顔'], ['angry', '怒り'], ['sad', '悲しみ'],
  ['surprised', '驚き'], ['relaxed', 'くつろぎ'], ['blink', '目を閉じる'],
];
// Rough first/second formants (Hz) of Japanese vowels (adult female), used to pick the mouth shape.
// Labels for the per-expression sliders (VRM preset names; left/right are the character's own).
const EXPRESSION_LABELS = {
  blink: '両目を閉じる', blinkLeft: '左目を閉じる', blinkRight: '右目を閉じる',
  aa: 'あ', ih: 'い', ou: 'う', ee: 'え', oh: 'お',
  happy: '笑顔', angry: '怒り', sad: '悲しみ', surprised: '驚き', relaxed: 'くつろぎ', neutral: '通常',
  lookUp: '上を見る', lookDown: '下を見る', lookLeft: '左を見る', lookRight: '右を見る',
};
const FORMANTS = { aa: [950, 1500], ih: [350, 2700], ou: [400, 1600], ee: [580, 2300], oh: [580, 1000] };

const $ = (id) => document.getElementById(id);
const canvas = $('canvas');
const statusEl = $('status');

// ---- renderer / scene: lighting for physically based (non-toon) materials ----
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.NeutralToneMapping;
renderer.toneMappingExposure = 1;

const scene = new THREE.Scene();
scene.background = new THREE.Color(getComputedStyle(document.documentElement).getPropertyValue('--bg').trim() || '#e7e4df');
const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
scene.environmentIntensity = 0.6;

const key = new THREE.DirectionalLight(0xfff4e8, 1.8);
key.position.set(0.8, 2.2, 1.6);
scene.add(key);
const rim = new THREE.DirectionalLight(0xdfe8ff, 0.7);
rim.position.set(-1.2, 1.8, -1.0);
scene.add(rim);
scene.add(new THREE.HemisphereLight(0xffffff, 0x6b6157, 0.35));

// A long lens, as portraits are shot: up close, perspective narrows the cheeks and enlarges the nose.
const camera = new THREE.PerspectiveCamera(14, 1, 0.01, 50);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 0.4;
controls.maxDistance = 12;

function resize() {
  const { clientWidth: w, clientHeight: h } = canvas;
  renderer.setSize(w, h, false);
  camera.aspect = w / Math.max(h, 1);
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(canvas);

// ---- model ----
const loader = new GLTFLoader();
loader.register((parser) => new VRMLoaderPlugin(parser));
let vrm = null;
const look = createLook();
let headY = 1.5;
let lockCamera = false;   // set by viewer.setCamera (automated comparisons with a fixed camera)
let emotion = 'neutral';
let emotionWeight = 1;
const current = {};   // smoothed expression weights
const mouthTarget = Object.fromEntries(MOUTH.map((m) => [m, 0]));

async function loadModel(url, label) {
  statusEl.textContent = '読み込み中';
  const gltf = await loader.loadAsync(url);
  if (vrm) {
    scene.remove(vrm.scene);
    VRMUtils.deepDispose(vrm.scene);
  }
  vrm = gltf.userData.vrm;
  VRMUtils.removeUnnecessaryVertices(gltf.scene);
  VRMUtils.combineSkeletons(gltf.scene);
  if (vrm.meta?.metaVersion === '0') VRMUtils.rotateVRM0(vrm);
  vrm.scene.traverse((o) => {
    if (!o.isMesh) return;
    o.frustumCulled = false;
  });
  look.reset();
  applyLook();
  scene.add(vrm.scene);
  applyStudio(studio);
  measurePosture();
  relaxPose();
  motion.attach(vrm);
  vrm.update(0);
  const hairMesh = findHairMesh(vrm.scene);
  hairfx = hairMesh ? [createHairFX(hairMesh)].filter(Boolean) : [];
  applyHairFX();
  const head = vrm.humanoid.getNormalizedBoneNode('head');
  headY = head ? head.getWorldPosition(new THREE.Vector3()).y : 1.5;
  for (const k of Object.keys(current)) delete current[k];
  buildExpressionButtons();
  buildManualSliders();
  sync = findPerfectSync(vrm);
  showTrackStatus();
  if (BASE_BLINK > 0) setManual('blink', BASE_BLINK);
  setView(motion.name === 'walk' ? 'full' : 'bust');
  if (HAIR_URL && !label) await loadHair(HAIR_URL).catch(showError);
  morph.clear();
  $('morph-panel').hidden = true;
  if (MORPH_URL && !label) {
    statusEl.textContent = '変身先を読み込み中';
    const r = await morph.prepare(vrm, MORPH_URL, { companions: hairCompanions }).catch(showError);
    if (r) {
      $('morph-panel').hidden = false;
      $('morph-status').textContent = `${r.moving} 個のメッシュが変形、${r.textures} 枚のテクスチャが入れ替わる` +
        (r.out || r.in ? `、頂点数の合わない ${r.out} 個は溶けて消え ${r.in} 個が現れる` : '');
    }
  }
  $('model-name').textContent = label || url;
  statusEl.textContent = '';
  document.body.dataset.ready = '1';
}

function bone(name) {
  return vrm?.humanoid.getNormalizedBoneNode(name);
}

// The MPFB skeleton these models share puts the head joint 23° ahead of the neck joint, and at rest the neck reads as
// craned forward. Lean past 12° is taken back at the neck, the head turned 60 % of it the other way so the face keeps
// looking ahead; the idle sway and face tracking add their motion on top of this.
const posture = { neck: 0, head: 0 };
function measurePosture() {
  posture.neck = posture.head = 0;
  const n = vrm.humanoid.getRawBoneNode('neck'), h = vrm.humanoid.getRawBoneNode('head');
  if (!n || !h) return;
  vrm.scene.updateMatrixWorld(true);
  const a = n.getWorldPosition(new THREE.Vector3()), b = h.getWorldPosition(new THREE.Vector3());
  const lean = Math.atan2(b.z - a.z, b.y - a.y);   // + = forward (the model faces +z)
  const fix = Math.max(0, lean - THREE.MathUtils.degToRad(12));
  posture.neck = -fix;
  posture.head = 0.6 * fix;
}

// Arms down from the T-pose, elbows and fingers a little bent (straight fingers read as a mannequin).
function relaxPose() {
  const set = (name, x, y, z) => bone(name)?.rotation.set(x, y, z);
  set('neck', posture.neck, 0, 0);
  set('head', posture.head, 0, 0);
  set('leftUpperArm', 0, 0, -1.32);
  set('rightUpperArm', 0, 0, 1.32);
  set('leftLowerArm', 0, -0.35, 0);
  set('rightLowerArm', 0, 0.35, 0);
  const curl = { Index: 0.25, Middle: 0.35, Ring: 0.45, Little: 0.55 };
  for (const [side, s] of [['left', -1], ['right', 1]]) {
    for (const [finger, k] of Object.entries(curl)) {
      for (const [seg, m] of [['Proximal', 1], ['Intermediate', 1.3], ['Distal', 0.9]]) set(side + finger + seg, 0, 0, s * k * m);
    }
    set(side + 'ThumbMetacarpal', 0, -s * 0.2, 0);
    set(side + 'ThumbProximal', 0, -s * 0.2, 0);
  }
}

// ---- views ----
const VIEWS = {
  face: { dist: 1.12, dy: 0.03 },
  bust: { dist: 1.93, dy: -0.04 },
  full: { dist: 6.9, dy: null },
};
function setView(name) {
  const v = VIEWS[name];
  const ty = v.dy === null ? headY * 0.52 : headY + v.dy;
  controls.target.set(0, ty, 0);
  camera.position.set(0, ty + (name === 'full' ? 0.1 : 0.02), v.dist);
  controls.update();
  document.querySelectorAll('[data-view]').forEach((b) => b.classList.toggle('on', b.dataset.view === name));
}
document.querySelectorAll('[data-view]').forEach((b) => b.addEventListener('click', () => setView(b.dataset.view)));

// ---- expressions ----
function buildExpressionButtons() {
  const box = $('expressions');
  box.innerHTML = '';
  const names = new Set((vrm.expressionManager?.expressions || [])
    .filter((e) => (e._binds?.length ?? e.binds?.length ?? 1) > 0).map((e) => e.expressionName));
  for (const [name, label] of EMOTIONS) {
    if (name !== 'neutral' && !names.has(name)) continue;
    const b = document.createElement('button');
    b.textContent = label;
    b.dataset.exp = name;
    b.addEventListener('click', () => setExpression(name));
    box.appendChild(b);
  }
  setExpression(emotion in Object.fromEntries(EMOTIONS) ? emotion : 'neutral');
}
function setExpression(name) {
  emotion = name;
  document.querySelectorAll('[data-exp]').forEach((b) => b.classList.toggle('on', b.dataset.exp === name));
}
$('exp-weight').addEventListener('input', (e) => {
  emotionWeight = +e.target.value;
  $('exp-weight-out').textContent = emotionWeight.toFixed(2);
});

// manual weights, one slider per expression the model has (added on top of the automatic ones)
const manual = {};
function buildManualSliders() {
  const box = $('manual-exp');
  box.innerHTML = '';
  for (const k of Object.keys(manual)) { vrm?.expressionManager?.setValue(k, 0); delete manual[k]; }
  // only expressions that actually move something (presets with no binds are left out)
  const names = (vrm.expressionManager?.expressions || [])
    .filter((e) => (e._binds?.length ?? e.binds?.length ?? 1) > 0)
    .map((e) => e.expressionName);
  const order = Object.keys(EXPRESSION_LABELS);
  names.sort((a, b) => (order.indexOf(a) + 1 || 99) - (order.indexOf(b) + 1 || 99));
  for (const name of names) {
    const label = document.createElement('label');
    label.className = 'slider';
    label.innerHTML = `<span></span><input type="range" min="0" max="1" step="0.01" value="0"><output>0.00</output>`;
    label.querySelector('span').textContent = EXPRESSION_LABELS[name] || name;
    const input = label.querySelector('input');
    const out = label.querySelector('output');
    input.dataset.exp = name;
    input.addEventListener('input', () => {
      manual[name] = +input.value;
      out.textContent = (+input.value).toFixed(2);
      if (!manual[name]) vrm?.expressionManager?.setValue(name, 0);   // nothing else resets the ARKit keys
    });
    box.appendChild(label);
  }
}
$('manual-reset').addEventListener('click', () => {
  for (const k of Object.keys(manual)) { vrm?.expressionManager?.setValue(k, 0); delete manual[k]; }
  document.querySelectorAll('#manual-exp input').forEach((i) => { i.value = 0; i.nextElementSibling.textContent = '0.00'; });
  if (BASE_BLINK > 0) setManual('blink', BASE_BLINK);
});
function setManual(name, value) {
  manual[name] = value;
  const input = document.querySelector(`#manual-exp input[data-exp="${name}"]`);
  if (input) { input.value = value; input.nextElementSibling.textContent = (+value).toFixed(2); }
}

// automatic blinking
let blinkT = 0;
let nextBlink = 2 + Math.random() * 3;
function blinkWeight(dt) {
  if (!$('auto-blink').checked || emotion === 'blink') return 0;
  blinkT += dt;
  if (blinkT > nextBlink) {
    const t = blinkT - nextBlink;
    if (t > 0.16) {
      blinkT = 0;
      nextBlink = 1.8 + Math.random() * 4;
      return 0;
    }
    return Math.sin((t / 0.16) * Math.PI);
  }
  return 0;
}

// ---- lip sync ----
const audio = $('audio');
let ctx = null;
let analyser = null;
let source = null;
let mediaSource = null;
let micStream = null;
let freq = null;
let wave = null;

function ensureAudio() {
  if (ctx) return;
  ctx = new AudioContext();
  analyser = ctx.createAnalyser();
  analyser.fftSize = 2048;
  analyser.smoothingTimeConstant = 0.5;
  freq = new Float32Array(analyser.frequencyBinCount);
  wave = new Float32Array(analyser.fftSize);
}

function stopLip() {
  audio.pause();
  if (micStream) {
    micStream.getTracks().forEach((t) => t.stop());
    micStream = null;
  }
  if (source && source !== mediaSource) source.disconnect();
  source = null;
  document.querySelectorAll('#lip-sample,#lip-file,#lip-mic').forEach((b) => b.classList.remove('on'));
}

async function playAudio(url, button) {
  ensureAudio();
  stopLip();
  if (!mediaSource) {
    mediaSource = ctx.createMediaElementSource(audio);
    mediaSource.connect(analyser);
    analyser.connect(ctx.destination);
  }
  source = mediaSource;
  audio.src = url;
  await ctx.resume();
  await audio.play();
  button.classList.add('on');
}

async function startMic() {
  ensureAudio();
  stopLip();
  micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  source = ctx.createMediaStreamSource(micStream);
  source.connect(analyser); // not to the speakers: no echo
  await ctx.resume();
  $('lip-mic').classList.add('on');
}

audio.addEventListener('ended', () => document.querySelectorAll('#lip-sample,#lip-file').forEach((b) => b.classList.remove('on')));
$('lip-sample').addEventListener('click', () => playAudio(SAMPLE_VOICE, $('lip-sample')).catch(showError));
$('lip-file').addEventListener('click', () => $('audio-input').click());
$('audio-input').addEventListener('change', (e) => {
  const f = e.target.files[0];
  if (f) playAudio(URL.createObjectURL(f), $('lip-file')).catch(showError);
});
$('lip-mic').addEventListener('click', () => startMic().catch(showError));
$('lip-stop').addEventListener('click', stopLip);
for (const id of ['lip-gain', 'lip-max']) {
  $(id).addEventListener('input', (e) => { $(id + '-out').textContent = (+e.target.value).toFixed(id === 'lip-gain' ? 1 : 2); });
}

function peakHz(lo, hi, binHz) {
  let best = -Infinity, at = lo;
  for (let i = Math.ceil(lo / binHz); i <= Math.floor(hi / binHz); i++) {
    // light smoothing across neighbouring bins
    const v = (freq[i - 2] + 2 * freq[i - 1] + 3 * freq[i] + 2 * freq[i + 1] + freq[i + 2]) / 9;
    if (v > best) { best = v; at = i * binHz; }
  }
  return at;
}

// Volume -> how far the mouth opens; F1/F2 peaks -> which vowel shape.
function analyseVoice() {
  for (const m of MOUTH) mouthTarget[m] = 0;
  if (!source || (source === mediaSource && audio.paused)) return 0;
  analyser.getFloatTimeDomainData(wave);
  let sum = 0;
  for (const s of wave) sum += s * s;
  const rms = Math.sqrt(sum / wave.length);
  const level = Math.min(1, Math.max(0, (rms - 0.01) * 7 * +$('lip-gain').value));
  if (level <= 0) return 0;
  analyser.getFloatFrequencyData(freq);
  const binHz = ctx.sampleRate / analyser.fftSize;
  const f1 = peakHz(280, 1200, binHz);
  const f2 = peakHz(Math.max(f1 + 250, 850), 3200, binHz);
  let total = 0;
  const score = {};
  for (const [v, [a, b]] of Object.entries(FORMANTS)) {
    const d = Math.log(f1 / a) ** 2 + Math.log(f2 / b) ** 2;
    score[v] = Math.exp(-d / (2 * 0.18 ** 2));
    total += score[v];
  }
  const open = level * +$('lip-max').value;
  for (const v of MOUTH) mouthTarget[v] = total > 0 ? open * score[v] / total : (v === 'aa' ? open : 0);
  return level;
}

// ---- idle motion ----
let time = 0;
function idle(dt) {
  if (!$('idle').checked || motion.playing) return;   // a body motion has its own breathing and sway
  time += dt;
  const breath = Math.sin(time * 1.6);
  bone('spine')?.rotation.set(breath * 0.012, 0, 0);
  bone('chest')?.rotation.set(breath * 0.01, 0, 0);
  if (tracking) return;   // the camera drives the neck and head
  bone('neck')?.rotation.set(posture.neck + Math.sin(time * 0.5) * 0.02, Math.sin(time * 0.31) * 0.04, 0);
  bone('head')?.rotation.set(posture.head + Math.sin(time * 0.43 + 1) * 0.02, Math.sin(time * 0.27 + 2) * 0.05, Math.sin(time * 0.37) * 0.015);
}

// ---- face tracking (webcam) ----
const tracker = createFaceTracker();
let tracking = false;
let sync = null;   // ARKit-named expressions/morphs the model has (perfect sync)
const tracked = {};   // smoothed values from the camera
function smooth(name, target, dt) {
  const speed = 4 + 36 * (1 - +$('track-smooth').value);
  const cur = tracked[name] ?? target;
  return (tracked[name] = cur + (target - cur) * Math.min(1, dt * speed));
}
const PERFECT_SYNC_MIN = 40;   // treat the model as perfect sync when it has most of the 52
function showTrackStatus() {
  const el = $('track-status');
  if (!tracking) return;
  const mode = sync?.count >= PERFECT_SYNC_MIN
    ? `パーフェクトシンク（${sync.count} / 52 個をそのまま）`
    : 'モデルの表情に割り当て（目・口・笑顔・怒り・悲しみ・驚き）';
  el.textContent = `${tracker.found ? '顔を検出中' : '顔が見つからない'}。${mode}`;
  el.classList.toggle('found', tracker.found);
}
async function startTracking() {
  const b = $('track-start');
  if (tracking) {
    tracker.stop();
    tracking = false;
    b.classList.remove('on');
    b.textContent = 'カメラで動かす';
    $('cam').hidden = true;
    $('track-status').textContent = '停止した。';
    $('track-status').classList.remove('found');
    for (const k of Object.keys(tracked)) delete tracked[k];
    if (vrm?.lookAt) { vrm.lookAt.yaw = 0; vrm.lookAt.pitch = 0; }
    return;
  }
  $('track-status').textContent = 'カメラと顔の検出モデルを準備中';
  await tracker.start($('cam'));
  tracking = true;
  b.classList.add('on');
  b.textContent = '止める';
  $('cam').hidden = !$('track-video').checked;
  showTrackStatus();
}
$('track-start').addEventListener('click', () => startTracking().catch((e) => {
  tracking = false;
  $('track-status').textContent = `カメラを使えない: ${e.message || e}`;
}));
$('track-front').addEventListener('click', () => tracker.calibrate());
$('track-video').addEventListener('change', (e) => { $('cam').hidden = !tracking || !e.target.checked; });
$('track-smooth').addEventListener('input', (e) => { $('track-smooth-out').textContent = (+e.target.value).toFixed(2); });
$('track-gain').addEventListener('input', (e) => { $('track-gain-out').textContent = (+e.target.value).toFixed(1); });

// Returns { presets } (or {} for perfect sync) when the camera drove the face this frame, else null.
let lastFound = null;
function trackFace(dt) {
  if (!tracking) return null;
  const found = tracker.update();
  if (found !== lastFound) { lastFound = found; showTrackStatus(); }
  if (!found && !Object.keys(tracked).length) return null;
  const { shapes, pitch, yaw, roll } = tracker.read($('track-mirror').checked);
  // when the face is lost, ease back to rest instead of freezing
  if (!found) { for (const k in shapes) shapes[k] = 0; }
  const hp = smooth('pitch', found ? pitch : 0, dt);
  const hy = smooth('yaw', found ? yaw : 0, dt);
  const hr = smooth('roll', found ? roll : 0, dt);
  if ($('track-head').checked) {
    bone('neck')?.rotation.set(posture.neck + hp * 0.4, hy * 0.4, hr * 0.4);
    bone('head')?.rotation.set(posture.head + hp * 0.6, hy * 0.6, hr * 0.6);
  }
  if (sync?.count >= PERFECT_SYNC_MIN) {
    const s = {};
    for (const [k, v] of Object.entries(boost(shapes, +$('track-gain').value))) s[k] = smooth(k, v, dt);
    applyPerfectSync(vrm, sync, s);
    // the presets stay at rest so the automatic blink or a button's expression doesn't add on top
    return { presets: Object.fromEntries(['blink', 'blinkLeft', 'blinkRight', ...MOUTH, 'happy', 'angry', 'sad', 'surprised', 'relaxed'].map((k) => [k, 0])) };
  }
  const p = toPresets(shapes);
  const presets = {};
  for (const k of ['blinkLeft', 'blinkRight', ...MOUTH, 'happy', 'angry', 'sad', 'surprised']) presets[k] = smooth(k, p[k], dt);
  // the voice still opens the mouth when it says more than the camera saw
  for (const m of MOUTH) presets[m] = Math.max(presets[m], mouthTarget[m]);
  presets.blink = 0;
  presets.relaxed = 0;
  if (vrm.lookAt && $('track-gaze').checked) {
    vrm.lookAt.target = null;
    vrm.lookAt.yaw = smooth('gazeYaw', p.gazeYaw, dt);
    vrm.lookAt.pitch = smooth('gazePitch', p.gazePitch, dt);
  }
  return { presets };
}

// ---- loop ----
const clock = new THREE.Clock();
function approach(name, target, dt, speed) {
  const cur = current[name] ?? 0;
  const next = cur + (target - cur) * Math.min(1, dt * speed);
  current[name] = next;
  return next;
}
function tick() {
  const dt = Math.min(clock.getDelta(), 0.1);
  if (!lockCamera) controls.update();
  if (vrm) {
    motion.update(dt);
    idle(dt);
    const em = vrm.expressionManager;
    const level = analyseVoice();
    const face = trackFace(dt);
    if (em && face) {
      if (face.presets) for (const [name, v] of Object.entries(face.presets)) if (em.getExpression(name)) em.setValue(name, v);
    } else if (em) {
      for (const [name] of EMOTIONS) {
        if (name === 'neutral' || !em.getExpression(name)) continue;
        const w = name === emotion ? emotionWeight : 0;
        em.setValue(name, approach(name, w, dt, 10));
      }
      // a wink, or eyes held mostly shut, pauses the automatic blink; a small base value (lids lowered) does not
      const winking = (manual.blinkLeft || 0) > 0 || (manual.blinkRight || 0) > 0 || (manual.blink || 0) >= 0.5;
      if (emotion !== 'blink') em.setValue('blink', winking ? 0 : blinkWeight(dt));
      for (const m of MOUTH) if (em.getExpression(m)) em.setValue(m, approach(m, mouthTarget[m], dt, 22));
    }
    if (em) {
      for (const [name, v] of Object.entries(manual)) {
        if (v > 0 && em.getExpression(name)) em.setValue(name, Math.min(1, Math.max(em.getValue(name) ?? 0, v)));
      }
    }
    $('meter-bar').style.width = `${Math.round(level * 100)}%`;
    const top = MOUTH.reduce((a, b) => (mouthTarget[a] >= mouthTarget[b] ? a : b));
    document.querySelectorAll('#vowels span').forEach((s) => s.classList.toggle('on', level > 0.05 && s.dataset.v === top));
    morph.update(dt);
    if (morph.ready) {
      $('morph-progress').value = morph.value;
      $('morph-go').textContent = morph.target > 0.5 ? '元に戻る' : '変身する';
    }
    vrm.update(dt);
    morph.updateSprings(dt);
  }
  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}

// ---- hair as 3D Gaussians (PLY in the model's rest space), carried by the head bone ----
let hair = null;
async function loadHair(url) {
  if (hair) {
    hair.holder.parent?.remove(hair.holder);
    hair.viewer.dispose?.();
    hair = null;
  }
  const head = vrm.humanoid.getRawBoneNode('head');
  vrm.scene.updateMatrixWorld(true);
  const holder = new THREE.Group();
  holder.matrixAutoUpdate = false;
  holder.matrix.copy(head.matrixWorld).invert();   // model space at rest -> head-local
  head.add(holder);
  const viewer = new Splats.DropInViewer({ sharedMemoryForWorkers: false, dynamicScene: true, sphericalHarmonicsDegree: 2 });
  await viewer.addSplatScene(url, { format: Splats.SceneFormat.Ply, showLoadingUI: false, splatAlphaRemovalThreshold: 5 });
  holder.add(viewer);
  hair = { holder, viewer };
  holder.visible = $('hair-splats').checked;
}
$('hair-splats').addEventListener('change', (e) => { if (hair) hair.holder.visible = e.target.checked; });

// ---- mesh hair look (hairfx.js): ?hairfx=0 starts with both off ----
let hairfx = [];   // the shown model's hair, plus the ?morph= model's hair once it is swapped in
if (params.get('hairfx') === '0') $('hair-shader').checked = $('hair-cards').checked = false;
function applyHairFX() {
  for (const fx of hairfx) {
    fx.setShader($('hair-shader').checked);
    fx.setCards($('hair-cards').checked);
  }
}
// morph.js swaps meshes it cannot morph; a hair mesh brings its cards and both of its materials along
function hairCompanions(mesh, incoming) {
  let fx = hairfx.find((f) => f.mesh === mesh);
  if (!fx && incoming && findHairMesh(mesh) === mesh) {
    fx = createHairFX(mesh);
    if (fx) { hairfx.push(fx); applyHairFX(); }
  }
  return fx ? { meshes: [fx.cards], materials: [fx.orig, fx.shell] } : null;
}
$('hair-shader').addEventListener('change', applyHairFX);
$('hair-cards').addEventListener('change', applyHairFX);

// ---- shape-shift into ?morph= ----
const morph = createMorph(loader, renderer);
$('morph-go').addEventListener('click', () => morph.toggle());
$('morph-progress').addEventListener('input', (e) => morph.jump(+e.target.value));
$('morph-duration').addEventListener('input', (e) => {
  morph.setDuration(+e.target.value);
  $('morph-duration-out').textContent = (+e.target.value).toFixed(1);
});
$('morph-wobble').addEventListener('input', (e) => {
  morph.setWobble(+e.target.value);
  $('morph-wobble-out').textContent = (+e.target.value).toFixed(1);
});

// ---- look (skin / eye shading) ----
function applyLook() {
  if (vrm) look.apply(vrm.scene, { skin: $('look-skin').checked, eyes: $('look-eyes').checked });
}
$('look-skin').addEventListener('change', applyLook);
$('look-eyes').addEventListener('change', applyLook);

// ---- lighting controls ----
$('exposure').addEventListener('input', (e) => {
  renderer.toneMappingExposure = +e.target.value;
  $('exposure-out').textContent = (+e.target.value).toFixed(2);
});
$('env').addEventListener('input', (e) => {
  scene.environmentIntensity = +e.target.value;
  $('env-out').textContent = (+e.target.value).toFixed(2);
});

// ---- studio look (?studio=1): the Blender check renders' lighting (tripo/outfit/render_outfit.py) ----
// a light grey world that also lights the scene from every side, one sun from the upper left front, AgX
const STUDIO_GREY = new THREE.Color().setRGB(0.82, 0.83, 0.86, THREE.LinearSRGBColorSpace);
const studioEnv = (() => {
  const s = new THREE.Scene();
  s.background = STUDIO_GREY;
  return pmrem.fromScene(s, 0).texture;
})();
const studioSun = new THREE.DirectionalLight(0xffffff, 3);
studioSun.position.set(-0.383, 0.643, 0.663).multiplyScalar(3);   // Blender sun (50°, 0, -30°) in three's Y-up frame
studioSun.target.position.set(0, 0.8, 0);
studioSun.castShadow = true;   // the sun's shadows (under the bust, the collar on the neck) give the renders their depth
studioSun.shadow.mapSize.set(2048, 2048);
Object.assign(studioSun.shadow.camera, { left: -1, right: 1, top: 1, bottom: -1, near: 0.5, far: 6 });
studioSun.shadow.bias = -0.0003;
studioSun.shadow.normalBias = 0.01;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
const normal = {
  background: scene.background, environment: scene.environment, toneMapping: renderer.toneMapping,
  lights: [key, rim, ...scene.children.filter((o) => o.isHemisphereLight)],
};
function applyStudio(on) {
  scene.background = on ? new THREE.Color('#bdbec3') : normal.background;   // the grey as the renders show it
  scene.environment = on ? studioEnv : normal.environment;
  scene.environmentIntensity = on ? 0.75 : +$('env').value;   // no ambient occlusion here, so a little less than Blender's 0.9
  renderer.toneMapping = on ? THREE.AgXToneMapping : normal.toneMapping;
  for (const l of normal.lights) l.visible = !on;
  if (on) scene.add(studioSun, studioSun.target);
  else scene.remove(studioSun, studioSun.target);
  renderer.shadowMap.enabled = on;
  vrm?.scene.traverse((o) => {
    if (o.isMesh) o.castShadow = o.receiveShadow = on;
  });
  scene.traverse((o) => {
    if (o.material) for (const m of [o.material].flat()) m.needsUpdate = true;   // shadow on/off recompiles
  });
}
// studio by default; ?light=warm (or the old ?studio=0) for the warm key and rim lights
let studio = !(params.get('light') === 'warm' || params.get('studio') === '0');
function setLight(name) {
  studio = name === 'studio';
  document.querySelectorAll('[data-light]').forEach((b) => b.classList.toggle('on', b.dataset.light === name));
  applyStudio(studio);
}
document.querySelectorAll('[data-light]').forEach((b) => b.addEventListener('click', () => setLight(b.dataset.light)));
setLight(studio ? 'studio' : 'warm');

// ---- body motion: still (breathing only) or walking in place ----
const motion = createMotion();
function setMotion(name) {
  document.querySelectorAll('[data-motion]').forEach((b) => b.classList.toggle('on', b.dataset.motion === name));
  motion.play(name);
  if (vrm && name === 'none') {
    vrm.humanoid.resetNormalizedPose();
    relaxPose();
  }
  if (vrm && name === 'walk') setView('full');   // the legs are the point
}
document.querySelectorAll('[data-motion]').forEach((b) => b.addEventListener('click', () => setMotion(b.dataset.motion)));
setMotion(MOTION);

// ---- model file / drop ----
function openFile(f) {
  if (f) loadModel(URL.createObjectURL(f), f.name).catch(showError);
}
$('open-model').addEventListener('click', () => $('model-input').click());
$('model-input').addEventListener('change', (e) => openFile(e.target.files[0]));
addEventListener('dragover', (e) => e.preventDefault());
addEventListener('drop', (e) => {
  e.preventDefault();
  const f = [...e.dataTransfer.files].find((x) => /\.(vrm|glb)$/i.test(x.name));
  openFile(f);
});

function showError(err) {
  console.error(err);
  statusEl.textContent = `エラー: ${err.message || err}`;
}

// for automated checks
window.viewer = {
  setExpression, setView, setManual,
  orbit(yaw) {
    const t = controls.target, d = camera.position.distanceTo(t);
    camera.position.set(t.x + Math.sin(yaw) * d, camera.position.y, t.z + Math.cos(yaw) * d);
    controls.update();
  },
  lens(fov) {   // change the field of view, keeping the framing (distance scales with 1/tan)
    const t = controls.target, dir = camera.position.clone().sub(t);
    const k = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) / Math.tan(THREE.MathUtils.degToRad(fov / 2));
    camera.fov = fov; camera.updateProjectionMatrix();
    camera.position.copy(t).add(dir.multiplyScalar(k)); controls.update();
  },
  setCamera({ matrix, fov, offset }) {   // world matrix (column-major 16), vertical fov in degrees, [dx, dy] pixels of a size-px image
    lockCamera = true;
    camera.matrixAutoUpdate = false;
    camera.matrix.fromArray(matrix);
    camera.matrix.decompose(camera.position, camera.quaternion, camera.scale);
    camera.matrixAutoUpdate = true;
    camera.fov = fov;
    if (offset) camera.setViewOffset(offset[2], offset[2], -offset[0], -offset[1], offset[2], offset[2]);
    camera.updateProjectionMatrix();
  },
  setLook(skin, eyes) { $('look-skin').checked = skin; $('look-eyes').checked = eyes; applyLook(); },
  setHair(shader, cards) { $('hair-shader').checked = shader; $('hair-cards').checked = cards; applyHairFX(); },
  get hairfx() { return hairfx[0] || null; },
  morph, motion, setMotion, setLight,
  get posture() { return { ...posture }; },
  get vrm() { return vrm; },
  tracker,
  get tracked() { return { ...tracked }; },
  get sync() { return sync && { count: sync.count }; },
  get mouth() { return { ...mouthTarget }; },
  get current() { return { ...current }; },
};

resize();
tick();
loadModel(MODEL_URL).catch(showError);
