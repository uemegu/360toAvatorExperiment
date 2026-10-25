// Webcam face tracking with MediaPipe Face Landmarker: 52 ARKit-style blendshapes and the head pose.
// A model with ARKit-named expressions or morphs gets all 52 as they are (perfect sync); any other VRM
// gets them folded into its presets (blink, aa..oh, happy, ...), the head/neck bones and the eye gaze.
import * as THREE from 'three';
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

const WASM = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm';
const MODEL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

export const ARKIT = [
  'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight',
  'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight',
  'eyeBlinkLeft', 'eyeBlinkRight', 'eyeLookDownLeft', 'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight',
  'eyeLookOutLeft', 'eyeLookOutRight', 'eyeLookUpLeft', 'eyeLookUpRight', 'eyeSquintLeft', 'eyeSquintRight',
  'eyeWideLeft', 'eyeWideRight',
  'jawForward', 'jawLeft', 'jawOpen', 'jawRight',
  'mouthClose', 'mouthDimpleLeft', 'mouthDimpleRight', 'mouthFrownLeft', 'mouthFrownRight', 'mouthFunnel',
  'mouthLeft', 'mouthLowerDownLeft', 'mouthLowerDownRight', 'mouthPressLeft', 'mouthPressRight', 'mouthPucker',
  'mouthRight', 'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper',
  'mouthSmileLeft', 'mouthSmileRight', 'mouthStretchLeft', 'mouthStretchRight',
  'mouthUpperUpLeft', 'mouthUpperUpRight', 'noseSneerLeft', 'noseSneerRight', 'tongueOut',
];
const flip = (name) => name.replace(/Left|Right/, (s) => (s === 'Left' ? 'Right' : 'Left'));

export function createFaceTracker() {
  let landmarker = null;
  let stream = null;
  let video = null;
  let lastTime = -1;
  const shapes = Object.fromEntries(ARKIT.map((n) => [n, 0]));   // latest raw scores (MediaPipe's naming)
  const head = new THREE.Euler(0, 0, 0, 'YXZ');                  // latest raw head rotation (camera space)
  const zero = new THREE.Quaternion();                            // "facing front" offset, set by calibrate()
  const base = Object.fromEntries(ARKIT.map((n) => [n, 0]));     // the resting face, set by calibrate()
  const q = new THREE.Quaternion();
  const m = new THREE.Matrix4();
  let found = false;

  async function load() {
    if (landmarker) return;
    const files = await FilesetResolver.forVisionTasks(WASM);
    const options = {
      runningMode: 'VIDEO', numFaces: 1,
      outputFaceBlendshapes: true, outputFacialTransformationMatrixes: true,
    };
    try {
      landmarker = await FaceLandmarker.createFromOptions(files, { ...options, baseOptions: { modelAssetPath: MODEL, delegate: 'GPU' } });
    } catch {
      landmarker = await FaceLandmarker.createFromOptions(files, { ...options, baseOptions: { modelAssetPath: MODEL, delegate: 'CPU' } });
    }
  }

  async function start(videoEl) {
    video = videoEl;
    await load();
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' }, audio: false });
    video.srcObject = stream;
    await video.play();
    lastTime = -1;
  }

  function stop() {
    stream?.getTracks().forEach((t) => t.stop());
    stream = null;
    if (video) video.srcObject = null;
    found = false;
  }

  // Run detection on a new video frame. Returns true while a face is seen.
  function update() {
    if (!stream || !landmarker || video.readyState < 2 || video.currentTime === lastTime) return found;
    lastTime = video.currentTime;
    const r = landmarker.detectForVideo(video, performance.now());
    found = !!r.faceBlendshapes?.length;
    if (!found) return false;
    for (const c of r.faceBlendshapes[0].categories) if (c.categoryName in shapes) shapes[c.categoryName] = c.score;
    m.fromArray(r.facialTransformationMatrixes[0].data);
    q.setFromRotationMatrix(m).premultiply(zero);
    head.setFromQuaternion(q, 'YXZ');
    return true;
  }

  // Take the current head direction as "facing front" and the current face as neutral:
  // what reads as a slight pucker or a raised brow on this face at rest is subtracted from then on.
  function calibrate() {
    q.setFromRotationMatrix(m);
    zero.copy(q).invert();
    for (const n of ARKIT) base[n] = Math.min(shapes[n], 0.8);
  }

  // Scores and head rotation in the avatar's terms. MediaPipe names sides as the subject's own
  // (eyeBlinkLeft is the subject's left eye); `mirror` makes the avatar move like a reflection instead.
  function read(mirror) {
    const out = {};
    for (const n of ARKIT) out[mirror ? flip(n) : n] = Math.min(1, Math.max(0, (shapes[n] - base[n]) / (1 - base[n])));
    // the matrix is in the camera's frame (looking down -z): turning to the subject's left is +y
    const yaw = mirror ? -head.y : head.y;
    const roll = mirror ? -head.z : head.z;
    return { shapes: out, pitch: head.x, yaw, roll };
  }

  return {
    start, stop, update, calibrate, read,
    get running() { return !!stream; },
    get found() { return found; },
  };
}

// ---- driving the VRM ----

const clamp = (v) => Math.min(1, Math.max(0, v));
const lower = (s) => s.toLowerCase();

// Find what in the model answers to ARKit names: custom/preset expressions first, then raw morph targets.
export function findPerfectSync(vrm) {
  const want = new Map(ARKIT.map((n) => [lower(n), n]));
  const exprs = new Map();
  for (const e of vrm.expressionManager?.expressions || []) {
    const n = want.get(lower(e.expressionName));
    if (n) exprs.set(n, e.expressionName);
  }
  const morphs = new Map();
  vrm.scene.traverse((o) => {
    if (!o.isMesh || !o.morphTargetDictionary) return;
    for (const [key, i] of Object.entries(o.morphTargetDictionary)) {
      const n = want.get(lower(key.replace(/^.*[.:]/, '')));   // "blendShape1.eyeBlinkLeft" etc.
      if (!n || exprs.has(n)) continue;
      if (!morphs.has(n)) morphs.set(n, []);
      morphs.get(n).push([o, i]);
    }
  });
  const count = new Set([...exprs.keys(), ...morphs.keys()]).size;
  return { exprs, morphs, count };
}

export function applyPerfectSync(vrm, sync, shapes) {
  const em = vrm.expressionManager;
  for (const [n, name] of sync.exprs) em.setValue(name, shapes[n]);
  for (const [n, list] of sync.morphs) for (const [mesh, i] of list) mesh.morphTargetInfluences[i] = shapes[n];
}

// MediaPipe reads conservatively: shut eyes give about 0.5-0.7, a broad smile about half.
// Blinks are stretched so they close fully; everything but the gaze is scaled by `gain`.
export function boost(s, gain) {
  const out = {};
  for (const [k, v] of Object.entries(s)) {
    if (k.startsWith('eyeBlink')) out[k] = clamp((v - 0.1) / 0.45);
    else if (k.startsWith('eyeLook')) out[k] = v;
    else out[k] = clamp(v * gain);
  }
  return out;
}

// Fold the 52 scores into the VRM presets. Rough on purpose: presets mix many muscles into one shape.
export function toPresets(s) {
  const avg = (a, b) => (s[a] + s[b]) / 2;
  const jaw = clamp((s.jawOpen - 0.04) * 1.5);
  const pucker = clamp(s.mouthPucker * 1.3);
  const funnel = clamp(s.mouthFunnel * 1.5);
  const stretch = clamp(avg('mouthStretchLeft', 'mouthStretchRight') * 2 + avg('mouthSmileLeft', 'mouthSmileRight') * 0.6);
  const round = Math.max(pucker, funnel);
  const p = {
    blinkLeft: clamp((s.eyeBlinkLeft - 0.15) / 0.55),
    blinkRight: clamp((s.eyeBlinkRight - 0.15) / 0.55),
    ou: pucker * (1 - funnel * 0.5) * Math.max(0.35, 1 - jaw),
    oh: funnel * clamp(jaw * 2.5 + 0.3),
    aa: jaw * (1 - round) * (1 - stretch * 0.5),
    ee: jaw * stretch * (1 - round) * 0.9,
    ih: stretch * (1 - jaw) * (1 - round) * 0.8,
    happy: clamp((avg('mouthSmileLeft', 'mouthSmileRight') - 0.1) * 1.4),
    angry: clamp((avg('browDownLeft', 'browDownRight') - 0.15) * 1.6),
    surprised: clamp((s.browInnerUp - 0.25) * 1.2) * clamp(avg('eyeWideLeft', 'eyeWideRight') * 3 + jaw),
    sad: clamp((avg('mouthFrownLeft', 'mouthFrownRight') - 0.05) * 2) * clamp(s.browInnerUp * 2),
  };
  // the "happy" preset usually narrows the eyes too, so a smile shouldn't add a blink on top
  p.blinkLeft *= 1 - p.happy * 0.6;
  p.blinkRight *= 1 - p.happy * 0.6;
  // eye direction in degrees (avatar's own yaw: + is to its left)
  const yawL = s.eyeLookOutLeft - s.eyeLookInLeft;   // left eye looking out = to the avatar's left
  const yawR = s.eyeLookInRight - s.eyeLookOutRight;
  p.gazeYaw = ((yawL + yawR) / 2) * 30;
  p.gazePitch = (avg('eyeLookUpLeft', 'eyeLookUpRight') - avg('eyeLookDownLeft', 'eyeLookDownRight')) * 25;
  return p;
}
