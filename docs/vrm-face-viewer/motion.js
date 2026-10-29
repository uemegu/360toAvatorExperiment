// Body motions baked into motion-data.js (bake-motions.mjs: Mixamo clips retargeted onto VRM normalized bones,
// so no FBX ships with the page). Walking is played in place: the hips keep only their bob.
import * as THREE from 'three';
import { MOTION_DATA } from './motion-data.js?v=1';

function decode(b64) {
  const bin = atob(b64), u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Int16Array(u8.buffer);
}

function buildClip(name, vrm) {
  const m = MOTION_DATA[name];
  const times = Float32Array.from({ length: m.n }, (_, i) => m.start + i * m.dt);
  const hipsH = vrm.humanoid.normalizedRestPose.hips.position[1];
  const v0 = vrm.meta?.metaVersion === '0';
  const tracks = [];
  for (const [bone, kind, data] of m.tracks) {
    const node = vrm.humanoid.getNormalizedBoneNode(bone);
    if (!node) continue;
    const raw = decode(data);
    const stride = kind === 'q' ? 4 : 3;
    const t = raw.length === stride ? times.subarray(0, 1) : times;   // one key: the bone holds still
    if (kind === 'q') {
      const values = Float32Array.from(raw, (v, i) => (v0 && i % 2 === 0 ? -v : v) / 32767);
      tracks.push(new THREE.QuaternionKeyframeTrack(`${node.name}.quaternion`, t, values));
    } else {
      const values = Float32Array.from(raw, (v, i) => (v0 && i % 3 !== 1 ? -v : v) / 8192 * hipsH);
      if (name === 'walk') {   // keep the hips over the feet: drop the travel on x/z, keep the up-down bob
        for (let i = 0; i < values.length; i += 3) { values[i] = values[0]; values[i + 2] = values[2]; }
      }
      tracks.push(new THREE.VectorKeyframeTrack(`${node.name}.position`, t, values));
    }
  }
  return new THREE.AnimationClip(name, m.duration, tracks);
}

export function createMotion() {
  let mixer = null, action = null, vrm = null, name = 'none';
  return {
    get name() { return name; },
    get playing() { return !!action; },
    attach(v) {   // a new model: drop the old mixer, keep the chosen motion
      mixer?.stopAllAction();
      vrm = v;
      mixer = new THREE.AnimationMixer(v.scene);
      action = null;
      this.play(name);
    },
    play(next) {
      name = next;
      if (!vrm) return;   // played once a model is attached
      if (next === 'none') {   // the caller puts the bones back (the mixer leaves the last frame's pose)
        mixer.stopAllAction();
        action = null;
        return;
      }
      const a = mixer.clipAction(buildClip(next, vrm));
      a.reset().fadeIn(0.3).play();
      if (action) action.fadeOut(0.3);
      action = a;
    },
    update(dt) {
      mixer?.update(dt);
    },
  };
}
