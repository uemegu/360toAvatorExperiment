// What the skirt cloth needs from a VRM, shared by the viewer (viewer.js, CPU solver) and the test page
// (cloth-page.js): the dress mesh, the leg capsules fitted to the body's skin, and the height range where the dress
// turns from following the bones to cloth. Call with the model in its rest pose (legs as bound).
import * as THREE from 'three';

export function findDress(vrm) {
  let dress = null;
  vrm.scene.traverse((o) => { if (o.isSkinnedMesh && /dress/i.test(o.name + (o.material?.name ?? ''))) dress = o; });
  return dress;
}

export function rigDress(vrm) {
  const dress = findDress(vrm);
  if (!dress) return null;
  vrm.scene.updateMatrixWorld(true);
  const raw = (n) => vrm.humanoid.getRawBoneNode(n);
  // a point carried by a bone, placed at a rest-pose world position
  function anchor(boneName, x, y, z) {
    const o = new THREE.Object3D();
    raw(boneName).add(o);
    o.position.copy(raw(boneName).worldToLocal(new THREE.Vector3(x, y, z)));
    o.updateMatrixWorld(true);
    return o;
  }
  // the body: the largest skinned mesh that is not the dress
  let skin = null;
  vrm.scene.traverse((o) => {
    if (o.isSkinnedMesh && o !== dress && (!skin || o.geometry.attributes.position.count > skin.geometry.attributes.position.count)) skin = o;
  });
  // a capsule fitted to the skin a leg bone mainly drives: the axis through the centroids of the skin at 20% and 80%
  // down the bone, extended to the joints. Hand-set radii around the bone missed the calf, which bulges 6cm behind
  // the shin bone near the front of the leg.
  function fitLeg(boneName, childName) {
    const bi = skin.skeleton.bones.indexOf(raw(boneName));
    const P = skin.geometry.attributes.position, SI = skin.geometry.attributes.skinIndex, SW = skin.geometry.attributes.skinWeight;
    const p0 = raw(boneName).getWorldPosition(new THREE.Vector3()), p1 = raw(childName).getWorldPosition(new THREE.Vector3());
    const u = p1.clone().sub(p0), len = u.length();
    u.normalize();
    const pts = [], v = new THREE.Vector3();
    for (let i = 0; i < P.count; i++) {
      let w = 0;
      for (let k = 0; k < 4; k++) if (SI.getComponent(i, k) === bi) w += SW.getComponent(i, k);
      if (w < 0.5) continue;
      v.fromBufferAttribute(P, i).applyMatrix4(skin.matrixWorld);
      pts.push([v.clone(), v.clone().sub(p0).dot(u) / len]);
    }
    const slab = (lo, hi) => {
      const sel = pts.filter(([, t]) => t > lo && t < hi).map(([q]) => q);
      return [sel.reduce((c, q) => c.add(q), new THREE.Vector3()).divideScalar(Math.max(1, sel.length)), sel];
    };
    const [c0] = slab(0.1, 0.3), [c1] = slab(0.7, 0.9);
    const axis = c1.clone().sub(c0).normalize();
    const at = (t) => c0.clone().lerp(c1, (t - 0.2) / 0.6);   // the fitted line, by fraction down the bone
    // radius along the bone: the 95% radius in nine slabs, a straight line fitted through them and raised until it
    // covers every slab (two slabs at 20%/80% put on the ends left the calf 3-4mm out)
    const rs = [];
    for (let t = 0.1; t < 0.95; t += 0.1) {
      const [, sel] = slab(t - 0.05, t + 0.05);
      if (sel.length < 8) continue;
      const d = sel.map((q) => q.clone().sub(c0).cross(axis).length()).sort((x, y) => x - y);
      rs.push([t, d[Math.floor(d.length * 0.95)]]);
    }
    const mt = rs.reduce((s, [t]) => s + t, 0) / rs.length, mr = rs.reduce((s, [, r]) => s + r, 0) / rs.length;
    const slope = rs.reduce((s, [t, r]) => s + (t - mt) * (r - mr), 0) / Math.max(1e-9, rs.reduce((s, [t]) => s + (t - mt) ** 2, 0));
    const lift = Math.max(...rs.map(([t, r]) => r - (mr + slope * (t - mt))));
    const rAt = (t) => Math.max(0.02, mr + slope * (t - mt) + lift);
    const a = at(0), b = at(1);
    return { a: anchor(boneName, a.x, a.y, a.z), b: anchor(boneName, b.x, b.y, b.z), ra: rAt(0), rb: rAt(1) };
  }
  const hipsY = raw('hips').getWorldPosition(new THREE.Vector3()).y;
  // legs: fitted thigh and shin capsules. The buttocks reach 2cm further back than the thigh, so each gets its own
  // capsule, from the pelvis down the back of the thigh.
  const capsules = [];
  for (const s of ['left', 'right']) {
    const sx = Math.sign(raw(s + 'UpperLeg').getWorldPosition(new THREE.Vector3()).x);
    capsules.push(fitLeg(s + 'UpperLeg', s + 'LowerLeg'));
    // shins: 2cm clearance (the others 8mm). The hem points are 2-3cm apart, and a calf kicking back slipped between
    // them with less.
    capsules.push({ ...fitLeg(s + 'LowerLeg', s + 'Foot'), margin: 0.02, restAllowance: false });
    capsules.push({
      a: anchor('hips', sx * 0.05, hipsY - 0.01, 0), b: anchor(s + 'UpperLeg', sx * 0.055, hipsY - 0.13, 0), ra: 0.09, rb: 0.08,
    });
  }
  // cloth below the hips: following the bones 2cm above the hips joint, free 12cm below it
  return { dress, capsules, hipsY, opts: { top: hipsY + 0.02, bottom: hipsY - 0.12, capsules } };
}
