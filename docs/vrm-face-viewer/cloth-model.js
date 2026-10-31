// The cloth's data, shared by the GPU solver (cloth.js) and the CPU one (cloth-cpu.js), so both run the same steps
// on the same points and constraints. Above the waist the dress follows the bones. Below it the dress is a
// position-based cloth: Verlet with gravity, edge and bend lengths kept by Jacobi iterations, a weak pull back to the
// skinned shape (the draped dress is the rest shape, so it does not collapse), and tapered capsules on the legs that
// push it out. Skinning is done by the solvers too (this frame's bone matrices), so the cloth sees this frame's pose.
import * as THREE from 'three';

export function buildClothModel(mesh, opts) {
  const o = {
    top: 0.95, bottom: 0.82,   // world y: fully pinned above top, free below bottom
    substeps: 4, iterations: 8,
    damping: 0.995, gravity: -9.8,
    compress: 1, bend: 1,   // stiffness against squashing an edge, and against folding (structural stretch is 1);
    // softer than this, the tiers of the ruffled hem flatten and a flounce flaps out
    tetherHem: 0.01, tetherTop: 0.08,   // per-substep pull to the skinned shape, at the hem and just under the waist
    margin: 0.008,   // clearance outside each capsule; a capsule's own `margin` overrides it
    tackRadius: 0.02, tacks: 4, tack: 0.5,   // ties between the ruffle layers (see below)
    capsules: [],   // [{ a: Object3D, b: Object3D, ra, rb, margin, restAllowance }]
    ...opts,
  };
  mesh.updateMatrixWorld(true);
  const g = mesh.geometry;
  const P = g.attributes.position, SI = g.attributes.skinIndex, SW = g.attributes.skinWeight;
  const nV = P.count;

  // ---- weld the render vertices (UV seams) into simulation points ----
  const map = new Uint32Array(nV), first = [];
  const keyToId = new Map();
  for (let i = 0; i < nV; i++) {
    const k = `${P.getX(i).toFixed(5)},${P.getY(i).toFixed(5)},${P.getZ(i).toFixed(5)}`;
    let id = keyToId.get(k);
    if (id === undefined) { id = first.length; keyToId.set(k, id); first.push(i); }
    map[i] = id;
  }
  const U = first.length;
  const v = new THREE.Vector3();
  const restBind = new Float32Array(U * 4), restWorld = new Float32Array(U * 3);
  const joints = new Uint32Array(U * 4), weights = new Float32Array(U * 4);
  for (let u = 0; u < U; u++) {
    const i = first[u];
    v.fromBufferAttribute(P, i).applyMatrix4(mesh.bindMatrix);
    restBind.set([v.x, v.y, v.z, 1], u * 4);
    v.fromBufferAttribute(P, i).applyMatrix4(mesh.matrixWorld);
    restWorld.set([v.x, v.y, v.z], u * 3);
    for (let k = 0; k < 4; k++) { joints[u * 4 + k] = SI.getComponent(i, k); weights[u * 4 + k] = SW.getComponent(i, k); }
  }

  // ---- pin and tether weights from the rest height ----
  let minY = Infinity;
  for (let u = 0; u < U; u++) minY = Math.min(minY, restWorld[u * 3 + 1]);
  const ss = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
  const meta = new Float32Array(U * 4);   // pin, tether height factor
  for (let u = 0; u < U; u++) {
    const y = restWorld[u * 3 + 1];
    meta[u * 4] = ss(o.bottom, o.top, y);
    meta[u * 4 + 1] = ss(minY, o.top, y) ** 2;
  }

  // ---- constraints: triangle edges (structural) and the pair across each edge (bend) ----
  const idx = g.index.array, tris = [];
  for (let t = 0; t < idx.length; t += 3) {
    const a = map[idx[t]], b = map[idx[t + 1]], c = map[idx[t + 2]];
    if (a !== b && b !== c && a !== c) tris.push(a, b, c);
  }
  const edgeOpp = new Map();
  const ek = (a, b) => (a < b ? a * U + b : b * U + a);
  for (let t = 0; t < tris.length; t += 3) {
    for (let e = 0; e < 3; e++) {
      const a = tris[t + e], b = tris[t + (e + 1) % 3], c = tris[t + (e + 2) % 3];
      const k = ek(a, b);
      if (!edgeOpp.has(k)) edgeOpp.set(k, []);
      edgeOpp.get(k).push(c);
    }
  }
  const adj = Array.from({ length: U }, () => []);
  const dist = (a, b) => Math.hypot(restWorld[a * 3] - restWorld[b * 3], restWorld[a * 3 + 1] - restWorld[b * 3 + 1],
    restWorld[a * 3 + 2] - restWorld[b * 3 + 2]);
  let nBend = 0;
  for (const [k, opp] of edgeOpp) {
    const a = Math.floor(k / U), b = k % U, L = dist(a, b);
    adj[a].push(b, L); adj[b].push(a, L);
    if (opp.length === 2 && opp[0] !== opp[1]) {
      const [c, d] = opp, Lb = -Math.max(dist(c, d), 1e-5);   // negative: a bend pair
      adj[c].push(d, Lb); adj[d].push(c, Lb);
      nBend++;
    }
  }
  // tacks between layers: the skirt is tiers of ruffles and wrap flaps lying on one another, joined only at the top.
  // Skinned they move together; as cloth each flap swings alone, they part and the legs show between them. Each
  // point is tied to its nearest points (within TACK_R at rest) that the mesh does not already connect to it.
  // Encoded as rest length + 10.
  const TACK_R = o.tackRadius, cell = new Map(), ck = (x, y, z) => `${Math.floor(x / TACK_R)},${Math.floor(y / TACK_R)},${Math.floor(z / TACK_R)}`;
  for (let u = 0; u < U; u++) {
    const k = ck(restWorld[u * 3], restWorld[u * 3 + 1], restWorld[u * 3 + 2]);
    if (!cell.has(k)) cell.set(k, []);
    cell.get(k).push(u);
  }
  let nTack = 0;
  const tackDeg = new Uint8Array(U);   // tacks per point, capped at both ends (a point could collect 26 otherwise)
  for (let u = 0; u < U; u++) {
    if (meta[u * 4] >= 1) continue;
    const linked = new Set(adj[u].filter((_, k) => k % 2 === 0));
    const [x, y, z] = [restWorld[u * 3], restWorld[u * 3 + 1], restWorld[u * 3 + 2]];
    const cand = [];
    for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) for (let dz = -1; dz <= 1; dz++) {
      for (const w of cell.get(ck(x + dx * TACK_R, y + dy * TACK_R, z + dz * TACK_R)) ?? []) {
        if (w <= u || linked.has(w)) continue;
        const d = dist(u, w);
        if (d < TACK_R) cand.push([d, w]);
      }
    }
    cand.sort((a, b) => a[0] - b[0]);
    for (const [d, w] of cand) {
      if (tackDeg[u] >= o.tacks) break;
      if (tackDeg[w] >= o.tacks) continue;
      adj[u].push(w, d + 10); adj[w].push(u, d + 10);
      tackDeg[u]++; tackDeg[w]++; nTack++;
    }
  }
  const nbrPtr = new Uint32Array(U * 2);
  let total = 0;
  for (let u = 0; u < U; u++) { nbrPtr[u * 2] = total; nbrPtr[u * 2 + 1] = adj[u].length / 2; total += adj[u].length / 2; }
  const nbrJ = new Uint32Array(Math.max(1, total)), nbrL = new Float32Array(Math.max(1, total));
  for (let u = 0, n = 0; u < U; u++) for (let k = 0; k < adj[u].length; k += 2, n++) { nbrJ[n] = adj[u][k]; nbrL[n] = adj[u][k + 1]; }

  // faces around each point, for normals
  const vf = Array.from({ length: U }, () => []);
  for (let t = 0; t < tris.length / 3; t++) for (let e = 0; e < 3; e++) vf[tris[t * 3 + e]].push(t);
  const facePtr = new Uint32Array(U * 2), faceList = [];
  for (let u = 0; u < U; u++) { facePtr[u * 2] = faceList.length; facePtr[u * 2 + 1] = vf[u].length; faceList.push(...vf[u]); }
  const faces = new Uint32Array(tris.length / 3 * 4);
  for (let t = 0; t < tris.length / 3; t++) faces.set([tris[t * 3], tris[t * 3 + 1], tris[t * 3 + 2], 0], t * 4);

  // ---- how deep each point already sits in the legs at rest ----
  // The rest dress is not clear of the capsules everywhere (the buttock capsules stood out through its back, the
  // flaps of the slit lie inside the thigh). Pushed out to the full radius, the skirt stood out round the hips like a
  // box, and the springs pulled it back every frame (the hem shook). Each point is allowed its rest depth in each
  // capsule (allow[point * nCaps + capsule]): only a leg moving further in than at rest pushes it. Not for capsules with restAllowance: false (the shins: the
  // hem rests close to the calves, and the allowance let a calf kicking back through the centre-back hem again).
  const pa0 = new THREE.Vector3(), pb0 = new THREE.Vector3(), q0 = new THREE.Vector3(), ab0 = new THREE.Vector3();
  o.capsules.forEach((c) => { c.margin ??= o.margin; });
  const allow = new Float32Array(Math.max(1, U * o.capsules.length));
  o.restInside = o.capsules.map((c, ci) => {
    c.a.getWorldPosition(pa0); c.b.getWorldPosition(pb0);
    ab0.subVectors(pb0, pa0);
    const L2 = Math.max(ab0.lengthSq(), 1e-8);
    let n = 0;
    for (let u = 0; u < U; u++) {
      v.set(restWorld[u * 3], restWorld[u * 3 + 1], restWorld[u * 3 + 2]);
      const h = Math.min(1, Math.max(0, v.clone().sub(pa0).dot(ab0) / L2));
      q0.copy(pa0).addScaledVector(ab0, h);
      const depth = c.ra + (c.rb - c.ra) * h + c.margin - v.distanceTo(q0);
      if (depth > 0 && meta[u * 4] < 1) n++;
      if (c.restAllowance !== false) allow[u * o.capsules.length + ci] = Math.max(0, depth);
    }
    return n;
  });

  return {
    o, U, map, restBind, restWorld, joints, weights, meta, allow, tris, faces, facePtr, faceList,
    nbrPtr, nbrJ, nbrL, edges: edgeOpp.size, bends: nBend, tacks: nTack, nBones: mesh.skeleton.bones.length,
  };
}

// This frame's inputs: bone matrices taken to world space (matrixWorld × bindMatrixInverse × boneMatrix, 16 floats
// per bone) and the capsules, last frame's ends then this frame's (x, y, z, radius + margin; 16 floats per capsule).
export function createFrameInputs(mesh, model) {
  const bones = new Float32Array(model.nBones * 16), caps = new Float32Array(Math.max(1, model.o.capsules.length) * 16);
  const mWorldBindInv = new THREE.Matrix4(), tmp = new THREE.Matrix4(), pa = new THREE.Vector3(), pb = new THREE.Vector3();
  const invWorld = new THREE.Matrix4();
  return {
    bones, caps, invWorld,
    update(first) {
      mesh.updateMatrixWorld(true);
      mesh.skeleton.update();
      mWorldBindInv.multiplyMatrices(mesh.matrixWorld, mesh.bindMatrixInverse);
      const bm = mesh.skeleton.boneMatrices;
      for (let b = 0; b < model.nBones; b++) {
        tmp.fromArray(bm, b * 16).premultiply(mWorldBindInv);
        bones.set(tmp.elements, b * 16);
      }
      model.o.capsules.forEach((c, n) => {
        c.a.getWorldPosition(pa); c.b.getWorldPosition(pb);
        caps.copyWithin(n * 16, n * 16 + 8, n * 16 + 16);   // this frame's ends become last frame's
        caps.set([pa.x, pa.y, pa.z, c.ra + c.margin, pb.x, pb.y, pb.z, c.rb + c.margin], n * 16 + 8);
        if (first) caps.copyWithin(n * 16, n * 16 + 8, n * 16 + 16);
      });
      invWorld.copy(mesh.matrixWorld).invert();
    },
  };
}
