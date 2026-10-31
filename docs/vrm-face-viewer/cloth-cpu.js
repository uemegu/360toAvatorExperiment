// The cloth of cloth.js run on the CPU: the same data (cloth-model.js) and the same steps, pass for pass, so it runs
// in the WebGL viewer and in browsers without WebGPU. Positions are kept as float32 like the GPU buffers (the
// arithmetic in between is float64, so the two drift apart by rounding only). The mesh is drawn from the points: its
// position and normal attributes are rewritten every frame and the renderer's skinning is turned off.
import * as THREE from 'three';
import { buildClothModel, createFrameInputs } from './cloth-model.js?v=1';

export function createClothCPU(mesh, opts = {}) {
  const model = buildClothModel(mesh, opts);
  const { o, U, map, restBind, joints, weights, meta, allow, faces, facePtr, faceList, nbrPtr, nbrJ, nbrL } = model;
  const frame = createFrameInputs(mesh, model);
  const nCaps = o.capsules.length;
  const u = {   // the same knobs as the GPU version's uniforms
    uTack: { value: o.tack }, uCollide: { value: 1 }, uDamp: { value: o.damping }, uGrav: { value: o.gravity },
    uCompress: { value: o.compress }, uBend: { value: o.bend }, uTetherHem: { value: o.tetherHem },
    uTetherTop: { value: o.tetherTop }, uFree: { value: 1 },
  };
  const pos = new Float32Array(U * 4), pos2 = new Float32Array(U * 4), prev = new Float32Array(U * 4);
  const skin = new Float32Array(U * 4), skinPrev = new Float32Array(U * 4), normal = new Float32Array(U * 4);
  const tgt = new Float64Array(3);
  // the constraints decoded once: rest length and kind (0 stretch, 1 bend, 2 tack) per neighbour entry
  const restL = new Float32Array(nbrL.length), kind = new Uint8Array(nbrL.length);
  for (let k = 0; k < nbrL.length; k++) {
    const L = nbrL[k];
    kind[k] = L > 5 ? 2 : L > 0 ? 0 : 1;
    restL[k] = L > 5 ? L - 10 : Math.abs(L);
  }
  const wFree = new Float32Array(U);   // 1 - effective pin, per point (set each frame)
  const len3 = (x, y, z) => Math.sqrt(x * x + y * y + z * z);   // Math.hypot is several times slower in V8
  let step = 0, sub = o.substeps, h = 1 / 240;

  function skinPass() {
    const B = frame.bones;
    for (let i = 0; i < U; i++) {
      const px = restBind[i * 4], py = restBind[i * 4 + 1], pz = restBind[i * 4 + 2];
      let sx = 0, sy = 0, sz = 0;
      for (let k = 0; k < 4; k++) {
        const w = weights[i * 4 + k], b = joints[i * 4 + k] * 16;
        sx += (B[b] * px + B[b + 4] * py + B[b + 8] * pz + B[b + 12]) * w;
        sy += (B[b + 1] * px + B[b + 5] * py + B[b + 9] * pz + B[b + 13]) * w;
        sz += (B[b + 2] * px + B[b + 6] * py + B[b + 10] * pz + B[b + 14]) * w;
      }
      skinPrev[i * 4] = skin[i * 4]; skinPrev[i * 4 + 1] = skin[i * 4 + 1]; skinPrev[i * 4 + 2] = skin[i * 4 + 2];
      skin[i * 4] = sx; skin[i * 4 + 1] = sy; skin[i * 4 + 2] = sz;
    }
    step = 0;
  }
  function initPass() {
    skinPrev.set(skin); pos.set(skin); prev.set(skin);
  }
  const alpha = () => Math.min((step + 1) / sub, 1);
  function target(i, al) {
    for (let k = 0; k < 3; k++) tgt[k] = skinPrev[i * 4 + k] + (skin[i * 4 + k] - skinPrev[i * 4 + k]) * al;
  }
  const pinOf = (i) => Math.max(meta[i * 4], 1 - u.uFree.value);

  function predictPass() {
    const al = alpha(), damp = u.uDamp.value, gy = u.uGrav.value * h * h;
    for (let i = 0; i < U; i++) {
      const j = i * 4;
      const x = pos[j], y = pos[j + 1], z = pos[j + 2];
      let vx = (x - prev[j]) * damp, vy = (y - prev[j + 1]) * damp, vz = (z - prev[j + 2]) * damp;
      const vl = len3(vx, vy, vz);
      if (vl > 0.05) { const s = 0.05 / vl; vx *= s; vy *= s; vz *= s; }
      const pin = pinOf(i);
      target(i, al);
      prev[j] = x; prev[j + 1] = y; prev[j + 2] = z;
      pos[j] = x + vx + (tgt[0] - (x + vx)) * pin;
      pos[j + 1] = y + vy + gy + (tgt[1] - (y + vy + gy)) * pin;
      pos[j + 2] = z + vz + (tgt[2] - (z + vz)) * pin;
    }
  }

  function jacobi(src, dst) {
    const tack = u.uTack.value, compress = u.uCompress.value, bend = u.uBend.value;
    for (let i = 0; i < U; i++) {
      const xi = src[i * 4], yi = src[i * 4 + 1], zi = src[i * 4 + 2];
      const wi = wFree[i];
      const start = nbrPtr[i * 2], end = start + nbrPtr[i * 2 + 1];
      if (wi === 0) { dst[i * 4] = xi; dst[i * 4 + 1] = yi; dst[i * 4 + 2] = zi; continue; }   // pinned: no share
      let sx = 0, sy = 0, sz = 0;
      for (let k = start; k < end; k++) {
        const j = nbrJ[k] * 4;
        const dx = src[j] - xi, dy = src[j + 1] - yi, dz = src[j + 2] - zi;
        let len = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (len < 1e-6) len = 1e-6;
        const c = len - restL[k], t = kind[k];
        const stiff = t === 2 ? tack : t === 1 ? bend : (c > 0 ? 1 : compress);
        const ws = wi + wFree[nbrJ[k]];
        const f = c * stiff * (wi / (ws > 1e-6 ? ws : 1e-6)) / len;
        sx += dx * f; sy += dy * f; sz += dz * f;
      }
      const cnt = Math.max(end - start, 1);
      dst[i * 4] = xi + sx * 1.5 / cnt; dst[i * 4 + 1] = yi + sy * 1.5 / cnt; dst[i * 4 + 2] = zi + sz * 1.5 / cnt;
    }
  }

  const A = new Float64Array(4), Bc = new Float64Array(4), A0 = new Float64Array(4), B0 = new Float64Array(4);
  function capAt(out, c, end, al) {   // capsule c's end (0: a, 1: b) at this point of the frame
    const C = frame.caps, p = c * 16 + end * 4, q = p + 8;
    for (let k = 0; k < 4; k++) out[k] = C[p + k] + (C[q + k] - C[p + k]) * al;
  }
  function collidePass() {
    const al = alpha(), al0 = Math.max(al - 1 / sub, 0);
    const tHem = u.uTetherHem.value, tTop = u.uTetherTop.value, collide = u.uCollide.value;
    for (let i = 0; i < U; i++) {
      const j = i * 4, pin = pinOf(i);
      target(i, al);
      const tether = (tHem + (tTop - tHem) * meta[j + 1]) * (1 - pin);
      let x = pos[j] + (tgt[0] - pos[j]) * tether;
      let y = pos[j + 1] + (tgt[1] - pos[j + 1]) * tether;
      let z = pos[j + 2] + (tgt[2] - pos[j + 2]) * tether;
      const x0 = x, y0 = y, z0 = z;
      for (let c = 0; c < nCaps; c++) {
        capAt(A, c, 0, al); capAt(Bc, c, 1, al);
        const abx = Bc[0] - A[0], aby = Bc[1] - A[1], abz = Bc[2] - A[2];
        const ab2 = Math.max(abx * abx + aby * aby + abz * abz, 1e-8);
        const hh = Math.min(1, Math.max(0, ((x - A[0]) * abx + (y - A[1]) * aby + (z - A[2]) * abz) / ab2));
        const qx = A[0] + abx * hh, qy = A[1] + aby * hh, qz = A[2] + abz * hh;
        const allowed = allow[i * nCaps + c];
        const r = A[3] + (Bc[3] - A[3]) * hh - allowed;
        let dx = x - qx, dy = y - qy, dz = z - qz;
        if (len3(dx, dy, dz) >= r) continue;
        // the side it came from, if it was outside on the other side one substep ago (see cloth.js)
        capAt(A0, c, 0, al0); capAt(B0, c, 1, al0);
        const ax0 = B0[0] - A0[0], ay0 = B0[1] - A0[1], az0 = B0[2] - A0[2];
        const px = prev[j], py = prev[j + 1], pz = prev[j + 2];
        const h0 = Math.min(1, Math.max(0, ((px - A0[0]) * ax0 + (py - A0[1]) * ay0 + (pz - A0[2]) * az0)
          / Math.max(ax0 * ax0 + ay0 * ay0 + az0 * az0, 1e-8)));
        let wx = px - (A0[0] + ax0 * h0), wy = py - (A0[1] + ay0 * h0), wz = pz - (A0[2] + az0 * h0);
        const r0 = A0[3] + (B0[3] - A0[3]) * h0 - allowed;
        if (len3(wx, wy, wz) > r0 && dx * wx + dy * wy + dz * wz < 0) {
          const al2 = Math.max(len3(abx, aby, abz), 1e-6), ux = abx / al2, uy = aby / al2, uz = abz / al2;
          const wa = wx * ux + wy * uy + wz * uz;
          wx -= ux * wa; wy -= uy * wa; wz -= uz * wa;
          if (len3(wx, wy, wz) > 1e-4) { dx = wx; dy = wy; dz = wz; }
        }
        const s = r / Math.max(len3(dx, dy, dz), 1e-6);
        x = qx + dx * s; y = qy + dy * s; z = qz + dz * s;
      }
      y = Math.max(y, 0.003);
      const k = collide * Math.min(1, Math.max(0, (1 - pin) * 5));   // not where it (nearly) follows the bones
      pos[j] = x0 + (x - x0) * k; pos[j + 1] = y0 + (y - y0) * k; pos[j + 2] = z0 + (z - z0) * k;
    }
  }

  function normalPass() {
    for (let i = 0; i < U; i++) {
      let nx = 0, ny = 0, nz = 0;
      const start = facePtr[i * 2], end = start + facePtr[i * 2 + 1];
      for (let k = start; k < end; k++) {
        const f = faceList[k] * 4, a = faces[f] * 4, b = faces[f + 1] * 4, c = faces[f + 2] * 4;
        const ex = pos[b] - pos[a], ey = pos[b + 1] - pos[a + 1], ez = pos[b + 2] - pos[a + 2];
        const gx = pos[c] - pos[a], gy = pos[c + 1] - pos[a + 1], gz = pos[c + 2] - pos[a + 2];
        nx += ey * gz - ez * gy; ny += ez * gx - ex * gz; nz += ex * gy - ey * gx;
      }
      const l = Math.max(len3(nx, ny, nz), 1e-12);
      normal[i * 4] = nx / l; normal[i * 4 + 1] = ny / l; normal[i * 4 + 2] = nz / l;
    }
  }

  // ---- draw the mesh from the points (mesh-local positions; normals through the inverse world rotation) ----
  const geo = mesh.geometry;
  const P = geo.attributes.position, N = geo.attributes.normal;
  P.setUsage(THREE.DynamicDrawUsage); N?.setUsage(THREE.DynamicDrawUsage);
  const draw = o.draw !== false;
  if (draw) {
    mesh.isSkinnedMesh = false;   // the vertices come from the points, not the renderer's skinning
    mesh.frustumCulled = false;
  }
  const e = frame.invWorld.elements;
  function writeMesh() {
    const pa = P.array, na = N?.array;
    for (let v = 0; v < P.count; v++) {
      const j = map[v] * 4, x = pos[j], y = pos[j + 1], z = pos[j + 2];
      pa[v * 3] = e[0] * x + e[4] * y + e[8] * z + e[12];
      pa[v * 3 + 1] = e[1] * x + e[5] * y + e[9] * z + e[13];
      pa[v * 3 + 2] = e[2] * x + e[6] * y + e[10] * z + e[14];
      if (na) {
        const nx = normal[j], ny = normal[j + 1], nz = normal[j + 2];
        const lx = e[0] * nx + e[4] * ny + e[8] * nz, ly = e[1] * nx + e[5] * ny + e[9] * nz, lz = e[2] * nx + e[6] * ny + e[10] * nz;
        const l = Math.max(len3(lx, ly, lz), 1e-12);
        na[v * 3] = lx / l; na[v * 3 + 1] = ly / l; na[v * 3 + 2] = lz / l;
      }
    }
    P.needsUpdate = true;
    if (N) N.needsUpdate = true;
  }

  let started = false;
  return {
    points: U, edges: model.edges, bends: model.bends, tacks: model.tacks,
    uniforms: u,
    options: o,
    reset() { started = false; },
    readPositions: async () => pos.slice().buffer,   // same layout as the GPU version (xyzw per point)
    lastMs: 0,
    update(dt) {
      const t0 = performance.now();
      frame.update(!started);
      skinPass();
      if (!started) { initPass(); started = true; }
      sub = o.substeps;
      const free = 1 - u.uFree.value;
      for (let i = 0; i < U; i++) wFree[i] = 1 - Math.max(meta[i * 4], free);
      h = Math.min(dt, 1 / 30) / sub;
      for (let s = 0; s < sub; s++) {
        predictPass();
        for (let k = 0; k < o.iterations; k += 2) { jacobi(pos, pos2); jacobi(pos2, pos); }
        collidePass();
        step++;
      }
      normalPass();
      if (draw) writeMesh();
      this.lastMs = performance.now() - t0;
    },
  };
}
