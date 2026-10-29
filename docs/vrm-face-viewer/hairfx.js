// Hair look for a mesh hair shell (the Tripo hair grafted by tripo/hair_graft.py): a strand-aware shader
// (anisotropic highlight across the strands, strand bumps and colour streaks along them, light through the hair
// from behind) and translucent strand cards on the shell that soften the hairline and the silhouette.
// Both follow one strand direction field: the brush strokes of the baked texture where they are clear, else
// "towards the bun" (a bone named *bun*, an updo) or "down" (loose hair), smoothed over the mesh.
import * as THREE from 'three';

const ANALYSIS = 1024;   // texture analysis (structure tensor) resolution
const FLOW = 2048;       // strand flow map (line integral convolution) resolution
const LIC_STEPS = 12;    // texels walked each way along the strand

export function findHairMesh(root) {
  let hit = null;
  root.traverse((o) => {
    if (!hit && o.isSkinnedMesh && /hair/i.test([].concat(o.material)[0]?.name || o.name)) hit = o;
  });
  return hit;
}

export function createHairFX(mesh) {
  const orig = mesh.material;
  if (!orig.map?.image) return null;
  fillGaps(orig);
  const pix = imagePixels(orig.map.image, ANALYSIS);
  const tensor = structureTensor(pix, ANALYSIS);
  const field = strandField(mesh, tensor);
  mesh.geometry.setAttribute('tangent', new THREE.BufferAttribute(field.tangent, 4));
  const flow = flowMap(mesh.geometry, field, tensor);
  const shell = shellMaterial(orig, flow);
  const cards = buildCards(mesh, field, pix);
  return {
    mesh, orig, shell, cards,
    setShader(on) { mesh.material = on ? shell : orig; },
    setCards(on) { cards.visible = on; },
    dispose() {
      mesh.material = orig;
      cards.parent?.remove(cards);
      cards.geometry.dispose(); cards.material.map.dispose(); cards.material.dispose();
      shell.dispose(); flow.dispose();
    },
  };
}

// ---- texture gaps: the Tripo texture is black outside its UV charts, and the seams (the partings and cracks of
// the hair shell) sample that black into thin dark lines; fill the gaps with the charts' colours (push-pull) ----
function fillGaps(m) {
  const src = m.map, img = src.image;
  const n = Math.min(Math.max(img.width, img.height), 4096);
  const c = document.createElement('canvas');
  c.width = c.height = n;
  const g = c.getContext('2d', { willReadFrequently: true });
  g.drawImage(img, 0, 0, n, n);
  const im = g.getImageData(0, 0, n, n), d = im.data;
  // levels of colour sums and weights; a texel is a gap when it is (near) pure black
  const levels = [];
  let size = n, col = new Float32Array(n * n * 3), wt = new Float32Array(n * n);
  for (let i = 0; i < n * n; i++) {
    if (Math.max(d[4 * i], d[4 * i + 1], d[4 * i + 2]) > 3) {
      wt[i] = 1;
      col[3 * i] = d[4 * i]; col[3 * i + 1] = d[4 * i + 1]; col[3 * i + 2] = d[4 * i + 2];
    }
  }
  levels.push({ size, col, wt });
  while (size > 1) {   // push: average the valid texels of each 2x2 block
    const h = size >> 1, c2 = new Float32Array(h * h * 3), w2 = new Float32Array(h * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < h; x++) {
      let r = 0, gg = 0, b = 0, w = 0;
      for (const [dx, dy] of [[0, 0], [1, 0], [0, 1], [1, 1]]) {
        const j = (2 * y + dy) * size + 2 * x + dx, ww = wt[j];
        r += col[3 * j] * ww; gg += col[3 * j + 1] * ww; b += col[3 * j + 2] * ww; w += ww;
      }
      const k = y * h + x;
      if (w > 0) { c2[3 * k] = r / w; c2[3 * k + 1] = gg / w; c2[3 * k + 2] = b / w; w2[k] = 1; }
    }
    size = h; col = c2; wt = w2;
    levels.push({ size, col, wt });
  }
  for (let l = levels.length - 2; l >= 0; l--) {   // pull: a gap takes the coarser level's colour
    const f = levels[l], cr = levels[l + 1];
    for (let y = 0; y < f.size; y++) for (let x = 0; x < f.size; x++) {
      const i = y * f.size + x;
      if (f.wt[i]) continue;
      const j = Math.min(y >> 1, cr.size - 1) * cr.size + Math.min(x >> 1, cr.size - 1);
      f.col[3 * i] = cr.col[3 * j]; f.col[3 * i + 1] = cr.col[3 * j + 1]; f.col[3 * i + 2] = cr.col[3 * j + 2];
      f.wt[i] = 1;
    }
  }
  const top = levels[0];
  for (let i = 0; i < n * n; i++) {
    d[4 * i] = top.col[3 * i]; d[4 * i + 1] = top.col[3 * i + 1]; d[4 * i + 2] = top.col[3 * i + 2];
  }
  g.putImageData(im, 0, 0);
  const tex = new THREE.CanvasTexture(c);
  for (const k of ['flipY', 'colorSpace', 'wrapS', 'wrapT', 'magFilter', 'minFilter', 'channel']) tex[k] = src[k];
  tex.anisotropy = Math.max(src.anisotropy, 8);
  m.map = tex;
  m.needsUpdate = true;
}

// ---- texture analysis ----
function imagePixels(image, n) {
  const c = document.createElement('canvas');
  c.width = c.height = n;
  const g = c.getContext('2d', { willReadFrequently: true });
  g.drawImage(image, 0, 0, n, n);
  return g.getImageData(0, 0, n, n).data;
}

function boxBlur(a, n, r) {
  const t = new Float32Array(a.length);
  for (let y = 0; y < n; y++) {   // rows into t
    let s = 0;
    for (let x = -r; x <= r; x++) s += a[y * n + Math.min(Math.max(x, 0), n - 1)];
    for (let x = 0; x < n; x++) {
      t[y * n + x] = s / (2 * r + 1);
      s += a[y * n + Math.min(x + r + 1, n - 1)] - a[y * n + Math.max(x - r, 0)];
    }
  }
  for (let x = 0; x < n; x++) {   // columns back into a
    let s = 0;
    for (let y = -r; y <= r; y++) s += t[Math.min(Math.max(y, 0), n - 1) * n + x];
    for (let y = 0; y < n; y++) {
      a[y * n + x] = s / (2 * r + 1);
      s += t[Math.min(y + r + 1, n - 1) * n + x] - t[Math.max(y - r, 0) * n + x];
    }
  }
}

// Smoothed structure tensor of the luminance; strands run across the dominant gradient.
function structureTensor(pix, n) {
  const L = new Float32Array(n * n);
  for (let i = 0; i < n * n; i++) L[i] = (0.3 * pix[4 * i] + 0.59 * pix[4 * i + 1] + 0.11 * pix[4 * i + 2]) / 255;
  const xx = new Float32Array(n * n), xy = new Float32Array(n * n), yy = new Float32Array(n * n);
  for (let y = 1; y < n - 1; y++) for (let x = 1; x < n - 1; x++) {
    const i = y * n + x, gx = L[i + 1] - L[i - 1], gy = L[i + n] - L[i - n];
    xx[i] = gx * gx; xy[i] = gx * gy; yy[i] = gy * gy;
  }
  for (const a of [xx, xy, yy]) { boxBlur(a, n, 3); boxBlur(a, n, 3); }
  return {
    n, L,
    // strand direction in texture space (du, dv) and how much to trust it (0..1)
    at(u, v) {
      const x = Math.min(Math.max(Math.floor(u * n), 0), n - 1), y = Math.min(Math.max(Math.floor(v * n), 0), n - 1);
      const i = y * n + x, a = xx[i], b = xy[i], c = yy[i];
      const tr = a + c, d = Math.hypot((a - c) / 2, b);
      if (tr < 1e-9) return [1, 0, 0];
      const coh = (2 * d / tr) ** 2, strength = (tr / 2 + d) / (tr / 2 + d + 2e-4);
      const th = 0.5 * Math.atan2(2 * b, a - c);
      return [-Math.sin(th), Math.cos(th), coh * strength];
    },
  };
}

// ---- strand direction per vertex ----
function strandField(mesh, tensor) {
  const geo = mesh.geometry, pos = geo.attributes.position, nrm = geo.attributes.normal, uv = geo.attributes.uv;
  const idx = geo.index.array, nv = pos.count, nt = idx.length / 3;
  // weld UV-seam duplicates so the field is smoothed across charts
  const weld = new Int32Array(nv), keys = new Map();
  for (let i = 0; i < nv; i++) {
    const k = `${Math.round(pos.getX(i) * 2e4)},${Math.round(pos.getY(i) * 2e4)},${Math.round(pos.getZ(i) * 2e4)}`;
    if (!keys.has(k)) keys.set(k, keys.size);
    weld[i] = keys.get(k);
  }
  const nw = keys.size;
  const P = new Float32Array(nw * 3), N = new Float32Array(nw * 3);
  for (let i = 0; i < nv; i++) {
    const w = weld[i];
    P[3 * w] = pos.getX(i); P[3 * w + 1] = pos.getY(i); P[3 * w + 2] = pos.getZ(i);
    N[3 * w] += nrm.getX(i); N[3 * w + 1] += nrm.getY(i); N[3 * w + 2] += nrm.getZ(i);
  }
  for (let w = 0; w < nw; w++) norm3(N, w);
  const nb = Array.from({ length: nw }, () => new Set());

  // prior: towards the bun (updo; the bun itself swirls round its axis) or down (loose hair)
  geo.computeBoundingBox();
  const center = geo.boundingBox.getCenter(new THREE.Vector3());
  const bun = mesh.skeleton.bones.findIndex((b) => /bun/i.test(b.name));
  let bunPos = null, bunW = null;
  if (bun >= 0) {
    bunPos = new THREE.Vector3().setFromMatrixPosition(mesh.skeleton.boneInverses[bun].clone().invert())
      .applyMatrix4(mesh.bindMatrixInverse);
    const si = geo.attributes.skinIndex, sw = geo.attributes.skinWeight;
    bunW = new Float32Array(nw);
    for (let i = 0; i < nv; i++) for (let k = 0; k < 4; k++) if (si.getComponent(i, k) === bun) bunW[weld[i]] = Math.max(bunW[weld[i]], sw.getComponent(i, k));
  }
  const axis = bunPos ? bunPos.clone().sub(center).normalize() : null;
  const G = new Float32Array(nw * 3), p = new THREE.Vector3(), n = new THREE.Vector3(), g = new THREE.Vector3();
  for (let w = 0; w < nw; w++) {
    p.fromArray(P, 3 * w); n.fromArray(N, 3 * w);
    if (bunPos) {
      if (bunW[w] > 0.5) g.copy(p).sub(bunPos).cross(axis).negate();
      else g.copy(bunPos).sub(p);
    } else {
      g.set(0, -1, 0).addScaledVector(new THREE.Vector3(p.x - center.x, 0, p.z - center.z).normalize(), 0.3);
    }
    g.addScaledVector(n, -n.dot(g));
    if (g.lengthSq() < 1e-10) g.set(0, -1, 0).addScaledVector(n, -n.y);
    g.normalize().toArray(G, 3 * w);
  }

  // texture term: each triangle's brush-stroke direction, mapped from UV to 3D by the triangle's Jacobian
  const T = new Float32Array(nw * 6), area = new Float32Array(nw);
  const p0 = new THREE.Vector3(), e1 = new THREE.Vector3(), e2 = new THREE.Vector3(), fn = new THREE.Vector3();
  const Pu = new THREE.Vector3(), Pv = new THREE.Vector3(), t = new THREE.Vector3();
  for (let f = 0; f < nt; f++) {
    const a = idx[3 * f], b = idx[3 * f + 1], c = idx[3 * f + 2];
    const wa = weld[a], wb = weld[b], wc = weld[c];
    nb[wa].add(wb).add(wc); nb[wb].add(wa).add(wc); nb[wc].add(wa).add(wb);
    p0.fromBufferAttribute(pos, a);
    e1.fromBufferAttribute(pos, b).sub(p0); e2.fromBufferAttribute(pos, c).sub(p0);
    fn.crossVectors(e1, e2);
    const ar = fn.length() / 2;
    if (ar < 1e-12) continue;
    fn.normalize();
    for (const w of [wa, wb, wc]) area[w] += ar;
    const u0 = uv.getX(a), v0 = uv.getY(a);
    const du1 = uv.getX(b) - u0, dv1 = uv.getY(b) - v0, du2 = uv.getX(c) - u0, dv2 = uv.getY(c) - v0;
    const det = du1 * dv2 - dv1 * du2;
    if (Math.abs(det) < 1e-12) continue;
    Pu.copy(e1).multiplyScalar(dv2).addScaledVector(e2, -dv1).divideScalar(det);
    Pv.copy(e2).multiplyScalar(du1).addScaledVector(e1, -du2).divideScalar(det);
    const [du, dv, conf] = tensor.at(u0 + (du1 + du2) / 3, v0 + (dv1 + dv2) / 3);
    if (conf < 0.05) continue;
    t.copy(Pu).multiplyScalar(du).addScaledVector(Pv, dv);
    t.addScaledVector(fn, -fn.dot(t));
    if (t.lengthSq() < 1e-14) continue;
    t.normalize();
    for (const w of [wa, wb, wc]) addOuter(T, w, t, conf * ar);
  }
  // combine (texture averaged over the vertex's area, plus a weaker prior), then diffuse over the mesh
  for (let w = 0; w < nw; w++) {
    for (let k = 0; k < 6; k++) T[6 * w + k] /= Math.max(area[w], 1e-12);
    addOuter(T, w, g.fromArray(G, 3 * w), 0.3);
  }
  const nbs = nb.map((s) => [...s]);
  let A = T, B = new Float32Array(nw * 6);
  for (let it = 0; it < 6; it++) {
    for (let w = 0; w < nw; w++) {
      const l = nbs[w];
      for (let k = 0; k < 6; k++) {
        let s = 0;
        for (const j of l) s += A[6 * j + k];
        B[6 * w + k] = 0.5 * A[6 * w + k] + 0.5 * (l.length ? s / l.length : A[6 * w + k]);
      }
    }
    [A, B] = [B, A];
  }
  // principal direction (power iteration from the prior, which also fixes the sign)
  const D = new Float32Array(nw * 3);
  for (let w = 0; w < nw; w++) {
    const m = A.subarray(6 * w, 6 * w + 6);   // xx xy xz yy yz zz
    let x = G[3 * w], y = G[3 * w + 1], z = G[3 * w + 2];
    for (let it = 0; it < 12; it++) {
      const nx = m[0] * x + m[1] * y + m[2] * z, ny = m[1] * x + m[3] * y + m[4] * z, nz = m[2] * x + m[4] * y + m[5] * z;
      const l = Math.hypot(nx, ny, nz) || 1;
      x = nx / l; y = ny / l; z = nz / l;
    }
    const nd = x * N[3 * w] + y * N[3 * w + 1] + z * N[3 * w + 2];
    x -= nd * N[3 * w]; y -= nd * N[3 * w + 1]; z -= nd * N[3 * w + 2];
    if (x * G[3 * w] + y * G[3 * w + 1] + z * G[3 * w + 2] < 0) { x = -x; y = -y; z = -z; }
    D[3 * w] = x; D[3 * w + 1] = y; D[3 * w + 2] = z;
    norm3(D, w);
  }
  const tangent = new Float32Array(nv * 4);
  for (let i = 0; i < nv; i++) {
    const w = weld[i];
    t.fromArray(D, 3 * w); n.fromBufferAttribute(nrm, i);
    t.addScaledVector(n, -n.dot(t)).normalize();
    tangent[4 * i] = t.x; tangent[4 * i + 1] = t.y; tangent[4 * i + 2] = t.z; tangent[4 * i + 3] = 1;
  }
  // open edges of the shell (the hairline): each edge with its outward direction (away from the hair)
  const edgeCount = new Map();
  for (let f = 0; f < nt; f++) for (let k = 0; k < 3; k++) {
    const a = weld[idx[3 * f + k]], b = weld[idx[3 * f + (k + 1) % 3]], c = weld[idx[3 * f + (k + 2) % 3]];
    const key = a < b ? `${a},${b}` : `${b},${a}`;
    const e = edgeCount.get(key);
    if (e) e.n++; else edgeCount.set(key, { a, b, c, n: 1 });
  }
  const boundary = [...edgeCount.values()].filter((e) => e.n === 1);
  return { weld, P, N, D, tangent, center, boundary, nbs };
}

function norm3(a, i) {
  const l = Math.hypot(a[3 * i], a[3 * i + 1], a[3 * i + 2]) || 1;
  a[3 * i] /= l; a[3 * i + 1] /= l; a[3 * i + 2] /= l;
}

function addOuter(T, w, t, s) {
  T[6 * w] += s * t.x * t.x; T[6 * w + 1] += s * t.x * t.y; T[6 * w + 2] += s * t.x * t.z;
  T[6 * w + 3] += s * t.y * t.y; T[6 * w + 4] += s * t.y * t.z; T[6 * w + 5] += s * t.z * t.z;
}

// ---- strand flow map: noise smeared along the strands in texture space (bump + colour streaks) ----
function flowMap(geo, field, tensor) {
  const F = FLOW, pos = geo.attributes.position, uv = geo.attributes.uv, idx = geo.index.array, tan = field.tangent;
  const dirU = new Float32Array(F * F), dirV = new Float32Array(F * F), mask = new Uint8Array(F * F);
  const p0 = new THREE.Vector3(), e1 = new THREE.Vector3(), e2 = new THREE.Vector3();
  const Pu = new THREE.Vector3(), Pv = new THREE.Vector3(), t = new THREE.Vector3();
  for (let f = 0; f < idx.length / 3; f++) {
    const ids = [idx[3 * f], idx[3 * f + 1], idx[3 * f + 2]];
    const [a, b, c] = ids;
    p0.fromBufferAttribute(pos, a);
    e1.fromBufferAttribute(pos, b).sub(p0); e2.fromBufferAttribute(pos, c).sub(p0);
    const ua = uv.getX(a) * F, va = uv.getY(a) * F, ub = uv.getX(b) * F, vb = uv.getY(b) * F, uc = uv.getX(c) * F, vc = uv.getY(c) * F;
    const du1 = ub - ua, dv1 = vb - va, du2 = uc - ua, dv2 = vc - va, det = du1 * dv2 - dv1 * du2;
    if (Math.abs(det) < 1e-9) continue;
    Pu.copy(e1).multiplyScalar(dv2).addScaledVector(e2, -dv1).divideScalar(det);
    Pv.copy(e2).multiplyScalar(du1).addScaledVector(e1, -du2).divideScalar(det);
    // 3D -> texture direction: least squares through the 2x2 normal matrix of [Pu Pv]
    const g11 = Pu.dot(Pu), g12 = Pu.dot(Pv), g22 = Pv.dot(Pv), gd = g11 * g22 - g12 * g12;
    if (Math.abs(gd) < 1e-30) continue;
    const x0 = Math.max(Math.floor(Math.min(ua, ub, uc)), 0), x1 = Math.min(Math.ceil(Math.max(ua, ub, uc)), F - 1);
    const y0 = Math.max(Math.floor(Math.min(va, vb, vc)), 0), y1 = Math.min(Math.ceil(Math.max(va, vb, vc)), F - 1);
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
      const px = x + 0.5 - ua, py = y + 0.5 - va;
      const l1 = (px * dv2 - py * du2) / det, l2 = (py * du1 - px * dv1) / det, l0 = 1 - l1 - l2;
      if (l0 < -0.02 || l1 < -0.02 || l2 < -0.02) continue;
      t.set(0, 0, 0);
      [l0, l1, l2].forEach((l, k) => { t.x += l * tan[4 * ids[k]]; t.y += l * tan[4 * ids[k] + 1]; t.z += l * tan[4 * ids[k] + 2]; });
      const r1 = Pu.dot(t), r2 = Pv.dot(t);
      let du = (g22 * r1 - g12 * r2) / gd, dv = (g11 * r2 - g12 * r1) / gd;
      let l = Math.hypot(du, dv) || 1;
      du /= l; dv /= l;
      // the texel's own brush stroke where it is clear
      const [tu, tv, conf] = tensor.at((x + 0.5) / F, (y + 0.5) / F);
      const s = (tu * du + tv * dv < 0 ? -1 : 1) * Math.min(conf, 0.8);
      du = du * (1 - Math.abs(s)) + tu * s; dv = dv * (1 - Math.abs(s)) + tv * s;
      l = Math.hypot(du, dv) || 1;
      const i = y * F + x;
      dirU[i] = du / l; dirV[i] = dv / l; mask[i] = 1;
    }
  }
  // grow the charts a few texels so mip levels and the walk do not run into the gaps
  for (let it = 0; it < 4; it++) {
    const add = [];
    for (let y = 1; y < F - 1; y++) for (let x = 1; x < F - 1; x++) {
      const i = y * F + x;
      if (mask[i]) continue;
      let su = 0, sv = 0, c = 0;
      for (const j of [i - 1, i + 1, i - F, i + F]) if (mask[j]) { su += dirU[j]; sv += dirV[j]; c++; }
      if (c) add.push(i, su / c, sv / c);
    }
    for (let k = 0; k < add.length; k += 3) { dirU[add[k]] = add[k + 1]; dirV[add[k]] = add[k + 2]; mask[add[k]] = 1; }
  }
  // line integral convolution of white noise
  const noise = new Float32Array(F * F);
  for (let i = 0; i < F * F; i++) noise[i] = Math.random();
  const out = new Float32Array(F * F);
  let sum = 0, sum2 = 0, cnt = 0;
  for (let y = 0; y < F; y++) for (let x = 0; x < F; x++) {
    const i0 = y * F + x;
    if (!mask[i0]) continue;
    let s = noise[i0], c = 1;
    for (const sign of [1, -1]) {
      let px = x + 0.5, py = y + 0.5, du = dirU[i0] * sign, dv = dirV[i0] * sign;
      for (let k = 0; k < LIC_STEPS; k++) {
        px += du; py += dv;
        const xi = px | 0, yi = py | 0;
        if (xi < 0 || yi < 0 || xi >= F || yi >= F) break;
        const j = yi * F + xi;
        if (!mask[j]) break;
        let nu = dirU[j], nv = dirV[j];
        if (nu * du + nv * dv < 0) { nu = -nu; nv = -nv; }
        du = nu; dv = nv;
        s += noise[j]; c++;
      }
    }
    const v = s / c;
    out[i0] = v; sum += v; sum2 += v * v; cnt++;
  }
  const mean = sum / cnt, sd = Math.sqrt(Math.max(sum2 / cnt - mean * mean, 1e-8));
  const data = new Uint8Array(F * F * 4);
  for (let i = 0; i < F * F; i++) {
    const v = mask[i] ? Math.min(Math.max(0.5 + (out[i] - mean) / (4 * sd), 0), 1) : 0.5;
    data[4 * i] = data[4 * i + 1] = data[4 * i + 2] = Math.round(v * 255);
    data[4 * i + 3] = 255;
  }
  const tex = new THREE.DataTexture(data, F, F);
  tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.magFilter = THREE.LinearFilter;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.generateMipmaps = true;
  tex.anisotropy = 8;
  tex.needsUpdate = true;
  return tex;
}

// ---- materials ----
const HAIR = {
  roughness: 0.46, anisotropy: 0.8, anisotropyRotation: Math.PI / 2, specularIntensity: 0.45,
  sheen: 0.4, sheenRoughness: 0.5, streak: 0.35, trans: 0.9,
};

// light through the hair from behind (strongest at the rim), and colour streaks along the strands
function hairTerms(m, { streak = 0, cards = false } = {}) {
  m.userData.uniforms = { uStreak: { value: streak }, uTrans: { value: HAIR.trans } };
  m.customProgramCacheKey = () => `hairfx-${cards}`;
  m.onBeforeCompile = (sh) => {
    Object.assign(sh.uniforms, m.userData.uniforms);
    let fs = sh.fragmentShader.replace('#include <common>', '#include <common>\nuniform float uStreak;\nuniform float uTrans;');
    if (cards) fs = fs.replace('float faceDirection = gl_FrontFacing ? 1.0 : - 1.0;', 'float faceDirection = 1.0;');
    // edge-on, the strand bump tips the normal away from the eye (black rims round the bun): fade it out there
    fs = fs.replace('#include <normal_fragment_maps>', THREE.ShaderChunk.normal_fragment_maps.replace(
      'normal = perturbNormalArb( - vViewPosition, normal, dHdxy_fwd(), faceDirection );',
      'normal = perturbNormalArb( - vViewPosition, normal, dHdxy_fwd() * smoothstep( 0.15, 0.5, abs( dot( normal, normalize( vViewPosition ) ) ) ), faceDirection );'));
    fs = fs.replace('#include <map_fragment>', `#include <map_fragment>
#ifdef USE_BUMPMAP
{
  float hairness = 1.0 - smoothstep( 0.12, 0.3, dot( diffuseColor.rgb, vec3( 0.3, 0.59, 0.11 ) ) );
  diffuseColor.rgb *= 1.0 + uStreak * hairness * ( texture2D( bumpMap, vBumpMapUv ).r - 0.5 ) * 2.0;
}
#endif`);
    fs = fs.replace('#include <lights_fragment_end>', `#include <lights_fragment_end>
#if NUM_DIR_LIGHTS > 0
for ( int i = 0; i < NUM_DIR_LIGHTS; i ++ ) {
  float back = pow( saturate( dot( - directionalLights[ i ].direction, geometryViewDir ) ), 3.0 );
  float edge = 1.0 - saturate( abs( dot( geometryNormal, geometryViewDir ) ) );
  reflectedLight.directDiffuse += pow( diffuseColor.rgb, vec3( 0.75 ) ) * directionalLights[ i ].color * uTrans * back * ( 0.3 + edge * edge );
}
#endif`);
    sh.fragmentShader = fs;
  };
  return m;
}

function shellMaterial(orig, flow) {
  const m = new THREE.MeshPhysicalMaterial({
    name: `${orig.name}+fx`, map: orig.map, color: orig.color, side: orig.side,
    roughness: HAIR.roughness, metalness: 0, specularIntensity: HAIR.specularIntensity,
    anisotropy: HAIR.anisotropy, anisotropyRotation: HAIR.anisotropyRotation,
    bumpMap: flow, bumpScale: 0.4,   // stronger, the noise aliases into black specks and moire where the hair is seen edge-on
    sheen: HAIR.sheen, sheenRoughness: HAIR.sheenRoughness, sheenColor: new THREE.Color(0.35, 0.3, 0.27),
  });
  return hairTerms(m, { streak: HAIR.streak });
}

// ---- strand cards ----
function strandTexture() {
  const W = 256, H = 512, cell = W / 4;
  const c = document.createElement('canvas');
  c.width = W; c.height = H;
  const g = c.getContext('2d');
  for (let k = 0; k < 4; k++) {
    const count = 7 + k * 2;
    for (let s = 0; s < count; s++) {
      const x0 = k * cell + 5 + Math.random() * (cell - 10), y0 = Math.random() * H * 0.15, y1 = H * (0.8 + Math.random() * 0.2);
      const grad = g.createLinearGradient(0, y0, 0, y1);
      const a = 0.45 + Math.random() * 0.55;
      grad.addColorStop(0, 'rgba(255,255,255,0)');
      grad.addColorStop(0.2, `rgba(255,255,255,${a})`);
      grad.addColorStop(0.75, `rgba(255,255,255,${a})`);
      grad.addColorStop(1, 'rgba(255,255,255,0)');
      g.strokeStyle = grad;
      g.lineWidth = 1.2 + Math.random() * 1.6;
      const amp = Math.random() * 3, ph = Math.random() * 6;
      g.beginPath();
      for (let y = y0; y <= y1; y += 8) {
        const x = x0 + amp * Math.sin(ph + y / 60) + (y - y0) * (Math.random() - 0.5) * 0.004;
        if (y === y0) g.moveTo(x, y); else g.lineTo(x, y);
      }
      g.stroke();
    }
  }
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 8;
  return tex;
}

const CARDS = { lying: 3000, fins: 1200, hairline: 250, fringe: 500, stray: 40 };

function buildCards(mesh, field, pix) {
  const geo = mesh.geometry, pos = geo.attributes.position, nrm = geo.attributes.normal, uv = geo.attributes.uv;
  const si = geo.attributes.skinIndex, sw = geo.attributes.skinWeight, idx = geo.index.array, tan = field.tangent;
  const nt = idx.length / 3, center = field.center;
  const color = (u, v) => {   // linear colour of the baked texture, 3x3 texels around (u, v)
    let r = 0, g = 0, b = 0;
    const x = Math.floor(u * ANALYSIS), y = Math.floor(v * ANALYSIS);
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const i = 4 * (Math.min(Math.max(y + dy, 0), ANALYSIS - 1) * ANALYSIS + Math.min(Math.max(x + dx, 0), ANALYSIS - 1));
      r += pix[i]; g += pix[i + 1]; b += pix[i + 2];
    }
    return new THREE.Color().setRGB(r / 2295, g / 2295, b / 2295, THREE.SRGBColorSpace);
  };
  // outward-facing, hair-coloured triangles, picked by area
  const cum = [], tris = [], hairCols = [];
  let total = 0;
  const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3(), fn = new THREE.Vector3();
  for (let f = 0; f < nt; f++) {
    a.fromBufferAttribute(pos, idx[3 * f]); b.fromBufferAttribute(pos, idx[3 * f + 1]); c.fromBufferAttribute(pos, idx[3 * f + 2]);
    fn.crossVectors(b.clone().sub(a), c.clone().sub(a));
    const ar = fn.length() / 2;
    const mid = a.clone().add(b).add(c).divideScalar(3);
    if (ar < 1e-12 || fn.dot(mid.clone().sub(center)) <= 0) continue;
    const col = color((uv.getX(idx[3 * f]) + uv.getX(idx[3 * f + 1]) + uv.getX(idx[3 * f + 2])) / 3,
      (uv.getY(idx[3 * f]) + uv.getY(idx[3 * f + 1]) + uv.getY(idx[3 * f + 2])) / 3);
    if (col.r * 0.3 + col.g * 0.59 + col.b * 0.11 > 0.12) continue;   // scalp / skin patches of the atlas
    total += ar; cum.push(total); tris.push(f); hairCols.push(col);
  }
  hairCols.sort((x, y) => x.r - y.r);
  // rings from the skin-coloured part of the atlas (painted hairline, parting, scalp), per welded vertex
  const nw = field.P.length / 3, skinD = new Int32Array(nw).fill(99), first = new Int32Array(nw).fill(-1);
  for (let i = 0; i < pos.count; i++) {
    const w = field.weld[i], col = color(uv.getX(i), uv.getY(i));
    if (first[w] < 0) first[w] = i;
    if (col.r * 0.3 + col.g * 0.59 + col.b * 0.11 > 0.12) skinD[w] = 0;
  }
  for (let ring = 1; ring <= 3; ring++) {
    for (let w = 0; w < nw; w++) if (skinD[w] === 99 && field.nbs[w].some((j) => skinD[j] === ring - 1)) skinD[w] = ring;
  }
  const hairColor = hairCols[hairCols.length >> 1] || new THREE.Color(0.02, 0.014, 0.012);
  const P = [], N = [], T = [], UV = [], C = [], SI = [], SW = [], I = [];
  const SEG = 4;
  const sample = (f, l0, l1) => {   // surface point of triangle f at barycentric (l0, l1, 1 - l0 - l1)
    const ids = [idx[3 * f], idx[3 * f + 1], idx[3 * f + 2]], ls = [l0, l1, 1 - l0 - l1];
    const p = new THREE.Vector3(), n = new THREE.Vector3(), t = new THREE.Vector3();
    let u = 0, v = 0;
    ids.forEach((i, k) => {
      p.addScaledVector(a.fromBufferAttribute(pos, i), ls[k]);
      n.addScaledVector(a.fromBufferAttribute(nrm, i), ls[k]);
      t.x += ls[k] * tan[4 * i]; t.y += ls[k] * tan[4 * i + 1]; t.z += ls[k] * tan[4 * i + 2];
      u += ls[k] * uv.getX(i); v += ls[k] * uv.getY(i);
    });
    n.normalize();
    t.addScaledVector(n, -n.dot(t)).normalize();
    const skin = ids[ls.indexOf(Math.max(...ls))];
    return { p, n, t, u, v, skin, near: Math.min(...ids.map((i) => skinD[field.weld[i]])) };
  };
  // one ribbon: centre line p + t*s + n*(lift - curvature), width along `across` (b for lying cards, n for fins)
  const ribbon = (s, { len, width, lift, fin, tilt = 0, start = -0.5, cell, alpha }) => {
    const bt = new THREE.Vector3().crossVectors(s.n, s.t);
    const across = fin ? s.n : bt;
    let col = color(s.u, s.v);
    if (col.r * 0.3 + col.g * 0.59 + col.b * 0.11 > 0.12) col = hairColor.clone();   // rooted on a skin patch
    col.multiplyScalar(0.8 + Math.random() * 0.3);
    const base = P.length / 3;
    for (let k = 0; k <= SEG; k++) {
      const q = start + k / SEG, d = q * len;
      const out = lift - (d * d) / (2 * 0.085) * 0.85 + tilt * Math.max(q, 0) ** 2 * len;
      const cpt = s.p.clone().addScaledVector(s.t, d).addScaledVector(s.n, out);
      const e0 = fin ? cpt : cpt.clone().addScaledVector(across, -width / 2);
      const e1 = cpt.clone().addScaledVector(across, fin ? width : width / 2);
      for (const [e, uu] of [[e0, 0], [e1, 1]]) {
        P.push(e.x, e.y, e.z); N.push(s.n.x, s.n.y, s.n.z); T.push(s.t.x, s.t.y, s.t.z, 1);
        UV.push((cell + uu) / 4, k / SEG); C.push(col.r, col.g, col.b, alpha);
        for (let j = 0; j < 4; j++) { SI.push(si.getComponent(s.skin, j)); SW.push(sw.getComponent(s.skin, j)); }
      }
      if (k < SEG) { const v0 = base + 2 * k; I.push(v0, v0 + 1, v0 + 2, v0 + 1, v0 + 3, v0 + 2); }
    }
  };
  const pick = () => {
    const r = Math.random() * total;
    let lo = 0, hi = cum.length - 1;
    while (lo < hi) { const m = (lo + hi) >> 1; if (cum[m] < r) lo = m + 1; else hi = m; }
    let l0 = Math.random(), l1 = Math.random();
    if (l0 + l1 > 1) { l0 = 1 - l0; l1 = 1 - l1; }
    return sample(tris[lo], l0, l1);
  };
  const cell = () => Math.floor(Math.random() * 4);
  // near the parting and the hairline (painted as skin in the atlas) the cards are short and faint, not covering it
  const short = (s) => (s.near <= 2 ? 0.35 : 1);
  for (let k = 0; k < CARDS.lying; k++) {
    const s = pick();
    ribbon(s, { len: (0.012 + Math.random() * 0.022) * short(s), width: 0.003 + Math.random() * 0.003, lift: 0.0006 + Math.random() * 0.0012, cell: cell(), alpha: (0.5 + Math.random() * 0.4) * (0.4 + 0.6 * short(s)) });
  }
  for (let k = 0; k < CARDS.fins; k++) {
    const s = pick();
    ribbon(s, { len: (0.012 + Math.random() * 0.02) * short(s), width: 0.0015 + Math.random() * 0.0025, lift: -0.0004, fin: true, cell: cell(), alpha: (0.35 + Math.random() * 0.35) * (0.4 + 0.6 * short(s)) });
  }
  // the painted hairline / parting: wisps from the last hair vertex reaching onto the skin
  const fringe = [];
  for (let w = 0; w < nw; w++) {
    if (skinD[w] !== 1 || first[w] < 0) continue;
    const pw = new THREE.Vector3().fromArray(field.P, 3 * w), nw3 = new THREE.Vector3().fromArray(field.N, 3 * w);
    if (nw3.dot(pw.clone().sub(center)) <= 0) continue;
    for (const j of field.nbs[w]) if (skinD[j] === 0) fringe.push([w, j]);
  }
  for (let k = 0; fringe.length && k < CARDS.fringe; k++) {
    const [w, j] = fringe[Math.floor(Math.random() * fringe.length)];
    const ph = new THREE.Vector3().fromArray(field.P, 3 * w), ps = new THREE.Vector3().fromArray(field.P, 3 * j);
    const n = new THREE.Vector3().fromArray(field.N, 3 * w);
    const o = ps.clone().sub(ph);
    o.addScaledVector(n, -n.dot(o)).normalize();
    const t = new THREE.Vector3().fromArray(field.D, 3 * w);
    if (t.dot(o) < 0) t.negate();
    t.lerp(o, 0.6).addScaledVector(n, -n.dot(t)).normalize();
    const len = 0.003 + Math.random() * 0.005, i = first[w];
    const p = ph.lerp(ps, 0.15 + Math.random() * 0.35).addScaledVector(t, -len * 0.5);
    ribbon({ p, n, t, u: uv.getX(i), v: uv.getY(i), skin: i },
      { len, width: 0.0008 + Math.random() * 0.0008, lift: 0.0003, start: 0, tilt: -0.01, cell: cell(), alpha: 0.3 + Math.random() * 0.25 });
  }
  for (let k = 0; k < CARDS.stray; k++) {   // a few loose hairs that leave the surface
    ribbon(pick(), { len: 0.025 + Math.random() * 0.03, width: 0.0025, lift: 0.0008, start: 0, tilt: 0.06 + Math.random() * 0.1, cell: 0, alpha: 0.5 });
  }
  // the hairline: short wisps rooted just inside the open edge of the shell, reaching out past it
  const edges = field.boundary.filter((e) => {
    const pa = new THREE.Vector3().fromArray(field.P, 3 * e.a);
    return pa.y > center.y - 0.06;   // not the inner rim at the nape / under the bun
  });
  const elen = edges.map((e) => Math.hypot(field.P[3 * e.a] - field.P[3 * e.b], field.P[3 * e.a + 1] - field.P[3 * e.b + 1], field.P[3 * e.a + 2] - field.P[3 * e.b + 2]));
  const etot = elen.reduce((s, x) => s + x, 0);
  if (etot > 0) {
    const ecum = [];
    elen.reduce((s, x) => (ecum.push(s + x), s + x), 0);
    for (let k = 0; k < CARDS.hairline; k++) {
      const r = Math.random() * etot;
      const e = edges[ecum.findIndex((x) => x >= r)];
      const q = Math.random();
      const pa = new THREE.Vector3().fromArray(field.P, 3 * e.a), pb = new THREE.Vector3().fromArray(field.P, 3 * e.b);
      const pc = new THREE.Vector3().fromArray(field.P, 3 * e.c);
      const p = pa.clone().lerp(pb, q);
      const n = new THREE.Vector3().fromArray(field.N, 3 * e.a).lerp(new THREE.Vector3().fromArray(field.N, 3 * e.b), q).normalize();
      const out = p.clone().sub(pc).addScaledVector(n, -n.dot(p.clone().sub(pc)));
      const edge = pb.clone().sub(pa).normalize();
      out.addScaledVector(edge, -edge.dot(out)).normalize();
      const t = new THREE.Vector3().fromArray(field.D, 3 * e.a);
      if (t.dot(out) < 0) t.negate();
      t.lerp(out, 0.5).addScaledVector(n, -n.dot(t)).normalize();
      const len = 0.003 + Math.random() * 0.005;
      // nearest vertex of the edge for skinning and colour
      let best = 0, bd = Infinity;
      for (let i = 0; i < pos.count; i++) {
        if (field.weld[i] !== (q < 0.5 ? e.a : e.b)) continue;
        best = i; bd = 0; break;
      }
      if (bd) continue;
      ribbon({ p: p.addScaledVector(t, -len * 0.4), n, t, u: uv.getX(best), v: uv.getY(best), skin: best },
        { len, width: 0.0012 + Math.random() * 0.001, lift: 0.0003, start: 0, tilt: -0.02, cell: cell(), alpha: 0.3 + Math.random() * 0.25 });
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(P, 3));
  g.setAttribute('normal', new THREE.Float32BufferAttribute(N, 3));
  g.setAttribute('tangent', new THREE.Float32BufferAttribute(T, 4));
  g.setAttribute('uv', new THREE.Float32BufferAttribute(UV, 2));
  g.setAttribute('color', new THREE.Float32BufferAttribute(C, 4));
  g.setAttribute('skinIndex', new THREE.Uint16BufferAttribute(SI, 4));
  g.setAttribute('skinWeight', new THREE.Float32BufferAttribute(SW, 4));
  g.setIndex(I);
  const m = hairTerms(new THREE.MeshPhysicalMaterial({
    name: 'hair_cards', map: strandTexture(), vertexColors: true, side: THREE.DoubleSide,
    transparent: true, depthWrite: false, alphaTest: 0.02,   // blended: sub-pixel strands fade instead of speckling
    // layered cards each catch the highlight, so far less specular than the shell
    roughness: HAIR.roughness, metalness: 0, specularIntensity: 0.35,
    anisotropy: HAIR.anisotropy, anisotropyRotation: HAIR.anisotropyRotation,
    sheen: 0.15, sheenRoughness: HAIR.sheenRoughness, sheenColor: new THREE.Color(0.35, 0.3, 0.27),
  }), { cards: true });
  const cards = new THREE.SkinnedMesh(g, m);
  cards.name = 'hair_cards';
  cards.userData.companion = true;   // follows its hair mesh (morph.js does not pair it)
  cards.frustumCulled = false;
  cards.position.copy(mesh.position); cards.quaternion.copy(mesh.quaternion); cards.scale.copy(mesh.scale);
  mesh.parent.add(cards);
  cards.bind(mesh.skeleton, mesh.bindMatrix);
  return cards;
}
