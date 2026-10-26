// Shape-shift between the shown VRM and a second VRM with the same meshes (e.g. two faces fitted onto the same MPFB base).
// Vertices start at different times (a smooth noise over the rest position) and bulge along the normal while they move;
// textures are mixed on the GPU with a noise mask whose moving edge glows a little.
// Meshes without a same-named, same-sized partner (hair, clothes) are swapped instead: the shown one dissolves and the
// other VRM's one, moved onto the shown skeleton (its own extra bones and spring bones come along), dissolves in,
// with the same noise and timing as the vertices so the swap front runs with the morph.
import * as THREE from 'three';
import { VRMUtils } from '@pixiv/three-vrm';

const MAP_KEYS = ['map', 'shadeMultiplyTexture', 'roughnessMap', 'metalnessMap', 'emissiveMap'];
const SPREAD = 0.55;   // share of the transition used by the per-vertex start delay

// meshes another effect hangs on a mesh (userData.companion, e.g. hair cards) are not paired, they follow their mesh
const meshesOf = (root) => { const out = []; root.traverse((o) => { if (o.isMesh && !o.userData.companion) out.push(o); }); return out; };
const smooth = (u) => u * u * (3 - 2 * u);
// 0..1, smooth blobs a few cm across; the same formula is used for UVs in the texture shader
const noise = (x, y, z) => 0.5 + 0.25 * Math.sin(x * 23 + Math.sin(y * 17) * 1.7) + 0.25 * Math.sin(y * 29 + Math.sin(z * 19 + x * 7) * 1.3);

// the two files carry their own copies of shared textures (clothes, teeth...); compare small thumbnails
function sameImage(a, b) {
  if (a === b) return true;
  if (a.width !== b.width || a.height !== b.height) return false;
  const thumb = (img) => {
    const c = document.createElement('canvas'); c.width = c.height = 48;
    const g = c.getContext('2d', { willReadFrequently: true }); g.drawImage(img, 0, 0, 48, 48);
    return g.getImageData(0, 0, 48, 48).data;
  };
  const x = thumb(a), y = thumb(b);
  for (let i = 0; i < x.length; i++) if (Math.abs(x[i] - y[i]) > 2) return false;
  return true;
}

const MIX_SHADER = {
  uniforms: { a: { value: null }, b: { value: null }, t: { value: 0 } },
  vertexShader: 'varying vec2 vUv; void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }',
  fragmentShader: `
    uniform sampler2D a, b; uniform float t; varying vec2 vUv;
    float n(vec2 p) { return 0.5 + 0.25 * sin(p.x * 23.0 + sin(p.y * 17.0) * 1.7) + 0.25 * sin(p.y * 29.0 + sin(p.x * 13.0) * 1.3); }
    void main() {
      float d = n(vUv) * ${SPREAD.toFixed(2)};
      float u = clamp((t - d) / ${(1 - SPREAD).toFixed(2)}, 0.0, 1.0);
      vec4 c = mix(texture2D(a, vUv), texture2D(b, vUv), smoothstep(0.0, 1.0, u));
      float edge = pow(sin(3.14159 * u), 12.0);
      c.rgb += vec3(0.02, 0.08, 0.11) * edge;
      gl_FragColor = c;
    }`,
};

// A swapped mesh rides the morphing body: each vertex carries the offset of its nearest morphing vertex (carryDelta,
// applied as that vertex moves - towards the other body when dissolving out, from the shown one when dissolving in)
// and is discarded / revealed when that vertex is half way (carryDelay), so two hairs over one patch of scalp swap
// together and the growing head never pokes through. The edge glows a little.
function dissolvable(mat, incoming, t) {
  if (mat.userData.dissolve) { mat.userData.dissolve.uDissolveIn.value = incoming ? 1 : 0; return; }
  const u = { uDissolveT: t, uDissolveIn: { value: incoming ? 1 : 0 } };
  mat.userData.dissolve = u;
  const prev = mat.onBeforeCompile, key = mat.customProgramCacheKey();
  const progress = (delay) => `clamp((uDissolveT - ${delay}) / ${(1 - SPREAD).toFixed(2)}, 0.0, 1.0)`;
  mat.onBeforeCompile = (sh, r) => {
    prev.call(mat, sh, r);
    Object.assign(sh.uniforms, u);
    const decl = '#include <common>\nvarying float vCarryDelay;\nuniform float uDissolveT;\nuniform float uDissolveIn;';
    sh.vertexShader = sh.vertexShader.replace('#include <common>', `${decl}\nattribute vec3 carryDelta;\nattribute float carryDelay;`)
      .replace('#include <begin_vertex>', `#include <begin_vertex>
{
  float cu = ${progress('carryDelay')};
  transformed += carryDelta * ( cu * cu * ( 3.0 - 2.0 * cu ) - uDissolveIn );
  vCarryDelay = carryDelay;
}`);
    sh.fragmentShader = sh.fragmentShader.replace('#include <common>', decl)
      .replace('#include <clipping_planes_fragment>', `#include <clipping_planes_fragment>
float dsU = ${progress('vCarryDelay')};
float dsV = uDissolveIn > 0.5 ? dsU - 0.3 : 0.7 - dsU;   // both show round the middle, so nothing is bared between them
if ( dsV < 0.0 ) discard;`)
      .replace('#include <emissivemap_fragment>', '#include <emissivemap_fragment>\ntotalEmissiveRadiance += vec3( 0.03, 0.12, 0.16 ) * ( 1.0 - smoothstep( 0.0, 0.012, dsV ) );');
  };
  mat.customProgramCacheKey = () => `${key}|dissolve`;
  mat.needsUpdate = true;
}

// nearest of many points, through a uniform grid
function nearestFinder(pts, cell = 0.01) {
  const grid = new Map(), n = pts.length / 3;
  const key = (x, y, z) => `${x},${y},${z}`;
  for (let i = 0; i < n; i++) {
    const k = key(Math.floor(pts[3 * i] / cell), Math.floor(pts[3 * i + 1] / cell), Math.floor(pts[3 * i + 2] / cell));
    if (!grid.has(k)) grid.set(k, []);
    grid.get(k).push(i);
  }
  return (x, y, z) => {
    const cx = Math.floor(x / cell), cy = Math.floor(y / cell), cz = Math.floor(z / cell);
    let best = -1, bd = Infinity;
    for (let r = 0; r <= 12 && best < 0; r++) {   // grow the shell until something is found, then one more ring
      for (let dx = -r; dx <= r; dx++) for (let dy = -r; dy <= r; dy++) for (let dz = -r; dz <= r; dz++) {
        if (Math.max(Math.abs(dx), Math.abs(dy), Math.abs(dz)) !== r) continue;
        for (const i of grid.get(key(cx + dx, cy + dy, cz + dz)) || []) {
          const d = (pts[3 * i] - x) ** 2 + (pts[3 * i + 1] - y) ** 2 + (pts[3 * i + 2] - z) ** 2;
          if (d < bd) { bd = d; best = i; }
        }
      }
      if (best >= 0 && r < 12) for (let dx = -r - 1; dx <= r + 1; dx++) for (let dy = -r - 1; dy <= r + 1; dy++) for (let dz = -r - 1; dz <= r + 1; dz++) {
        if (Math.max(Math.abs(dx), Math.abs(dy), Math.abs(dz)) !== r + 1) continue;
        for (const i of grid.get(key(cx + dx, cy + dy, cz + dz)) || []) {
          const d = (pts[3 * i] - x) ** 2 + (pts[3 * i + 1] - y) ** 2 + (pts[3 * i + 2] - z) ** 2;
          if (d < bd) { bd = d; best = i; }
        }
      }
    }
    return best;
  };
}

export function createMorph(loader, renderer) {
  let pairs = [];
  let maps = [];
  let swaps = [];          // { mesh, incoming, extra: companion meshes }
  let moved = [];          // the other VRM's bones / colliders re-homed onto the shown skeleton
  let springsB = null;     // the other VRM's spring bones (its hair), updated after the shown VRM
  const dissolveT = { value: 0 };
  let t = 0, target = 0, time = 0;
  let duration = 3, wobble = 1;
  const quadScene = new THREE.Scene();
  const quadCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.ShaderMaterial(MIX_SHADER));
  quadScene.add(quad);

  function clear() {
    for (const m of maps) { m.mat[m.key] = m.a; m.rt.dispose(); }
    for (const s of swaps) if (s.incoming) for (const o of [s.mesh, ...s.extra]) o.parent?.remove(o);
    for (const o of moved) o.parent?.remove(o);
    pairs = []; maps = []; swaps = []; moved = []; springsB = null; t = 0; target = 0;
    dissolveT.value = 0;
  }

  // companions(mesh, incoming) -> { meshes, materials } that should dissolve with a swapped mesh (e.g. its hair cards)
  async function prepare(vrmA, urlB, { companions } = {}) {
    clear();
    const gltf = await loader.loadAsync(urlB);
    const vrmB = gltf.userData.vrm;
    VRMUtils.removeUnnecessaryVertices(gltf.scene);   // same processing as the shown model, so vertex order matches
    VRMUtils.combineSkeletons(gltf.scene);
    gltf.scene.updateMatrixWorld(true);
    const A = meshesOf(vrmA.scene), B = meshesOf(gltf.scene);
    const byName = new Map(B.map((m) => [m.name, m])), paired = new Set(), outgoing = [];
    A.forEach((ma) => {
      const mb = byName.get(ma.name);
      const pa = ma.geometry.attributes.position, pb = mb?.geometry.attributes.position;
      if (!pb || pa.count !== pb.count) { outgoing.push(ma); return; }
      paired.add(mb);
      const base = Float32Array.from(pa.array), delta = new Float32Array(pa.array.length);
      for (let k = 0; k < delta.length; k++) delta[k] = pb.array[k] - base[k];
      const na = ma.geometry.attributes.normal, nb = mb.geometry.attributes.normal;
      const nBase = Float32Array.from(na.array), nDelta = new Float32Array(na.array.length);
      for (let k = 0; k < nDelta.length; k++) nDelta[k] = nb.array[k] - nBase[k];
      const delay = new Float32Array(pa.count), phase = new Float32Array(pa.count);
      for (let v = 0; v < pa.count; v++) {
        const x = base[v * 3], y = base[v * 3 + 1], z = base[v * 3 + 2];
        delay[v] = noise(x, y, z) * SPREAD;
        phase[v] = noise(z * 1.7, x * 1.3, y * 2.1) * 6.283;
      }
      const moving = delta.some((d) => Math.abs(d) > 1e-5);
      pairs.push({ mesh: ma, base, delta, nBase, nDelta, delay, phase, moving });
      // textures that differ between the two
      const matsA = [].concat(ma.material), matsB = [].concat(mb.material);
      matsA.forEach((mat, j) => {
        const other = matsB[j];
        for (const key of MAP_KEYS) {
          const a = mat[key], b = other?.[key];
          if (!a?.image || !b?.image || sameImage(a.image, b.image) || maps.some((m) => m.mat === mat && m.key === key)) continue;
          const rt = new THREE.WebGLRenderTarget(a.image.width, a.image.height, { generateMipmaps: true, minFilter: THREE.LinearMipmapLinearFilter, magFilter: THREE.LinearFilter });
          rt.texture.colorSpace = a.colorSpace;
          for (const p of ['flipY', 'wrapS', 'wrapT', 'channel', 'anisotropy']) rt.texture[p] = a[p];
          rt.texture.offset.copy(a.offset); rt.texture.repeat.copy(a.repeat);
          maps.push({ mat, key, a, b, rt, last: -1 });
        }
      });
    });
    const incoming = B.filter((m) => !paired.has(m) && m.isSkinnedMesh);
    if (incoming.length) adopt(vrmA, vrmB, incoming);
    // carriers: every vertex of the morphing meshes, where it starts (shown body), ends (other body), and when it moves
    const moving = pairs.filter((p) => p.moving);
    const from = new Float32Array(moving.reduce((n, p) => n + p.base.length, 0)), to = new Float32Array(from.length);
    const cDelta = new Float32Array(from.length), cDelay = new Float32Array(from.length / 3);
    let o = 0;
    for (const p of moving) {
      from.set(p.base, o); cDelta.set(p.delta, o); cDelay.set(p.delay, o / 3);
      for (let k = 0; k < p.base.length; k++) to[o + k] = p.base[k] + p.delta[k];
      o += p.base.length;
    }
    const nearFrom = cDelay.length ? nearestFinder(from) : null, nearTo = cDelay.length ? nearestFinder(to) : null;
    const carry = (geo, inc) => {
      const pos = geo.attributes.position, n = pos.count, d = new Float32Array(n * 3), w = new Float32Array(n);
      for (let v = 0; v < n; v++) {
        const x = pos.getX(v), y = pos.getY(v), z = pos.getZ(v);
        const k = (inc ? nearTo : nearFrom)?.(x, y, z) ?? -1;
        if (k < 0) { w[v] = noise(x, y, z) * SPREAD; continue; }
        d[3 * v] = cDelta[3 * k]; d[3 * v + 1] = cDelta[3 * k + 1]; d[3 * v + 2] = cDelta[3 * k + 2];
        w[v] = cDelay[k];
      }
      geo.setAttribute('carryDelta', new THREE.BufferAttribute(d, 3));
      geo.setAttribute('carryDelay', new THREE.BufferAttribute(w, 1));
    };
    const swap = (mesh, inc) => {
      const c = companions?.(mesh, inc) || {};
      for (const m of [mesh, ...(c.meshes || [])]) carry(m.geometry, inc);
      for (const m of [mesh, ...(c.meshes || [])].flatMap((x) => [].concat(x.material)).concat(c.materials || [])) dissolvable(m, inc, dissolveT);
      swaps.push({ mesh, incoming: inc, extra: c.meshes || [] });
    };
    outgoing.forEach((m) => swap(m, false));
    incoming.forEach((m) => swap(m, true));
    apply(0);
    return { meshes: pairs.length, moving: pairs.filter((p) => p.moving).length, textures: maps.length, out: outgoing.length, in: incoming.length };
  }

  // Put the other VRM's unpaired meshes on the shown skeleton. Shared bones (same name) skin with the shown model's
  // rest, so at rest each vertex sits where it was in its own file - where the morphed body ends up. Bones only the
  // other VRM has (its hair chains) and its spring colliders move under the shown bone of their nearest named
  // ancestor, keeping their rest world transform, and its spring bones keep running on them.
  function adopt(vrmA, vrmB, meshes) {
    const rest = new Map();   // shown bone -> rest world matrix, from the inverse bind matrices
    vrmA.scene.traverse((o) => {
      if (!o.isSkinnedMesh) return;
      o.skeleton.bones.forEach((b, i) => { if (!rest.has(b)) rest.set(b, o.skeleton.boneInverses[i].clone().invert()); });
    });
    const boneA = new Map([...rest.keys()].map((b) => [b.name, b]));
    const done = new Set();
    const rehome = (o) => {
      if (done.has(o)) return true;
      if (o.parent?.isBone && !boneA.has(o.parent.name)) { done.add(o); return rehome(o.parent); }   // moves with its parent
      let anc = o.parent;
      while (anc && !boneA.has(anc.name)) anc = anc.parent;
      if (!anc) return false;
      const a = boneA.get(anc.name);
      rest.get(a).clone().invert().multiply(o.matrixWorld).decompose(o.position, o.quaternion, o.scale);
      a.add(o);
      done.add(o); moved.push(o);
      return true;
    };
    for (const mb of meshes) {
      const sk = mb.skeleton, bones = [], inverses = [];
      sk.bones.forEach((b, i) => {
        const a = boneA.get(b.name);
        if (a) { bones.push(a); inverses.push(rest.get(a).clone().invert()); return; }
        rehome(b);
        bones.push(b); inverses.push(sk.boneInverses[i].clone());
      });
      mb.matrixWorld.decompose(mb.position, mb.quaternion, mb.scale);
      vrmA.scene.add(mb);
      mb.bind(new THREE.Skeleton(bones, inverses), mb.bindMatrix.clone());
      mb.frustumCulled = false;
    }
    const sb = vrmB?.springBoneManager;
    if (sb?.joints.size) {
      for (const c of sb.colliders) rehome(c);
      vrmA.scene.updateMatrixWorld(true);
      sb.setInitState();
      springsB = sb;
    }
  }

  function apply(tt) {
    const amp = 0.006 * wobble;   // metres of bulge at the moving front
    for (const p of pairs) {
      const pos = p.mesh.geometry.attributes.position, nor = p.mesh.geometry.attributes.normal;
      const P = pos.array, N = nor.array;
      if (!p.moving) continue;
      for (let v = 0, n = p.delay.length; v < n; v++) {
        const u = Math.min(1, Math.max(0, (tt - p.delay[v]) / (1 - SPREAD)));
        const e = smooth(u);
        const k = v * 3;
        const nx = p.nBase[k] + p.nDelta[k] * e, ny = p.nBase[k + 1] + p.nDelta[k + 1] * e, nz = p.nBase[k + 2] + p.nDelta[k + 2] * e;
        const bulge = amp * Math.sin(Math.PI * u) * (0.55 + 0.45 * Math.sin(time * 9 + p.phase[v]));
        P[k] = p.base[k] + p.delta[k] * e + nx * bulge;
        P[k + 1] = p.base[k + 1] + p.delta[k + 1] * e + ny * bulge;
        P[k + 2] = p.base[k + 2] + p.delta[k + 2] * e + nz * bulge;
        N[k] = nx; N[k + 1] = ny; N[k + 2] = nz;
      }
      pos.needsUpdate = true; nor.needsUpdate = true;
    }
    dissolveT.value = tt;
    const prevTarget = renderer.getRenderTarget();
    for (const m of maps) {
      if (tt <= 0) { m.mat[m.key] = m.a; m.last = tt; continue; }
      if (tt >= 1) { m.mat[m.key] = m.b; m.last = tt; continue; }
      if (m.last === tt) continue;
      quad.material.uniforms.a.value = m.a; quad.material.uniforms.b.value = m.b; quad.material.uniforms.t.value = tt;
      renderer.setRenderTarget(m.rt);
      renderer.render(quadScene, quadCam);
      m.mat[m.key] = m.rt.texture;
      m.last = tt;
    }
    renderer.setRenderTarget(prevTarget);
  }

  return {
    prepare,
    clear,
    get ready() { return pairs.length > 0 || swaps.length > 0; },
    get value() { return t; },
    get target() { return target; },
    toggle() { target = target > 0.5 ? 0 : 1; return target; },
    jump(v) { t = target = v; apply(t); },
    setDuration(s) { duration = s; },
    setWobble(w) { wobble = w; },
    // after the shown VRM's own update (its bones are posed): the adopted hair's spring bones
    updateSprings(dt) { springsB?.update(dt); },
    update(dt) {
      if (!pairs.length && !swaps.length) return;
      time += dt;
      if (t === target) return;
      t = target > t ? Math.min(target, t + dt / duration) : Math.max(target, t - dt / duration);
      apply(t);
    },
  };
}
