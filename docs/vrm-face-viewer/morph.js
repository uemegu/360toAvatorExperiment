// Shape-shift between the shown VRM and a second VRM with the same meshes (e.g. two faces fitted onto the same MPFB base).
// Vertices start at different times (a smooth noise over the rest position) and bulge along the normal while they move;
// textures are mixed on the GPU with a noise mask whose moving edge glows a little.
import * as THREE from 'three';
import { VRMUtils } from '@pixiv/three-vrm';

const MAP_KEYS = ['map', 'shadeMultiplyTexture', 'roughnessMap', 'metalnessMap', 'emissiveMap'];
const SPREAD = 0.55;   // share of the transition used by the per-vertex start delay

const meshesOf = (root) => { const out = []; root.traverse((o) => { if (o.isMesh) out.push(o); }); return out; };
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

export function createMorph(loader, renderer) {
  let pairs = [];
  let maps = [];
  let t = 0, target = 0, time = 0;
  let duration = 3, wobble = 1;
  const quadScene = new THREE.Scene();
  const quadCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.ShaderMaterial(MIX_SHADER));
  quadScene.add(quad);

  function clear() {
    for (const m of maps) { m.mat[m.key] = m.a; m.rt.dispose(); }
    pairs = []; maps = []; t = 0; target = 0;
  }

  async function prepare(vrmA, urlB) {
    clear();
    const gltf = await loader.loadAsync(urlB);
    VRMUtils.removeUnnecessaryVertices(gltf.scene);   // same processing as the shown model, so vertex order matches
    VRMUtils.combineSkeletons(gltf.scene);
    const A = meshesOf(vrmA.scene), B = meshesOf(gltf.scene);
    let skipped = 0;
    A.forEach((ma, i) => {
      const mb = B[i];
      const pa = ma.geometry.attributes.position, pb = mb?.geometry.attributes.position;
      if (!pb || pa.count !== pb.count) { skipped++; return; }
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
    apply(0);
    return { meshes: pairs.length, moving: pairs.filter((p) => p.moving).length, textures: maps.length, skipped };
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
    get ready() { return pairs.length > 0; },
    get value() { return t; },
    get target() { return target; },
    toggle() { target = target > 0.5 ? 0 : 1; return target; },
    jump(v) { t = target = v; apply(t); },
    setDuration(s) { duration = s; },
    setWobble(w) { wobble = w; },
    update(dt) {
      if (!pairs.length) return;
      time += dt;
      if (t === target) return;
      t = target > t ? Math.min(target, t + dt / duration) : Math.max(target, t - dt / duration);
      apply(t);
    },
  };
}
