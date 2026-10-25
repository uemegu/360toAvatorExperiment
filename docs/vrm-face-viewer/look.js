// Shading touches for realistic VRMs: wrapped (subsurface-like) skin, pore-scale bumps, wet eyes with lid shadow.
// Works on the materials three's GLTFLoader makes (MeshStandardMaterial); originals are kept so it can be switched off.
import * as THREE from 'three';

const SKIN = /body|skin|face/i;
const EYE = /eye(?!lash|brow)|high-poly|iris/i;

// Diffuse wrap per channel: red light reaches further past the terminator, as in skin.
const WRAP_DIFFUSE = `
	#ifdef SKIN_WRAP
		float nlw = dot( geometryNormal, directLight.direction );
		vec3 wrapW = vec3( 0.42, 0.2, 0.14 );
		vec3 wrapped = saturate( ( vec3( nlw ) + wrapW ) / ( 1.0 + wrapW ) );
		reflectedLight.directDiffuse += wrapped * directLight.color * BRDF_Lambert( material.diffuseContribution );
	#else
		reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseContribution );
	#endif`;
const DIFFUSE_LINE = 'reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseContribution );';

function skinChunk() {
  const chunk = THREE.ShaderChunk.lights_physical_pars_fragment;
  const at = chunk.lastIndexOf(DIFFUSE_LINE);
  if (at < 0) return null; // three changed: fall back to plain lighting
  return chunk.slice(0, at) + WRAP_DIFFUSE + chunk.slice(at + DIFFUSE_LINE.length);
}

// Tileable pore noise -> normal map (RGB) and roughness map (G).
function poreTextures(size = 512, seed = 7) {
  let s = seed;
  const rand = () => ((s = (s * 16807) % 2147483647) / 2147483647);
  const h = new Float32Array(size * size);
  const wrap = (v) => (v + size) % size;
  for (let k = 0; k < 2600; k++) {             // pores: small soft pits
    const cx = rand() * size, cy = rand() * size, r = 0.8 + rand() * 1.6, depth = 0.5 + rand() * 0.5;
    for (let dy = -4; dy <= 4; dy++) for (let dx = -4; dx <= 4; dx++) {
      const d2 = (dx * dx + dy * dy) / (r * r);
      if (d2 < 4) h[wrap(Math.round(cy) + dy) * size + wrap(Math.round(cx) + dx)] -= depth * Math.exp(-d2);
    }
  }
  const fine = new Float32Array(size * size);  // fine grain, smoothed once
  for (let i = 0; i < fine.length; i++) fine[i] = rand() - 0.5;
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    let a = 0;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) a += fine[wrap(y + dy) * size + wrap(x + dx)];
    h[y * size + x] += a * 0.06;
  }
  const nrm = new Uint8Array(size * size * 4);
  const rough = new Uint8Array(size * size * 4);
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    const i = y * size + x;
    const gx = h[y * size + wrap(x + 1)] - h[y * size + wrap(x - 1)];
    const gy = h[wrap(y + 1) * size + x] - h[wrap(y - 1) * size + x];
    const n = new THREE.Vector3(-gx * 1.5, -gy * 1.5, 1).normalize();
    nrm.set([(n.x * 0.5 + 0.5) * 255, (n.y * 0.5 + 0.5) * 255, (n.z * 0.5 + 0.5) * 255, 255], i * 4);
    const g = THREE.MathUtils.clamp(0.5 - h[i] * 0.12 + (rand() - 0.5) * 0.08, 0.3, 0.75) * 255;  // pits are duller
    rough.set([255, g, 255, 255], i * 4);
  }
  const tex = (data) => {
    const t = new THREE.DataTexture(data, size, size, THREE.RGBAFormat);
    t.wrapS = t.wrapT = THREE.RepeatWrapping;
    t.colorSpace = THREE.NoColorSpace;
    t.generateMipmaps = true;
    t.minFilter = THREE.LinearMipmapLinearFilter;
    t.magFilter = THREE.LinearFilter;
    t.needsUpdate = true;
    return t;
  };
  return { normal: tex(nrm), roughness: tex(rough) };
}

function physicalFrom(m) {
  const p = new THREE.MeshPhysicalMaterial();
  for (const k of ['name', 'map', 'color', 'transparent', 'opacity', 'alphaTest', 'side', 'depthWrite', 'roughness', 'metalness', 'normalMap', 'normalScale', 'roughnessMap', 'metalnessMap', 'alphaMap', 'emissive', 'emissiveMap', 'aoMap']) {
    if (m[k] !== undefined && m[k] !== null) p[k] = m[k]?.clone && !(m[k] instanceof THREE.Texture) ? m[k].clone() : m[k];
  }
  return p;
}

function skinMaterial(m, pores, repeat) {
  const p = physicalFrom(m);
  const chunk = skinChunk();
  p.normalMap = pores.normal.clone();
  p.normalMap.repeat.set(repeat, repeat);
  p.normalScale = new THREE.Vector2(0.18, 0.18);
  if (m.roughnessMap) {
    p.roughness = 1.0;               // the model's own map (e.g. glossy lips) wins; pores stay in the normal map
  } else {
    p.roughnessMap = pores.roughness.clone();
    p.roughnessMap.repeat.set(repeat, repeat);
    p.roughness = 1.0;               // the map carries 0.3-0.75
  }
  p.specularIntensity = 0.7;
  p.sheen = 0.25;                    // faint peach fuzz at grazing angles
  p.sheenRoughness = 0.8;
  p.sheenColor = new THREE.Color(1.0, 0.86, 0.8);
  if (chunk) {
    p.defines = { ...(p.defines || {}), SKIN_WRAP: '' };
    // the model's map marks hair painted on the scalp as fully rough (> 0.8): no skin sheen or highlight there
    const matte = Boolean(m.roughnessMap);
    if (matte) p.defines.MATTE_HAIR = '';
    p.onBeforeCompile = (shader) => {
      shader.fragmentShader = shader.fragmentShader
        .replace('#include <lights_physical_pars_fragment>', chunk)
        // glossy areas (lips) are smooth: fade the pore bumps out where the roughness map is low
        .replace('#include <normal_fragment_maps>', '#include <normal_fragment_maps>\nnormal = normalize( mix( nonPerturbedNormal, normal, smoothstep( 0.3, 0.55, roughnessFactor ) ) );')
        .replace('#include <lights_physical_fragment>', `#include <lights_physical_fragment>
#ifdef MATTE_HAIR
{
  float hair = smoothstep( 0.8, 0.95, roughnessFactor );
  material.sheenColor *= 1.0 - hair;
  material.specularColor *= 1.0 - 0.85 * hair;
  material.specularF90 *= 1.0 - 0.85 * hair;
}
#endif`);
    };
    p.customProgramCacheKey = () => (matte ? 'skin-wrap-matte' : 'skin-wrap');
  }
  return p;
}

// Eyeballs: darker top (upper lid shadow) and corners, a clear wet coat on top.
function eyeMaterial(m, geometry) {
  const p = physicalFrom(m);
  const pos = geometry.attributes.position;
  const sides = [[], []];
  for (let i = 0; i < pos.count; i++) sides[pos.getX(i) > 0 ? 0 : 1].push(i);
  const centre = (ids) => {
    const box = new THREE.Box3();
    for (const i of ids) box.expandByPoint(new THREE.Vector3().fromBufferAttribute(pos, i));
    return { c: box.getCenter(new THREE.Vector3()), r: Math.max(box.max.x - box.min.x, 1e-4) / 2 };
  };
  const [a, b] = sides.map(centre);
  p.color = new THREE.Color(1, 1, 1);
  p.emissive = new THREE.Color(0.07, 0.07, 0.075);   // whites stay white under the dim key light
  p.emissiveMap = p.map;
  // Light it as a perfect sphere: the cornea mesh's own normals smeared the highlight into a streak;
  // with sphere normals and a smooth surface the key light leaves one round catchlight.
  p.roughness = 0.1;
  p.onBeforeCompile = (shader) => {
    shader.uniforms.eyeA = { value: new THREE.Vector4(a.c.x, a.c.y, a.c.z, a.r) };
    shader.uniforms.eyeB = { value: new THREE.Vector4(b.c.x, b.c.y, b.c.z, b.r) };
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vEyeLocal;\nuniform vec4 eyeA;\nuniform vec4 eyeB;')
      .replace('#include <common>', '#include <common>\nvarying vec3 vEyeOff;')
      .replace('#include <begin_vertex>', '#include <begin_vertex>\nvEyeLocal = position;')
      // offset from the eyeball centre, carried to view space like a direction (skinned): it is linear across
      // a triangle, so normalising it per pixel gives an exact sphere normal and a round catchlight
      .replace('#include <skinning_vertex>', `#include <skinning_vertex>
        vec3 eOff = position - ( position.x > 0.0 ? eyeA.xyz : eyeB.xyz );
        #ifdef USE_SKINNING
          vec4 eS = bindMatrix * vec4( eOff, 0.0 );
          vec4 eK = boneMatX * eS * skinWeight.x + boneMatY * eS * skinWeight.y + boneMatZ * eS * skinWeight.z + boneMatW * eS * skinWeight.w;
          eOff = ( bindMatrixInverse * eK ).xyz;
        #endif
        vEyeOff = ( modelViewMatrix * vec4( eOff, 0.0 ) ).xyz;`);
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vEyeLocal;\nvarying vec3 vEyeOff;\nuniform vec4 eyeA;\nuniform vec4 eyeB;')
      .replace('#include <normal_fragment_begin>', '#include <normal_fragment_begin>\nnormal = normalize( vEyeOff );\nnonPerturbedNormal = normal;')
      // one round catchlight, upper right of the pupil, as a photographer's light leaves it
      .replace('#include <emissivemap_fragment>', `#include <emissivemap_fragment>
        vec3 cV = normalize( vViewPosition );
        vec3 cL = normalize( vec3( 0.45, 0.5, 1.0 ) );
        float cat = pow( max( dot( normal, normalize( cL + cV ) ), 0.0 ), 450.0 );
        totalEmissiveRadiance += vec3( 1.6 ) * smoothstep( 0.35, 0.6, cat );`)
      // the room's panel lights reflected as square glints: keep only a little of the environment's reflection
      .replace('#include <lights_fragment_maps>', '#include <lights_fragment_maps>\n#if defined( RE_IndirectSpecular )\nradiance *= 0.03;\n#endif')
      .replace('#include <map_fragment>', `#include <map_fragment>
        vec4 eyeC = vEyeLocal.x > 0.0 ? eyeA : eyeB;
        vec3 ed = ( vEyeLocal - eyeC.xyz ) / eyeC.w;
        float lid = smoothstep( -0.35, 0.35, ed.y );
        float corner = smoothstep( 0.3, 0.8, abs( ed.x ) );
        float below = smoothstep( 0.3, 0.7, -ed.y );
        diffuseColor.rgb *= clamp( 1.0 - 0.3 * lid - 0.12 * corner - 0.06 * below, 0.45, 1.0 );`);
  };
  p.customProgramCacheKey = () => 'eye-lid';
  return p;
}

export function createLook() {
  let pores = null;
  const originals = new Map();   // mesh -> original material
  const upgraded = new Map();    // mesh -> { skin?, eye? }

  function apply(root, { skin = true, eyes = true } = {}) {
    pores ||= poreTextures();
    root.traverse((o) => {
      if (!o.isMesh || Array.isArray(o.material)) return;
      if (!originals.has(o)) originals.set(o, o.material);
      const orig = originals.get(o);
      const name = `${o.name} ${orig.name}`;
      const cache = upgraded.get(o) || {};
      let next = orig;
      if (skin && SKIN.test(name) && !EYE.test(name)) next = cache.skin ||= skinMaterial(orig, pores, 18);
      if (eyes && EYE.test(name)) next = cache.eye ||= eyeMaterial(orig, o.geometry);
      upgraded.set(o, cache);
      o.material = next;
    });
  }

  function reset() {
    originals.clear();
    upgraded.clear();
  }
  return { apply, reset };
}
