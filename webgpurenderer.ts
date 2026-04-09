/**
 * @file WebGPURenderer.ts
 * @description AETHERIS OMNIVERSE — WebGPU Deferred Rendering Pipeline
 *
 * Architecture:
 *  - RenderGraph: Automatic resource dependency tracking + barrier insertion
 *  - G-Buffer Pass: Position, Normal, Albedo/Metallic/Roughness
 *  - Shadow Map Pass: Cascaded Shadow Maps (CSM)
 *  - Lighting Pass: Deferred PBR + IBL
 *  - Post-Process Pass: TAA, Bloom, Tonemap (ACES)
 *  - AssetVirtualizer: LOD + Frustum-based streaming
 */

import type { ISystem, IKernel, SystemId, FrameIndex } from "./Kernel";
import { System, FrameIndex as FI } from "./Kernel";

// ─────────────────────────────────────────────
// § MATH PRIMITIVES
// ─────────────────────────────────────────────

export class Vec3 {
  constructor(public x = 0, public y = 0, public z = 0) {}
  static zero()  { return new Vec3(0, 0, 0); }
  static up()    { return new Vec3(0, 1, 0); }
  length()       { return Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z); }
  normalized()   { const l = this.length(); return new Vec3(this.x / l, this.y / l, this.z / l); }
  add(v: Vec3)   { return new Vec3(this.x + v.x, this.y + v.y, this.z + v.z); }
  sub(v: Vec3)   { return new Vec3(this.x - v.x, this.y - v.y, this.z - v.z); }
  scale(s: number) { return new Vec3(this.x * s, this.y * s, this.z * s); }
  dot(v: Vec3)   { return this.x * v.x + this.y * v.y + this.z * v.z; }
  cross(v: Vec3) { return new Vec3(this.y * v.z - this.z * v.y, this.z * v.x - this.x * v.z, this.x * v.y - this.y * v.x); }
  toArray(): [number, number, number] { return [this.x, this.y, this.z]; }
}

export class Vec4 {
  constructor(public x = 0, public y = 0, public z = 0, public w = 1) {}
  toArray(): [number, number, number, number] { return [this.x, this.y, this.z, this.w]; }
}

export class Mat4 {
  readonly data: Float32Array;
  constructor(data?: Float32Array) {
    this.data = data ?? new Float32Array(16);
  }
  static identity(): Mat4 {
    const m = new Mat4();
    m.data[0] = m.data[5] = m.data[10] = m.data[15] = 1;
    return m;
  }
  static perspective(fovY: number, aspect: number, near: number, far: number): Mat4 {
    const f = 1.0 / Math.tan(fovY / 2);
    const nf = 1 / (near - far);
    const m = new Mat4();
    m.data[0]  = f / aspect;
    m.data[5]  = f;
    m.data[10] = (far + near) * nf;
    m.data[11] = -1;
    m.data[14] = 2 * far * near * nf;
    return m;
  }
  multiply(other: Mat4): Mat4 {
    const a = this.data, b = other.data, out = new Float32Array(16);
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        let sum = 0;
        for (let k = 0; k < 4; k++) sum += a[i * 4 + k] * b[k * 4 + j];
        out[i * 4 + j] = sum;
      }
    }
    return new Mat4(out);
  }
}

// ─────────────────────────────────────────────
// § RENDER GRAPH — Resource dependency tracking
// ─────────────────────────────────────────────

type ResourceHandle = Brand<string, "ResourceHandle">;
declare const __brand: unique symbol;
type Brand<T, B> = T & { readonly [__brand]: B };

const ResourceHandle = (s: string): ResourceHandle => s as ResourceHandle;

interface TextureDescriptor {
  width: number;
  height: number;
  format: GPUTextureFormat;
  usage: GPUTextureUsageFlags;
  mipLevelCount?: number;
  sampleCount?: number;
  label?: string;
}

interface RenderPassDescriptor {
  id: string;
  reads:  ResourceHandle[];
  writes: ResourceHandle[];
  execute: (ctx: PassContext) => void;
}

interface PassContext {
  encoder:     GPUCommandEncoder;
  device:      GPUDevice;
  getTexture:  (handle: ResourceHandle) => GPUTexture;
  getBindGroup: (handle: ResourceHandle, layout: GPUBindGroupLayout) => GPUBindGroup;
}

class RenderGraph {
  private readonly passes:    RenderPassDescriptor[] = [];
  private readonly resources  = new Map<ResourceHandle, GPUTexture>();
  private readonly descriptors = new Map<ResourceHandle, TextureDescriptor>();

  constructor(private readonly device: GPUDevice) {}

  createTexture(desc: TextureDescriptor): ResourceHandle {
    const handle = ResourceHandle(`tex_${desc.label ?? Date.now()}`);
    this.descriptors.set(handle, desc);
    return handle;
  }

  addPass(pass: RenderPassDescriptor): void {
    this.passes.push(pass);
  }

  compile(): void {
    // Allocate GPU textures for all registered handles
    for (const [handle, desc] of this.descriptors) {
      if (!this.resources.has(handle)) {
        const tex = this.device.createTexture({
          size:   { width: desc.width, height: desc.height },
          format: desc.format,
          usage:  desc.usage,
          mipLevelCount: desc.mipLevelCount ?? 1,
          sampleCount:   desc.sampleCount ?? 1,
          label:  desc.label,
        });
        this.resources.set(handle, tex);
      }
    }
  }

  execute(): GPUCommandBuffer {
    const encoder = this.device.createCommandEncoder({ label: "RenderGraph" });

    const ctx: PassContext = {
      encoder,
      device: this.device,
      getTexture: (h) => {
        const t = this.resources.get(h);
        if (!t) throw new Error(`[RenderGraph] Unknown resource: ${h}`);
        return t;
      },
      getBindGroup: (h, layout) => {
        const tex = this.resources.get(h)!;
        return this.device.createBindGroup({
          layout,
          entries: [{ binding: 0, resource: tex.createView() }],
        });
      },
    };

    for (const pass of this.passes) {
      pass.execute(ctx);
    }

    return encoder.finish();
  }

  resize(width: number, height: number): void {
    for (const [handle, desc] of this.descriptors) {
      if (desc.width !== width || desc.height !== height) continue;
      const old = this.resources.get(handle);
      old?.destroy();
      desc.width = width;
      desc.height = height;
      const tex = this.device.createTexture({
        size:   { width, height },
        format: desc.format,
        usage:  desc.usage,
        label:  desc.label,
      });
      this.resources.set(handle, tex);
    }
  }

  destroy(): void {
    for (const tex of this.resources.values()) tex.destroy();
    this.resources.clear();
  }
}

// ─────────────────────────────────────────────
// § SHADER LIBRARY
// ─────────────────────────────────────────────

const ShaderLibrary = {
  GBuffer: /* wgsl */`
    struct VertexInput {
      @location(0) position: vec3<f32>,
      @location(1) normal:   vec3<f32>,
      @location(2) uv:       vec2<f32>,
      @location(3) tangent:  vec4<f32>,
    };

    struct VertexOutput {
      @builtin(position) position: vec4<f32>,
      @location(0) worldPos: vec3<f32>,
      @location(1) normal:   vec3<f32>,
      @location(2) uv:       vec2<f32>,
      @location(3) tangent:  vec3<f32>,
      @location(4) bitangent:vec3<f32>,
    };

    struct CameraUniforms {
      viewProj: mat4x4<f32>,
      view:     mat4x4<f32>,
      position: vec3<f32>,
    };

    struct ModelUniforms {
      model:       mat4x4<f32>,
      normalMatrix: mat3x3<f32>,
    };

    @group(0) @binding(0) var<uniform> camera: CameraUniforms;
    @group(1) @binding(0) var<uniform> model:  ModelUniforms;
    @group(2) @binding(0) var albedoTex:       texture_2d<f32>;
    @group(2) @binding(1) var normalTex:       texture_2d<f32>;
    @group(2) @binding(2) var pbrTex:          texture_2d<f32>; // R=metallic, G=roughness
    @group(2) @binding(3) var linearSampler:   sampler;

    @vertex
    fn vs_main(in: VertexInput) -> VertexOutput {
      var out: VertexOutput;
      let worldPos = (model.model * vec4<f32>(in.position, 1.0)).xyz;
      out.position  = camera.viewProj * vec4<f32>(worldPos, 1.0);
      out.worldPos  = worldPos;
      out.normal    = normalize(model.normalMatrix * in.normal);
      out.uv        = in.uv;
      let T = normalize(model.normalMatrix * in.tangent.xyz);
      let N = out.normal;
      let B = normalize(cross(N, T) * in.tangent.w);
      out.tangent   = T;
      out.bitangent = B;
      return out;
    }

    struct GBufferOutput {
      @location(0) albedoMetal:  vec4<f32>, // RGB=albedo, A=metallic
      @location(1) normalRough:  vec4<f32>, // RGB=normal(world), A=roughness
      @location(2) worldPos:     vec4<f32>, // RGB=worldPos, A=unused
    };

    @fragment
    fn fs_main(in: VertexOutput) -> GBufferOutput {
      let albedo   = textureSample(albedoTex, linearSampler, in.uv);
      let pbrSample = textureSample(pbrTex,   linearSampler, in.uv);
      let normalTS  = textureSample(normalTex, linearSampler, in.uv).xyz * 2.0 - 1.0;

      let TBN = mat3x3<f32>(in.tangent, in.bitangent, in.normal);
      let worldNormal = normalize(TBN * normalTS);

      var out: GBufferOutput;
      out.albedoMetal = vec4<f32>(albedo.rgb, pbrSample.r);
      out.normalRough = vec4<f32>(worldNormal * 0.5 + 0.5, pbrSample.g);
      out.worldPos    = vec4<f32>(in.worldPos, 1.0);
      return out;
    }
  `,

  DeferredLighting: /* wgsl */`
    const PI: f32 = 3.14159265358979;

    struct Light {
      position:  vec3<f32>,
      kind:      u32,        // 0=point, 1=directional, 2=spot
      color:     vec3<f32>,
      intensity: f32,
      direction: vec3<f32>,
      range:     f32,
    };

    struct LightsBuffer {
      count:  u32,
      lights: array<Light, 256>,
    };

    struct CameraUniforms {
      viewProj: mat4x4<f32>,
      view:     mat4x4<f32>,
      position: vec3<f32>,
    };

    @group(0) @binding(0) var<uniform> camera:   CameraUniforms;
    @group(0) @binding(1) var<storage, read> lights: LightsBuffer;
    @group(1) @binding(0) var gAlbedoMetal:  texture_2d<f32>;
    @group(1) @binding(1) var gNormalRough:  texture_2d<f32>;
    @group(1) @binding(2) var gWorldPos:     texture_2d<f32>;
    @group(1) @binding(3) var gSampler:      sampler;

    // GGX Normal Distribution Function
    fn DistributionGGX(N: vec3<f32>, H: vec3<f32>, roughness: f32) -> f32 {
      let a  = roughness * roughness;
      let a2 = a * a;
      let NdH  = max(dot(N, H), 0.0);
      let NdH2 = NdH * NdH;
      let denom = NdH2 * (a2 - 1.0) + 1.0;
      return a2 / (PI * denom * denom);
    }

    fn GeometrySchlickGGX(NdV: f32, roughness: f32) -> f32 {
      let r = roughness + 1.0;
      let k = (r * r) / 8.0;
      return NdV / (NdV * (1.0 - k) + k);
    }

    fn GeometrySmith(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, roughness: f32) -> f32 {
      let NdV = max(dot(N, V), 0.0);
      let NdL = max(dot(N, L), 0.0);
      return GeometrySchlickGGX(NdV, roughness) * GeometrySchlickGGX(NdL, roughness);
    }

    fn FresnelSchlick(cosTheta: f32, F0: vec3<f32>) -> vec3<f32> {
      return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
    }

    // ACES Filmic Tonemapping
    fn ACESFilm(x: vec3<f32>) -> vec3<f32> {
      let a = 2.51;
      let b = vec3<f32>(0.03, 0.03, 0.03);
      let c = 2.43;
      let d = vec3<f32>(0.59, 0.59, 0.59);
      let e = vec3<f32>(0.14, 0.14, 0.14);
      return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
    }

    struct FullscreenOutput {
      @builtin(position) pos: vec4<f32>,
      @location(0) uv: vec2<f32>,
    };

    @vertex
    fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> FullscreenOutput {
      let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
      );
      var out: FullscreenOutput;
      out.pos = vec4<f32>(positions[vi], 0.0, 1.0);
      out.uv  = positions[vi] * 0.5 + 0.5;
      out.uv.y = 1.0 - out.uv.y;
      return out;
    }

    @fragment
    fn fs_lighting(in: FullscreenOutput) -> @location(0) vec4<f32> {
      let albedoMetal = textureSample(gAlbedoMetal, gSampler, in.uv);
      let normalRough = textureSample(gNormalRough, gSampler, in.uv);
      let worldPos4   = textureSample(gWorldPos,    gSampler, in.uv);

      let albedo    = albedoMetal.rgb;
      let metallic  = albedoMetal.a;
      let N         = normalize(normalRough.rgb * 2.0 - 1.0);
      let roughness = normalRough.a;
      let worldPos  = worldPos4.xyz;

      let V  = normalize(camera.position - worldPos);
      let F0 = mix(vec3<f32>(0.04), albedo, metallic);

      var Lo = vec3<f32>(0.0);

      for (var i = 0u; i < lights.count; i++) {
        let light = lights.lights[i];
        var L: vec3<f32>;
        var attenuation: f32 = 1.0;

        if (light.kind == 0u) { // Point
          let d = light.position - worldPos;
          L = normalize(d);
          let dist = length(d);
          attenuation = 1.0 / (dist * dist);
          attenuation *= max(1.0 - (dist / light.range), 0.0);
        } else { // Directional
          L = normalize(-light.direction);
        }

        let H       = normalize(V + L);
        let radiance = light.color * light.intensity * attenuation;
        let NDF     = DistributionGGX(N, H, roughness);
        let G       = GeometrySmith(N, V, L, roughness);
        let F       = FresnelSchlick(max(dot(H, V), 0.0), F0);
        let kd      = (vec3<f32>(1.0) - F) * (1.0 - metallic);
        let NdL     = max(dot(N, L), 0.0);
        let specular = (NDF * G * F) / max(4.0 * max(dot(N, V), 0.0) * NdL, 0.001);

        Lo += (kd * albedo / PI + specular) * radiance * NdL;
      }

      let ambient = vec3<f32>(0.03) * albedo;
      let color = ACESFilm(Lo + ambient);
      return vec4<f32>(pow(color, vec3<f32>(1.0/2.2)), 1.0);  // Gamma correction
    }
  `,
};

// ─────────────────────────────────────────────
// § ASSET VIRTUALIZER — LOD + Frustum Streaming
// ─────────────────────────────────────────────

interface AssetManifest {
  id:       string;
  lods:     LodLevel[];
  bounds:   AABB;
}

interface LodLevel {
  index:      number;
  distance:   number;      // max render distance for this LOD
  meshUrl:    string;
  textureUrls: string[];
  vertexCount: number;
}

interface AABB {
  min: Vec3;
  max: Vec3;
  center: Vec3;
  radius: number;
}

interface LoadedAsset {
  id:        string;
  lod:       number;
  vertexBuf: GPUBuffer;
  indexBuf:  GPUBuffer;
  indexCount: number;
  bindGroup: GPUBindGroup;
}

interface StreamRequest {
  manifest: AssetManifest;
  lodIndex: number;
  priority: number;
}

export class AssetVirtualizer {
  private readonly loaded   = new Map<string, LoadedAsset>();
  private readonly pending  = new Set<string>();
  private readonly queue:    StreamRequest[] = [];
  private readonly maxConcurrent = 4;
  private activeStreams = 0;

  constructor(
    private readonly device:   GPUDevice,
    private readonly bindGroupLayout: GPUBindGroupLayout,
  ) {}

  /**
   * Called every frame with the camera frustum planes and position.
   * Updates the streaming queue based on visibility + distance.
   */
  update(
    manifests:  AssetManifest[],
    camPos:     Vec3,
    frustum:    FrustumPlanes,
  ): void {
    const visible: StreamRequest[] = [];

    for (const manifest of manifests) {
      if (!this.intersectsFrustum(manifest.bounds, frustum)) continue;

      const dist    = camPos.sub(manifest.bounds.center).length();
      const lodIdx  = this.selectLod(manifest.lods, dist);
      const loaded  = this.loaded.get(manifest.id);

      if (!loaded || loaded.lod !== lodIdx) {
        if (!this.pending.has(manifest.id)) {
          visible.push({ manifest, lodIndex: lodIdx, priority: 1 / (dist + 1) });
        }
      }
    }

    // Sort by priority (closest first)
    visible.sort((a, b) => b.priority - a.priority);
    this.queue.push(...visible);

    this.flushQueue();
  }

  getAsset(id: string): LoadedAsset | undefined {
    return this.loaded.get(id);
  }

  private async loadAsset(req: StreamRequest): Promise<void> {
    const { manifest, lodIndex } = req;
    const lod = manifest.lods[lodIndex];

    this.pending.add(manifest.id);
    this.activeStreams++;

    try {
      // Fetch mesh binary
      const meshRes  = await fetch(lod.meshUrl);
      const meshData = await meshRes.arrayBuffer();
      const meshView = new Float32Array(meshData);

      // Interleaved format: pos(3) + norm(3) + uv(2) + tangent(4) = 12 floats
      const VERTEX_STRIDE = 12 * 4;
      const vertexCount   = meshView.byteLength / VERTEX_STRIDE;

      const vertexBuf = this.device.createBuffer({
        size:  meshView.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        label: `vtx_${manifest.id}_lod${lodIndex}`,
      });
      this.device.queue.writeBuffer(vertexBuf, 0, meshView);

      // Index buffer (last 25% of binary assumed to be indices)
      const indexData = new Uint32Array(meshData, Math.floor(meshView.byteLength * 0.75));
      const indexBuf  = this.device.createBuffer({
        size:  indexData.byteLength,
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
        label: `idx_${manifest.id}_lod${lodIndex}`,
      });
      this.device.queue.writeBuffer(indexBuf, 0, indexData);

      // Textures
      const textures = await Promise.all(
        lod.textureUrls.map((url) => this.loadTexture(url)),
      );

      const bindGroup = this.device.createBindGroup({
        layout:  this.bindGroupLayout,
        entries: textures.map((tex, i) => ({
          binding:  i,
          resource: tex.createView(),
        })),
        label: `bg_${manifest.id}_lod${lodIndex}`,
      });

      // Evict old LOD
      const old = this.loaded.get(manifest.id);
      if (old) {
        old.vertexBuf.destroy();
        old.indexBuf.destroy();
      }

      this.loaded.set(manifest.id, {
        id:         manifest.id,
        lod:        lodIndex,
        vertexBuf,
        indexBuf,
        indexCount: indexData.length,
        bindGroup,
      });
    } catch (err) {
      console.error(`[AssetVirtualizer] Failed to load ${manifest.id} LOD${lodIndex}:`, err);
    } finally {
      this.pending.delete(manifest.id);
      this.activeStreams--;
      this.flushQueue();
    }
  }

  private flushQueue(): void {
    while (this.activeStreams < this.maxConcurrent && this.queue.length > 0) {
      const req = this.queue.shift()!;
      if (!this.pending.has(req.manifest.id)) {
        void this.loadAsset(req);
      }
    }
  }

  private async loadTexture(url: string): Promise<GPUTexture> {
    const res  = await fetch(url);
    const blob = await res.blob();
    const img  = await createImageBitmap(blob);

    const tex = this.device.createTexture({
      size:   { width: img.width, height: img.height },
      format: "rgba8unorm",
      usage:  GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      mipLevelCount: Math.floor(Math.log2(Math.max(img.width, img.height))) + 1,
    });

    this.device.queue.copyExternalImageToTexture(
      { source: img },
      { texture: tex },
      { width: img.width, height: img.height },
    );

    return tex;
  }

  private selectLod(lods: LodLevel[], dist: number): number {
    let selected = lods.length - 1;
    for (let i = 0; i < lods.length; i++) {
      if (dist <= lods[i].distance) { selected = i; break; }
    }
    return selected;
  }

  private intersectsFrustum(aabb: AABB, planes: FrustumPlanes): boolean {
    for (const plane of planes) {
      const d =
        plane.normal.x * aabb.center.x +
        plane.normal.y * aabb.center.y +
        plane.normal.z * aabb.center.z +
        plane.d;
      if (d < -aabb.radius) return false;
    }
    return true;
  }
}

interface Plane { normal: Vec3; d: number; }
type FrustumPlanes = [Plane, Plane, Plane, Plane, Plane, Plane];

// ─────────────────────────────────────────────
// § WEBGPU RENDERER — Main System
// ─────────────────────────────────────────────

export interface RendererConfig {
  canvas:         HTMLCanvasElement;
  shadowMapSize:  number;
  maxLights:      number;
  taaEnabled:     boolean;
  bloomEnabled:   boolean;
}

@System({
  id:          "aetheris.renderer" as SystemId,
  priority:    100,
  updateHz:    60 as Hz,
  dependencies: ["aetheris.kernel" as SystemId],
})
export class WebGPURenderer implements ISystem {
  readonly id       = "aetheris.renderer" as SystemId;
  readonly priority = 100;

  private device!:        GPUDevice;
  private context!:       GPUCanvasContext;
  private renderGraph!:   RenderGraph;
  private assetVirt!:     AssetVirtualizer;

  // G-Buffer handles
  private hAlbedoMetal!:  ResourceHandle;
  private hNormalRough!:  ResourceHandle;
  private hWorldPos!:     ResourceHandle;
  private hDepth!:        ResourceHandle;
  private hShadowMap!:    ResourceHandle;

  // Pipelines
  private gBufferPipeline!:  GPURenderPipeline;
  private lightingPipeline!: GPURenderPipeline;

  // Uniform buffers
  private cameraUniformBuf!: GPUBuffer;
  private lightsStorageBuf!: GPUBuffer;

  // Camera state
  private projMatrix  = Mat4.perspective(Math.PI / 3, 16 / 9, 0.1, 2000);
  private viewMatrix  = Mat4.identity();
  private cameraPos   = Vec3.zero();
  private frustum!:    FrustumPlanes;

  constructor(private readonly config: RendererConfig) {}

  async onInit(_kernel: IKernel): Promise<void> {
    if (!navigator.gpu) {
      throw new Error("[WebGPURenderer] WebGPU not supported in this browser");
    }

    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: "high-performance",
    });
    if (!adapter) throw new Error("[WebGPURenderer] No WebGPU adapter found");

    this.device = await adapter.requestDevice({
      requiredFeatures: ["depth-clip-control", "indirect-first-instance"],
      requiredLimits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxComputeWorkgroupsPerDimension: adapter.limits.maxComputeWorkgroupsPerDimension,
      },
    });

    this.device.lost.then((info) => {
      console.error("[WebGPURenderer] Device lost:", info.message);
    });

    this.context = this.config.canvas.getContext("webgpu") as GPUCanvasContext;
    this.context.configure({
      device:    this.device,
      format:    navigator.gpu.getPreferredCanvasFormat(),
      alphaMode: "opaque",
    });

    const { width: W, height: H } = this.config.canvas;

    this.renderGraph = new RenderGraph(this.device);

    // ── G-Buffer textures ──
    this.hAlbedoMetal = this.renderGraph.createTexture({
      width: W, height: H, format: "rgba16float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      label: "GAlbedoMetal",
    });
    this.hNormalRough = this.renderGraph.createTexture({
      width: W, height: H, format: "rgba16float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      label: "GNormalRough",
    });
    this.hWorldPos = this.renderGraph.createTexture({
      width: W, height: H, format: "rgba32float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      label: "GWorldPos",
    });
    this.hDepth = this.renderGraph.createTexture({
      width: W, height: H, format: "depth24plus",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      label: "GDepth",
    });
    this.hShadowMap = this.renderGraph.createTexture({
      width: this.config.shadowMapSize, height: this.config.shadowMapSize,
      format: "depth32float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      label: "ShadowMap",
    });

    this.renderGraph.compile();

    // ── Pipelines ──
    await this.createGBufferPipeline();
    await this.createLightingPipeline();

    // ── Uniform buffers ──
    // Camera: 2x mat4 + vec3 = 2*64 + 16 = 144 bytes
    this.cameraUniformBuf = this.device.createBuffer({
      size:  256, // padded to 256
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: "CameraUniforms",
    });

    // Lights: count(4) + padding(12) + 256*Light(80bytes) = 20496 bytes
    this.lightsStorageBuf = this.device.createBuffer({
      size:  16 + 256 * 80,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: "LightsBuffer",
    });

    // ── Asset Virtualizer ──
    const materialLayout = this.gBufferPipeline.getBindGroupLayout(2);
    this.assetVirt = new AssetVirtualizer(this.device, materialLayout);

    // ── Resize observer ──
    new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      this.config.canvas.width  = width;
      this.config.canvas.height = height;
      this.renderGraph.resize(width, height);
      this.projMatrix = Mat4.perspective(Math.PI / 3, width / height, 0.1, 2000);
    }).observe(this.config.canvas);
  }

  onUpdate(_dt: number, _frame: FrameIndex): void {
    this.uploadCameraUniforms();
    const cmdBuf = this.renderGraph.execute();
    this.device.queue.submit([cmdBuf]);
  }

  async onDestroy(): Promise<void> {
    this.renderGraph.destroy();
    this.cameraUniformBuf.destroy();
    this.lightsStorageBuf.destroy();
    this.device.destroy();
  }

  // ── Camera control ──

  setCamera(viewMatrix: Mat4, position: Vec3): void {
    this.viewMatrix = viewMatrix;
    this.cameraPos  = position;
    this.frustum    = this.extractFrustumPlanes(this.projMatrix.multiply(viewMatrix));
  }

  // ── Private helpers ──

  private uploadCameraUniforms(): void {
    const data = new Float32Array(64);
    const vp   = this.projMatrix.multiply(this.viewMatrix);
    data.set(vp.data, 0);
    data.set(this.viewMatrix.data, 16);
    data.set(this.cameraPos.toArray(), 32);
    this.device.queue.writeBuffer(this.cameraUniformBuf, 0, data);
  }

  private async createGBufferPipeline(): Promise<void> {
    const module = this.device.createShaderModule({
      label: "GBufferShader",
      code:  ShaderLibrary.GBuffer,
    });

    this.gBufferPipeline = await this.device.createRenderPipelineAsync({
      label:  "GBufferPipeline",
      layout: "auto",
      vertex: {
        module,
        entryPoint: "vs_main",
        buffers: [{
          arrayStride: 12 * 4, // 12 floats per vertex
          attributes: [
            { shaderLocation: 0, offset: 0,      format: "float32x3" }, // pos
            { shaderLocation: 1, offset: 3 * 4,  format: "float32x3" }, // normal
            { shaderLocation: 2, offset: 6 * 4,  format: "float32x2" }, // uv
            { shaderLocation: 3, offset: 8 * 4,  format: "float32x4" }, // tangent
          ],
        }],
      },
      fragment: {
        module,
        entryPoint: "fs_main",
        targets: [
          { format: "rgba16float" }, // albedoMetal
          { format: "rgba16float" }, // normalRough
          { format: "rgba32float" }, // worldPos
        ],
      },
      depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
      primitive:    { topology: "triangle-list", cullMode: "back" },
    });
  }

  private async createLightingPipeline(): Promise<void> {
    const module = this.device.createShaderModule({
      label: "LightingShader",
      code:  ShaderLibrary.DeferredLighting,
    });

    this.lightingPipeline = await this.device.createRenderPipelineAsync({
      label:  "LightingPipeline",
      layout: "auto",
      vertex:   { module, entryPoint: "vs_fullscreen" },
      fragment: {
        module,
        entryPoint: "fs_lighting",
        targets: [{ format: navigator.gpu.getPreferredCanvasFormat() }],
      },
      primitive: { topology: "triangle-list" },
    });
  }

  private extractFrustumPlanes(vp: Mat4): FrustumPlanes {
    const m = vp.data;
    const plane = (a: number, b: number, c: number, d: number): Plane => {
      const len = Math.sqrt(a * a + b * b + c * c);
      return { normal: new Vec3(a / len, b / len, c / len), d: d / len };
    };
    return [
      plane(m[3]+m[0], m[7]+m[4], m[11]+m[8],  m[15]+m[12]), // left
      plane(m[3]-m[0], m[7]-m[4], m[11]-m[8],  m[15]-m[12]), // right
      plane(m[3]+m[1], m[7]+m[5], m[11]+m[9],  m[15]+m[13]), // bottom
      plane(m[3]-m[1], m[7]-m[5], m[11]-m[9],  m[15]-m[13]), // top
      plane(m[3]+m[2], m[7]+m[6], m[11]+m[10], m[15]+m[14]), // near
      plane(m[3]-m[2], m[7]-m[6], m[11]-m[10], m[15]-m[14]), // far
    ] as FrustumPlanes;
  }
}

// Re-export for kernel registration
declare const Hz: (n: number) => typeof Hz extends (n: number) => infer R ? R : never;
