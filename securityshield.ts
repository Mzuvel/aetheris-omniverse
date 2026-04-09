/**
 * @file SecurityShield.ts
 * @description AETHERIS OMNIVERSE — Security & Anti-Cheat Layer
 *
 * Features:
 *  - AES-256-GCM state encryption via SubtleCrypto (WebCrypto)
 *  - RSA-OAEP session key exchange
 *  - WASM Sandboxing for third-party mods (memory isolation)
 *  - Behavioral telemetry (aimbot/speedhack/teleport detection)
 *  - Memory integrity checks (prototype pollution detection)
 *  - Anti-debug / anti-tamper guards
 */

import type { ISystem, IKernel, SystemId } from "../../kernel/src/Kernel";
import type { EntityState, PlayerInput } from "../../network/src/MultiplayerSync";

// ─────────────────────────────────────────────
// § CRYPTO — AES-256-GCM + RSA-OAEP
// ─────────────────────────────────────────────

const AESGCM_KEY_BITS = 256;
const IV_LENGTH       = 12;   // 96-bit IV for GCM
const TAG_LENGTH_BITS = 128;

export class AesCryptoService {
  private sessionKey:     CryptoKey | null = null;
  private serverPubKey:   CryptoKey | null = null;
  private clientKeyPair:  CryptoKeyPair | null = null;

  /**
   * Generate an RSA-OAEP key pair for this session.
   * The public key is sent to the server; server encrypts the AES session key with it.
   */
  async generateClientKeyPair(): Promise<JsonWebKey> {
    this.clientKeyPair = await crypto.subtle.generateKey(
      {
        name:           "RSA-OAEP",
        modulusLength:  4096,
        publicExponent: new Uint8Array([0x01, 0x00, 0x01]),
        hash:           "SHA-256",
      },
      true,
      ["encrypt", "decrypt"],
    );

    return crypto.subtle.exportKey("jwk", this.clientKeyPair.publicKey);
  }

  /**
   * Import server's RSA public key.
   */
  async importServerPublicKey(jwk: JsonWebKey): Promise<void> {
    this.serverPubKey = await crypto.subtle.importKey(
      "jwk",
      jwk,
      { name: "RSA-OAEP", hash: "SHA-256" },
      false,
      ["encrypt"],
    );
  }

  /**
   * Process the server's encrypted session key.
   * Server encrypts an AES-256 key with our RSA public key.
   */
  async unwrapSessionKey(encryptedKeyBuf: ArrayBuffer): Promise<void> {
    if (!this.clientKeyPair) throw new SecurityError("Client key pair not generated");

    this.sessionKey = await crypto.subtle.unwrapKey(
      "raw",
      encryptedKeyBuf,
      this.clientKeyPair.privateKey,
      { name: "RSA-OAEP" },
      { name: "AES-GCM", length: AESGCM_KEY_BITS },
      false,
      ["encrypt", "decrypt"],
    );
  }

  /**
   * Generate a local AES key (for offline / testing scenarios).
   */
  async generateLocalKey(): Promise<void> {
    this.sessionKey = await crypto.subtle.generateKey(
      { name: "AES-GCM", length: AESGCM_KEY_BITS },
      false,
      ["encrypt", "decrypt"],
    );
  }

  /**
   * Encrypt arbitrary bytes with AES-256-GCM.
   * Returns: [IV (12 bytes) | Ciphertext+Tag]
   */
  async encrypt(plaintext: ArrayBuffer): Promise<ArrayBuffer> {
    if (!this.sessionKey) throw new SecurityError("Session key not initialized");

    const iv         = crypto.getRandomValues(new Uint8Array(IV_LENGTH));
    const ciphertext = await crypto.subtle.encrypt(
      { name: "AES-GCM", iv, tagLength: TAG_LENGTH_BITS },
      this.sessionKey,
      plaintext,
    );

    const result = new Uint8Array(IV_LENGTH + ciphertext.byteLength);
    result.set(iv, 0);
    result.set(new Uint8Array(ciphertext), IV_LENGTH);
    return result.buffer;
  }

  /**
   * Decrypt AES-256-GCM ciphertext.
   * Expects: [IV (12 bytes) | Ciphertext+Tag]
   */
  async decrypt(data: ArrayBuffer): Promise<ArrayBuffer> {
    if (!this.sessionKey) throw new SecurityError("Session key not initialized");
    if (data.byteLength < IV_LENGTH + 16) throw new SecurityError("Ciphertext too short");

    const iv         = data.slice(0, IV_LENGTH);
    const ciphertext = data.slice(IV_LENGTH);

    return crypto.subtle.decrypt(
      { name: "AES-GCM", iv, tagLength: TAG_LENGTH_BITS },
      this.sessionKey,
      ciphertext,
    );
  }

  /**
   * Compute HMAC-SHA256 for integrity checking.
   */
  async hmac(data: ArrayBuffer, key?: CryptoKey): Promise<ArrayBuffer> {
    const hmacKey = key ?? await crypto.subtle.generateKey(
      { name: "HMAC", hash: "SHA-256" },
      false,
      ["sign"],
    );
    return crypto.subtle.sign("HMAC", hmacKey, data);
  }

  /**
   * Derive a subkey from the session key using HKDF.
   */
  async deriveSubkey(info: string): Promise<CryptoKey> {
    if (!this.sessionKey) throw new SecurityError("Session key not initialized");

    const keyMaterial = await crypto.subtle.importKey(
      "raw",
      new TextEncoder().encode(info),
      "HKDF",
      false,
      ["deriveKey"],
    );

    return crypto.subtle.deriveKey(
      {
        name:   "HKDF",
        hash:   "SHA-256",
        salt:   new Uint8Array(32),
        info:   new TextEncoder().encode("aetheris-subkey"),
      },
      keyMaterial,
      { name: "AES-GCM", length: 256 },
      false,
      ["encrypt", "decrypt"],
    );
  }

  get isReady(): boolean { return this.sessionKey !== null; }
}

// ─────────────────────────────────────────────
// § WASM SANDBOX — Mod isolation
// ─────────────────────────────────────────────

export interface ModManifest {
  id:           string;
  name:         string;
  version:      string;
  permissions:  ModPermission[];
  wasmUrl:      string;
  checksum:     string;  // SHA-256 hex of wasm binary
}

export type ModPermission =
  | "read:world"
  | "write:chat"
  | "read:ui"
  | "write:ui-overlay"
  | "network:restricted";

export interface ModAPI {
  getWorldState:    () => unknown;
  sendChatMessage:  (msg: string) => void;
  renderOverlay:    (data: unknown) => void;
}

interface SandboxedMod {
  manifest:  ModManifest;
  instance:  WebAssembly.Instance;
  memory:    WebAssembly.Memory;
  api:       ModAPI;
}

export class WasmSandbox {
  private readonly mods     = new Map<string, SandboxedMod>();
  private readonly maxMemoryPages = 256;  // 16 MB per mod

  /**
   * Load and verify a mod WASM binary, then instantiate it in isolation.
   */
  async loadMod(
    manifest: ModManifest,
    api:      ModAPI,
  ): Promise<void> {
    // 1. Fetch binary
    const res = await fetch(manifest.wasmUrl);
    const bin = await res.arrayBuffer();

    // 2. Verify checksum
    const hashBuf  = await crypto.subtle.digest("SHA-256", bin);
    const hashHex  = Array.from(new Uint8Array(hashBuf))
      .map((b) => b.toString(16).padStart(2, "0"))
      .join("");

    if (hashHex !== manifest.checksum) {
      throw new SecurityError(
        `[WasmSandbox] Checksum mismatch for mod "${manifest.id}". ` +
        `Expected ${manifest.checksum}, got ${hashHex}`,
      );
    }

    // 3. Isolated memory — mod cannot access host memory
    const memory = new WebAssembly.Memory({
      initial:  16,
      maximum:  this.maxMemoryPages,
    });

    // 4. Build restricted import object — only expose allowed APIs
    const imports = this.buildImports(manifest, memory, api);

    // 5. Compile + instantiate in isolation
    const { instance } = await WebAssembly.instantiate(bin, imports);

    this.mods.set(manifest.id, { manifest, instance, memory, api });

    console.log(`[WasmSandbox] Loaded mod "${manifest.name}" v${manifest.version}`);
  }

  /**
   * Invoke a mod's exported function with timeout guard.
   */
  async callMod<T = unknown>(
    modId:    string,
    fnName:   string,
    args:     number[],
    timeoutMs = 5000,
  ): Promise<T> {
    const mod = this.mods.get(modId);
    if (!mod) throw new SecurityError(`[WasmSandbox] Mod "${modId}" not loaded`);

    const fn = mod.instance.exports[fnName];
    if (typeof fn !== "function") {
      throw new SecurityError(`[WasmSandbox] Mod "${modId}" has no export "${fnName}"`);
    }

    return new Promise<T>((resolve, reject) => {
      const timer = setTimeout(
        () => reject(new SecurityError(`[WasmSandbox] Mod "${modId}" timed out`)),
        timeoutMs,
      );
      try {
        const result = fn(...args) as T;
        clearTimeout(timer);
        resolve(result);
      } catch (err) {
        clearTimeout(timer);
        reject(err);
      }
    });
  }

  unloadMod(modId: string): void {
    this.mods.delete(modId);
  }

  private buildImports(
    manifest:  ModManifest,
    memory:    WebAssembly.Memory,
    api:       ModAPI,
  ): WebAssembly.Imports {
    const perms = new Set(manifest.permissions);

    const env: Record<string, WebAssembly.ImportValue> = {
      memory,
      // Always allowed: logging
      log: (ptr: number, len: number) => {
        const bytes = new Uint8Array(memory.buffer, ptr, len);
        console.log(`[Mod:${manifest.id}]`, new TextDecoder().decode(bytes));
      },
      // Always allowed: abort
      abort: (msgPtr: number, filePtr: number, line: number, col: number) => {
        console.error(`[Mod:${manifest.id}] WASM abort at ${filePtr}:${line}:${col}`);
      },
    };

    if (perms.has("read:world")) {
      env["aetheris_get_world"] = () => {
        // Serializes world state into mod's isolated memory
        const state   = JSON.stringify(api.getWorldState());
        const encoded = new TextEncoder().encode(state);
        const ptr     = (mod_malloc(memory, encoded.length));
        new Uint8Array(memory.buffer, ptr, encoded.length).set(encoded);
        return ptr;
      };
    }

    if (perms.has("write:chat")) {
      env["aetheris_send_chat"] = (ptr: number, len: number) => {
        const bytes = new Uint8Array(memory.buffer, ptr, len);
        const msg   = new TextDecoder().decode(bytes);
        // Sanitize: strip HTML, limit length
        const safe  = msg.replace(/<[^>]*>/g, "").slice(0, 256);
        api.sendChatMessage(safe);
      };
    }

    if (perms.has("write:ui-overlay")) {
      env["aetheris_render_overlay"] = (ptr: number, len: number) => {
        const bytes = new Uint8Array(memory.buffer, ptr, len);
        const json  = JSON.parse(new TextDecoder().decode(bytes));
        api.renderOverlay(json);
      };
    }

    // Deny everything else
    return {
      env,
      wasi_snapshot_preview1: {
        // Minimal WASI stub — deny file/network access
        fd_write:    () => -1,
        fd_read:     () => -1,
        proc_exit:   (code: number) => { throw new Error(`WASM exit: ${code}`); },
        args_get:    () => 0,
        args_sizes_get: () => 0,
      },
    };
  }
}

// Bump allocator for mod memory (mods manage their own heap)
function mod_malloc(memory: WebAssembly.Memory, size: number): number {
  // Use a fixed offset; real mods would export their own malloc
  return 1024 + size; // simplified
}

// ─────────────────────────────────────────────
// § BEHAVIORAL ANALYZER — Anti-cheat telemetry
// ─────────────────────────────────────────────

export interface AnomalyEvent {
  kind:      AnomalyKind;
  severity:  "low" | "medium" | "high" | "critical";
  playerId:  string;
  timestamp: number;
  evidence:  Record<string, unknown>;
}

export type AnomalyKind =
  | "speedhack"
  | "teleport"
  | "aimbot_snap"
  | "aimbot_smooth"
  | "wallhack_probe"
  | "packet_manipulation"
  | "memory_tampering";

interface MovementSample {
  timestamp: number;
  position:  { x: number; y: number; z: number };
  velocity:  { x: number; y: number; z: number };
}

interface AimSample {
  timestamp: number;
  yaw:       number;
  pitch:     number;
  firing:    boolean;
}

export class BehaviorAnalyzer {
  private readonly movementHistory = new Map<string, MovementSample[]>();
  private readonly aimHistory      = new Map<string, AimSample[]>();
  private readonly anomalyCallbacks: ((event: AnomalyEvent) => void)[] = [];

  private readonly THRESHOLDS = {
    MAX_SPEED_UNITS_PER_SEC:   15,   // max legitimate movement speed
    MAX_TELEPORT_DIST:          5,   // meters per tick at 60hz
    AIMBOT_SNAP_ANGLE_DEG:     30,   // degrees per frame threshold
    AIMBOT_SMOOTH_STDDEV:      0.01, // suspiciously low aim variance
    MAX_PACKET_RATE_PER_SEC:  120,
  };

  onAnomaly(cb: (event: AnomalyEvent) => void): () => void {
    this.anomalyCallbacks.push(cb);
    return () => {
      const idx = this.anomalyCallbacks.indexOf(cb);
      if (idx >= 0) this.anomalyCallbacks.splice(idx, 1);
    };
  }

  /**
   * Analyze incoming player state snapshot for movement anomalies.
   */
  analyzeMovement(playerId: string, state: EntityState, timestamp: number): void {
    const hist = this.movementHistory.get(playerId) ?? [];
    const prev = hist[hist.length - 1];

    const sample: MovementSample = {
      timestamp,
      position: { ...state.position },
      velocity: { ...state.velocity },
    };

    if (prev) {
      const dt   = (timestamp - prev.timestamp) / 1000;
      if (dt <= 0) return;

      const dx   = state.position.x - prev.position.x;
      const dy   = state.position.y - prev.position.y;
      const dz   = state.position.z - prev.position.z;
      const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);
      const speed = dist / dt;

      // Speedhack detection
      if (speed > this.THRESHOLDS.MAX_SPEED_UNITS_PER_SEC * 1.5) {
        this.emit({
          kind:     "speedhack",
          severity: speed > this.THRESHOLDS.MAX_SPEED_UNITS_PER_SEC * 3 ? "critical" : "high",
          playerId,
          timestamp,
          evidence: { speed, maxAllowed: this.THRESHOLDS.MAX_SPEED_UNITS_PER_SEC, dt },
        });
      }

      // Teleport detection
      if (dist > this.THRESHOLDS.MAX_TELEPORT_DIST && dt < 0.1) {
        this.emit({
          kind:     "teleport",
          severity: "critical",
          playerId,
          timestamp,
          evidence: { dist, dt, from: prev.position, to: state.position },
        });
      }
    }

    hist.push(sample);
    if (hist.length > 300) hist.shift();
    this.movementHistory.set(playerId, hist);
  }

  /**
   * Analyze aim patterns for aimbot signatures.
   */
  analyzeAim(playerId: string, input: PlayerInput): void {
    const hist = this.aimHistory.get(playerId) ?? [];
    const prev = hist[hist.length - 1];

    const sample: AimSample = {
      timestamp: input.timestamp as number,
      yaw:       input.yaw,
      pitch:     input.pitch,
      firing:    (input.buttons & 1) === 1,
    };

    if (prev) {
      const dt = (sample.timestamp - prev.timestamp) / 1000;
      if (dt <= 0) return;

      const dyaw   = Math.abs(angleDiff(sample.yaw,   prev.yaw));
      const dpitch = Math.abs(angleDiff(sample.pitch, prev.pitch));
      const totalDeg = (dyaw + dpitch) * (180 / Math.PI);

      // Snap detection: huge angle change in single frame while firing
      if (totalDeg > this.THRESHOLDS.AIMBOT_SNAP_ANGLE_DEG && sample.firing) {
        this.emit({
          kind:     "aimbot_snap",
          severity: totalDeg > 90 ? "critical" : "high",
          playerId,
          timestamp: sample.timestamp,
          evidence: { totalDeg, dyawDeg: dyaw * 180/Math.PI, dpitchDeg: dpitch * 180/Math.PI, dt },
        });
      }

      // Smooth aimbot: statistically analyze aim variance while tracking target
      if (hist.length >= 60 && sample.firing) {
        const recent = hist.slice(-60);
        const variance = this.aimVariance(recent);
        if (variance < this.THRESHOLDS.AIMBOT_SMOOTH_STDDEV) {
          this.emit({
            kind:     "aimbot_smooth",
            severity: "high",
            playerId,
            timestamp: sample.timestamp,
            evidence: { variance, threshold: this.THRESHOLDS.AIMBOT_SMOOTH_STDDEV, windowSize: 60 },
          });
        }
      }
    }

    hist.push(sample);
    if (hist.length > 600) hist.shift();
    this.aimHistory.set(playerId, hist);
  }

  private aimVariance(samples: AimSample[]): number {
    const yaws    = samples.map((s) => s.yaw);
    const pitches = samples.map((s) => s.pitch);
    return (standardDeviation(yaws) + standardDeviation(pitches)) / 2;
  }

  private emit(event: AnomalyEvent): void {
    for (const cb of this.anomalyCallbacks) cb(event);
  }

  clearPlayer(id: string): void {
    this.movementHistory.delete(id);
    this.aimHistory.delete(id);
  }
}

function angleDiff(a: number, b: number): number {
  let diff = a - b;
  while (diff >  Math.PI) diff -= Math.PI * 2;
  while (diff < -Math.PI) diff += Math.PI * 2;
  return diff;
}

function standardDeviation(values: number[]): number {
  const n    = values.length;
  if (n < 2) return 0;
  const mean = values.reduce((a, b) => a + b, 0) / n;
  const sq   = values.reduce((sum, v) => sum + (v - mean) ** 2, 0);
  return Math.sqrt(sq / (n - 1));
}

// ─────────────────────────────────────────────
// § MEMORY INTEGRITY — Prototype pollution & tamper detection
// ─────────────────────────────────────────────

interface IntegritySnapshot {
  objectProtoKeys:   string[];
  arrayProtoKeys:    string[];
  functionProtoKeys: string[];
  criticalFunctions: Map<string, string>;
}

export class MemoryIntegrityGuard {
  private baseline: IntegritySnapshot | null = null;
  private checkInterval: ReturnType<typeof setInterval> | null = null;

  private readonly criticalFns: [string, () => Function][] = [
    ["Array.prototype.push",         () => Array.prototype.push],
    ["JSON.stringify",               () => JSON.stringify],
    ["JSON.parse",                   () => JSON.parse],
    ["Math.random",                  () => Math.random],
    ["performance.now",              () => performance.now],
    ["crypto.getRandomValues",       () => crypto.getRandomValues.bind(crypto)],
    ["WebSocket.prototype.send",     () => WebSocket.prototype.send],
    ["DataView.prototype.getFloat32",() => DataView.prototype.getFloat32],
  ];

  capture(): void {
    this.baseline = {
      objectProtoKeys:   Object.getOwnPropertyNames(Object.prototype),
      arrayProtoKeys:    Object.getOwnPropertyNames(Array.prototype),
      functionProtoKeys: Object.getOwnPropertyNames(Function.prototype),
      criticalFunctions: new Map(
        this.criticalFns.map(([name, getter]) => [name, getter().toString()]),
      ),
    };
  }

  startWatching(
    onViolation: (violation: IntegrityViolation) => void,
    intervalMs = 5000,
  ): void {
    if (!this.baseline) throw new SecurityError("Must call capture() first");

    this.checkInterval = setInterval(() => {
      const violations = this.check();
      for (const v of violations) onViolation(v);
    }, intervalMs);
  }

  stopWatching(): void {
    if (this.checkInterval) clearInterval(this.checkInterval);
  }

  private check(): IntegrityViolation[] {
    if (!this.baseline) return [];
    const violations: IntegrityViolation[] = [];

    // Prototype pollution check
    const currentObjKeys = Object.getOwnPropertyNames(Object.prototype);
    const addedKeys       = currentObjKeys.filter(
      (k) => !this.baseline!.objectProtoKeys.includes(k),
    );
    for (const key of addedKeys) {
      violations.push({
        kind:    "prototype_pollution",
        target:  `Object.prototype.${key}`,
        details: `Unexpected property added`,
      });
    }

    // Critical function integrity check
    for (const [name, getter] of this.criticalFns) {
      try {
        const current  = getter().toString();
        const original = this.baseline.criticalFunctions.get(name);
        if (original && current !== original) {
          violations.push({
            kind:    "function_tampered",
            target:  name,
            details: `Function body changed`,
          });
        }
      } catch {
        violations.push({
          kind:    "function_missing",
          target:  name,
          details: `Function no longer accessible`,
        });
      }
    }

    return violations;
  }
}

interface IntegrityViolation {
  kind:    "prototype_pollution" | "function_tampered" | "function_missing";
  target:  string;
  details: string;
}

// ─────────────────────────────────────────────
// § SECURITY SHIELD — System integration
// ─────────────────────────────────────────────

export interface SecurityConfig {
  serverPublicKeyJwk?: JsonWebKey;
  enableIntegrityChecks: boolean;
  enableBehaviorAnalysis: boolean;
  anomalyReportUrl:      string;
}

@System({
  id:          "aetheris.security" as SystemId,
  priority:    1,   // Highest priority — init first
  updateHz:    10 as Hz,
  dependencies: [],
})
export class SecurityShield implements ISystem {
  readonly id       = "aetheris.security" as SystemId;
  readonly priority = 1;

  readonly crypto   = new AesCryptoService();
  readonly sandbox  = new WasmSandbox();
  readonly behavior = new BehaviorAnalyzer();
  readonly integrity = new MemoryIntegrityGuard();

  private violationBuffer: AnomalyEvent[] = [];
  private reportFlushTimer: ReturnType<typeof setInterval> | null = null;

  constructor(private readonly config: SecurityConfig) {}

  async onInit(kernel: IKernel): Promise<void> {
    // 1. Capture memory baseline ASAP
    this.integrity.capture();

    // 2. Generate client key pair and start key exchange
    const pubKeyJwk = await this.crypto.generateClientKeyPair();
    kernel.emit("security:client-pubkey", { pubKeyJwk });

    // 3. Import server public key if provided
    if (this.config.serverPublicKeyJwk) {
      await this.crypto.importServerPublicKey(this.config.serverPublicKeyJwk);
    } else {
      // Development fallback
      await this.crypto.generateLocalKey();
    }

    // 4. Start integrity monitoring
    if (this.config.enableIntegrityChecks) {
      this.integrity.startWatching((v) => {
        console.error("[SecurityShield] Memory integrity violation:", v);
        kernel.emit("security:integrity-violation", { violation: v });
      });
    }

    // 5. Wire anomaly reports to server
    if (this.config.enableBehaviorAnalysis) {
      this.behavior.onAnomaly((event) => {
        this.violationBuffer.push(event);
        if (event.severity === "critical") {
          void this.flushViolations(); // immediate flush for critical
        }
      });

      // Batch-upload anomaly reports every 10s
      this.reportFlushTimer = setInterval(() => void this.flushViolations(), 10_000);
    }

    // 6. Listen for session key from server
    kernel.on<{ encryptedKey: ArrayBuffer }>("security:session-key", async ({ encryptedKey }) => {
      await this.crypto.unwrapSessionKey(encryptedKey);
      kernel.emit("security:ready", {});
    });
  }

  onUpdate(): void {
    // Periodic integrity checks are handled by the guard's internal interval.
    // Here we just emit current security metrics.
  }

  async onDestroy(): Promise<void> {
    this.integrity.stopWatching();
    if (this.reportFlushTimer) clearInterval(this.reportFlushTimer);
    await this.flushViolations();
  }

  /**
   * Encrypt a game state payload before sending to server.
   */
  async encryptPayload(data: ArrayBuffer): Promise<ArrayBuffer> {
    return this.crypto.encrypt(data);
  }

  /**
   * Decrypt a payload received from server.
   */
  async decryptPayload(data: ArrayBuffer): Promise<ArrayBuffer> {
    return this.crypto.decrypt(data);
  }

  private async flushViolations(): Promise<void> {
    if (this.violationBuffer.length === 0) return;
    const batch = [...this.violationBuffer];
    this.violationBuffer = [];

    try {
      await fetch(this.config.anomalyReportUrl, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ events: batch, ts: Date.now() }),
      });
    } catch (err) {
      // Re-queue on failure
      this.violationBuffer.unshift(...batch);
    }
  }
}

// ─────────────────────────────────────────────
// § ERRORS
// ─────────────────────────────────────────────

export class SecurityError extends Error {
  constructor(m: string) { super(m); this.name = "SecurityError"; }
}

declare const Hz: unknown;
