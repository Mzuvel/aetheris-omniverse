/**
 * @file MultiplayerSync.ts
 * @description AETHERIS OMNIVERSE — Multiplayer Netcode Engine
 *
 * Architecture (Source Engine-style):
 *  - Client-Side Prediction: Apply inputs locally before server ack
 *  - Server Reconciliation: Re-simulate from last acked state on correction
 *  - Lag Compensation: Server rewinds world state for hit registration
 *  - Entity Interpolation: Smooth rendering of remote entities
 *  - Delta Compression: Only send changed state fields
 */

import type { ISystem, IKernel, SystemId, FrameIndex, Timestamp } from "../../kernel/src/Kernel";
import { System } from "../../kernel/src/Kernel";

// ─────────────────────────────────────────────
// § DOMAIN TYPES
// ─────────────────────────────────────────────

declare const __brand: unique symbol;
type Brand<T, B> = T & { readonly [__brand]: B };

export type PlayerId  = Brand<string,  "PlayerId">;
export type SessionId = Brand<string,  "SessionId">;
export type TickNum   = Brand<number,  "TickNum">;
export type Seq       = Brand<number,  "Seq">;

export const PlayerId  = (s: string): PlayerId   => s as PlayerId;
export const SessionId = (s: string): SessionId  => s as SessionId;
export const TickNum   = (n: number): TickNum     => n as TickNum;
export const Seq       = (n: number): Seq         => n as Seq;

// ─────────────────────────────────────────────
// § STATE INTERFACES
// ─────────────────────────────────────────────

export interface Vec3Like { x: number; y: number; z: number; }

export interface PlayerInput {
  seq:        Seq;
  tick:       TickNum;
  timestamp:  Timestamp;
  moveX:      number;    // -1..1
  moveZ:      number;    // -1..1
  yaw:        number;    // radians
  pitch:      number;    // radians
  buttons:    number;    // bitmask: bit0=fire, bit1=jump, bit2=crouch, etc.
}

export interface EntityState {
  id:          PlayerId;
  position:    Vec3Like;
  velocity:    Vec3Like;
  yaw:         number;
  pitch:       number;
  health:      number;
  animation:   number;
  flags:       number;
  lastTick:    TickNum;
}

export interface WorldSnapshot {
  tick:      TickNum;
  timestamp: Timestamp;
  entities:  Map<PlayerId, EntityState>;
}

export interface ServerAck {
  seq:          Seq;
  tick:         TickNum;
  serverState:  EntityState;   // authoritative state for this client
  worldTick:    TickNum;
}

export interface DeltaUpdate {
  tick:       TickNum;
  timestamp:  Timestamp;
  entities:   DeltaEntityState[];
}

export interface DeltaEntityState {
  id:       PlayerId;
  mask:     number;        // bitmask of which fields changed
  position?: Vec3Like;
  velocity?: Vec3Like;
  yaw?:     number;
  pitch?:   number;
  health?:  number;
  animation?: number;
  flags?:   number;
}

// Bitmask constants for delta compression
export const DELTA_MASK = {
  POSITION:  0b0000001,
  VELOCITY:  0b0000010,
  YAW:       0b0000100,
  PITCH:     0b0001000,
  HEALTH:    0b0010000,
  ANIMATION: 0b0100000,
  FLAGS:     0b1000000,
} as const;

// ─────────────────────────────────────────────
// § PHYSICS SIMULATION (deterministic)
// ─────────────────────────────────────────────

export interface PhysicsParams {
  moveSpeed:     number;  // units/sec
  jumpVelocity:  number;
  gravity:       number;
  friction:      number;
  airControl:    number;
}

const DEFAULT_PHYSICS: PhysicsParams = {
  moveSpeed:    6.0,
  jumpVelocity: 8.0,
  gravity:      -20.0,
  friction:     8.0,
  airControl:   0.3,
};

function simulateEntity(
  state: EntityState,
  input: PlayerInput,
  dt: number,
  physics: PhysicsParams = DEFAULT_PHYSICS,
): EntityState {
  const next = deepCloneEntity(state);
  next.yaw   = input.yaw;
  next.pitch = input.pitch;

  const isGrounded = next.position.y <= 0.001;
  const jumpBit    = (input.buttons >> 1) & 1;
  const crouchBit  = (input.buttons >> 2) & 1;

  // Horizontal movement — transform input by yaw
  const sinY = Math.sin(input.yaw);
  const cosY = Math.cos(input.yaw);
  const wishX = input.moveX * cosY + input.moveZ * sinY;
  const wishZ = -input.moveX * sinY + input.moveZ * cosY;
  const wishSpeed = Math.sqrt(wishX * wishX + wishZ * wishZ);

  const accel = isGrounded ? 1.0 : physics.airControl;
  const targetSpeed = Math.min(wishSpeed, 1.0) * physics.moveSpeed * (crouchBit ? 0.5 : 1.0);

  if (wishSpeed > 0) {
    next.velocity.x = lerp(next.velocity.x, (wishX / wishSpeed) * targetSpeed, accel * dt * 10);
    next.velocity.z = lerp(next.velocity.z, (wishZ / wishSpeed) * targetSpeed, accel * dt * 10);
  } else if (isGrounded) {
    const friction = Math.max(0, 1 - physics.friction * dt);
    next.velocity.x *= friction;
    next.velocity.z *= friction;
  }

  // Vertical
  if (isGrounded && jumpBit) {
    next.velocity.y = physics.jumpVelocity;
  } else if (!isGrounded) {
    next.velocity.y += physics.gravity * dt;
  }

  // Integrate position
  next.position.x += next.velocity.x * dt;
  next.position.y += next.velocity.y * dt;
  next.position.z += next.velocity.z * dt;

  // Ground clamp
  if (next.position.y < 0) {
    next.position.y = 0;
    next.velocity.y = 0;
  }

  next.lastTick = TickNum(state.lastTick + 1);
  return next;
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * Math.min(1, Math.max(0, t));
}

function lerpVec3(a: Vec3Like, b: Vec3Like, t: number): Vec3Like {
  return { x: lerp(a.x, b.x, t), y: lerp(a.y, b.y, t), z: lerp(a.z, b.z, t) };
}

function deepCloneEntity(e: EntityState): EntityState {
  return {
    ...e,
    position: { ...e.position },
    velocity: { ...e.velocity },
  };
}

// ─────────────────────────────────────────────
// § CLIENT-SIDE PREDICTION
// ─────────────────────────────────────────────

interface PendingInput {
  input:  PlayerInput;
  state:  EntityState;   // state AFTER applying this input
}

export class ClientPrediction {
  private pendingInputs: PendingInput[] = [];
  private currentState:  EntityState;
  private lastAckedSeq:  Seq = Seq(-1);
  private seqCounter     = 0;

  constructor(
    initialState: EntityState,
    private readonly physics: PhysicsParams = DEFAULT_PHYSICS,
  ) {
    this.currentState = deepCloneEntity(initialState);
  }

  /**
   * Apply a new input locally and return the predicted state.
   * Input is stored in the pending buffer until server acks it.
   */
  applyInput(
    rawInput: Omit<PlayerInput, "seq">,
    dt: number,
  ): { seq: Seq; predictedState: EntityState } {
    const seq: Seq   = Seq(++this.seqCounter);
    const input:  PlayerInput = { ...rawInput, seq };
    const nextState = simulateEntity(this.currentState, input, dt, this.physics);

    this.pendingInputs.push({ input, state: nextState });
    this.currentState = nextState;

    // Cap pending buffer
    if (this.pendingInputs.length > 1024) {
      this.pendingInputs.shift();
    }

    return { seq, predictedState: nextState };
  }

  /**
   * Process server acknowledgment. If there's a mismatch, reconcile.
   */
  reconcile(ack: ServerAck): ReconcileResult {
    this.lastAckedSeq = ack.seq;

    // Drop all pending inputs that have been acknowledged
    this.pendingInputs = this.pendingInputs.filter((p) => p.input.seq > ack.seq);

    const serverState = ack.serverState;
    const error       = stateError(this.currentState, serverState);

    if (error < RECONCILE_THRESHOLD) {
      // Prediction was accurate — smooth correction
      this.currentState = blendStates(this.currentState, serverState, 0.3);
      return { kind: "smooth", error };
    }

    // Large divergence — full re-simulation from server state
    let replayState = deepCloneEntity(serverState);
    const dt = 1 / 60; // assume fixed tick

    for (const pending of this.pendingInputs) {
      replayState = simulateEntity(replayState, pending.input, dt, this.physics);
      pending.state = deepCloneEntity(replayState);
    }

    this.currentState = replayState;
    return { kind: "correction", error };
  }

  get state(): EntityState        { return this.currentState; }
  get pendingCount(): number       { return this.pendingInputs.length; }
  get lastAcked(): Seq             { return this.lastAckedSeq; }
}

const RECONCILE_THRESHOLD = 0.25; // meters; errors below this get smoothed

interface ReconcileResult {
  kind:  "smooth" | "correction";
  error: number;
}

function stateError(predicted: EntityState, authoritative: EntityState): number {
  const dx = predicted.position.x - authoritative.position.x;
  const dy = predicted.position.y - authoritative.position.y;
  const dz = predicted.position.z - authoritative.position.z;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function blendStates(a: EntityState, b: EntityState, t: number): EntityState {
  return {
    ...a,
    position: lerpVec3(a.position, b.position, t),
    velocity: lerpVec3(a.velocity, b.velocity, t),
    yaw:      lerp(a.yaw, b.yaw, t),
    health:   lerp(a.health, b.health, t),
  };
}

// ─────────────────────────────────────────────
// § ENTITY INTERPOLATION (remote players)
// ─────────────────────────────────────────────

interface SnapshotBuffer {
  snapshots: { timestamp: Timestamp; state: EntityState }[];
}

const INTERP_DELAY_MS    = 100;  // render 100ms behind to have snapshots to interpolate between
const MAX_SNAPSHOT_BUFFER = 32;

export class EntityInterpolator {
  private readonly buffers = new Map<PlayerId, SnapshotBuffer>();

  /**
   * Push a new server snapshot for a remote entity.
   */
  pushSnapshot(id: PlayerId, state: EntityState, timestamp: Timestamp): void {
    let buf = this.buffers.get(id);
    if (!buf) {
      buf = { snapshots: [] };
      this.buffers.set(id, buf);
    }

    buf.snapshots.push({ timestamp, state: deepCloneEntity(state) });
    if (buf.snapshots.length > MAX_SNAPSHOT_BUFFER) {
      buf.snapshots.shift();
    }
  }

  /**
   * Get the interpolated state for a remote entity at the given render time.
   */
  getState(id: PlayerId, renderTime: Timestamp): EntityState | undefined {
    const buf = this.buffers.get(id);
    if (!buf || buf.snapshots.length < 2) return buf?.snapshots[0]?.state;

    const targetTime = (renderTime - INTERP_DELAY_MS) as Timestamp;

    // Find the two snapshots that bracket targetTime
    let prev = buf.snapshots[0];
    let next = buf.snapshots[1];

    for (let i = 1; i < buf.snapshots.length; i++) {
      if (buf.snapshots[i].timestamp >= targetTime) {
        prev = buf.snapshots[i - 1];
        next = buf.snapshots[i];
        break;
      }
      // If all snapshots are older than target, use the latest two for extrapolation
      prev = buf.snapshots[buf.snapshots.length - 2];
      next = buf.snapshots[buf.snapshots.length - 1];
    }

    const range = next.timestamp - prev.timestamp;
    if (range === 0) return prev.state;

    const t = Math.min(1, Math.max(0, (targetTime - prev.timestamp) / range));
    return blendStates(prev.state, next.state, t);
  }

  removeEntity(id: PlayerId): void {
    this.buffers.delete(id);
  }

  get trackedCount(): number { return this.buffers.size; }
}

// ─────────────────────────────────────────────
// § LAG COMPENSATOR (server-side)
// ─────────────────────────────────────────────

const LAG_COMP_WINDOW_MS = 200;

export class LagCompensator {
  private readonly history = new Map<PlayerId, WorldSnapshot[]>();

  /**
   * Record a world snapshot (called every server tick).
   */
  recordSnapshot(snapshot: WorldSnapshot): void {
    for (const [id, state] of snapshot.entities) {
      let arr = this.history.get(id);
      if (!arr) {
        arr = [];
        this.history.set(id, arr);
      }
      arr.push({ ...snapshot, entities: new Map([[id, deepCloneEntity(state)]]) });

      // Trim old history
      const cutoff = (snapshot.timestamp - LAG_COMP_WINDOW_MS) as Timestamp;
      while (arr.length > 0 && arr[0].timestamp < cutoff) arr.shift();
    }
  }

  /**
   * Rewind world state to attackerTimestamp for lag-compensated hit detection.
   * Returns a snapshot of entity positions at that historical moment.
   */
  rewindTo(
    targetTimestamp: Timestamp,
    entityIds: PlayerId[],
  ): Map<PlayerId, EntityState> {
    const result = new Map<PlayerId, EntityState>();

    for (const id of entityIds) {
      const hist = this.history.get(id);
      if (!hist || hist.length === 0) continue;

      // Binary search for closest snapshot
      let lo = 0, hi = hist.length - 1;
      while (lo < hi) {
        const mid = (lo + hi + 1) >> 1;
        if (hist[mid].timestamp <= targetTimestamp) lo = mid;
        else hi = mid - 1;
      }

      const prevSnap = hist[lo];
      const nextSnap = hist[Math.min(lo + 1, hist.length - 1)];

      if (prevSnap === nextSnap) {
        result.set(id, deepCloneEntity(prevSnap.entities.get(id)!));
        continue;
      }

      const range = nextSnap.timestamp - prevSnap.timestamp;
      const t     = range === 0 ? 0 : (targetTimestamp - prevSnap.timestamp) / range;
      const pState = prevSnap.entities.get(id)!;
      const nState = nextSnap.entities.get(id)!;
      result.set(id, blendStates(pState, nState, t));
    }

    return result;
  }

  /**
   * Server-side hit test with lag compensation.
   */
  hitTest(
    attackerTimestamp: Timestamp,
    attackerPosition:  Vec3Like,
    direction:         Vec3Like,
    range:             number,
    candidates:        PlayerId[],
  ): HitResult[] {
    const rewound = this.rewindTo(attackerTimestamp, candidates);
    const results: HitResult[] = [];

    for (const [id, state] of rewound) {
      const dist = raySphereDist(attackerPosition, direction, state.position, 0.5);
      if (dist !== null && dist <= range) {
        results.push({ entityId: id, distance: dist, hitPosition: state.position });
      }
    }

    return results.sort((a, b) => a.distance - b.distance);
  }
}

interface HitResult {
  entityId:    PlayerId;
  distance:    number;
  hitPosition: Vec3Like;
}

function raySphereDist(
  origin: Vec3Like, dir: Vec3Like,
  center: Vec3Like, radius: number,
): number | null {
  const oc = { x: origin.x - center.x, y: origin.y - center.y, z: origin.z - center.z };
  const a  = dir.x*dir.x + dir.y*dir.y + dir.z*dir.z;
  const b  = 2 * (oc.x*dir.x + oc.y*dir.y + oc.z*dir.z);
  const c  = oc.x*oc.x + oc.y*oc.y + oc.z*oc.z - radius*radius;
  const disc = b*b - 4*a*c;
  if (disc < 0) return null;
  const t = (-b - Math.sqrt(disc)) / (2*a);
  return t >= 0 ? t : null;
}

// ─────────────────────────────────────────────
// § NETWORK TRANSPORT — WebSocket with reconnect
// ─────────────────────────────────────────────

type MessageHandler = (data: ArrayBuffer) => void;

export class NetworkTransport {
  private ws:          WebSocket | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private _rtt       = 0;
  private _jitter    = 0;
  private pingSeq    = 0;
  private pingSent   = new Map<number, number>();  // seq → sendTime
  private readonly handlers = new Map<number, MessageHandler>();
  private msgQueue:  ArrayBuffer[] = [];

  constructor(
    private readonly url:            string,
    private readonly onConnect:      () => void,
    private readonly onDisconnect:   () => void,
    private readonly onMessage:      (typeId: number, data: DataView) => void,
  ) {}

  connect(): void {
    if (this.ws?.readyState === WebSocket.OPEN) return;

    this.ws = new WebSocket(this.url, ["aetheris-v1"]);
    this.ws.binaryType = "arraybuffer";

    this.ws.onopen = () => {
      if (this.reconnectTimer) {
        clearTimeout(this.reconnectTimer);
        this.reconnectTimer = null;
      }
      // Flush queued messages
      for (const msg of this.msgQueue) this.ws!.send(msg);
      this.msgQueue = [];
      this.onConnect();
      this.startPingLoop();
    };

    this.ws.onclose = (ev) => {
      this.onDisconnect();
      if (!ev.wasClean) {
        this.scheduleReconnect();
      }
    };

    this.ws.onerror = () => {
      this.ws?.close();
    };

    this.ws.onmessage = (ev) => {
      const buf  = ev.data as ArrayBuffer;
      const view = new DataView(buf);
      const typeId = view.getUint8(0);

      if (typeId === MSG_TYPE.PONG) {
        this.handlePong(view);
        return;
      }

      this.onMessage(typeId, new DataView(buf, 1));
    };
  }

  send(typeId: number, payload: ArrayBuffer): void {
    const header  = new ArrayBuffer(1);
    new DataView(header).setUint8(0, typeId);
    const full    = concatBuffers(header, payload);

    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(full);
    } else {
      this.msgQueue.push(full);
    }
  }

  sendInput(input: PlayerInput): void {
    const buf  = new ArrayBuffer(INPUT_PACKET_SIZE);
    const view = new DataView(buf);
    let  off   = 0;

    view.setUint32(off, input.seq,       false); off += 4;
    view.setUint32(off, input.tick,      false); off += 4;
    view.setFloat64(off, input.timestamp, false); off += 8;
    view.setFloat32(off, input.moveX,    false); off += 4;
    view.setFloat32(off, input.moveZ,    false); off += 4;
    view.setFloat32(off, input.yaw,      false); off += 4;
    view.setFloat32(off, input.pitch,    false); off += 4;
    view.setUint32(off, input.buttons,   false);

    this.send(MSG_TYPE.INPUT, buf);
  }

  disconnect(): void {
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.ws?.close(1000, "Client disconnect");
    this.ws = null;
  }

  get rtt():    number { return this._rtt; }
  get jitter(): number { return this._jitter; }

  private startPingLoop(): void {
    const ping = () => {
      if (this.ws?.readyState !== WebSocket.OPEN) return;
      const seq  = this.pingSeq++;
      const buf  = new ArrayBuffer(4);
      new DataView(buf).setUint32(0, seq);
      this.pingSent.set(seq, performance.now());
      this.send(MSG_TYPE.PING, buf);
      setTimeout(ping, 1000);
    };
    setTimeout(ping, 1000);
  }

  private handlePong(view: DataView): void {
    const seq  = view.getUint32(1);
    const sent = this.pingSent.get(seq);
    if (sent === undefined) return;
    const rtt       = performance.now() - sent;
    this._jitter    = Math.abs(rtt - this._rtt) * 0.1 + this._jitter * 0.9;
    this._rtt       = rtt * 0.1 + this._rtt * 0.9;
    this.pingSent.delete(seq);
  }

  private scheduleReconnect(delay = 2000): void {
    this.reconnectTimer = setTimeout(() => {
      console.log("[Transport] Reconnecting...");
      this.connect();
    }, delay);
  }
}

const MSG_TYPE = {
  INPUT:      0x01,
  WORLD_DELTA: 0x02,
  SERVER_ACK:  0x03,
  PING:        0x10,
  PONG:        0x11,
  SPAWN:       0x20,
  DESPAWN:     0x21,
} as const;

const INPUT_PACKET_SIZE = 4 + 4 + 8 + 4 + 4 + 4 + 4 + 4; // 36 bytes

function concatBuffers(a: ArrayBuffer, b: ArrayBuffer): ArrayBuffer {
  const out = new Uint8Array(a.byteLength + b.byteLength);
  out.set(new Uint8Array(a), 0);
  out.set(new Uint8Array(b), a.byteLength);
  return out.buffer;
}

// ─────────────────────────────────────────────
// § MULTIPLAYER SYNC SYSTEM (kernel integration)
// ─────────────────────────────────────────────

export interface MultiplayerSyncConfig {
  serverUrl: string;
  localPlayerId: PlayerId;
  sessionId:     SessionId;
  tickRate:      number;
}

@System({
  id:          "aetheris.multiplayer" as SystemId,
  priority:    50,
  updateHz:    60 as Hz,
  dependencies: ["aetheris.kernel" as SystemId],
})
export class MultiplayerSync implements ISystem {
  readonly id       = "aetheris.multiplayer" as SystemId;
  readonly priority = 50;

  private prediction!:    ClientPrediction;
  private interpolator!:  EntityInterpolator;
  private lagCompensator: LagCompensator = new LagCompensator();
  private transport!:     NetworkTransport;

  private localState!:    EntityState;
  private worldEntities   = new Map<PlayerId, EntityState>();
  private connected       = false;
  private serverTick      = TickNum(0);

  private reconcileStats = {
    total:       0,
    corrections: 0,
    avgError:    0,
  };

  constructor(private readonly config: MultiplayerSyncConfig) {}

  async onInit(kernel: IKernel): Promise<void> {
    this.localState = this.createDefaultPlayerState(this.config.localPlayerId);
    this.prediction  = new ClientPrediction(this.localState);
    this.interpolator = new EntityInterpolator();

    this.transport = new NetworkTransport(
      this.config.serverUrl,
      ()  => { this.connected = true;  kernel.emit("multiplayer:connected",    {}); },
      ()  => { this.connected = false; kernel.emit("multiplayer:disconnected", {}); },
      (typeId, data) => this.handleMessage(typeId, data),
    );

    this.transport.connect();
  }

  onUpdate(dt: number, frame: FrameIndex): void {
    if (!this.connected) return;
    // Polling inputs is done by the InputSystem; here we just advance
    // interpolation and expose current states to the render system
    const now = performance.now() as Timestamp;

    for (const [id] of this.worldEntities) {
      if (id === this.config.localPlayerId) continue;
      const interp = this.interpolator.getState(id, now);
      if (interp) this.worldEntities.set(id, interp);
    }
  }

  async onDestroy(): Promise<void> {
    this.transport.disconnect();
  }

  /**
   * Called by InputSystem with raw player input each frame.
   */
  submitInput(rawInput: Omit<PlayerInput, "seq">, dt: number): void {
    if (!this.connected) return;
    const { seq, predictedState } = this.prediction.applyInput(rawInput, dt);
    this.localState = predictedState;
    this.worldEntities.set(this.config.localPlayerId, predictedState);
    this.transport.sendInput({ ...rawInput, seq });
  }

  /**
   * Get all entity states for rendering (local predicted + remote interpolated).
   */
  getWorldState(): ReadonlyMap<PlayerId, EntityState> {
    return this.worldEntities;
  }

  get rtt():    number { return this.transport.rtt; }
  get jitter(): number { return this.transport.jitter; }
  get stats()          { return { ...this.reconcileStats, connected: this.connected }; }

  // ── Private ──

  private handleMessage(typeId: number, data: DataView): void {
    switch (typeId) {
      case MSG_TYPE.SERVER_ACK:
        this.handleServerAck(data); break;
      case MSG_TYPE.WORLD_DELTA:
        this.handleWorldDelta(data); break;
      case MSG_TYPE.SPAWN:
        this.handleSpawn(data); break;
      case MSG_TYPE.DESPAWN:
        this.handleDespawn(data); break;
    }
  }

  private handleServerAck(view: DataView): void {
    let off = 0;
    const seq       = Seq(view.getUint32(off, false));   off += 4;
    const tick      = TickNum(view.getUint32(off, false)); off += 4;
    const serverState = this.readEntityState(view, off);

    const ack: ServerAck = { seq, tick, serverState, worldTick: this.serverTick };
    const result = this.prediction.reconcile(ack);

    this.reconcileStats.total++;
    if (result.kind === "correction") this.reconcileStats.corrections++;
    this.reconcileStats.avgError =
      (this.reconcileStats.avgError * 0.95) + (result.error * 0.05);
  }

  private handleWorldDelta(view: DataView): void {
    let off = 0;
    const tick      = TickNum(view.getUint32(off, false));  off += 4;
    const timestamp = view.getFloat64(off, false) as Timestamp; off += 8;
    const count     = view.getUint16(off, false);           off += 2;

    this.serverTick = tick;

    for (let i = 0; i < count; i++) {
      const idLen  = view.getUint8(off); off += 1;
      const idBytes = new Uint8Array(view.buffer, view.byteOffset + off, idLen);
      const id     = PlayerId(new TextDecoder().decode(idBytes)); off += idLen;
      const mask   = view.getUint8(off); off += 1;

      const existing = this.worldEntities.get(id) ?? this.createDefaultPlayerState(id);
      const updated  = this.applyDelta(existing, mask, view, off);
      off += deltaSize(mask);

      this.worldEntities.set(id, updated);

      if (id !== this.config.localPlayerId) {
        this.interpolator.pushSnapshot(id, updated, timestamp);
        this.lagCompensator.recordSnapshot({
          tick,
          timestamp,
          entities: new Map([[id, updated]]),
        });
      }
    }
  }

  private handleSpawn(view: DataView): void {
    const idLen  = view.getUint8(0);
    const idBytes = new Uint8Array(view.buffer, view.byteOffset + 1, idLen);
    const id     = PlayerId(new TextDecoder().decode(idBytes));
    this.worldEntities.set(id, this.createDefaultPlayerState(id));
  }

  private handleDespawn(view: DataView): void {
    const idLen  = view.getUint8(0);
    const idBytes = new Uint8Array(view.buffer, view.byteOffset + 1, idLen);
    const id     = PlayerId(new TextDecoder().decode(idBytes));
    this.worldEntities.delete(id);
    this.interpolator.removeEntity(id);
  }

  private readEntityState(view: DataView, off: number): EntityState {
    return {
      id:        this.config.localPlayerId,
      position:  { x: view.getFloat32(off, false), y: view.getFloat32(off+4, false), z: view.getFloat32(off+8, false) },
      velocity:  { x: view.getFloat32(off+12, false), y: view.getFloat32(off+16, false), z: view.getFloat32(off+20, false) },
      yaw:       view.getFloat32(off+24, false),
      pitch:     view.getFloat32(off+28, false),
      health:    view.getFloat32(off+32, false),
      animation: view.getUint8(off+36),
      flags:     view.getUint32(off+37, false),
      lastTick:  TickNum(0),
    };
  }

  private applyDelta(base: EntityState, mask: number, view: DataView, off: number): EntityState {
    const next = deepCloneEntity(base);
    if (mask & DELTA_MASK.POSITION) {
      next.position = { x: view.getFloat32(off, false), y: view.getFloat32(off+4, false), z: view.getFloat32(off+8, false) };
      off += 12;
    }
    if (mask & DELTA_MASK.VELOCITY) {
      next.velocity = { x: view.getFloat32(off, false), y: view.getFloat32(off+4, false), z: view.getFloat32(off+8, false) };
      off += 12;
    }
    if (mask & DELTA_MASK.YAW)      { next.yaw       = view.getFloat32(off, false); off += 4; }
    if (mask & DELTA_MASK.PITCH)    { next.pitch      = view.getFloat32(off, false); off += 4; }
    if (mask & DELTA_MASK.HEALTH)   { next.health     = view.getFloat32(off, false); off += 4; }
    if (mask & DELTA_MASK.ANIMATION){ next.animation  = view.getUint8(off);          off += 1; }
    if (mask & DELTA_MASK.FLAGS)    { next.flags      = view.getUint32(off, false);  off += 4; }
    return next;
  }

  private createDefaultPlayerState(id: PlayerId): EntityState {
    return {
      id, position: { x: 0, y: 0, z: 0 }, velocity: { x: 0, y: 0, z: 0 },
      yaw: 0, pitch: 0, health: 100, animation: 0, flags: 0, lastTick: TickNum(0),
    };
  }
}

function deltaSize(mask: number): number {
  let size = 0;
  if (mask & DELTA_MASK.POSITION)  size += 12;
  if (mask & DELTA_MASK.VELOCITY)  size += 12;
  if (mask & DELTA_MASK.YAW)       size += 4;
  if (mask & DELTA_MASK.PITCH)     size += 4;
  if (mask & DELTA_MASK.HEALTH)    size += 4;
  if (mask & DELTA_MASK.ANIMATION) size += 1;
  if (mask & DELTA_MASK.FLAGS)     size += 4;
  return size;
}

declare const Hz: unknown;
