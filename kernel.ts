/**
 * @file Kernel.ts
 * @description AETHERIS OMNIVERSE — Core Gaming OS Kernel
 * System Orchestrator with DI Container, Fixed-Timestep Scheduler,
 * Object Pooling, and typed EventBus backbone.
 *
 * Architecture: Domain-Driven Design + Hexagonal Architecture
 * TypeScript: 5.4+ with decorators, advanced generics, utility types
 */

import "reflect-metadata";

// ─────────────────────────────────────────────
// § BRANDED TYPES & DOMAIN PRIMITIVES
// ─────────────────────────────────────────────

declare const __brand: unique symbol;
type Brand<T, B> = T & { readonly [__brand]: B };

export type SystemId   = Brand<string, "SystemId">;
export type FrameIndex = Brand<number, "FrameIndex">;
export type Timestamp  = Brand<number, "Timestamp">;
export type Hz         = Brand<number, "Hz">;

export const SystemId   = (s: string): SystemId   => s as SystemId;
export const FrameIndex = (n: number): FrameIndex => n as FrameIndex;
export const Timestamp  = (n: number): Timestamp  => n as Timestamp;
export const Hz         = (n: number): Hz         => n as Hz;

// ─────────────────────────────────────────────
// § DECORATORS (TypeScript 5.4 Stage-3)
// ─────────────────────────────────────────────

const INJECTABLE_META  = Symbol("aetheris:injectable");
const INJECT_TOKENS    = Symbol("aetheris:inject-tokens");
const SYSTEM_META      = Symbol("aetheris:system");

export function Injectable(): ClassDecorator {
  return (target) => {
    Reflect.defineMetadata(INJECTABLE_META, true, target);
  };
}

export function Inject(token: string | symbol): ParameterDecorator {
  return (target, _, paramIndex) => {
    const existing: Array<string | symbol> =
      Reflect.getOwnMetadata(INJECT_TOKENS, target) ?? [];
    existing[paramIndex] = token;
    Reflect.defineMetadata(INJECT_TOKENS, existing, target);
  };
}

export interface SystemMeta {
  id: SystemId;
  priority: number;
  updateHz: Hz;
  dependencies?: SystemId[];
}

export function System(meta: SystemMeta): ClassDecorator {
  return (target) => {
    Reflect.defineMetadata(SYSTEM_META, meta, target);
    Reflect.defineMetadata(INJECTABLE_META, true, target);
  };
}

// ─────────────────────────────────────────────
// § INTERFACES — HEXAGONAL PORTS
// ─────────────────────────────────────────────

export interface ISystem {
  readonly id: SystemId;
  readonly priority: number;
  onInit(kernel: IKernel): Promise<void>;
  onUpdate(dt: number, frame: FrameIndex): void;
  onDestroy(): Promise<void>;
}

export interface IKernel {
  readonly frame: FrameIndex;
  readonly time: Timestamp;
  readonly deltaTime: number;
  registerSystem(system: ISystem): void;
  getSystem<T extends ISystem>(id: SystemId): T;
  emit<T>(event: string, payload: T): void;
  on<T>(event: string, handler: (payload: T) => void): () => void;
  resolve<T>(token: string | symbol): T;
}

export interface IPoolable {
  reset(): void;
}

// ─────────────────────────────────────────────
// § OBJECT POOL — Zero-allocation hot path
// ─────────────────────────────────────────────

export class ObjectPool<T extends IPoolable> {
  private readonly pool: T[] = [];
  private readonly active = new Set<T>();
  private totalCreated = 0;

  constructor(
    private readonly factory: () => T,
    private readonly initialSize: number,
    private readonly maxSize: number,
    private readonly poolName: string,
  ) {
    for (let i = 0; i < initialSize; i++) {
      this.pool.push(this.factory());
      this.totalCreated++;
    }
  }

  acquire(): T {
    let obj = this.pool.pop();

    if (!obj) {
      if (this.active.size >= this.maxSize) {
        throw new PoolExhaustionError(
          `[ObjectPool:${this.poolName}] Pool exhausted. max=${this.maxSize}`,
        );
      }
      obj = this.factory();
      this.totalCreated++;
    }

    this.active.add(obj);
    return obj;
  }

  release(obj: T): void {
    if (!this.active.has(obj)) {
      console.warn(`[ObjectPool:${this.poolName}] Releasing unknown object`);
      return;
    }
    this.active.delete(obj);
    obj.reset();
    this.pool.push(obj);
  }

  releaseAll(): void {
    for (const obj of this.active) {
      obj.reset();
      this.pool.push(obj);
    }
    this.active.clear();
  }

  get stats() {
    return {
      pooled: this.pool.length,
      active: this.active.size,
      totalCreated: this.totalCreated,
      utilization: this.active.size / this.maxSize,
    };
  }
}

// ─────────────────────────────────────────────
// § EVENT BUS — Typed pub/sub with wildcards
// ─────────────────────────────────────────────

type EventMap = Record<string, unknown>;
type Handler<T> = (payload: T) => void;

interface EventSubscription {
  id: number;
  event: string;
  handler: Handler<unknown>;
  once: boolean;
}

export class TypedEventBus<TEvents extends EventMap = EventMap> {
  private readonly subscriptions = new Map<string, EventSubscription[]>();
  private subIdCounter = 0;
  private readonly metrics = new Map<string, number>();

  on<K extends keyof TEvents & string>(
    event: K,
    handler: Handler<TEvents[K]>,
  ): () => void {
    return this.addSubscription(event, handler as Handler<unknown>, false);
  }

  once<K extends keyof TEvents & string>(
    event: K,
    handler: Handler<TEvents[K]>,
  ): () => void {
    return this.addSubscription(event, handler as Handler<unknown>, true);
  }

  emit<K extends keyof TEvents & string>(event: K, payload: TEvents[K]): void {
    const subs = this.subscriptions.get(event);
    if (!subs?.length) return;

    this.metrics.set(event, (this.metrics.get(event) ?? 0) + 1);

    const toRemove: EventSubscription[] = [];
    for (const sub of subs) {
      try {
        sub.handler(payload);
      } catch (err) {
        console.error(`[EventBus] Error in handler for "${event}":`, err);
      }
      if (sub.once) toRemove.push(sub);
    }

    if (toRemove.length) {
      const remaining = subs.filter((s) => !toRemove.includes(s));
      this.subscriptions.set(event, remaining);
    }
  }

  private addSubscription(
    event: string,
    handler: Handler<unknown>,
    once: boolean,
  ): () => void {
    const id = ++this.subIdCounter;
    const sub: EventSubscription = { id, event, handler, once };

    const existing = this.subscriptions.get(event) ?? [];
    this.subscriptions.set(event, [...existing, sub]);

    return () => {
      const subs = this.subscriptions.get(event) ?? [];
      this.subscriptions.set(
        event,
        subs.filter((s) => s.id !== id),
      );
    };
  }

  getMetrics(): ReadonlyMap<string, number> {
    return this.metrics;
  }
}

// ─────────────────────────────────────────────
// § DI CONTAINER
// ─────────────────────────────────────────────

type Constructor<T = unknown> = new (...args: unknown[]) => T;
type Factory<T> = () => T;
type Binding<T> = { kind: "singleton" | "transient"; value: T | Factory<T> };

export class DIContainer {
  private readonly bindings = new Map<string | symbol, Binding<unknown>>();
  private readonly singletons = new Map<string | symbol, unknown>();

  bindSingleton<T>(token: string | symbol, factory: Factory<T>): this {
    this.bindings.set(token, { kind: "singleton", value: factory });
    return this;
  }

  bindTransient<T>(token: string | symbol, factory: Factory<T>): this {
    this.bindings.set(token, { kind: "transient", value: factory });
    return this;
  }

  bindValue<T>(token: string | symbol, value: T): this {
    this.singletons.set(token, value);
    return this;
  }

  resolve<T>(token: string | symbol): T {
    if (this.singletons.has(token)) {
      return this.singletons.get(token) as T;
    }

    const binding = this.bindings.get(token);
    if (!binding) {
      throw new DIResolutionError(`[DIContainer] No binding for token: ${String(token)}`);
    }

    if (binding.kind === "singleton") {
      const existing = this.singletons.get(token);
      if (existing !== undefined) return existing as T;

      const instance = (binding.value as Factory<T>)();
      this.singletons.set(token, instance);
      return instance;
    }

    return (binding.value as Factory<T>)();
  }

  resolveClass<T>(ctor: Constructor<T>): T {
    const isInjectable = Reflect.getMetadata(INJECTABLE_META, ctor);
    if (!isInjectable) {
      throw new DIResolutionError(`[DIContainer] Class ${ctor.name} is not @Injectable`);
    }

    const tokens: Array<string | symbol> =
      Reflect.getOwnMetadata(INJECT_TOKENS, ctor) ?? [];

    const args = tokens.map((token) => this.resolve(token));
    return new ctor(...(args as ConstructorParameters<typeof ctor>));
  }
}

// ─────────────────────────────────────────────
// § SCHEDULER — Fixed-timestep with spiral of death prevention
// ─────────────────────────────────────────────

interface SchedulerConfig {
  physicsHz: Hz;
  renderHz: Hz;
  maxCatchupFrames: number;
}

export class FixedTimestepScheduler {
  private readonly physicsInterval: number;
  private readonly renderInterval: number;
  private accumulator = 0;
  private lastTime = 0;
  private _running = false;
  private rafHandle = 0;
  private _frameIndex = FrameIndex(0);

  private readonly physicsSystems: ISystem[] = [];
  private readonly renderSystems:  ISystem[] = [];

  constructor(
    private readonly config: SchedulerConfig,
    private readonly onPhysicsFrame: (dt: number, frame: FrameIndex) => void,
    private readonly onRenderFrame:  (alpha: number, frame: FrameIndex) => void,
    private readonly onMetrics:      (metrics: SchedulerMetrics) => void,
  ) {
    this.physicsInterval = 1000 / config.physicsHz;
    this.renderInterval  = 1000 / config.renderHz;
  }

  start(): void {
    if (this._running) return;
    this._running = true;
    this.lastTime = performance.now();
    this.loop(this.lastTime);
  }

  stop(): void {
    this._running = false;
    cancelAnimationFrame(this.rafHandle);
  }

  private loop = (now: number): void => {
    if (!this._running) return;

    const elapsed = now - this.lastTime;
    this.lastTime = now;

    // Spiral-of-death prevention: cap accumulator
    const cappedElapsed = Math.min(
      elapsed,
      this.physicsInterval * this.config.maxCatchupFrames,
    );

    this.accumulator += cappedElapsed;

    let physicsSteps = 0;
    const physicsStart = performance.now();

    while (this.accumulator >= this.physicsInterval) {
      this.onPhysicsFrame(this.physicsInterval / 1000, this._frameIndex);
      this._frameIndex = FrameIndex(this._frameIndex + 1);
      this.accumulator -= this.physicsInterval;
      physicsSteps++;
    }

    const physicsMs = performance.now() - physicsStart;

    // Alpha for interpolation between physics states
    const alpha = this.accumulator / this.physicsInterval;
    const renderStart = performance.now();
    this.onRenderFrame(alpha, this._frameIndex);
    const renderMs = performance.now() - renderStart;

    this.onMetrics({
      frameIndex: this._frameIndex,
      elapsed,
      physicsSteps,
      physicsMs,
      renderMs,
      accumulator: this.accumulator,
    });

    this.rafHandle = requestAnimationFrame(this.loop);
  };

  get frameIndex(): FrameIndex { return this._frameIndex; }
  get isRunning(): boolean      { return this._running; }
}

interface SchedulerMetrics {
  frameIndex: FrameIndex;
  elapsed: number;
  physicsSteps: number;
  physicsMs: number;
  renderMs: number;
  accumulator: number;
}

// ─────────────────────────────────────────────
// § KERNEL EVENTS
// ─────────────────────────────────────────────

export interface KernelEvents {
  "kernel:ready":           { timestamp: Timestamp };
  "kernel:shutdown":        { reason: string };
  "kernel:system:init":     { systemId: SystemId };
  "kernel:system:error":    { systemId: SystemId; error: Error };
  "kernel:frame:physics":   { frame: FrameIndex; dt: number };
  "kernel:frame:render":    { frame: FrameIndex; alpha: number };
  "kernel:metrics":         { metrics: SchedulerMetrics };
}

// ─────────────────────────────────────────────
// § KERNEL IMPLEMENTATION
// ─────────────────────────────────────────────

export class AetherisKernel implements IKernel {
  private readonly systems       = new Map<SystemId, ISystem>();
  private readonly eventBus      = new TypedEventBus<KernelEvents & Record<string, unknown>>();
  private readonly container     = new DIContainer();
  private readonly scheduler:    FixedTimestepScheduler;

  private _frame     = FrameIndex(0);
  private _time      = Timestamp(0);
  private _deltaTime = 0;
  private _initialized = false;

  constructor(config: KernelConfig = DEFAULT_KERNEL_CONFIG) {
    this.scheduler = new FixedTimestepScheduler(
      {
        physicsHz:         config.physicsHz,
        renderHz:          config.renderHz,
        maxCatchupFrames:  config.maxCatchupFrames,
      },
      this.onPhysicsUpdate.bind(this),
      this.onRenderUpdate.bind(this),
      (m) => this.eventBus.emit("kernel:metrics", { metrics: m }),
    );

    this.container.bindValue("kernel", this);
  }

  // ── Lifecycle ──

  async init(): Promise<void> {
    if (this._initialized) throw new KernelError("Kernel already initialized");

    const sorted = this.topologicalSort();

    for (const system of sorted) {
      try {
        await system.onInit(this);
        this.eventBus.emit("kernel:system:init", { systemId: system.id });
      } catch (err) {
        const error = err instanceof Error ? err : new Error(String(err));
        this.eventBus.emit("kernel:system:error", { systemId: system.id, error });
        throw new KernelError(`System ${system.id} failed to init: ${error.message}`);
      }
    }

    this._initialized = true;
    this.eventBus.emit("kernel:ready", { timestamp: Timestamp(performance.now()) });
    this.scheduler.start();
  }

  async shutdown(reason = "User requested"): Promise<void> {
    this.scheduler.stop();
    this.eventBus.emit("kernel:shutdown", { reason });

    const sorted = this.topologicalSort().reverse();
    for (const system of sorted) {
      await system.onDestroy().catch((e) =>
        console.error(`[Kernel] System ${system.id} shutdown error:`, e),
      );
    }
  }

  // ── System Management ──

  registerSystem(system: ISystem): void {
    if (this._initialized) {
      throw new KernelError("Cannot register systems after initialization");
    }
    if (this.systems.has(system.id)) {
      throw new KernelError(`System ${system.id} already registered`);
    }
    this.systems.set(system.id, system);
  }

  getSystem<T extends ISystem>(id: SystemId): T {
    const sys = this.systems.get(id);
    if (!sys) throw new KernelError(`System "${id}" not found`);
    return sys as T;
  }

  // ── EventBus delegation ──

  emit<T>(event: string, payload: T): void {
    this.eventBus.emit(event as keyof KernelEvents & string, payload as never);
  }

  on<T>(event: string, handler: (payload: T) => void): () => void {
    return this.eventBus.on(event as keyof KernelEvents & string, handler as never);
  }

  // ── DI delegation ──

  resolve<T>(token: string | symbol): T {
    return this.container.resolve<T>(token);
  }

  get container_(): DIContainer { return this.container; }

  // ── Getters ──

  get frame():     FrameIndex { return this._frame; }
  get time():      Timestamp  { return this._time; }
  get deltaTime(): number     { return this._deltaTime; }

  // ── Private ──

  private onPhysicsUpdate(dt: number, frame: FrameIndex): void {
    this._frame     = frame;
    this._deltaTime = dt;
    this._time      = Timestamp(this._time + dt * 1000);

    for (const system of this.systemsByPriority()) {
      system.onUpdate(dt, frame);
    }

    this.eventBus.emit("kernel:frame:physics", { frame, dt });
  }

  private onRenderUpdate(alpha: number, frame: FrameIndex): void {
    this.eventBus.emit("kernel:frame:render", { frame, alpha });
  }

  private systemsByPriority(): ISystem[] {
    return [...this.systems.values()].sort((a, b) => a.priority - b.priority);
  }

  private topologicalSort(): ISystem[] {
    const visited  = new Set<SystemId>();
    const sorted:   ISystem[] = [];

    const visit = (id: SystemId): void => {
      if (visited.has(id)) return;
      const sys = this.systems.get(id);
      if (!sys) throw new KernelError(`Missing dependency: ${id}`);

      const meta: SystemMeta | undefined = Reflect.getMetadata(SYSTEM_META, sys.constructor);
      for (const dep of meta?.dependencies ?? []) {
        visit(dep);
      }

      visited.add(id);
      sorted.push(sys);
    };

    for (const id of this.systems.keys()) {
      visit(id);
    }
    return sorted;
  }
}

// ─────────────────────────────────────────────
// § CONFIG & ERRORS
// ─────────────────────────────────────────────

export interface KernelConfig {
  physicsHz:        Hz;
  renderHz:         Hz;
  maxCatchupFrames: number;
}

export const DEFAULT_KERNEL_CONFIG: KernelConfig = {
  physicsHz:        Hz(120),
  renderHz:         Hz(60),
  maxCatchupFrames: 5,
};

export class KernelError        extends Error { constructor(m: string) { super(m); this.name = "KernelError"; } }
export class DIResolutionError  extends Error { constructor(m: string) { super(m); this.name = "DIResolutionError"; } }
export class PoolExhaustionError extends Error { constructor(m: string) { super(m); this.name = "PoolExhaustionError"; } }
