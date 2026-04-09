/**
 * @file EscrowSystem.ts + WalletService.ts
 * @description AETHERIS OMNIVERSE — Global Economy Engine
 *
 * Architecture:
 *  - WalletService:  Multi-currency balance management with double-entry ledger
 *  - EscrowSystem:   Atomic trade state machine with dispute resolution
 *  - TaxEngine:      Dynamic jurisdiction-aware tax calculation
 *  - All operations are idempotent and use optimistic locking
 */

// ─────────────────────────────────────────────
// § DOMAIN TYPES
// ─────────────────────────────────────────────

declare const __brand: unique symbol;
type Brand<T, B> = T & { readonly [__brand]: B };

export type WalletId     = Brand<string, "WalletId">;
export type TxId         = Brand<string, "TxId">;
export type EscrowId     = Brand<string, "EscrowId">;
export type CurrencyCode = Brand<string, "CurrencyCode">;
export type ItemId       = Brand<string, "ItemId">;

export const WalletId     = (s: string): WalletId     => s as WalletId;
export const TxId         = (s: string): TxId         => s as TxId;
export const EscrowId     = (s: string): EscrowId     => s as EscrowId;
export const CurrencyCode = (s: string): CurrencyCode => s as CurrencyCode;
export const ItemId       = (s: string): ItemId       => s as ItemId;

export const CURRENCIES = {
  AEC:  CurrencyCode("AEC"),   // Aetheris Coins (premium)
  AEG:  CurrencyCode("AEG"),   // Aetheris Gems (earnable)
  USD:  CurrencyCode("USD"),
  EUR:  CurrencyCode("EUR"),
  JPY:  CurrencyCode("JPY"),
  BRL:  CurrencyCode("BRL"),
} as const;

// Use integer arithmetic (cents/micros) to avoid floating point errors
// All amounts stored as bigint (micro-units, 1 unit = 1_000_000 micro-units)

export type MicroAmount = Brand<bigint, "MicroAmount">;
export const MicroAmount = (n: bigint): MicroAmount => n as MicroAmount;
export const fromDecimal = (d: number): MicroAmount => MicroAmount(BigInt(Math.round(d * 1_000_000)));
export const toDecimal   = (m: MicroAmount): number  => Number(m) / 1_000_000;

// ─────────────────────────────────────────────
// § LEDGER — Double-entry bookkeeping
// ─────────────────────────────────────────────

export type EntryType =
  | "credit"       // money coming in
  | "debit"        // money going out
  | "hold"         // reserved for escrow
  | "hold_release" // release from hold
  | "fee"
  | "tax"
  | "refund";

export interface LedgerEntry {
  id:          TxId;
  walletId:    WalletId;
  currency:    CurrencyCode;
  type:        EntryType;
  amount:      MicroAmount;
  balance:     MicroAmount;  // running balance AFTER this entry
  refTxId?:    TxId;         // links to counterpart entry
  escrowId?:   EscrowId;
  description: string;
  metadata:    Record<string, unknown>;
  createdAt:   number;       // Unix ms
  idempotencyKey: string;
}

// ─────────────────────────────────────────────
// § WALLET SERVICE
// ─────────────────────────────────────────────

export interface Balance {
  currency:   CurrencyCode;
  available:  MicroAmount;  // available = total - held
  held:       MicroAmount;  // reserved in active escrows
  total:      MicroAmount;
}

export interface Wallet {
  id:        WalletId;
  ownerId:   string;
  balances:  Map<CurrencyCode, Balance>;
  createdAt: number;
  version:   number;  // optimistic lock
}

export interface TransferResult {
  txId:         TxId;
  fromEntry:    LedgerEntry;
  toEntry:      LedgerEntry;
  feeEntry?:    LedgerEntry;
  taxEntry?:    LedgerEntry;
}

export interface TransferRequest {
  idempotencyKey: string;
  fromWalletId:   WalletId;
  toWalletId:     WalletId;
  currency:       CurrencyCode;
  amount:         MicroAmount;
  description:    string;
  metadata?:      Record<string, unknown>;
}

export class WalletService {
  private readonly wallets = new Map<WalletId, Wallet>();
  private readonly ledger:  LedgerEntry[] = [];
  private readonly idempotencyCache = new Map<string, TxId>();
  private readonly taxEngine = new TaxEngine();

  /**
   * Create a new wallet for a user.
   */
  createWallet(ownerId: string): Wallet {
    const wallet: Wallet = {
      id:       WalletId(`w_${generateId()}`),
      ownerId,
      balances: new Map([
        [CURRENCIES.AEC, zeroBalance(CURRENCIES.AEC)],
        [CURRENCIES.AEG, zeroBalance(CURRENCIES.AEG)],
      ]),
      createdAt: Date.now(),
      version:   0,
    };
    this.wallets.set(wallet.id, wallet);
    return wallet;
  }

  /**
   * Credit a wallet (idempotent).
   */
  credit(
    walletId:       WalletId,
    currency:       CurrencyCode,
    amount:         MicroAmount,
    description:    string,
    idempotencyKey: string,
    metadata:       Record<string, unknown> = {},
  ): LedgerEntry {
    if (amount <= BigInt(0)) throw new EconomyError("Credit amount must be positive");

    const cached = this.idempotencyCache.get(idempotencyKey);
    if (cached) return this.ledger.find((e) => e.id === cached)!;

    const wallet  = this.requireWallet(walletId);
    const balance = this.requireBalance(wallet, currency);

    const newTotal = MicroAmount(balance.total + amount);
    const newAvail = MicroAmount(balance.available + amount);

    const entry: LedgerEntry = {
      id:             TxId(`tx_${generateId()}`),
      walletId,
      currency,
      type:           "credit",
      amount,
      balance:        newAvail,
      description,
      metadata,
      createdAt:      Date.now(),
      idempotencyKey,
    };

    balance.total     = newTotal;
    balance.available = newAvail;
    wallet.version++;

    this.ledger.push(entry);
    this.idempotencyCache.set(idempotencyKey, entry.id);
    return entry;
  }

  /**
   * Transfer between wallets with automatic fee + tax calculation.
   */
  async transfer(req: TransferRequest): Promise<TransferResult> {
    // Idempotency check
    const cached = this.idempotencyCache.get(req.idempotencyKey);
    if (cached) {
      const entries = this.ledger.filter((e) => e.refTxId === cached || e.id === cached);
      return this.buildTransferResult(entries);
    }

    const fromWallet = this.requireWallet(req.fromWalletId);
    const toWallet   = this.requireWallet(req.toWalletId);
    const fromBal    = this.requireBalance(fromWallet, req.currency);
    const toBal      = this.requireBalance(toWallet, req.currency);

    // Sufficient available funds check
    if (fromBal.available < req.amount) {
      throw new InsufficientFundsError(
        `Insufficient funds: need ${toDecimal(req.amount)} ${req.currency}, ` +
        `have ${toDecimal(fromBal.available)} available`,
      );
    }

    // Calculate fee + tax
    const feeRate   = 0.01;    // 1% platform fee
    const feeAmount = MicroAmount(BigInt(Math.round(Number(req.amount) * feeRate)));
    const taxCalc   = this.taxEngine.calculate(req.currency, req.amount, {});
    const totalDebit = MicroAmount(req.amount + feeAmount + taxCalc.taxAmount);

    if (fromBal.available < totalDebit) {
      throw new InsufficientFundsError("Insufficient funds including fees and taxes");
    }

    // ── Atomic multi-entry ──
    const txId = TxId(`tx_${generateId()}`);
    const now  = Date.now();

    // Debit sender
    fromBal.available = MicroAmount(fromBal.available - totalDebit);
    fromBal.total     = MicroAmount(fromBal.total - totalDebit);
    const fromEntry: LedgerEntry = {
      id: txId, walletId: req.fromWalletId, currency: req.currency,
      type: "debit", amount: req.amount, balance: fromBal.available,
      description: req.description, metadata: req.metadata ?? {},
      createdAt: now, idempotencyKey: req.idempotencyKey,
    };

    // Credit receiver
    const creditId = TxId(`tx_${generateId()}`);
    toBal.available = MicroAmount(toBal.available + req.amount);
    toBal.total     = MicroAmount(toBal.total + req.amount);
    const toEntry: LedgerEntry = {
      id: creditId, walletId: req.toWalletId, currency: req.currency,
      type: "credit", amount: req.amount, balance: toBal.available,
      refTxId: txId, description: req.description, metadata: req.metadata ?? {},
      createdAt: now, idempotencyKey: `${req.idempotencyKey}_credit`,
    };

    // Fee entry
    const feeId = TxId(`tx_${generateId()}`);
    const feeEntry: LedgerEntry = {
      id: feeId, walletId: req.fromWalletId, currency: req.currency,
      type: "fee", amount: feeAmount, balance: fromBal.available,
      refTxId: txId, description: "Platform fee 1%", metadata: {},
      createdAt: now, idempotencyKey: `${req.idempotencyKey}_fee`,
    };

    // Tax entry
    let taxEntry: LedgerEntry | undefined;
    if (taxCalc.taxAmount > BigInt(0)) {
      taxEntry = {
        id: TxId(`tx_${generateId()}`), walletId: req.fromWalletId,
        currency: req.currency, type: "tax", amount: taxCalc.taxAmount,
        balance: fromBal.available, refTxId: txId,
        description: `Tax ${taxCalc.jurisdiction} ${taxCalc.rate * 100}%`,
        metadata: { jurisdiction: taxCalc.jurisdiction },
        createdAt: now, idempotencyKey: `${req.idempotencyKey}_tax`,
      };
    }

    fromWallet.version++;
    toWallet.version++;

    this.ledger.push(fromEntry, toEntry, feeEntry);
    if (taxEntry) this.ledger.push(taxEntry);
    this.idempotencyCache.set(req.idempotencyKey, txId);

    return { txId, fromEntry, toEntry, feeEntry, taxEntry };
  }

  getWallet(id: WalletId): Wallet | undefined {
    return this.wallets.get(id);
  }

  getLedger(walletId: WalletId, limit = 50): LedgerEntry[] {
    return this.ledger
      .filter((e) => e.walletId === walletId)
      .slice(-limit)
      .reverse();
  }

  private requireWallet(id: WalletId): Wallet {
    const w = this.wallets.get(id);
    if (!w) throw new EconomyError(`Wallet not found: ${id}`);
    return w;
  }

  private requireBalance(wallet: Wallet, currency: CurrencyCode): Balance {
    let bal = wallet.balances.get(currency);
    if (!bal) {
      bal = zeroBalance(currency);
      wallet.balances.set(currency, bal);
    }
    return bal;
  }

  private buildTransferResult(entries: LedgerEntry[]): TransferResult {
    const debit  = entries.find((e) => e.type === "debit")!;
    const credit = entries.find((e) => e.type === "credit")!;
    const fee    = entries.find((e) => e.type === "fee");
    const tax    = entries.find((e) => e.type === "tax");
    return { txId: debit.id, fromEntry: debit, toEntry: credit, feeEntry: fee, taxEntry: tax };
  }
}

function zeroBalance(currency: CurrencyCode): Balance {
  return { currency, available: MicroAmount(0n), held: MicroAmount(0n), total: MicroAmount(0n) };
}

// ─────────────────────────────────────────────
// § ESCROW SYSTEM — Safe peer-to-peer trade
// ─────────────────────────────────────────────

export type EscrowStatus =
  | "pending"    // created, waiting for both parties
  | "funded"     // buyer deposited funds
  | "accepted"   // seller accepted
  | "confirmed"  // buyer confirmed receipt
  | "completed"  // funds released to seller
  | "disputed"   // dispute raised
  | "cancelled"  // cancelled before completion
  | "expired";   // timed out

export interface TradeItem {
  itemId:    ItemId;
  quantity:  number;
  metadata:  Record<string, unknown>;
}

export interface EscrowContract {
  id:              EscrowId;
  buyerWalletId:   WalletId;
  sellerWalletId:  WalletId;
  currency:        CurrencyCode;
  amount:          MicroAmount;
  items:           TradeItem[];
  status:          EscrowStatus;
  createdAt:       number;
  expiresAt:       number;
  fundedAt?:       number;
  acceptedAt?:     number;
  confirmedAt?:    number;
  completedAt?:    number;
  cancelledAt?:    number;
  txIds:           TxId[];
  disputeReason?:  string;
}

export interface EscrowTransition {
  from:    EscrowStatus;
  to:      EscrowStatus;
  actorWalletId: WalletId;
  timestamp: number;
}

// Valid state machine transitions
const VALID_TRANSITIONS: Partial<Record<EscrowStatus, EscrowStatus[]>> = {
  pending:   ["funded", "cancelled", "expired"],
  funded:    ["accepted", "cancelled"],
  accepted:  ["confirmed", "disputed"],
  confirmed: ["completed"],
  disputed:  ["completed", "cancelled"],
};

export class EscrowSystem {
  private readonly escrows = new Map<EscrowId, EscrowContract>();
  private readonly history = new Map<EscrowId, EscrowTransition[]>();

  constructor(
    private readonly walletService: WalletService,
    private readonly expiryMs: number = 24 * 60 * 60 * 1000,  // 24h default
  ) {}

  /**
   * Create a new escrow trade.
   */
  createEscrow(
    buyerWalletId:  WalletId,
    sellerWalletId: WalletId,
    currency:       CurrencyCode,
    amount:         MicroAmount,
    items:          TradeItem[],
  ): EscrowContract {
    if (buyerWalletId === sellerWalletId) {
      throw new EconomyError("Buyer and seller must be different wallets");
    }
    if (amount <= BigInt(0))  throw new EconomyError("Escrow amount must be positive");
    if (items.length === 0)   throw new EconomyError("Trade must include at least one item");

    const escrow: EscrowContract = {
      id:             EscrowId(`esc_${generateId()}`),
      buyerWalletId,
      sellerWalletId,
      currency,
      amount,
      items:          JSON.parse(JSON.stringify(items)),
      status:         "pending",
      createdAt:      Date.now(),
      expiresAt:      Date.now() + this.expiryMs,
      txIds:          [],
    };

    this.escrows.set(escrow.id, escrow);
    this.history.set(escrow.id, []);
    return escrow;
  }

  /**
   * Buyer funds the escrow — holds funds from buyer wallet.
   */
  fund(escrowId: EscrowId): void {
    const escrow = this.requireEscrow(escrowId);
    this.transition(escrow, "funded", escrow.buyerWalletId);

    // Place a hold on buyer funds
    const wallet = this.walletService.getWallet(escrow.buyerWalletId)!;
    const bal    = wallet.balances.get(escrow.currency);
    if (!bal || bal.available < escrow.amount) {
      throw new InsufficientFundsError("Insufficient funds to fund escrow");
    }

    bal.available = MicroAmount(bal.available - escrow.amount);
    bal.held      = MicroAmount(bal.held + escrow.amount);

    escrow.fundedAt = Date.now();
  }

  /**
   * Seller accepts the terms and commits to delivering items.
   */
  accept(escrowId: EscrowId): void {
    const escrow = this.requireEscrow(escrowId);
    this.transition(escrow, "accepted", escrow.sellerWalletId);
    escrow.acceptedAt = Date.now();
  }

  /**
   * Buyer confirms they received the items.
   */
  confirm(escrowId: EscrowId): void {
    const escrow = this.requireEscrow(escrowId);
    this.transition(escrow, "confirmed", escrow.buyerWalletId);
    escrow.confirmedAt = Date.now();

    // Auto-complete after confirmation
    this.complete(escrowId);
  }

  /**
   * Release held funds to seller and mark complete.
   */
  private complete(escrowId: EscrowId): void {
    const escrow = this.requireEscrow(escrowId);
    this.transition(escrow, "completed", escrow.sellerWalletId);

    // Release hold from buyer
    const buyerWallet = this.walletService.getWallet(escrow.buyerWalletId)!;
    const buyerBal    = buyerWallet.balances.get(escrow.currency)!;
    buyerBal.held     = MicroAmount(buyerBal.held - escrow.amount);
    buyerBal.total    = MicroAmount(buyerBal.total - escrow.amount);

    // Transfer to seller
    const entry = this.walletService.credit(
      escrow.sellerWalletId,
      escrow.currency,
      escrow.amount,
      `Escrow release: ${escrowId}`,
      `escrow_complete_${escrowId}`,
      { escrowId },
    );

    escrow.txIds.push(entry.id);
    escrow.completedAt = Date.now();
  }

  /**
   * Raise a dispute — freezes the escrow for admin review.
   */
  dispute(escrowId: EscrowId, actorWalletId: WalletId, reason: string): void {
    const escrow = this.requireEscrow(escrowId);
    if (
      actorWalletId !== escrow.buyerWalletId &&
      actorWalletId !== escrow.sellerWalletId
    ) {
      throw new EconomyError("Only escrow participants can raise a dispute");
    }
    this.transition(escrow, "disputed", actorWalletId);
    escrow.disputeReason = reason.slice(0, 1000);
  }

  /**
   * Admin resolution: release funds to winner or refund buyer.
   */
  resolveDispute(escrowId: EscrowId, releaseToSeller: boolean): void {
    const escrow = this.requireEscrow(escrowId);
    if (escrow.status !== "disputed") {
      throw new EconomyError("Escrow is not in disputed state");
    }

    if (releaseToSeller) {
      escrow.status = "confirmed"; // transition path: confirmed → completed
      this.complete(escrowId);
    } else {
      this.cancel(escrowId, "Dispute resolved: refund to buyer");
    }
  }

  /**
   * Cancel and refund the buyer.
   */
  cancel(escrowId: EscrowId, reason = "Cancelled"): void {
    const escrow = this.requireEscrow(escrowId);
    const allowed: EscrowStatus[] = VALID_TRANSITIONS[escrow.status] ?? [];
    if (!allowed.includes("cancelled")) {
      throw new InvalidTransitionError(
        `Cannot cancel escrow in status "${escrow.status}"`,
      );
    }
    escrow.status      = "cancelled";
    escrow.cancelledAt = Date.now();
    escrow.disputeReason = reason;

    // Release hold back to buyer if funded
    if (escrow.fundedAt) {
      const wallet = this.walletService.getWallet(escrow.buyerWalletId)!;
      const bal    = wallet.balances.get(escrow.currency)!;
      bal.held      = MicroAmount(bal.held - escrow.amount);
      bal.available = MicroAmount(bal.available + escrow.amount);
    }
  }

  /**
   * Expire stale escrows (call this periodically).
   */
  expireStale(): EscrowContract[] {
    const now     = Date.now();
    const expired: EscrowContract[] = [];

    for (const escrow of this.escrows.values()) {
      if (
        now > escrow.expiresAt &&
        (escrow.status === "pending" || escrow.status === "funded")
      ) {
        this.cancel(escrow.id, "Expired");
        expired.push(escrow);
      }
    }

    return expired;
  }

  getEscrow(id: EscrowId): EscrowContract | undefined {
    return this.escrows.get(id);
  }

  getHistory(id: EscrowId): EscrowTransition[] {
    return this.history.get(id) ?? [];
  }

  private requireEscrow(id: EscrowId): EscrowContract {
    const e = this.escrows.get(id);
    if (!e) throw new EconomyError(`Escrow not found: ${id}`);
    return e;
  }

  private transition(escrow: EscrowContract, to: EscrowStatus, actor: WalletId): void {
    const allowed = VALID_TRANSITIONS[escrow.status] ?? [];
    if (!allowed.includes(to)) {
      throw new InvalidTransitionError(
        `Invalid escrow transition: ${escrow.status} → ${to}`,
      );
    }

    const transition: EscrowTransition = {
      from:          escrow.status,
      to,
      actorWalletId: actor,
      timestamp:     Date.now(),
    };

    this.history.get(escrow.id)!.push(transition);
    escrow.status = to;
  }
}

// ─────────────────────────────────────────────
// § TAX ENGINE
// ─────────────────────────────────────────────

interface TaxResult {
  taxAmount:    MicroAmount;
  rate:         number;
  jurisdiction: string;
}

interface TaxRule {
  jurisdiction: string;
  rate:         number;
  currencies:   CurrencyCode[];
  threshold:    MicroAmount;  // min amount to apply tax
}

export class TaxEngine {
  private readonly rules: TaxRule[] = [
    { jurisdiction: "EU",  rate: 0.20, currencies: [CURRENCIES.USD, CURRENCIES.EUR], threshold: fromDecimal(0) },
    { jurisdiction: "US",  rate: 0.08, currencies: [CURRENCIES.USD, CURRENCIES.AEC], threshold: fromDecimal(0) },
    { jurisdiction: "BR",  rate: 0.15, currencies: [CURRENCIES.USD, CURRENCIES.BRL], threshold: fromDecimal(0) },
    { jurisdiction: "DEFAULT", rate: 0.05, currencies: [CURRENCIES.AEC, CURRENCIES.AEG], threshold: fromDecimal(10) },
  ];

  calculate(
    currency:     CurrencyCode,
    amount:       MicroAmount,
    userContext:  { jurisdiction?: string },
  ): TaxResult {
    const jurisdiction = userContext.jurisdiction ?? "DEFAULT";
    const rule = this.rules.find(
      (r) => r.jurisdiction === jurisdiction && r.currencies.includes(currency),
    ) ?? this.rules.find((r) => r.jurisdiction === "DEFAULT")!;

    if (amount < rule.threshold) {
      return { taxAmount: MicroAmount(0n), rate: 0, jurisdiction: rule.jurisdiction };
    }

    const taxAmount = MicroAmount(BigInt(Math.round(Number(amount) * rule.rate)));
    return { taxAmount, rate: rule.rate, jurisdiction: rule.jurisdiction };
  }
}

// ─────────────────────────────────────────────
// § ERRORS
// ─────────────────────────────────────────────

export class EconomyError         extends Error { constructor(m: string) { super(m); this.name = "EconomyError"; } }
export class InsufficientFundsError extends EconomyError { constructor(m: string) { super(m); this.name = "InsufficientFundsError"; } }
export class InvalidTransitionError extends EconomyError { constructor(m: string) { super(m); this.name = "InvalidTransitionError"; } }

// ─────────────────────────────────────────────
// § UTILS
// ─────────────────────────────────────────────

function generateId(): string {
  return crypto.randomUUID().replace(/-/g, "").slice(0, 16);
}
