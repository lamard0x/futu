import logging
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

from src.config import Config
from src.indicators import compute_all
from src.strategy import (
    detect_regime, detect_htf_bias, scan_main, scan_alert,
    Signal, SignalType, SignalSource, Regime, HTFBias,
)
from src.risk import RiskManager

logger = logging.getLogger("futu.backtest")


@dataclass
class Trade:
    entry_time: str
    exit_time: str
    side: str
    entry_price: float
    exit_price: float
    sl_price: float
    tp_price: float
    pnl: float
    pnl_pct: float
    regime: str
    source: str
    reason: str


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    initial_balance: float = 0.0
    final_balance: float = 0.0
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    avg_trade_duration: float = 0.0


class BacktestEngine:
    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.risk = RiskManager(config=self.config.risk)

    def run(
        self,
        candles_15m: list[dict],
        candles_1m: list[dict] | None = None,
        candles_htf: list[dict] | None = None,
    ) -> BacktestResult:
        df_15m = compute_all(candles_15m, self.config.indicators)
        df_1m = compute_all(candles_1m, self.config.indicators) if candles_1m else None
        df_htf = compute_all(candles_htf, self.config.indicators) if candles_htf else None

        trades: list[Trade] = []
        balance = self.config.risk.account_balance
        initial_balance = balance
        peak_balance = balance
        max_dd = 0.0
        position = None
        cooldown = 0

        lookback = 50
        for i in range(lookback, len(df_15m)):
            window = df_15m.iloc[i - lookback:i + 1]
            row = df_15m.iloc[i]

            if cooldown > 0:
                cooldown -= 1
                continue

            if position is not None:
                position["candle_count"] += 1
                exit_price, exit_reason = self._check_exit(row, position)

                if exit_price is not None:
                    side_mult = 1 if position["side"] == "long" else -1
                    pnl_pct = side_mult * (exit_price - position["entry_price"]) / position["entry_price"]
                    pnl = pnl_pct * position["position_value"]
                    fees = position["position_value"] * 0.0002 * 2
                    pnl -= fees

                    balance += pnl
                    peak_balance = max(peak_balance, balance)
                    dd = (peak_balance - balance) / peak_balance if peak_balance > 0 else 0
                    max_dd = max(max_dd, dd)

                    trades.append(Trade(
                        entry_time=str(position["entry_time"]),
                        exit_time=str(row.name),
                        side=position["side"],
                        entry_price=position["entry_price"],
                        exit_price=exit_price,
                        sl_price=position["sl"],
                        tp_price=position["tp1"],
                        pnl=pnl,
                        pnl_pct=pnl_pct * 100,
                        regime=position["regime"],
                        source=position["source"],
                        reason=exit_reason,
                    ))

                    if pnl < 0:
                        cooldown = self.config.risk.cooldown_candles

                    position = None
                continue

            # Get H4 bias for current timestamp
            bias = HTFBias.NEUTRAL
            if df_htf is not None:
                htf_before = df_htf[df_htf.index <= row.name]
                if len(htf_before) >= 50:
                    htf_window = htf_before.iloc[-50:]
                    bias = detect_htf_bias(htf_window)

            signal = scan_main(window, self.config.strategy, bias)
            if signal is None:
                continue

            logger.debug(
                "Signal: %s %s @ %.0f | regime=%s | htf=%s",
                signal.type.value, signal.source.value,
                signal.entry_price, signal.regime.value, bias.value,
            )

            risk_pct = self.config.risk.risk_per_trade_main
            risk_amount = balance * risk_pct
            sl_dist_pct = abs(signal.entry_price - signal.sl_price) / signal.entry_price
            if sl_dist_pct == 0:
                continue
            position_value = risk_amount / sl_dist_pct

            rr_ok, rr = self.risk.check_rr(signal)
            if not rr_ok:
                continue

            position = {
                "side": "long" if signal.type == SignalType.LONG else "short",
                "entry_price": signal.entry_price,
                "entry_time": row.name,
                "sl": signal.sl_price,
                "tp1": signal.tp1_price,
                "tp2": signal.tp2_price,
                "position_value": position_value,
                "candle_count": 0,
                "regime": signal.regime.value,
                "source": signal.source.value,
                "partial_closed": False,
            }

        return self._compile_result(trades, initial_balance, balance, max_dd)

    def _check_exit(self, row, pos: dict) -> tuple[float | None, str]:
        high = row["high"]
        low = row["low"]
        is_long = pos["side"] == "long"

        if is_long and low <= pos["sl"]:
            return pos["sl"], "SL hit"
        if not is_long and high >= pos["sl"]:
            return pos["sl"], "SL hit"

        if is_long and high >= pos["tp1"]:
            if pos["tp2"] and not pos["partial_closed"]:
                pos["partial_closed"] = True
                pos["sl"] = pos["entry_price"]
                return None, ""
            return pos["tp1"], "TP hit"
        if not is_long and low <= pos["tp1"]:
            if pos["tp2"] and not pos["partial_closed"]:
                pos["partial_closed"] = True
                pos["sl"] = pos["entry_price"]
                return None, ""
            return pos["tp1"], "TP hit"

        if pos["tp2"]:
            if is_long and high >= pos["tp2"]:
                return pos["tp2"], "TP2 hit"
            if not is_long and low <= pos["tp2"]:
                return pos["tp2"], "TP2 hit"

        if pos["regime"] == "ranging":
            if pos["candle_count"] >= self.config.strategy.ranging_max_candles:
                return row["close"], "time exit"

        if "chandelier_long" in row.index and pos["candle_count"] > 3:
            if is_long and row["chandelier_long"] > pos["sl"]:
                pos["sl"] = row["chandelier_long"]
            if not is_long and row["chandelier_short"] < pos["sl"]:
                pos["sl"] = row["chandelier_short"]

        return None, ""

    def _compile_result(
        self, trades: list[Trade], initial: float, final: float, max_dd: float
    ) -> BacktestResult:
        if not trades:
            return BacktestResult(initial_balance=initial, final_balance=final)

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        pnls = [t.pnl for t in trades]

        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 1

        returns = pd.Series(pnls)
        sharpe = 0.0
        if returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * (252 ** 0.5)

        return BacktestResult(
            trades=trades,
            initial_balance=initial,
            final_balance=final,
            total_trades=len(trades),
            wins=len(wins),
            losses=len(losses),
            win_rate=len(wins) / len(trades) * 100 if trades else 0,
            total_pnl=final - initial,
            total_pnl_pct=(final - initial) / initial * 100,
            max_drawdown_pct=max_dd * 100,
            sharpe_ratio=sharpe,
            profit_factor=gross_profit / gross_loss if gross_loss > 0 else 0,
            avg_win=np.mean([t.pnl for t in wins]) if wins else 0,
            avg_loss=np.mean([t.pnl for t in losses]) if losses else 0,
            best_trade=max(pnls) if pnls else 0,
            worst_trade=min(pnls) if pnls else 0,
        )


def print_result(r: BacktestResult):
    print("\n" + "=" * 50)
    print("BACKTEST RESULT")
    print("=" * 50)
    print(f"Initial:       ${r.initial_balance:.2f}")
    print(f"Final:         ${r.final_balance:.2f}")
    print(f"PnL:           ${r.total_pnl:.2f} ({r.total_pnl_pct:+.1f}%)")
    print(f"Trades:        {r.total_trades}")
    print(f"Win/Loss:      {r.wins}/{r.losses}")
    print(f"Win Rate:      {r.win_rate:.1f}%")
    print(f"Profit Factor: {r.profit_factor:.2f}")
    print(f"Sharpe:        {r.sharpe_ratio:.2f}")
    print(f"Max Drawdown:  {r.max_drawdown_pct:.1f}%")
    print(f"Avg Win:       ${r.avg_win:.2f}")
    print(f"Avg Loss:      ${r.avg_loss:.2f}")
    print(f"Best Trade:    ${r.best_trade:.2f}")
    print(f"Worst Trade:   ${r.worst_trade:.2f}")
    print("=" * 50)

    if r.trades:
        print(f"\n{'#':>3} {'Side':>5} {'Entry':>10} {'Exit':>10} {'PnL':>8} {'Regime':>10} {'Reason':>10}")
        print("-" * 65)
        for i, t in enumerate(r.trades[:20], 1):
            print(f"{i:>3} {t.side:>5} {t.entry_price:>10.2f} {t.exit_price:>10.2f} {t.pnl:>+8.2f} {t.regime:>10} {t.reason:>10}")
        if len(r.trades) > 20:
            print(f"  ... and {len(r.trades) - 20} more trades")
