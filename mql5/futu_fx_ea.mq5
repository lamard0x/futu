//+------------------------------------------------------------------+
//| FUTU FX EA — Session-based BB ranging + Trending breakout        |
//| Asian (00-09 UTC): BB mean reversion on M5                       |
//| London/NY (09-21 UTC): Trending breakout on H1                   |
//| Auto TP/SL, trailing to breakeven, partial close                 |
//+------------------------------------------------------------------+
#property copyright "FUTU"
#property version   "2.0"
#property strict

//── Inputs ──
input int    MagicNumber    = 202603;
input double RiskPercent    = 2.0;     // Risk % per trade
input double BalanceStart   = 199.0;   // Fixed sizing balance
input int    Slippage       = 20;

// Telegram
input string TG_BotToken    = "8719620921:AAHpOk-lWg8MAv2oUmhJlUhoQS4gjT3pfQk";
input string TG_ChatID      = "1876038514";
input bool   TG_Enabled     = true;

// BB Ranging params
input int    BB_Period      = 20;
input double BB_StdDev      = 2.0;
input double BB_TouchPct    = 0.8;     // BB touch tolerance %
input int    RSI_Period     = 10;
input double RSI_BullOS     = 42.0;    // RSI oversold (bullish bias)
input double RSI_BullOB     = 58.0;    // RSI overbought (bullish bias)
input double RSI_BearOS     = 35.0;
input double RSI_BearOB     = 58.0;
input double RSI_NeutOS     = 45.0;
input double RSI_NeutOB     = 55.0;
input double VolMult        = 0.4;     // Volume SMA multiplier
input double RangingSL_ATR  = 0.5;     // SL in ATR multiples
input double RangingMinRR   = 1.2;     // Min R:R for ranging
input int    ADX_Period     = 14;
input double ADX_Trending   = 35.0;    // ADX threshold (ranging < this)

// Trending params
input double TrendADX_Min   = 30.0;
input double TrendVolMult   = 1.2;
input int    TrendLookback  = 20;
input double TrendBodyPct   = 0.5;
input double TrendSL_ATR    = 1.5;
input double TrendMinRR     = 1.5;

// EMA params (for bias)
input int    EMA_Fast       = 9;
input int    EMA_Mid        = 21;

// Spread filter
input double MaxSpreadPips     = 3.0;     // Max spread (pips) for FX pairs
input double MaxSpreadPipsXAU  = 25.0;    // Max spread (pips) for XAU pairs

// Trailing
input double TrailBE_Pct    = 50.0;    // Move SL to BE at X% of TP distance
input double PartialPct     = 50.0;    // Close X% at halfway to TP

// Session hours (UTC)
input int    AsianStart     = 0;
input int    AsianEnd       = 9;
input int    LondonStart    = 9;
input int    LondonEnd      = 21;

//── Globals ──
int handleRSI, handleBB, handleADX, handleATR, handleEMA9, handleEMA21, handleVolSMA;
int handleRSI_H4, handleEMA9_H4, handleEMA21_H4;
int handleADX_H1, handleRSI_H1, handleATR_H1, handleEMA9_H1, handleEMA21_H1, handleVolSMA_H1;
datetime lastRangingScan = 0;
datetime lastTrendingScan = 0;

//+------------------------------------------------------------------+
int OnInit() {
   // M5 indicators
   handleRSI    = iRSI(_Symbol, PERIOD_M5, RSI_Period, PRICE_CLOSE);
   handleBB     = iBands(_Symbol, PERIOD_M5, BB_Period, 0, BB_StdDev, PRICE_CLOSE);
   handleADX    = iADX(_Symbol, PERIOD_M5, ADX_Period);
   handleATR    = iATR(_Symbol, PERIOD_M5, 14);
   handleEMA9   = iMA(_Symbol, PERIOD_M5, EMA_Fast, 0, MODE_EMA, PRICE_CLOSE);
   handleEMA21  = iMA(_Symbol, PERIOD_M5, EMA_Mid, 0, MODE_EMA, PRICE_CLOSE);
   handleVolSMA = iMA(_Symbol, PERIOD_M5, 20, 0, MODE_SMA, PRICE_CLOSE); // placeholder

   // H4 bias indicators
   handleEMA9_H4  = iMA(_Symbol, PERIOD_H4, EMA_Fast, 0, MODE_EMA, PRICE_CLOSE);
   handleEMA21_H4 = iMA(_Symbol, PERIOD_H4, EMA_Mid, 0, MODE_EMA, PRICE_CLOSE);

   // H1 trending indicators
   handleADX_H1    = iADX(_Symbol, PERIOD_H1, ADX_Period);
   handleRSI_H1    = iRSI(_Symbol, PERIOD_H1, RSI_Period, PRICE_CLOSE);
   handleATR_H1    = iATR(_Symbol, PERIOD_H1, 14);
   handleEMA9_H1   = iMA(_Symbol, PERIOD_H1, EMA_Fast, 0, MODE_EMA, PRICE_CLOSE);
   handleEMA21_H1  = iMA(_Symbol, PERIOD_H1, EMA_Mid, 0, MODE_EMA, PRICE_CLOSE);

   Print("FUTU FX EA started on ", _Symbol, " Magic=", MagicNumber);
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason) {
   Print("FUTU FX EA stopped.");
}

void OnTradeTransaction(const MqlTradeTransaction &trans,
                        const MqlTradeRequest &request,
                        const MqlTradeResult &result) {
   // Notify on position close (TP/SL/manual)
   if (trans.type == TRADE_TRANSACTION_DEAL_ADD && trans.deal > 0) {
      // Check if this is a close deal
      if (HistoryDealSelect(trans.deal)) {
         long entry = HistoryDealGetInteger(trans.deal, DEAL_ENTRY);
         long magic = HistoryDealGetInteger(trans.deal, DEAL_MAGIC);
         if (entry == DEAL_ENTRY_OUT && magic == MagicNumber) {
            string symbol = HistoryDealGetString(trans.deal, DEAL_SYMBOL);
            double profit = HistoryDealGetDouble(trans.deal, DEAL_PROFIT);
            double volume = HistoryDealGetDouble(trans.deal, DEAL_VOLUME);
            string reason = HistoryDealGetString(trans.deal, DEAL_COMMENT);
            string tag = profit >= 0 ? "WIN" : "LOSS";
            string msg = "[FX] " + tag + " " + symbol + "\n"
               + "PnL: " + DoubleToString(profit, 2) + " Vol: " + DoubleToString(volume, 2) + "\n"
               + reason + "\n"
               + "Bal: " + DoubleToString(AccountInfoDouble(ACCOUNT_BALANCE), 2);
            SendTelegram(msg);
         }
      }
   }
}

//+------------------------------------------------------------------+
void OnTick() {
   // Check session
   MqlDateTime dt;
   TimeCurrent(dt);
   int hour = dt.hour; // Server time (UTC for Exness)
   int dow  = dt.day_of_week;

   // Weekend — no trading
   if (dow == 0 || dow == 6) return;

   // Manage existing positions (trailing, partial close)
   ManagePositions();

   // Already have position on this symbol?
   if (HasPosition()) return;

   // Asian session — BB ranging on M5
   if (hour >= AsianStart && hour < AsianEnd) {
      // Scan every new M5 bar
      datetime barTime = iTime(_Symbol, PERIOD_M5, 0);
      if (barTime != lastRangingScan) {
         lastRangingScan = barTime;
         ScanRanging();
      }
   }

   // London/NY session — Trending breakout on H1
   if (hour >= LondonStart && hour < LondonEnd) {
      datetime barTime = iTime(_Symbol, PERIOD_H1, 0);
      if (barTime != lastTrendingScan) {
         lastTrendingScan = barTime;
         ScanTrending();
      }
   }
}

//+------------------------------------------------------------------+
//  HTF Bias (H4 EMA9 vs EMA21)
//+------------------------------------------------------------------+
string GetBias() {
   double ema9[], ema21[];
   CopyBuffer(handleEMA9_H4, 0, 0, 1, ema9);
   CopyBuffer(handleEMA21_H4, 0, 0, 1, ema21);
   if (ema9[0] > ema21[0]) return "bullish";
   if (ema9[0] < ema21[0]) return "bearish";
   return "neutral";
}

//+------------------------------------------------------------------+
//  BB Ranging Strategy (M5)
//+------------------------------------------------------------------+
void ScanRanging() {
   // Use closed candle [1]
   double close1 = iClose(_Symbol, PERIOD_M5, 1);
   double open1  = iOpen(_Symbol, PERIOD_M5, 1);
   double high1  = iHigh(_Symbol, PERIOD_M5, 1);
   double low1   = iLow(_Symbol, PERIOD_M5, 1);
   double close2 = iClose(_Symbol, PERIOD_M5, 2);

   // Get indicators
   double rsi[], bb_upper[], bb_lower[], bb_mid[], adx_main[], atr[];
   CopyBuffer(handleRSI, 0, 1, 1, rsi);
   CopyBuffer(handleBB, 1, 1, 1, bb_upper);  // upper
   CopyBuffer(handleBB, 2, 1, 1, bb_lower);  // lower
   CopyBuffer(handleBB, 0, 1, 1, bb_mid);    // mid
   CopyBuffer(handleADX, 0, 1, 1, adx_main); // main ADX
   CopyBuffer(handleATR, 0, 1, 1, atr);

   if (atr[0] <= 0 || adx_main[0] >= ADX_Trending) return;

   double bb_width = bb_upper[0] - bb_lower[0];
   if (bb_width <= 0) return;
   double candle_range = high1 - low1;
   if (candle_range <= 0) return;

   // Prev candle BB for anti-breakout
   double prev_bbl[], prev_bbu[];
   CopyBuffer(handleBB, 2, 2, 1, prev_bbl);
   CopyBuffer(handleBB, 1, 2, 1, prev_bbu);

   string bias = GetBias();
   double oversold, overbought;
   GetRSIThresholds(bias, oversold, overbought);

   // Volume check (tick volume)
   long vol1 = iTickVolume(_Symbol, PERIOD_M5, 1);
   long vol_avg = 0;
   for (int i = 1; i <= 20; i++) vol_avg += iTickVolume(_Symbol, PERIOD_M5, i);
   vol_avg /= 20;

   // ── LONG ──
   bool touch_lower = low1 <= bb_lower[0] * (1 + BB_TouchPct / 100);
   bool close_inside = close1 > bb_lower[0];
   bool prev_above = close2 > prev_bbl[0];
   bool mid_room = bb_mid[0] > close1;

   if (touch_lower && close_inside && prev_above && mid_room) {
      double lower_wick = MathMin(close1, open1) - low1;
      double wick_pct = lower_wick / candle_range;

      int opt = 0;
      if (wick_pct >= 0.15) opt++;
      if (close1 > open1) opt++;
      if (rsi[0] <= oversold) opt++;
      if (vol_avg > 0 && vol1 > vol_avg * VolMult) opt++;

      if (opt >= 3) {
         double sl = close1 - RangingSL_ATR * atr[0];
         double tp = close1 + (bb_mid[0] - close1) * 0.50;
         double risk = MathAbs(close1 - sl);
         double reward = MathAbs(tp - close1);
         if (risk > 0 && reward / risk >= RangingMinRR) {
            double lots = CalcLots(close1, sl);
            if (lots > 0 && CheckSpread()) {
               OpenOrder("buy", lots, sl, tp, "FUTU RANGE LONG");
            }
         }
      }
   }

   // ── SHORT ──
   bool touch_upper = high1 >= bb_upper[0] * (1 - BB_TouchPct / 100);
   bool close_inside_s = close1 < bb_upper[0];
   bool prev_below = close2 < prev_bbu[0];
   bool mid_room_s = bb_mid[0] < close1;

   if (touch_upper && close_inside_s && prev_below && mid_room_s) {
      double upper_wick = high1 - MathMax(close1, open1);
      double wick_pct = upper_wick / candle_range;

      int opt = 0;
      if (wick_pct >= 0.15) opt++;
      if (close1 < open1) opt++;
      if (rsi[0] >= overbought) opt++;
      if (vol_avg > 0 && vol1 > vol_avg * VolMult) opt++;

      if (opt >= 3) {
         double sl = close1 + RangingSL_ATR * atr[0];
         double tp = close1 - (close1 - bb_mid[0]) * 0.50;
         double risk = MathAbs(close1 - sl);
         double reward = MathAbs(tp - close1);
         if (risk > 0 && reward / risk >= RangingMinRR) {
            double lots = CalcLots(close1, sl);
            if (lots > 0 && CheckSpread()) {
               OpenOrder("sell", lots, sl, tp, "FUTU RANGE SHORT");
            }
         }
      }
   }
}

//+------------------------------------------------------------------+
//  Trending Breakout Strategy (H1)
//+------------------------------------------------------------------+
void ScanTrending() {
   double close1 = iClose(_Symbol, PERIOD_H1, 1);
   double open1  = iOpen(_Symbol, PERIOD_H1, 1);
   double high1  = iHigh(_Symbol, PERIOD_H1, 1);
   double low1   = iLow(_Symbol, PERIOD_H1, 1);

   double adx[], plus_di[], minus_di[], rsi[], atr[], ema9[], ema21[];
   CopyBuffer(handleADX_H1, 0, 1, 2, adx);        // ADX main
   CopyBuffer(handleADX_H1, 1, 1, 1, plus_di);     // +DI
   CopyBuffer(handleADX_H1, 2, 1, 1, minus_di);    // -DI
   CopyBuffer(handleRSI_H1, 0, 1, 1, rsi);
   CopyBuffer(handleATR_H1, 0, 1, 1, atr);
   CopyBuffer(handleEMA9_H1, 0, 1, 1, ema9);
   CopyBuffer(handleEMA21_H1, 0, 1, 1, ema21);

   if (atr[0] <= 0 || adx[0] < TrendADX_Min) return;
   if (adx[0] < adx[1] - 1) return; // ADX falling

   double candle_range = high1 - low1;
   if (candle_range <= 0) return;
   double body = MathAbs(close1 - open1);
   if (body / candle_range < TrendBodyPct) return;

   // Volume check
   long vol1 = iTickVolume(_Symbol, PERIOD_H1, 1);
   long vol_avg = 0;
   for (int i = 1; i <= 20; i++) vol_avg += iTickVolume(_Symbol, PERIOD_H1, i);
   vol_avg /= 20;
   if (vol_avg <= 0 || vol1 < vol_avg * TrendVolMult) return;

   // Recent high/low
   double recent_high = 0, recent_low = 999999;
   for (int i = 2; i <= TrendLookback + 1; i++) {
      double h = iHigh(_Symbol, PERIOD_H1, i);
      double l = iLow(_Symbol, PERIOD_H1, i);
      if (h > recent_high) recent_high = h;
      if (l < recent_low) recent_low = l;
   }

   string bias = GetBias();
   double sl_dist = TrendSL_ATR * atr[0];

   // LONG breakout
   if (plus_di[0] > minus_di[0] && close1 > recent_high && close1 > open1
       && ema9[0] > ema21[0] && rsi[0] > 50 && rsi[0] < 80
       && (bias == "bullish" || bias == "neutral")) {
      double sl = ema9[0] - sl_dist;
      if (sl > close1 - 0.5 * atr[0]) sl = close1 - 0.5 * atr[0];
      double risk = MathAbs(close1 - sl);
      double tp = close1 + 2.0 * risk;
      if (risk > 0 && (tp - close1) / risk >= TrendMinRR) {
         double lots = CalcLots(close1, sl);
         if (lots > 0 && CheckSpread()) OpenOrder("buy", lots, sl, tp, "FUTU TREND LONG");
      }
   }

   // SHORT breakout
   if (minus_di[0] > plus_di[0] && close1 < recent_low && close1 < open1
       && ema9[0] < ema21[0] && rsi[0] > 20 && rsi[0] < 50
       && (bias == "bearish" || bias == "neutral")) {
      double sl = ema9[0] + sl_dist;
      if (sl < close1 + 0.5 * atr[0]) sl = close1 + 0.5 * atr[0];
      double risk = MathAbs(close1 - sl);
      double tp = close1 - 2.0 * risk;
      if (risk > 0 && (close1 - tp) / risk >= TrendMinRR) {
         double lots = CalcLots(close1, sl);
         if (lots > 0 && CheckSpread()) OpenOrder("sell", lots, sl, tp, "FUTU TREND SHORT");
      }
   }
}

//+------------------------------------------------------------------+
//  Position Management — Trailing + Partial Close
//+------------------------------------------------------------------+
void ManagePositions() {
   for (int i = PositionsTotal() - 1; i >= 0; i--) {
      ulong ticket = PositionGetTicket(i);
      if (ticket == 0) continue;
      if (PositionGetInteger(POSITION_MAGIC) != MagicNumber) continue;
      if (PositionGetString(POSITION_SYMBOL) != _Symbol) continue;

      double entry = PositionGetDouble(POSITION_PRICE_OPEN);
      double sl    = PositionGetDouble(POSITION_SL);
      double tp    = PositionGetDouble(POSITION_TP);
      double vol   = PositionGetDouble(POSITION_VOLUME);
      long   type  = PositionGetInteger(POSITION_TYPE);
      double bid   = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      double ask   = SymbolInfoDouble(_Symbol, SYMBOL_ASK);

      double tp_dist = MathAbs(tp - entry);
      double half_tp = tp_dist * TrailBE_Pct / 100.0;

      if (type == POSITION_TYPE_BUY) {
         double mid_target = entry + half_tp;
         // Move SL to breakeven
         if (bid >= mid_target && sl < entry) {
            ModifySL(ticket, entry, tp);
         }
         // Partial close at halfway
         if (bid >= mid_target && vol > 0.02) {
            double close_vol = NormalizeDouble(vol * PartialPct / 100.0, 2);
            if (close_vol >= 0.01) {
               ClosePartial(ticket, close_vol);
            }
         }
      }
      else if (type == POSITION_TYPE_SELL) {
         double mid_target = entry - half_tp;
         if (ask <= mid_target && sl > entry) {
            ModifySL(ticket, entry, tp);
         }
         if (ask <= mid_target && vol > 0.02) {
            double close_vol = NormalizeDouble(vol * PartialPct / 100.0, 2);
            if (close_vol >= 0.01) {
               ClosePartial(ticket, close_vol);
            }
         }
      }
   }
}

//+------------------------------------------------------------------+
//  Spread Check — returns true if spread is acceptable
//+------------------------------------------------------------------+
bool CheckSpread() {
   double spread = SymbolInfoDouble(_Symbol, SYMBOL_ASK) - SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double point  = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   if (point <= 0) return false;
   double spread_pips = spread / (point * 10);

   double max_spread = MaxSpreadPips;
   if (StringFind(_Symbol, "XAU") >= 0) max_spread = MaxSpreadPipsXAU;

   if (spread_pips > max_spread) {
      Print("SPREAD SKIP: ", _Symbol, " spread=", DoubleToString(spread_pips, 1),
            " pips > max=", DoubleToString(max_spread, 1));
      return false;
   }
   return true;
}

//+------------------------------------------------------------------+
//  Helpers
//+------------------------------------------------------------------+
void GetRSIThresholds(string bias, double &os, double &ob) {
   if (bias == "bullish")  { os = RSI_BullOS; ob = RSI_BullOB; }
   else if (bias == "bearish") { os = RSI_BearOS; ob = RSI_BearOB; }
   else { os = RSI_NeutOS; ob = RSI_NeutOB; }
}

bool HasPosition() {
   for (int i = PositionsTotal() - 1; i >= 0; i--) {
      ulong ticket = PositionGetTicket(i);
      if (ticket == 0) continue;
      if (PositionGetInteger(POSITION_MAGIC) == MagicNumber
          && PositionGetString(POSITION_SYMBOL) == _Symbol)
         return true;
   }
   return false;
}

double CalcLots(double entry, double sl) {
   double pip_size = SymbolInfoDouble(_Symbol, SYMBOL_POINT) * 10;
   if (pip_size <= 0) pip_size = 0.0001;
   double sl_pips = MathAbs(entry - sl) / pip_size;
   if (sl_pips <= 0) return 0;

   // Pip value per lot (approximate)
   double pip_value = 10.0; // default for USD pairs
   string base = StringSubstr(_Symbol, 3, 3);
   if (base == "JPY") pip_value = 6.5;

   double risk_usd = BalanceStart * RiskPercent / 100.0;
   double lots = risk_usd / (sl_pips * pip_value);
   lots = NormalizeDouble(lots, 2);
   if (lots < 0.01) lots = 0;

   double max_lots = BalanceStart * 200 / 100000.0; // leverage cap
   if (lots > max_lots) lots = max_lots;

   return lots;
}

void OpenOrder(string side, double lots, double sl, double tp, string comment) {
   MqlTradeRequest request = {};
   MqlTradeResult result = {};

   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   request.action    = TRADE_ACTION_DEAL;
   request.symbol    = _Symbol;
   request.volume    = lots;
   request.type      = (side == "buy") ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
   request.price     = (side == "buy") ? SymbolInfoDouble(_Symbol, SYMBOL_ASK)
                                       : SymbolInfoDouble(_Symbol, SYMBOL_BID);
   request.sl        = NormalizeDouble(sl, digits);
   request.tp        = NormalizeDouble(tp, digits);
   request.deviation = Slippage;
   request.magic     = MagicNumber;
   request.comment   = comment;
   request.type_filling = ORDER_FILLING_IOC;

   bool ok = OrderSend(request, result);
   if (ok && result.retcode == TRADE_RETCODE_DONE) {
      Print("ORDER OK: ", comment, " ", lots, " lots @ ", result.price,
            " SL=", sl, " TP=", tp, " ticket=", result.order);
      double rr = (MathAbs(tp - result.price) > 0 && MathAbs(result.price - sl) > 0)
                  ? MathAbs(tp - result.price) / MathAbs(result.price - sl) : 0;
      int dig = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
      string msg = "[FX] " + comment + "\n"
         + _Symbol + " @ " + DoubleToString(result.price, dig) + "\n"
         + "SL: " + DoubleToString(sl, dig) + " TP: " + DoubleToString(tp, dig) + "\n"
         + "Size: " + DoubleToString(lots, 2) + " lots RR 1:" + DoubleToString(rr, 1) + "\n"
         + "Bal: " + DoubleToString(AccountInfoDouble(ACCOUNT_BALANCE), 2);
      SendTelegram(msg);
   } else {
      Print("ORDER FAIL: ", comment, " retcode=", result.retcode,
            " comment=", result.comment);
   }
}

void ModifySL(ulong ticket, double new_sl, double tp) {
   MqlTradeRequest request = {};
   MqlTradeResult result = {};
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);

   request.action   = TRADE_ACTION_SLTP;
   request.position = ticket;
   request.symbol   = _Symbol;
   request.sl       = NormalizeDouble(new_sl, digits);
   request.tp       = NormalizeDouble(tp, digits);

   OrderSend(request, result);
}

//+------------------------------------------------------------------+
//  Telegram Notification
//+------------------------------------------------------------------+
void SendTelegram(string msg) {
   if (!TG_Enabled || TG_BotToken == "" || TG_ChatID == "") return;

   string url = "https://api.telegram.org/bot" + TG_BotToken + "/sendMessage";

   // Build JSON body
   StringReplace(msg, "\"", "'");
   StringReplace(msg, "\n", "\\n");
   string json = "{\"chat_id\":\"" + TG_ChatID + "\",\"text\":\"" + msg + "\"}";

   char post[];
   StringToCharArray(json, post, 0, WHOLE_ARRAY, CP_UTF8);

   char result[];
   string resHeaders;
   string reqHeaders = "Content-Type: application/json\r\n";

   ResetLastError();
   int res = WebRequest("POST", url, reqHeaders, 5000, post, result, resHeaders);
   if (res == 200) {
      Print("Telegram sent OK");
   } else {
      string errBody = CharArrayToString(result, 0, WHOLE_ARRAY, CP_UTF8);
      Print("Telegram failed: ", res, " err=", GetLastError(), " body=", errBody);
   }
}

void ClosePartial(ulong ticket, double vol) {
   MqlTradeRequest request = {};
   MqlTradeResult result = {};

   long type = PositionGetInteger(POSITION_TYPE);
   request.action    = TRADE_ACTION_DEAL;
   request.symbol    = _Symbol;
   request.volume    = vol;
   request.type      = (type == POSITION_TYPE_BUY) ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
   request.price     = (request.type == ORDER_TYPE_BUY)
                        ? SymbolInfoDouble(_Symbol, SYMBOL_ASK)
                        : SymbolInfoDouble(_Symbol, SYMBOL_BID);
   request.deviation = Slippage;
   request.magic     = MagicNumber;
   request.position  = ticket;
   request.comment   = "FUTU PARTIAL";
   request.type_filling = ORDER_FILLING_IOC;

   bool ok = OrderSend(request, result);
   if (ok) Print("PARTIAL CLOSE ticket=", ticket, " vol=", vol);
}
