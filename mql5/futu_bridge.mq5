//+------------------------------------------------------------------+
//| FUTU Bridge EA - File-based signal bridge for Python bot         |
//| Reads signal.json -> execute orders                              |
//| Writes candles CSV + positions + account for Python              |
//+------------------------------------------------------------------+
#property copyright "FUTU"
#property version   "1.0"
#property strict

input int    CheckIntervalMs = 3000;
input int    MagicNumber     = 202603;
input string SignalFile      = "signal.json";
input string ResultFile      = "result.json";
input int    CandleCount     = 200;
input int    Slippage        = 20;

int OnInit() {
   EventSetMillisecondTimer(CheckIntervalMs);
   Print("FUTU Bridge EA started. Magic=", MagicNumber);
   WriteCandleData();
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason) {
   EventKillTimer();
   Print("FUTU Bridge EA stopped.");
}

void OnTimer() {
   CheckSignalFile();
   WriteCandleData();
}

void OnTick() {
   CheckSignalFile();
}

void CheckSignalFile() {
   if (!FileIsExist(SignalFile)) return;

   int handle = FileOpen(SignalFile, FILE_READ|FILE_TXT|FILE_ANSI);
   if (handle == INVALID_HANDLE) return;

   string content = "";
   while (!FileIsEnding(handle))
      content += FileReadString(handle) + "\n";
   FileClose(handle);
   FileDelete(SignalFile);

   if (StringLen(content) < 5) return;

   string action = GetJsonString(content, "action");
   string symbol = GetJsonString(content, "symbol");
   double volume = GetJsonDouble(content, "volume");
   double sl     = GetJsonDouble(content, "sl");
   double tp     = GetJsonDouble(content, "tp");
   int    ticket = (int)GetJsonDouble(content, "ticket");
   double new_sl = GetJsonDouble(content, "new_sl");

   if (action == "buy" || action == "sell")
      ExecuteOrder(action, symbol, volume, sl, tp);
   else if (action == "modify_sl")
      ModifySL(ticket, symbol, new_sl, tp);
   else if (action == "close" || action == "close_partial")
      ClosePosition(ticket, symbol, volume);
}

void ExecuteOrder(string action, string symbol, double volume, double sl, double tp) {
   MqlTradeRequest request = {};
   MqlTradeResult result = {};

   request.action    = TRADE_ACTION_DEAL;
   request.symbol    = symbol;
   request.volume    = NormalizeDouble(volume, 2);
   request.type      = (action == "buy") ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
   request.price     = (action == "buy") ? SymbolInfoDouble(symbol, SYMBOL_ASK)
                                         : SymbolInfoDouble(symbol, SYMBOL_BID);
   request.sl        = sl;
   request.tp        = tp;
   request.deviation = Slippage;
   request.magic     = MagicNumber;
   request.comment   = "FUTU FX";
   request.type_filling = ORDER_FILLING_IOC;

   bool ok = OrderSend(request, result);
   WriteResult(ok, result.retcode, result.order, result.price, result.volume, result.comment, "order");
   Print("ORDER ", action, " ", symbol, " ", volume, " lots: ",
         ok ? "OK" : "FAIL", " retcode=", result.retcode, " ticket=", result.order);
}

void ModifySL(int ticket, string symbol, double new_sl, double tp) {
   MqlTradeRequest request = {};
   MqlTradeResult result = {};

   request.action    = TRADE_ACTION_SLTP;
   request.position  = ticket;
   request.symbol    = symbol;
   request.sl        = new_sl;
   request.tp        = tp;

   bool ok = OrderSend(request, result);
   WriteResult(ok, result.retcode, ticket, 0, 0, "", "modify_sl");
   Print("MODIFY SL ticket=", ticket, " new_sl=", new_sl, ": ", ok ? "OK" : "FAIL");
}

void ClosePosition(int ticket, string symbol, double volume) {
   MqlTradeRequest request = {};
   MqlTradeResult result = {};

   if (!PositionSelectByTicket(ticket)) {
      Print("Position not found: ", ticket);
      return;
   }

   request.action    = TRADE_ACTION_DEAL;
   request.symbol    = symbol;
   request.volume    = (volume > 0) ? volume : PositionGetDouble(POSITION_VOLUME);
   request.type      = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY)
                        ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
   request.price     = (request.type == ORDER_TYPE_BUY)
                        ? SymbolInfoDouble(symbol, SYMBOL_ASK)
                        : SymbolInfoDouble(symbol, SYMBOL_BID);
   request.deviation = Slippage;
   request.magic     = MagicNumber;
   request.position  = ticket;
   request.comment   = "FUTU FX CLOSE";
   request.type_filling = ORDER_FILLING_IOC;

   bool ok = OrderSend(request, result);
   WriteResult(ok, result.retcode, ticket, 0, 0, "", "close");
   Print("CLOSE ticket=", ticket, " vol=", request.volume, ": ", ok ? "OK" : "FAIL");
}

void WriteResult(bool ok, int retcode, long ticket, double price, double volume, string comment, string action) {
   int fh = FileOpen(ResultFile, FILE_WRITE|FILE_TXT|FILE_ANSI);
   if (fh == INVALID_HANDLE) return;
   FileWriteString(fh, StringFormat(
      "{\"ok\":%s,\"retcode\":%d,\"ticket\":%d,\"price\":%.5f,\"volume\":%.2f,\"comment\":\"%s\",\"action\":\"%s\"}",
      ok ? "true" : "false", retcode, ticket, price, volume, comment, action));
   FileClose(fh);
}

void WriteCandleData() {
   string symbols[] = {"EURUSDm", "GBPUSDm", "AUDUSDm", "USDJPYm", "GBPJPYm"};
   ENUM_TIMEFRAMES tfs[] = {PERIOD_M5, PERIOD_H1, PERIOD_H4};
   string tfnames[] = {"5m", "1h", "4h"};

   for (int s = 0; s < ArraySize(symbols); s++) {
      for (int t = 0; t < ArraySize(tfs); t++) {
         string fname = "candles_" + symbols[s] + "_" + tfnames[t] + ".csv";
         MqlRates rates[];
         int copied = CopyRates(symbols[s], tfs[t], 0, CandleCount, rates);
         if (copied <= 0) continue;

         int fh = FileOpen(fname, FILE_WRITE|FILE_CSV|FILE_ANSI, ',');
         if (fh == INVALID_HANDLE) continue;

         FileWriteString(fh, "timestamp,open,high,low,close,volume\n");
         for (int i = 0; i < copied; i++) {
            FileWriteString(fh, StringFormat("%d,%.5f,%.5f,%.5f,%.5f,%d\n",
               (long)rates[i].time, rates[i].open, rates[i].high,
               rates[i].low, rates[i].close, (long)rates[i].tick_volume));
         }
         FileClose(fh);
      }
   }

   // Write positions
   int fh = FileOpen("positions.csv", FILE_WRITE|FILE_CSV|FILE_ANSI, ',');
   if (fh != INVALID_HANDLE) {
      FileWriteString(fh, "ticket,symbol,type,volume,price_open,sl,tp,profit,magic\n");
      for (int i = 0; i < PositionsTotal(); i++) {
         ulong ticket = PositionGetTicket(i);
         if (ticket == 0) continue;
         if (PositionGetInteger(POSITION_MAGIC) != MagicNumber) continue;
         FileWriteString(fh, StringFormat("%d,%s,%d,%.2f,%.5f,%.5f,%.5f,%.2f,%d\n",
            ticket, PositionGetString(POSITION_SYMBOL),
            PositionGetInteger(POSITION_TYPE), PositionGetDouble(POSITION_VOLUME),
            PositionGetDouble(POSITION_PRICE_OPEN), PositionGetDouble(POSITION_SL),
            PositionGetDouble(POSITION_TP), PositionGetDouble(POSITION_PROFIT),
            PositionGetInteger(POSITION_MAGIC)));
      }
      FileClose(fh);
   }

   // Write account info
   fh = FileOpen("account.json", FILE_WRITE|FILE_TXT|FILE_ANSI);
   if (fh != INVALID_HANDLE) {
      FileWriteString(fh, StringFormat(
         "{\"balance\":%.2f,\"equity\":%.2f,\"leverage\":%d}",
         AccountInfoDouble(ACCOUNT_BALANCE), AccountInfoDouble(ACCOUNT_EQUITY),
         AccountInfoInteger(ACCOUNT_LEVERAGE)));
      FileClose(fh);
   }
}

string GetJsonString(string json, string key) {
   string search = "\"" + key + "\":\"";
   int pos = StringFind(json, search);
   if (pos < 0) return "";
   pos += StringLen(search);
   int end = StringFind(json, "\"", pos);
   if (end < 0) return "";
   return StringSubstr(json, pos, end - pos);
}

double GetJsonDouble(string json, string key) {
   string search1 = "\"" + key + "\":";
   int pos = StringFind(json, search1);
   if (pos < 0) return 0;
   pos += StringLen(search1);
   while (pos < StringLen(json) && (StringGetCharacter(json, pos) == ' ' || StringGetCharacter(json, pos) == '"'))
      pos++;
   string num = "";
   while (pos < StringLen(json)) {
      ushort ch = StringGetCharacter(json, pos);
      if ((ch >= '0' && ch <= '9') || ch == '.' || ch == '-')
         num += ShortToString(ch);
      else
         break;
      pos++;
   }
   return StringToDouble(num);
}
