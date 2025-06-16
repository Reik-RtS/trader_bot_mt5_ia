#␊ metatrader5Bot.pyw

import os
import json
import time
import threading
import tempfile
import mplfinance as mpf
import logging
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import joblib
import keyring
import tkinter as tk
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone
from queue import Queue, Empty
from functools import lru_cache
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from tkinter import messagebox, filedialog
from tkinter.scrolledtext import ScrolledText
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

FEATURE_NAMES_30 = [
    'return','range','vol_rel','rsi14','sma5',
    'balance','equity','margin','margin_free','margin_level','leverage',
    'profit_acc',
    'spread','volume_min','volume_max','volume_step',
    'swap_long','swap_short','margin_initial','margin_maintenance',
    'digits','point',
    'bid','ask','last_price','time_msc',
    'open_positions','floating_profit','pending_ord','terminal_version'
]

# ----------------------
# Archivos y servicios
# ----------------------
CONFIG_FILE     = "config.json"
MODEL_FILE      = "unsup_model.pkl"
SCALER_FILE     = "scaler.pkl"
SUPERVISED_FILE = "supervised_model.pkl"
FEEDBACK_FILE   = "feedback.json"
LOG_FILE        = "bot_logs.txt"
HISTORY_FILE    = "history.json"
SYMBOLS         = ["AAPL", "BTCUSD", "XAUUSD"]
KEYRING_SERVICE = "MT5BotService"

log_queue = Queue()
done_queue = Queue()

# ----------------------
# Logging (UTF-8)
# ----------------------
logging.basicConfig(
    handlers=[logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')],
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ"
)
def log(msg: str, level=logging.INFO):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    logging.log(level, msg)
    log_queue.put(f"{ts} {msg}")

# ----------------------
# Escritura atómica
# ----------------------
def save_atomic(obj, dest_path, log_func):
    dirn = os.path.dirname(dest_path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dirn)
    os.close(fd)
    joblib.dump(obj, tmp_path)
    try:
        os.replace(tmp_path, dest_path)
        log_func(f"💾 Guardado atómico: {dest_path}")
    except PermissionError as e:
        log_func(f"⚠️ WinError 32 al reemplazar {dest_path}: {e}", logging.WARNING)
        joblib.dump(obj, dest_path)
        log_func(f"💾 Guardado directo tras fallback: {dest_path}")

# ----------------------
# Configuración
# ----------------------
def load_config():
    defaults = {
        "interval":       1,
        "max_ops":        5,
        "min_confidence": 0.75,
        "backtest_days":  182,
        "auto":           False,
        "login":          None,
        "server":         None
    }

    if not os.path.exists(CONFIG_FILE):
        # no existe → creamos
        save_config(defaults)
        log(f"🆕 Se creó {CONFIG_FILE} con valores por defecto.")
        return defaults

    # existe: intentamos cargar
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        # rellenar valores faltantes
        for k, v in defaults.items():
            cfg.setdefault(k, v)
        return cfg

    except (json.JSONDecodeError, ValueError) as e:
        # corrupto: renombrar y crear uno nuevo
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        corrupt = f"{CONFIG_FILE}.bak_{ts}"
        os.replace(CONFIG_FILE, corrupt)
        log(f"⚠️ {CONFIG_FILE} corrupto renombrado a {corrupt}", logging.WARNING)
        save_config(defaults)
        log(f"🆕 Se creó {CONFIG_FILE} con valores por defecto tras corrupción.")
        return defaults

def save_config(cfg):
    txt = json.dumps(cfg, indent=2)
    fd, tmp = tempfile.mkstemp(dir=".")
    os.close(fd)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(txt)
    os.replace(tmp, CONFIG_FILE)

def load_feedback():
    try:
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("Feedback no es un dict")
            return data
    except (json.JSONDecodeError, ValueError, FileNotFoundError) as e:
        # Renombramos el fichero corrupto/incorrecto si existe
        if os.path.exists(FEEDBACK_FILE):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            corrupt_name = f"{FEEDBACK_FILE}.bak_{ts}"
            os.replace(FEEDBACK_FILE, corrupt_name)
            log(f"⚠️ Feedback corrupto o inválido renombrado a {corrupt_name}", logging.WARNING)
        # Creamos uno nuevo vacío
        with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f)
        log("🆕 Nuevo feedback.json inicializado.", logging.INFO)
        return {}

# ----------------------
# Historial de trading
# ----------------------
def load_history():
    if os.path.exists(HISTORY_FILE):
        data = json.load(open(HISTORY_FILE, "r", encoding="utf-8"))
        if isinstance(data, list):
            return {"initial_balance": None, "trades": data}
        return {
            "initial_balance": data.get("initial_balance"),
            "trades": data.get("trades", [])
        }
    return {"initial_balance": None, "trades": []}

def save_history(hist):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2)

# ----------------------
# MT5 Connection
# ----------------------
def initialize_mt5(login=None, server=None, retries=3):
    pwd = keyring.get_password(KEYRING_SERVICE, str(login))
    if pwd is None:
        log("❌ Contraseña no encontrada en keyring.", logging.ERROR)
        return False
    for attempt in range(1, retries+1):
        if mt5.initialize(login=login, password=pwd, server=server):
            log("✅ MT5 iniciado y logueado.")
            return True
        log(f"⚠️ Intento {attempt} de login fallido: {mt5.last_error()}", logging.WARNING)
        time.sleep(2)
    return False

def shutdown_mt5():
    try:
        mt5.shutdown()
        log("✅ MT5 desconectado correctamente.")
    except Exception as e:
        log(f"❌ Error al desconectar MT5: {e}", logging.ERROR)

def select_symbol(sym):
    ok = mt5.symbol_select(sym, True)
    info = mt5.symbol_info(sym)
    if not (ok and info and info.visible):
        log(f"⚠️ No se pudo agregar {sym}", logging.WARNING)
    else:
        log(f"🔹 Símbolo agregado: {sym}")

# ----------------------
# Caching de metadatos
# ----------------------
@lru_cache(maxsize=32)
def get_symbol_info(sym):
    return mt5.symbol_info(sym)

# ----------------------
# Descarga de datos
# ----------------------
def get_historical_data(symbol, tf, count):
    now = datetime.now(timezone.utc)
    try:
        rates = mt5.copy_rates_from(symbol, tf, now, count)
        if rates is None:
            raise RuntimeError(mt5.last_error())
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        return df
    except Exception as e:
        log(f"❌ Error get_historical_data({symbol}): {e}", logging.ERROR)
        return pd.DataFrame()

# ----------------------
# Unsupervised Model
# ----------------------
class UnsupervisedModel:
    def __init__(self, n_clusters=5, batch_size=100):
        self.lock = threading.Lock()
        self.n_clusters, self.batch_size = n_clusters, batch_size

        # Carga o inicialización de scaler y KMeans
        if os.path.exists(SCALER_FILE):
            self.scaler = joblib.load(SCALER_FILE)
            log("🔄 Scaler cargado.")
        else:
            self.scaler = StandardScaler()

        if os.path.exists(MODEL_FILE):
            self.model = joblib.load(MODEL_FILE)
            self.is_fitted = True
            log("🔄 KMeans cargado.")
        else:
            self.model = MiniBatchKMeans(n_clusters=n_clusters,
                                         batch_size=batch_size,
                                         random_state=42)
            self.is_fitted = False

        # Mapa de dirección histórica cluster → BUY(+1) / SELL(-1)
        self.cluster_signal = {}
        # Mapa de confianza histórica cluster → probabilidad de éxito
        self.cluster_confidence = {}

    def save(self):
        save_atomic(self.scaler, SCALER_FILE, log)
        save_atomic(self.model, MODEL_FILE, log)

    def fit_initial(self, X):
        log(f"🎓 KMeans inicial con {len(X)} muestras.")
        # usar lista única de 25
        df = pd.DataFrame(X, columns=FEATURE_NAMES_30)
        Xs = self.scaler.fit_transform(X)
        self.model.partial_fit(Xs)
        self.is_fitted = True

        df['cluster'] = self.model.predict(Xs)

        # Señal BUY/SELL y confianza
        self.cluster_signal = {
            int(c): (1 if df.loc[df.cluster==c, 'return'].mean() > 0 else -1)
            for c in df.cluster.unique()
        }
        counts    = df.groupby('cluster').size().to_dict()
        successes = df[df['return'] > 0].groupby('cluster').size().to_dict()
        self.cluster_confidence = {
            int(c): successes.get(c, 0) / counts[c]
            for c in counts
        }

        log(f"🔢 Cluster→señal: {self.cluster_signal}")
        log(f"🔎 Cluster→confianza: {self.cluster_confidence}")
        self.save()

    def predict(self, X):
        """Devuelve BUY o SELL según cluster_signal."""
        Xs = self.scaler.transform(X)
        cluster = self.model.predict(Xs)[0]
        return mt5.ORDER_TYPE_BUY if self.cluster_signal.get(cluster, 0) > 0 else mt5.ORDER_TYPE_SELL

    def predict_proba(self, X):
        """Devuelve la confianza histórica de ese cluster (0.0–1.0)."""
        Xs = self.scaler.transform(X)
        cluster = int(self.model.predict(Xs)[0])
        return self.cluster_confidence.get(cluster, 0.5)

    def update(self, X_new):
        try:
            with self.lock:
                # Normalizar y entrenar incrementalmente
                Xs = self.scaler.transform(X_new)
                if not self.is_fitted:
                    self.fit_initial(X_new)
                else:
                    self.model.partial_fit(Xs)
                    log(f"🔄 KMeans update +{len(X_new)} muestras.")

                    # --- recalcular confianza usando feedback histórico ---
                    if os.path.exists(FEEDBACK_FILE):
                        fb = load_feedback()
                        counts = defaultdict(int)
                        successes = defaultdict(int)

                        for ticket, label in fb.items():
                            # Convertir feedback a booleano
                            was_correct = bool(label)

                            # Cargar las features guardadas para ese ticket
                            feats_file = f"feat_{ticket}.json"
                            if not os.path.exists(feats_file):
                                continue
                            data = json.load(open(feats_file, "r", encoding="utf-8"))
                            row = []
                            for name in FEATURE_NAMES_30:
                                val = data.get(name, np.nan)
                                if np.isnan(val):
                                    # misma lógica de imputación que para el df
                                    if name in ['rsi14','sma5','vol_rel','profit_acc']:
                                        val = historical_means[name]
                                    else:
                                        val = 0
                                row.append(val)
                            feat = np.array([row])
                            cid = int(self.model.predict(self.scaler.transform(feat))[0])

                            counts[cid] += 1

                            # Contamos éxito sólo si el feedback fue True
                            # y la señal del cluster coincide
                            pred = self.cluster_signal[cid]  # +1 o –1  
                            actual = +1 if data['return'] > 0 else -1  
                            if (pred == actual) == was_correct:
                                successes[cid] += 1

                        # Actualizar cluster_confidence evitando división por cero
                        for cid, tot in counts.items():
                            if tot == 0:
                                self.cluster_confidence[cid] = 0.5
                            else:
                                self.cluster_confidence[cid] = successes[cid] / tot

                        log(f"🔎 Cluster→confianza (actualizada): {self.cluster_confidence}")

                    # Guardar los objetos actualizados
                    self.save()
        except Exception as e:
            log(f"❌ Error en UModel.update: {e}", logging.ERROR)

# ----------------------
# Supervised Model
# ----------------------
class SupervisedModel:
    def __init__(self):
        if os.path.exists(SUPERVISED_FILE):
            self.clf = joblib.load(SUPERVISED_FILE)
            log("🔄 Supervisado cargado.")
        else:
            self.clf = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight="balanced"
            )
        self._warned_unfitted = False

    def predict_proba(self, feat):
        """
        Devuelve la probabilidad de que la operación sea positiva.
        feat: [return, range]
        """
        try:
            probs = self.clf.predict_proba([feat])[0]
            # índice 1 corresponde a la clase “éxito”
            return probs[1]
        except NotFittedError:
            if not self._warned_unfitted:
                log("⚠️ Modelo supervisado sin entrenar, usando probabilidad neutra.", logging.WARNING)
                self._warned_unfitted = True
            return 0.5
        except Exception as e:
            log(f"❌ Error predict_proba RF: {e}", logging.ERROR)
            return 0.5  # fallback neutro

    def save(self):
        save_atomic(self.clf, SUPERVISED_FILE, log)

    def train_from_feedback(self):
        if not os.path.exists(FEEDBACK_FILE):
            return
        fb = load_feedback()
        X, y = [], []
        for ticket, label in fb.items():
            feats_file = f"feat_{ticket}.json"
            if os.path.exists(feats_file):
                data = json.load(open(feats_file, "r"))
                X.append([data['return'], data['range']])
                y.append(int(label))
        if X:
            try:
                self.clf.fit(X, y)
                self._warned_unfitted = False
                log(f"🏋️ Supervisado entrenado con {len(X)} ejemplos.")
                self.save()
            except Exception as e:
                log(f"❌ Error entrenando RF: {e}", logging.ERROR)

    def predict(self, feat):
        try:
            return bool(self.clf.predict([feat])[0])
        except:
            return True

# ----------------------
# Trading Bot
# ----------------------
class TradingBot:
    def __init__(self, symbols, u_model, s_model, min_confidence=0.75):
        self.symbols = symbols
        self.u_model = u_model
        self.s_model = s_model
        # umbral de confianza mínimo para abrir un trade
        self.min_confidence = min_confidence
        self.training_mode = False
        self.trading_mode  = False
    def start_trading(self, interval, max_ops):
        if not self.trading_mode:
            self.trading_mode = True
            threading.Thread(
                target=self.trading_loop,
                args=(interval, max_ops),
                daemon=True
            ).start()

    def generate_features(self, df, symbol):
        """
        Devuelve un array de 30 features:
          - indicadores técnicos
          - datos de cuenta
          - datos de símbolo
          - último tick
          - exposición (posiciones abiertas)
          - datos adicionales de terminal y órdenes
        """
        from ta.momentum import RSIIndicator
        from ta.trend    import SMAIndicator

        # 1) Limpieza y cálculo de indicadores básicos
        df = df.copy().dropna()
        df['return']   = df['close'].pct_change()
        df['range']    = (df['high'] - df['low']) / df['open']
        df['vol_rel']  = df['tick_volume'] / df['tick_volume'].rolling(20).mean()
        df['rsi14']    = RSIIndicator(df['close'], window=14).rsi()
        df['sma5']     = SMAIndicator(df['close'], window=5).sma_indicator()
        df = df.dropna()

        # 2) Datos de cuenta
        acc = mt5.account_info()
        df['balance']      = acc.balance
        df['equity']       = acc.equity
        df['margin']       = acc.margin
        df['margin_free']  = acc.margin_free
        df['margin_level'] = acc.margin_level
        df['leverage']     = acc.leverage

        # 3) Datos de símbolo
        info = get_symbol_info(symbol)
        df['spread']        = info.spread
        df['volume_min']    = info.volume_min
        df['volume_max']    = info.volume_max
        df['volume_step']   = info.volume_step
        df['swap_long']     = info.swap_long
        df['swap_short']    = info.swap_short
        df['margin_initial']= info.margin_initial
        df['margin_maintenance'] = info.margin_maintenance

        # 4) Último tick
        tick = mt5.symbol_info_tick(symbol)
        df['bid']        = tick.bid
        df['ask']        = tick.ask
        df['last_price'] = tick.last
        df['tick_vol']   = tick.volume

        # 5) Exposición
        open_pos = mt5.positions_get(symbol=symbol) or []
        df['open_positions']   = len(open_pos)
        df['floating_profit']  = sum(p.profit for p in open_pos)

        # 6) Columnas adicionales
        # 6.1) Profit acumulado (ejemplo: suma acumulada de return)
        df['profit_acc'] = df['return'].cumsum()
        # 6.2) Precisión y punto mínimo del símbolo
        df['digits'] = info.digits
        df['point']  = info.point
        # 6.3) Marca de tiempo en milisegundos
        df['time_msc'] = (df['time'].astype('int64') // 1_000_000).astype(int)
        # 6.4) Órdenes pendientes
        pending = mt5.orders_get(symbol=symbol) or []
        df['pending_ord'] = len(pending)
        # 6.5) Número de build del terminal
        term = mt5.terminal_info()
        df['terminal_version'] = getattr(term, 'build', 0)

        # ——— IMPUTACIÓN ÚNICA DE NaNs ———
        # Definimos qué columnas van con media y cuáles con 0
        mean_cols = ['rsi14','sma5','vol_rel','profit_acc']
        zero_cols = [c for c in FEATURE_NAMES_30 if c not in mean_cols]

        # Calculamos medias (ignora columnas vacías)
        means = {c: df[c].mean() for c in mean_cols}

        # Rellenamos
        for c in mean_cols:
            df[c] = df[c].fillna(means[c])
        for c in zero_cols:
            df[c] = df[c].fillna(0)

        # 7) Extracción en el orden correcto
        feats = df[FEATURE_NAMES_30].values
        return feats, df

    def initial_training(self):
        log("🔍 Entrenamiento inicial (histórico).")
        all_feats = []
        for s in self.symbols:
            df = get_historical_data(s, mt5.TIMEFRAME_D1, 1000)
            feats, _ = self.generate_features(df, s)
            if feats.size:
                all_feats.append(feats)
        if all_feats:
            X = np.vstack(all_feats)
            self.u_model.fit_initial(X)
        self.s_model.train_from_feedback()

    def backtest_training(self, lookback_days=90):
        cfg = load_config()
        interval       = cfg["interval"]
        max_ops        = cfg["max_ops"]
        min_confidence = cfg["min_confidence"]

        end   = datetime.now(timezone.utc)
        start = end - timedelta(days=lookback_days)
        log(f"📊 Backtest entrenamiento {lookback_days}d iniciando desde {start.date()}…")

        # descargamos todo el histórico de M1 de una vez por símbolo
        for sym in self.symbols:
            rates = mt5.copy_rates_range(sym, mt5.TIMEFRAME_M1, start, end)
            df = pd.DataFrame(rates)

            if df.empty or 'time' not in df.columns:
                log(f"⚠️ No hay datos históricos para {sym}, salto backtest.") 
                continue

            df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
            for i in range(0, len(df), interval):
                window = df.iloc[i:i+interval]
                if len(window) < interval:
                    continue
                feats, _ = self.generate_features(window, sym)  # <<<--- MODIFICAR
                if feats.size:
                    self.u_model.update(feats)

            log(f"✅ Backtest {sym} completado ({len(df)} minutos procesados).")

        # una vez fuera del bucle de símbolos:
        self.u_model.save()
        log("✅ Backtest entrenamiento completo. Modelo y scaler guardados.")

        if getattr(self, 'ui_panel', None):
            done_queue.put('backtest_done')

    def live_training_loop(self):
        self.training_mode = True
        log("🔄 Entrenamiento en vivo iniciado.")
        while self.training_mode:
            try:
                for s in self.symbols:
                    df = get_historical_data(s, mt5.TIMEFRAME_M1, 100)
                    feats, _ = self.generate_features(df, s)
                    if feats.size:
                        self.u_model.update(feats)
                time.sleep(5)
            except Exception as e:
                log(f"❌ Error live_training_loop: {e}", logging.ERROR)
        log("⏹️ Entrenamiento en vivo detenido.")

    def start_live_training(self):
        if not self.training_mode:
            threading.Thread(target=self.live_training_loop, daemon=True).start()

    def open_order(self, sym, otype, feat):
        info = get_symbol_info(sym)
        tick = mt5.symbol_info_tick(sym)
        if not info or not tick:
            return log(f"❌ No data para {sym}", logging.ERROR)

        step  = info.volume_step
        vol   = round(min(max(info.volume_min, 0.1),
                          info.volume_max) / step) * step
        price = tick.ask if otype==mt5.ORDER_TYPE_BUY else tick.bid
        sl    = price * (0.99 if otype==mt5.ORDER_TYPE_BUY else 1.01)
        tp    = price * (1.02 if otype==mt5.ORDER_TYPE_BUY else 0.98)

        request = {
            "action":      mt5.TRADE_ACTION_DEAL,
            "symbol":      sym,
            "volume":      vol,
            "type":        otype,
            "price":       price,
            "sl":          sl,
            "tp":          tp,
            "deviation":   10,
            "magic":       1000,
            "comment":     "bot_entry",
            "type_time":   mt5.ORDER_TIME_GTC,
            "type_filling":mt5.ORDER_FILLING_IOC
        }

        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_MARKET_CLOSED:
            log(f"⚠️ Mercado cerrado para {sym}, omitiendo orden (retcode {result.retcode})", logging.WARNING)
            return None
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            log(f"❌ Error abrir {sym}: retcode {result.retcode}", logging.ERROR)
            return False

        log(f"🟢 Abierta {sym}, ticket={result.order}")
        feat_data = dict(zip(
            FEATURE_NAMES_30,
            [float(x) for x in feat.flatten().tolist()]
        ))

        with open(f"feat_{result.order}.json", "w", encoding="utf-8") as f:
            json.dump(feat_data, f)
        return result.order

    def close_position(self, pos):
        try:
            tick = mt5.symbol_info_tick(pos.symbol)
            price = tick.bid if pos.type==0 else tick.ask
            otype = mt5.ORDER_TYPE_SELL if pos.type==0 else mt5.ORDER_TYPE_BUY
            res = mt5.order_send({
                "action":      mt5.TRADE_ACTION_DEAL,
                "position":    pos.ticket,
                "symbol":      pos.symbol,
                "volume":      pos.volume,
                "type":        otype,
                "price":       price,
                "deviation":   10,
                "magic":       pos.magic,
                "comment":     "bot_close",
                "type_time":   mt5.ORDER_TIME_GTC,
                "type_filling":mt5.ORDER_FILLING_IOC
            })
            if res.retcode == mt5.TRADE_RETCODE_DONE:
                pnl = (price - pos.price_open) * pos.volume * (1 if pos.type==0 else -1)
                hist = load_history()
                hist["trades"].append({
                    "ticket":    pos.ticket,
                    "symbol":    pos.symbol,
                    "pnl":       pnl,
                    "fecha":     datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                save_history(hist)
                # Log y notificación a UI
                msg = f"🟢 Cerrada {pos.ticket} con P&L {pnl:.2f}"
                log(msg)
                last = hist["trades"][-1]
                if getattr(bot, 'ui_panel', None):
                    bot.ui_panel.after(0, bot.ui_panel.add_trade_to_list, last)
                return True
            else:
                log(f"❌ Error cerrar {pos.ticket}: {res.retcode}", logging.ERROR)
        except Exception as e:
            log(f"❌ Excepción close_position: {e}", logging.ERROR)
        return False

    def manage_positions(self):
        try:
            for pos in mt5.positions_get() or []:
                entry = pos.price_open
                tick  = mt5.symbol_info_tick(pos.symbol)
                if not tick: continue
                current = tick.bid if pos.type==0 else tick.ask
                loss = ((current-entry)/entry)*100*(1 if pos.type==0 else -1)
                if loss < -1.0:
                    log(f"🔻 Cerrando {pos.ticket}: loss {loss:.2f}%")
                    self.close_position(pos)
        except Exception as e:
            log(f"❌ Error manage_positions: {e}", logging.ERROR)

    def trading_loop(self, interval, max_ops):
        self.trading_mode = True
        log("▶️ Trading automático iniciado.")
        # Inicializar historial si hace falta
        hist = load_history()
        if hist["initial_balance"] is None:
            acc = mt5.account_info()
            if acc:
                hist["initial_balance"] = acc.balance
                save_history(hist)

        alpha = 0.7  # peso para el modelo supervisado en la confianza combinada

        while self.trading_mode:
            try:
                # Recarga parámetros dinámicos
                cfg = load_config()
                interval = cfg.get("interval", interval)
                max_ops  = cfg.get("max_ops", max_ops)
                start_time = time.time()

                # 1) Gestionar stops y cierres forzosos
                self.manage_positions()

                # 2) Comprobar cuántas posiciones hay abiertas
                positions = mt5.positions_get() or []
                if len(positions) < max_ops:
                    # 3) Generar lista de candidatos
                    candidates = []
                    for sym in self.symbols:
                        df = get_historical_data(sym, mt5.TIMEFRAME_M1, 20)
                        feats, _ = self.generate_features(df, sym)
                        if feats.size == 0:
                            continue

                        # última muestra
                        last_feat = feats[-1].reshape(1, -1)

                        # dirección sin supervisión
                        unsig_type = self.u_model.predict(last_feat)
                        # confianza no supervisada
                        conf_unsup = self.u_model.predict_proba(last_feat)
                        # confianza supervisada
                        feat_sup = [float(last_feat[0][0]), float(last_feat[0][1])]
                        conf_sup = self.s_model.predict_proba(feat_sup)
                        # confianza combinada
                        combined_conf = alpha * conf_sup + (1 - alpha) * conf_unsup

                        if combined_conf >= self.min_confidence:
                            score = abs(last_feat[0][0]) * combined_conf
                            candidates.append((sym, unsig_type, score, combined_conf, last_feat))

                    # 4) Ordenar candidatos por score descendente
                    candidates.sort(key=lambda x: x[2], reverse=True)

                    # 5) Abrir hasta completar max_ops
                    slots = max_ops - len(positions)
                    ops_count = 0
                    for sym, otype, score, conf, vec in candidates:
                        if slots <= 0:
                            break
                        ticket = self.open_order(sym, otype, vec)
                        if ticket:
                            ops_count += 1
                            slots -= 1
                            log(f"✅ Trade #{ops_count} enviado para {sym} (conf={conf:.2f})")
                else:
                    log(f"⏸️ Ya hay {len(positions)} posiciones abiertas (límite={max_ops}).")

                # 6) Logging post-apertura
                positions = mt5.positions_get() or []
                if positions:
                    for pos in positions:
                        log(f"📌 Posición abierta: {pos.symbol} #{pos.ticket} vol={pos.volume} open={pos.price_open}")
                else:
                    log("📌 No hay posiciones abiertas este ciclo.")

                # 7) Logging de deals del día
                now = datetime.now(timezone.utc)
                deals = mt5.history_deals_get(now.replace(hour=0, minute=0, second=0), now) or []
                if deals:
                    for d in deals:
                        log(f"📑 Deal: ticket={d.ticket}, sym={d.symbol}, price={d.price}, vol={d.volume}")
                else:
                    log("📑 Sin deals hoy.")

                # 8) Esperar hasta el siguiente ciclo
                elapsed = time.time() - start_time
                time.sleep(max(0, interval * 60 - elapsed))

            except Exception as e:
                log(f"❌ Error trading_loop: {e}", logging.ERROR)

        log("⏹️ Trading detenido.")

# Al iniciar el bot:
u_model = UnsupervisedModel()
s_model = SupervisedModel()
cfg     = load_config()
bot     = TradingBot(SYMBOLS, u_model, s_model, min_confidence=cfg.get("min_confidence", 0.75))

# ----------------------
# GUI con Tkinter
# ----------------------
class DarkStyle:
    bg, fg             = "#2e2e2e", "#ffffff"
    btn_bg, btn_fg     = "#444444", "#ffffff"
    entry_bg, entry_fg = "#3e3e3e", "#ffffff"
    log_bg, log_fg     = "#1e1e1e", "#ffffff"

class LoginWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Login MT5")
        self.configure(bg=DarkStyle.bg)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.columnconfigure(1, weight=1)
        cfg = load_config()

        tk.Label(self, text="Login:",   bg=DarkStyle.bg, fg=DarkStyle.fg)\
            .grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.e_login = tk.Entry(self, bg=DarkStyle.entry_bg, fg=DarkStyle.entry_fg)
        self.e_login.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        tk.Label(self, text="Contraseña:", bg=DarkStyle.bg, fg=DarkStyle.fg)\
            .grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.e_pwd = tk.Entry(self, show="*", bg=DarkStyle.entry_bg, fg=DarkStyle.entry_fg)
        self.e_pwd.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        tk.Label(self, text="Servidor:", bg=DarkStyle.bg, fg=DarkStyle.fg)\
            .grid(row=2, column=0, padx=5, pady=5, sticky="e")
        self.e_srv = tk.Entry(self, bg=DarkStyle.entry_bg, fg=DarkStyle.entry_fg)
        self.e_srv.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

        self.var_auto = tk.BooleanVar(value=cfg.get("auto", False))
        tk.Checkbutton(self, text="Autologueo", variable=self.var_auto,
                       bg=DarkStyle.bg, fg=DarkStyle.fg, selectcolor=DarkStyle.bg)\
            .grid(row=3, column=0, columnspan=2)

        if cfg.get("auto"):
            self.e_login.insert(0, cfg["login"])
            self.e_srv.insert(0,   cfg["server"])
            self.after(100, self.login)

        tk.Button(self, text="Entrar", command=self.login,
                  bg=DarkStyle.btn_bg, fg=DarkStyle.btn_fg)\
            .grid(row=4, column=0, columnspan=2, pady=10, sticky="ew")

    def _on_close(self):
        shutdown_mt5()
        self.destroy()

    def login(self):
        cfg = load_config()
        try:
            login = int(self.e_login.get())
        except ValueError:
            return messagebox.showerror("Error", "Login debe ser numérico")
        pwd, srv = self.e_pwd.get(), self.e_srv.get()
        keyring.set_password(KEYRING_SERVICE, str(login), pwd)
        log("🔐 Credenciales seguras en keyring.")
        cfg.update({
            "auto":  self.var_auto.get(),
            "login": login,
            "server": srv
        })
        save_config(cfg)

        if not initialize_mt5(login, srv):
            return messagebox.showerror("Error", "MT5 init failed")

        for pos in mt5.positions_get() or []:
            bot.close_position(pos)

        open_syms, closed_syms = [], []
        for s in SYMBOLS:
            select_symbol(s)
            info = mt5.symbol_info(s)
            if info and info.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL:
                log(f"🔓 Mercado abierto: {s}")
                open_syms.append(s)
            else:
                log(f"🔒 Mercado cerrado: {s}")
                closed_syms.append(s)

        if not open_syms:
            messagebox.showwarning(
                "Atención",
                "Ningún símbolo está abierto para trading. Saldrá del programa."
            )
            shutdown_mt5()
            self.destroy()
            return

        bot.symbols = open_syms
        bot.initial_training()
        self.destroy()
        panel = PanelWindow()
        bot.ui_panel = panel

class PanelWindow(tk.Tk):
    WINDOW_SIZE = 100
    def __init__(self):
        super().__init__()
        self.view_start = None
        self.view_end   = None

        self.title("Panel IA bot MT5")
        self.configure(bg=DarkStyle.bg)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        # —– configuración de la ventana, widgets… —–
        self.after(1000, self._update_chart)
        self.after(1000, self._refresh_positions_list)

        # Bind del evento
        self.bind('<<BacktestDone>>', self._on_backtest_done)

        # Arrancamos el polling para escuchar done_queue
        self._poll_backtest_done()

    def _poll_backtest_done(self):
        try:
            token = done_queue.get_nowait()
        except Empty:
            pass
        else:
            if token == 'backtest_done':
                # dispara tu evento virtual
                self.event_generate('<<BacktestDone>>')
        finally:
            self.after(100, self._poll_backtest_done)

    def _finish_backtest(self):
        # Restauramos el botón
        self.hist_btn.config(bg=DarkStyle.btn_bg, text="Entrenamiento Histórico")
        # Mostramos el mensaje
        messagebox.showinfo(
            "Entrenamiento Histórico",
            f"✅ Backtest de {load_config().get('backtest_days')} días finalizado."
        )

    def _start_backtest(self):
        cfg = load_config()
        days = cfg.get("backtest_days", 90)

        # Indicamos visualmente que está corriendo
        self.hist_btn.config(bg="#ffaa00")
        self.hist_btn.config(text="Backtest…")

        # Lanzamos el hilo
        threading.Thread(
            target=lambda: bot.backtest_training(lookback_days=days),
            daemon=True
        ).start()
        log(f"▶️ Backtest entrenamiento histórico lanzado ({days} días).")

    def _on_backtest_done(self, event=None):
        self.hist_btn.config(bg=DarkStyle.btn_bg, text="Entrenamiento Histórico")
        messagebox.showinfo(
            "Entrenamiento Histórico",
            f"✅ Backtest de {load_config().get('backtest_days')} días finalizado."
        )

    def _refresh_positions_list(self):
        """
        Vuelve a poblar el listbox mostrando:
        1) Las posiciones ABIERTAS con su PnL en tiempo real.
        2) Las operaciones CERRADAS históricas (igual que antes).
        """
        # 1) Limpiamos la lista
        self.trade_listbox.delete(0, tk.END)

        # 2) Posiciones ABIERTAS
        open_positions = mt5.positions_get() or []
        for pos in open_positions:
            tick = mt5.symbol_info_tick(pos.symbol)
            if not tick:
                continue
            current = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
            pnl_pct = ((current - pos.price_open) / pos.price_open) * 100 * (1 if pos.type==mt5.ORDER_TYPE_BUY else -1)
            text = f"🔄 OPEN {pos.symbol} #{pos.ticket}  PnL: {pnl_pct:+.2f}%"
            i = self.trade_listbox.size()
            self.trade_listbox.insert("end", text)
            color = "#00FF00" if pnl_pct > 0 else "#FF0000" if pnl_pct < 0 else "#888888"
            self.trade_listbox.itemconfig(i, fg=color)

            # Cerrar si supera -1% de pérdida
            if pnl_pct < -1.0:
                log(f"🔻 Auto-cerrando {pos.ticket}: pérdida {pnl_pct:.2f}%")
                bot.close_position(pos)

        # 3) Operaciones CERRADAS (histórico)
        history = load_history().get("trades", [])
        for ses in history:
            pnl = ses.get("pnl", 0)
            marker = "▲" if pnl > 0 else "▼" if pnl < 0 else "▽"
            text = f"{ses['fecha']}   {marker}  PnL: {pnl:+.2f}"
            i = self.trade_listbox.size()
            self.trade_listbox.insert("end", text)
            color = "#00FF00" if pnl > 0 else "#FF0000" if pnl < 0 else "#888888"
            self.trade_listbox.itemconfig(i, fg=color)

        # 4) Scroll automático al final y reprogramación
        self.trade_listbox.see(tk.END)
        self.after(1000, self._refresh_positions_list)

    def plot_recent_trades(self):
        now = datetime.now(timezone.utc)
        deals = mt5.history_deals_get(now.replace(hour=0, minute=0, second=0), now) or []
        trades = {}
        for d in deals:
            trades.setdefault(d.ticket, []).append(d)

        for ds in trades.values():
            if len(ds) < 2:
                continue
            od = min(ds, key=lambda d: d.time)
            cd = max(ds, key=lambda d: d.time)
            t_open = datetime.fromtimestamp(od.time, tz=timezone.utc)
            t_close = datetime.fromtimestamp(cd.time, tz=timezone.utc)
            po, pc = od.price, cd.price
            pnl = (pc - po) / 0.0001 * (1 if od.type == mt5.ORDER_TYPE_BUY else -1)
            tri_color = 'green' if pnl > 0 else 'red' if pnl < 0 else 'gray'

            marker = self.ax.scatter(
                t_open, po, marker='v', s=80, color=tri_color, zorder=5
            )
            txt = self.ax.text(
                (mdates.date2num(t_open) + mdates.date2num(t_close)) / 2,
                (po + pc) / 2,
                f"{pnl:+.1f} pips",
                color=tri_color, fontsize=8,
                ha='center', backgroundcolor=DarkStyle.bg
            )

            marker.ticket = cd.ticket
            marker.pnl    = pnl
            marker.set_picker(True)

    def add_trade_to_list(self, entry):
        i = self.trade_listbox.size()
        marker = "▲" if entry['pnl']>0 else "▼" if entry['pnl']<0 else "▽"
        text = f"{i+1:2d}. {entry['fecha']}   {marker}  PnL: {entry['pnl']:+.2f}"
        self.trade_listbox.insert("end", text)
        self.trade_listbox.itemconfig(i,
            fg="#00FF00" if entry['pnl']>0 else "#FF0000" if entry['pnl']<0 else "#888888")
        self.trade_listbox.see(tk.END)
        self._update_chart()
        
    def ask_yes_no_cancel(self, title, prompt):
        dlg = tk.Toplevel(self)
        dlg.title(title)
        dlg.resizable(False, False)
        dlg.configure(bg=DarkStyle.bg)
        dlg.protocol("WM_DELETE_WINDOW", lambda: dlg.destroy())

        lbl = tk.Label(
            dlg,
            text=prompt,
            justify="left",
            wraplength=300,
            bg=DarkStyle.bg,
            fg=DarkStyle.fg
        )
        lbl.pack(padx=20, pady=(20,10))
        frm = tk.Frame(dlg, bg=DarkStyle.bg)
        frm.pack(pady=(0,20))
        result = {"value": None}
        def on_choice(val):
            result["value"] = val
            dlg.destroy()
        for text, val in [("Sí", True), ("No", False), ("Cancelar", None)]:
            btn = tk.Button(
                frm,
                text=text,
                width=8,
                command=lambda v=val: on_choice(v),
                bg=DarkStyle.btn_bg,
                fg=DarkStyle.btn_fg,
                activebackground=DarkStyle.entry_bg,
                activeforeground=DarkStyle.btn_fg,
                relief="flat"
            )
            btn.pack(side="left", padx=5)
        dlg.transient(self)
        dlg.grab_set()
        self.wait_window(dlg)
        return result["value"]

    def __init__(self):
        super().__init__()
        self.title("Panel IA bot MT5")
        self.configure(bg=DarkStyle.bg)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.rowconfigure(6, weight=0)
        self.log_widget = ScrolledText(
            self,
            state='disabled',
            bg=DarkStyle.log_bg,
            fg=DarkStyle.log_fg,
            font=("Consolas", 10),
            wrap='none',
            height=8
        )
        self.log_widget.grid(row=6, column=0, columnspan=8, sticky="nsew", padx=5, pady=5)
        self._process_log_queue()

        for i in range(8):
            self.columnconfigure(i, weight=1)

        cfg = load_config()
        self.var_interval = tk.StringVar(value=str(cfg.get("interval",1)))
        self.var_maxops   = tk.StringVar(value=str(cfg.get("max_ops",5)))
        self.var_backtest = tk.StringVar(value=str(cfg.get("backtest_days",90)))
        self.var_interval.trace_add("write", self._on_params)
        self.var_maxops.trace_add("write", self._on_params)
        self.var_backtest.trace_add("write", self._on_params)

        # Fila 0: Parámetros y botones Logout/Historial
        tk.Label(self, text="Minutos ciclo:", bg=DarkStyle.bg, fg=DarkStyle.fg)\
            .grid(row=0, column=0, padx=5, pady=5, sticky="e")
        tk.Entry(self, textvariable=self.var_interval,
                 bg=DarkStyle.entry_bg, fg=DarkStyle.entry_fg)\
            .grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        tk.Button(self, text="Exportar Modelo", command=self._export_model,
                  bg=DarkStyle.btn_bg, fg=DarkStyle.btn_fg)\
            .grid(row=0, column=2, padx=5, pady=5, sticky="ew")
        # logs
        self.logs_btn = tk.Button(
            self, text="Logs", command=self.show_logs,
            bg=DarkStyle.btn_bg, fg=DarkStyle.btn_fg
        )
        self.logs_btn.grid(row=0, column=6, padx=5, pady=5, sticky="ew")

        # Logout
        tk.Button(self, text="Logout", command=self._logout,
                  bg=DarkStyle.btn_bg, fg=DarkStyle.btn_fg)\
            .grid(row=0, column=7, padx=5, pady=5, sticky="ew")

        # Cerrar operaciones
        self.force_btn = tk.Button(
            self,
            text="Cerrar Operaciones",
            command=self.force_close_all,
            bg="#aa5500",
            fg=DarkStyle.fg
        )
        self.force_btn.grid(row=1, column=7, padx=5, pady=5, sticky="ew")
        
        # Fila 1: Ops/ciclo e importar modelo
        tk.Label(self, text="Ops/ciclo:", bg=DarkStyle.bg, fg=DarkStyle.fg) \
            .grid(row=1, column=0, padx=5, pady=5, sticky="e")
        tk.Entry(self, textvariable=self.var_maxops,
                 bg=DarkStyle.entry_bg, fg=DarkStyle.entry_fg) \
            .grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        tk.Button(self, text="Importar Modelo", command=self._import_model,
                  bg=DarkStyle.btn_bg, fg=DarkStyle.btn_fg) \
            .grid(row=1, column=2, padx=5, pady=5, sticky="ew")
        tk.Label(self, text="Días de Entrenar:", bg=DarkStyle.bg, fg=DarkStyle.fg) \
            .grid(row=1, column=3, padx=5, pady=5, sticky="e")
        tk.Entry(self, textvariable=self.var_backtest,
                 bg=DarkStyle.entry_bg, fg=DarkStyle.entry_fg) \
            .grid(row=1, column=4, padx=5, pady=5, sticky="ew")

        # Fila 2: Copiar LOG. / Trading / Training
        self.copy_btn = tk.Button(
            self, text="Copiar LOG.", command=self.copy_log,
            bg=DarkStyle.btn_bg, fg=DarkStyle.btn_fg
        )
        self.copy_btn.grid(row=2, column=0, padx=5, pady=5, sticky="ew")
        self.trade_btn = tk.Button(
            self, text="Iniciar Trading", command=self._toggle_trading,
            bg=DarkStyle.btn_bg, fg=DarkStyle.btn_fg
        )
        self.trade_btn.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        self.hist_btn = tk.Button(
            self, text="Entrenamiento Histórico",
            command=self._start_backtest,
            bg=DarkStyle.btn_bg,
            fg=DarkStyle.btn_fg
        )
        self.hist_btn.grid(row=2, column=2, padx=5, pady=5, sticky="ew")
        self.bind('<<BacktestDone>>', lambda e: self._finish_backtest())
        self.train_btn = tk.Button(
            self, text="Entrenamiento", command=self.toggle_training,
            bg=DarkStyle.btn_bg, fg=DarkStyle.btn_fg
        )
        self.train_btn.grid(row=2, column=3, padx=5, pady=5, sticky="ew")

        # Fila 3:  Historial de transacciones IA
        self.rowconfigure(3, weight=1, minsize=150)
        frame_hist = tk.Frame(self, bg=DarkStyle.bg)
        frame_hist.grid(row=3, column=0, columnspan=8, sticky="nsew", padx=5, pady=5)
        frame_hist.columnconfigure(0, weight=1)
        frame_hist.rowconfigure(0, weight=1)

        lb = tk.Listbox(
            frame_hist,
            bg=DarkStyle.entry_bg,
            fg=DarkStyle.fg,
            selectbackground=DarkStyle.btn_bg,
            selectforeground=DarkStyle.fg,
            font=("Consolas", 10),
            borderwidth=0,
            highlightthickness=0
        )
        lb.grid(row=0, column=0, sticky="nsew")
        self.trade_listbox = lb
        sb = tk.Scrollbar(frame_hist, orient="vertical", command=lb.yview)
        sb.grid(row=0, column=1, sticky="ns")
        lb.config(yscrollcommand=sb.set)
        history = load_history()
        if isinstance(history, dict) and "trades" in history:
            history = history["trades"]

        for i, ses in enumerate(history):
            pnl = ses.get("pnl", 0)
            if pnl > 0:
                marker, color = "▲", "#00FF00"
            elif pnl < 0:
                marker, color = "▼", "#FF0000"
            else:
                marker, color = "▽", "#888888"
            text = f"{i+1:2d}. {ses.get('fecha')}   {marker}  PnL: {pnl:+.2f}"
            lb.insert("end", text)
            lb.itemconfig(i, fg=color)
        lb.see(tk.END)

        def on_select():
            sel = lb.curselection()
            if not sel:
                return
            idx = sel[0]
            ses = history[idx]
            respuesta = self.ask_yes_no_cancel(
                "Feedback Sesión",
                f"Sesión {ses['fecha']}\nPnL = {ses['pnl']:+.2f}\n\n¿Fue satisfactoria?"
            )
            log(f"📝 Feedback sesión {ses['fecha']}: {respuesta}")
            if respuesta is not None:
                #
                pass

        lb.bind("<Double-Button-1>", lambda e: on_select())

        # Fila 4: Gráfico
        self.rowconfigure(4, weight=1)
        self.fig = plt.Figure(figsize=(6,3), facecolor=DarkStyle.bg)
        self.ax = self.fig.add_subplot(111, facecolor=DarkStyle.bg)
        self.ax.grid(True, color="#555555")
        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        widget = self.canvas.get_tk_widget()
        widget.grid(row=4, column=0, columnspan=8, sticky="nsew")
        widget.configure(bg=DarkStyle.bg)
        tb_frame = tk.Frame(self, bg=DarkStyle.bg)
        tb_frame.grid(row=5, column=0, columnspan=8, sticky="ew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, tb_frame)
        self.toolbar.update()
        self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.fig.canvas.mpl_connect('pick_event',   self._on_pick)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_hover)
        self.view_start = self.view_end = None
        self.after(1000, self._update_chart)

    def force_close_all(self):
        count = 0
        for pos in mt5.positions_get() or []:
            if bot.close_position(pos):
                count += 1
        log(f"🛑 Se forzaron cierre de {count} posiciones.")
        try:
            open(LOG_FILE, "w", encoding="utf-8").close()
        except Exception as e:
            log(f"❌ No se pudo limpiar archivo de log: {e}", logging.ERROR)
        self.log_widget.configure(state='normal')
        self.log_widget.delete("1.0", tk.END)
        self.log_widget.configure(state='disabled')
        self.trade_listbox.delete(0, tk.END)

    def _toggle_trading(self):
        if bot.trading_mode:
            bot.trading_mode = False
            self.trade_btn.config(text="Iniciar Trading", bg=DarkStyle.btn_bg)
            log("⏹️ Se detuvo el trading de la IA.")
        else:
            self._start_bal = mt5.account_info().balance
            cfg = load_config()
            interval, max_ops = cfg.get("interval",1), cfg.get("max_ops",5)
#            bot.trading_mode = True
            threading.Thread(target=bot.start_trading, args=(interval, max_ops), daemon=True).start()
            self.trade_btn.config(text="Detener Trading", bg="#aa3333")
            log("▶️ Se comenzó el trading de la IA.")

    def copy_log(self):
        self.log_widget.configure(state='normal')
        txt = self.log_widget.get("1.0", tk.END)
        self.log_widget.configure(state='disabled')
        self.clipboard_clear()
        self.clipboard_append(txt)
        log("📋 Logs copiados al portapapeles.")

    def _on_pick(self, event):
        artista = event.artist
        ticket  = getattr(artista, "ticket", None)
        pnl     = getattr(artista, "pnl", None)
        log(f"🖱️ Pick en ticket={ticket}, pnl={pnl}")
        try:
            if pnl is not None:
                msg = f"Ganancia/Pérdida: {pnl:+.1f} pips"
            else:
                msg = "No hay información de PnL."
            correcto = self.ask_yes_no_cancel(
                "Feedback transacción",
                msg + "\n\n¿Fue esta operación CORRECTA?"
            )

            if ticket is not None:
                if os.path.exists(FEEDBACK_FILE):
                    fb = load_feedback()
                else:
                    fb = {}
                fb[str(ticket)] = int(correcto)
                with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
                    json.dump(fb, f, indent=2)
                log(f"📝 Feedback guardado ticket={ticket}: {correcto}")
                self.u_model.save()
                self.s_model.train_from_feedback()
                
        except Exception as e:
            log(f"⚠️ Error en pick_event: {e}", logging.WARNING)

    def _on_hover(self, event):
        for txt in self.ax.texts:
            cont, _ = txt.contains(event)
            if cont:
                ticket = getattr(txt, 'ticket', None)
                if ticket:
                    message = txt.get_text()
                    self.tooltip = tk.Toplevel(self)
                    tk.Label(self.tooltip, text=f"Ticket {ticket}: {message}").pack()
                    self.tooltip.after(1000, self.tooltip.destroy)
                break

    def _on_close(self):
        bot.training_mode = False
        bot.trading_mode = False
        shutdown_mt5()
        self.destroy()

    def _process_log_queue(self):
        try:
            while True:
                msg = log_queue.get_nowait()
                self.log_widget.configure(state='normal')
                self.log_widget.insert(tk.END, msg+"\n")
                self.log_widget.see(tk.END)
                self.log_widget.configure(state='disabled')
        except Empty:
            pass
        self.after(100, self._process_log_queue)

    def toggle_training(self):
        if bot.training_mode:
            bot.training_mode = False
            self.train_btn.config(text="Entrenamiento")
            log("⏹️ Deteniendo entrenamiento en vivo.")
        else:
            bot.start_live_training()
            self.train_btn.config(text="Detener Entrenamiento")
            log("▶️ Entrenamiento iniciado desde UI.")

    def _on_params(self, *args):
        try:
            cfg = load_config()
            cfg["interval"] = int(self.var_interval.get())
            cfg["max_ops"]  = int(self.var_maxops.get())
            cfg["backtest_days"]  = int(self.var_backtest.get())
            save_config(cfg)
        except ValueError:
            pass

    def _export_model(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".pkl",
            filetypes=[("Pickle","*.pkl"),("All","*.*")],
            title="Exportar Modelo"
        )
        if path:
            joblib.dump(u_model.model, path)
            log(f"📤 KMeans exportado a {path}")

    def _import_model(self):
        path = filedialog.askopenfilename(
            filetypes=[("Pickle","*.pkl"),("All","*.*")],
            title="Importar Modelo"
        )
        if path:
            u_model.model = joblib.load(path)
            u_model.is_fitted = True
            u_model.save()
            log(f"📥 KMeans importado desde {path}")

    def _logout(self):
        cfg = load_config()
        cfg["auto"] = False
        save_config(cfg)
        shutdown_mt5()
        self.destroy()
        LoginWindow()

    def _on_scroll(self, event):
        if self.view_start is None:
            return

        shift = PanelWindow.WINDOW_SIZE // 10 * (1 if event.step > 0 else -1)
        new_start = self.view_start + shift
        new_end   = new_start + PanelWindow.WINDOW_SIZE

        max_index = len(self.xnums)
        if new_start < 0:
            new_start = 0
            new_end = min(PanelWindow.WINDOW_SIZE, max_index)
        if new_end > max_index:
            new_end = max_index
            new_start = max(0, max_index - PanelWindow.WINDOW_SIZE)

        self.view_start, self.view_end = new_start, new_end
        self.ax.set_xlim(self.xnums[self.view_start], self.xnums[self.view_end-1])
        self.canvas.draw()

    def _update_chart(self):
        cfg      = load_config()
        interval = cfg.get("interval", 1)

        # 1) Carga de datos
        df = get_historical_data("BTCUSD", mt5.TIMEFRAME_M1, interval*60)

        # <-- Protección: si no hay filas, reprograma y retorna -->
        if df.empty:
            self.after(2000, self._update_chart)
            return

        # 2) Ventana de visualización
        total = len(df)
        N = min(total, PanelWindow.WINDOW_SIZE)
        if total == 0 or N == 0:
            # no hay nada que mostrar
            self.after(2000, self._update_chart)
            return

        if self.view_start is None:
            self.view_start = total - N
            self.view_end   = total
        data = df.iloc[self.view_start:self.view_end].copy()

        if data.empty:
            self.after(2000, self._update_chart)
            return

        self.xnums = mdates.date2num(data['time'])

        # 3) Configuración de estilo
        mc = mpf.make_marketcolors(up='cyan', down='magenta', inherit=True)
        s  = mpf.make_mpf_style(
            marketcolors=mc,
            gridcolor='#555555',
            facecolor=DarkStyle.bg,
            edgecolor=DarkStyle.fg
        )
        mpf_df = data.set_index('time')[['open','high','low','close']]

        # <-- Protección final justo antes de plot -->
        if mpf_df.empty:
            self.after(2000, self._update_chart)
            return

        # 4) Dibujo de velas
        self.ax.clear()
        self.ax.set_facecolor(DarkStyle.bg)
        mpf.plot(
            mpf_df, type='candle', style=s,
            ax=self.ax, datetime_format='%H:%M',
            volume=False, show_nontrading=True
        )
        self.ax.grid(True, color="#555555")
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        self.ax.tick_params(colors=DarkStyle.fg)

        # 5) Título dinámico con saldo
        acc = mt5.account_info()
        if acc:
            balance = acc.balance
            title_text = f"Saldo:({balance:.2f}) - Mercados:(AAPL, BTCUSD, XAUUSD) M1"
        else:
            title_text = "Saldo:(N/D) - Mercados:(AAPL, BTCUSD, XAUUSD) M1"
        self.ax.set_title(title_text, color=DarkStyle.fg)

        # 6) Añadir trades cerrados
        self.plot_recent_trades()

        # 7) Render y siguiente actualización
        self.fig.tight_layout()
        self.canvas.draw()
        self.after(2000, self._update_chart)

    def show_logs(self):
        win = tk.Toplevel(self)
        win.title("Logs del sistema")
        win.configure(bg=DarkStyle.bg)
        win.transient(self)
        win.grab_set()
        win.geometry("700x400")

        txt = ScrolledText(
            win,
            state='normal',
            bg=DarkStyle.log_bg,
            fg=DarkStyle.log_fg,
            font=("Consolas", 10),
            wrap='none'
        )
        txt.pack(fill='both', expand=True, padx=10, pady=10)

        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            content = f"⚠️ Error leyendo {LOG_FILE}: {e}"

        txt.insert("1.0", content)
        txt.configure(state='disabled')
        txt.see(tk.END)

#        log_scroll = tk.Scrollbar(self, orient="vertical", command=self.log_widget.yview)
#        self.log_widget.configure(yscrollcommand=log_scroll.set)
#        log_scroll.grid(row=6, column=8, sticky="ns")

#        xsb.pack(fill='x', side='bottom')
        xsb = tk.Scrollbar(win, orient="horizontal", command=txt.xview)
        txt.configure(xscrollcommand=xsb.set)
        xsb.pack(fill='x', side='bottom')

        def poll_queue():
            try:
                while True:
                    line = log_queue.get_nowait()
                    txt.configure(state='normal')
                    txt.insert(tk.END, line + "\n")
                    txt.configure(state='disabled')
                    txt.see(tk.END)
            except Empty:
                pass
            win.after(100, poll_queue)

        # Iniciar el polling
        win.after(100, poll_queue)

    def _record_history(self, start_balance, end_balance):
        if os.path.exists(HISTORY_FILE):
            history = json.load(open(HISTORY_FILE, "r", encoding="utf-8"))
        else:
            history = []
        entry = {
            "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "start_balance": start_balance,
            "end_balance":   end_balance,
            "pnl":           end_balance - start_balance
        }
        history.append(entry)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

if __name__ == "__main__":
    LoginWindow().mainloop()
