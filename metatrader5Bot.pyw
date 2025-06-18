# metatrader5Bot.pyw

from datetime import datetime
print(f"{datetime.now():%Y-%m-%d %H:%M:%S} ✅ Ejecución de inicio.")

import os
import csv
import json
import time
import glob
import joblib
import logging
import keyring
import tempfile
import requests
import threading
import numpy as np
import pandas as pd
import tkinter as tk
import yfinance as yf
import mplfinance as mpf
import MetaTrader5 as mt5
import tkinter.ttk as ttk
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from io import StringIO
from queue import Queue, Empty
from functools import lru_cache
from scipy.special import expit
from ta.trend import SMAIndicator
from collections import defaultdict
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from tkinter import messagebox, filedialog
from sklearn.cluster import MiniBatchKMeans
from tkinter.scrolledtext import ScrolledText
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler
from datetime import timezone, timedelta
from sklearn.ensemble import RandomForestClassifier
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
KEYRING_SERVICE = "MT5BotService"
META_FILE = "meta.json"
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
VALID_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
# Para descargas con yfinance
YF_TICKERS = {
  "BTCUSD": "BTC-USD",
  "AAPL":   "AAPL",
  "XAUUSD": "GC=F",
}
# Para MT5
MT5_SYMBOLS = {
  "BTCUSD": "BTCUSD",
  "AAPL":   "AAPL",
  "XAUUSD": "XAUUSD",
}
# Lista de claves lógicas
SYMBOL_KEYS = ["AAPL", "BTCUSD", "XAUUSD"]
INTRADAY_DAYS = 29
WINDOW_DAYS   = 7

START_DATE_STR = "2024-06-17"
END_DATE_STR   = "2025-06-17"
START_DATE     = datetime.strptime(START_DATE_STR, "%Y-%m-%d")
END_DATE       = datetime.strptime(END_DATE_STR,   "%Y-%m-%d")

log_queue = Queue()
done_queue = Queue()

def log(msg: str, level=logging.INFO):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    logging.log(level, msg)
    log_queue.put(f"{ts} {msg}")

# ----------------------
# Logging (UTF-8)
# ----------------------
logging.basicConfig(
    handlers=[logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')],
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ"
)

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
        "window_size": 20,
        "interval":       1,
        "max_ops":        5,
        "min_confidence": 0.75,
        "auto":           False,
        "login":          None,
        "server":         None,
        "alpha":          0.7,
        "atr_factor":     2.0,   # múltiplo de ATR para TP
        "risk_pct":       0.01,  # % de balance arriesgado
        "conf_exponent":  1.0,    # exponente aplicado a la confianza
        "max_risk_pct": 0.02,
        "use_sigmoid":  False,
        "sigmoid_a":    10.0,
        "months_to_fetch": 12,
        "alpha_vantage_key": "XN8PA6Y72IUJ81ZA"
    }

    if not os.path.exists(CONFIG_FILE):
        save_config(defaults)
        log(f"🆕 Se creó {CONFIG_FILE} con valores por defecto.")
        return defaults

    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        # rellenar valores faltantes
        for k, v in defaults.items():
            cfg.setdefault(k, v)
        return cfg

    except (json.JSONDecodeError, ValueError) as e:
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

def load_meta():
    if os.path.exists(META_FILE):
        return json.load(open(META_FILE, "r", encoding="utf-8"))
    return {}

def save_meta(meta):
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_feedback():
    """
    Carga feedback.json.  Si está corrupto o vacío,
    lo reinicia en el mismo archivo (evitando renombrados).
    """
    try:
        # Abrimos y cargamos
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("Feedback no es un dict")
        return data

    except (json.JSONDecodeError, ValueError, FileNotFoundError):
        # Si falla la lectura o JSON inválido, recreamos feedback.json
        with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f)
        log("🆕 feedback.json reinicializado.", logging.WARNING)
        return {}

def validar_csv(path: str) -> bool:
    """
    Comprueba que el CSV exista y tenga exactamente las columnas esperadas.
    path: ruta al archivo CSV
    returns: True si las columnas de la cabecera coinciden con VALID_COLUMNS
    """
    try:
        cols = list(pd.read_csv(path, nrows=0).columns)
        return set(cols) == set(VALID_COLUMNS)
    except Exception:
        return False


def guardar_csv(df: pd.DataFrame,
                tipo: str,
                user_ticker: str,
                fecha_inicio: datetime = None,
                fecha_fin: datetime = None):

    ticker_sane = user_ticker.replace("/", "-")

    if tipo == "raw":
        # Si nos pasan fechas, analizamos duración
        if fecha_inicio is not None and fecha_fin is not None:
            delta = (fecha_fin - fecha_inicio).days
            prefix = "mes_" if delta <= 31 else "meses_"
            fname = (
                f"{prefix}{ticker_sane}_"
                f"{fecha_inicio.strftime('%Y-%m-%d')}_"
                f"{fecha_fin.strftime('%Y-%m-%d')}.csv"
            )
        else:
            # histórico completo (sin rango explícito)
            fname = f"meses_{ticker_sane}_{START_DATE_STR}_{END_DATE_STR}.csv"
    elif tipo == "intradia":
        fname = (
            f"intradia_{ticker_sane}_"
            f"{fecha_inicio.strftime('%Y-%m-%d')}_"
            f"{fecha_fin.strftime('%Y-%m-%d')}.csv"
        )
    else:
        raise ValueError(f"Tipo desconocido: {tipo}")

    path = os.path.join(DATA_DIR, fname)

    if os.path.exists(path):
        log(f"[SKIP] {fname} ya existe.")
        return
    if df.empty:
        log(f"[SKIP] {fname}: DataFrame vacío.")
        return

    df.to_csv(path, index=False)
    if validar_csv(path):
        log(f"[OK]   {fname} guardado correctamente.")
    else:
        log(f"[ERR]  {fname} inválido, se borra.")
        os.remove(path)


def descarga_diaria(ticker_sym: str, user_ticker: str):
    """Descarga todo el historial diario y guarda meses_<TICKER>_… o mes_<TICKER>_…"""
    log(f"[Daily] Descargando {ticker_sym} ({START_DATE_STR} → {END_DATE_STR}) …")
    df = yf.download(
        ticker_sym,
        start=START_DATE_STR,
        end=END_DATE_STR,
        interval="1d",
        auto_adjust=False,
        progress=False
    )
    df = df.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df.columns = VALID_COLUMNS
    guardar_csv(df,
                tipo="raw",
                user_ticker=user_ticker,
                fecha_inicio=START_DATE,
                fecha_fin=END_DATE)
    time.sleep(1)


def descarga_intradia(ticker_sym: str, user_ticker: str):
    """Descarga intradía 1 min últimos INTRADAY_DAYS días, segmentado en WINDOW_DAYS."""
    f_fin = datetime.now()
    f_ini = f_fin - timedelta(days=INTRADAY_DAYS)
    all_dfs = []
    cursor = f_ini
    log(f"[Intraday] {ticker_sym}: últimos {INTRADAY_DAYS} días…")
    while cursor < f_fin:
        sub_end = min(cursor + timedelta(days=WINDOW_DAYS), f_fin)
        log(f"    Ventana: {cursor.date()} → {sub_end.date()}")
        try:
            df = yf.download(
                ticker_sym,
                start=cursor.strftime("%Y-%m-%d"),
                end=sub_end.strftime("%Y-%m-%d"),
                interval="1m",
                auto_adjust=False,
                progress=False,
                threads=False
            )
        except Exception as e:
            log(f"    [WARN] no hay intradía para {ticker_sym} {cursor.date()}–{sub_end.date()}: {e}")
            df = pd.DataFrame()

        if not df.empty:
            df = df.reset_index()[["Datetime", "Open", "High", "Low", "Close", "Volume"]]
            df.columns = VALID_COLUMNS
            all_dfs.append(df)
        cursor = sub_end
        time.sleep(1)

    if all_dfs:
        df_full = pd.concat(all_dfs, ignore_index=True)
        df_full = df_full.drop_duplicates(subset=["timestamp"], keep="first")
        guardar_csv(df_full,
                    tipo="intradia",
                    user_ticker=user_ticker,
                    fecha_inicio=f_ini,
                    fecha_fin=f_fin)
        log(f"[OK]   intradia_{user_ticker}_{f_ini.strftime('%Y-%m-%d')}_{f_fin.strftime('%Y-%m-%d')}.csv guardado correctamente.")
    else:
        log(f"[SKIP] intradia_{user_ticker}: ningún dato descargado.")

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
        raise MT5InitError("No credentials in keyring")
    for attempt in range(1, retries+1):
        if mt5.initialize(login=login, password=pwd, server=server):
            log("✅ MT5 iniciado y logueado.")
            return True
        log(f"⚠️ Intento {attempt} de login fallido: {mt5.last_error()}", logging.WARNING)
        time.sleep(2)
    raise MT5InitError("MT5.initialize() failed after retries")

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
    # Forzar visibilidad en MarketWatch
    if not mt5.symbol_select(sym, True):
        log(f"⚠️ No se pudo seleccionar {sym} en MarketWatch")
    info = mt5.symbol_info(sym)
    if info is None:
        log(f"❌ symbol_info devolvió None para {sym}: {mt5.last_error()}")
    return info

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
        self.historical_means = {}
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
        df = pd.DataFrame(X, columns=FEATURE_NAMES_30)

        # Guardamos las medias históricas para imputación
        mean_cols = ['rsi14', 'sma5', 'vol_rel', 'profit_acc']
        self.historical_means = {c: df[c].mean() for c in mean_cols}

        Xs = self.scaler.fit_transform(X)
        self.model.partial_fit(Xs)
        self.is_fitted = True

        df['cluster'] = self.model.predict(Xs)

        # Señal BUY/SELL y confianza
        self.cluster_signal = {
            int(c): (1 if df.loc[df.cluster == c, 'return'].mean() > 0 else -1)
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
                            was_correct = bool(label)
                            feats_file = f"feat_{ticket}.json"
                            if not os.path.exists(feats_file):
                                continue
                            data = json.load(open(feats_file, "r", encoding="utf-8"))
                            row = []
                            for name in FEATURE_NAMES_30:
                                val = data.get(name, np.nan)
                                if np.isnan(val):
                                    if name in self.historical_means:
                                        val = self.historical_means[name]
                                    else:
                                        val = 0
                                row.append(val)
                            feat = np.array([row])
                            cid = int(self.model.predict(self.scaler.transform(feat))[0])

                            counts[cid] += 1

                            pred = self.cluster_signal[cid]
                            actual = +1 if data['return'] > 0 else -1
                            if (pred == actual) == was_correct:
                                successes[cid] += 1

                        for cid, tot in counts.items():
                            if tot == 0:
                                self.cluster_confidence[cid] = 0.5
                            else:
                                self.cluster_confidence[cid] = successes[cid] / tot

                        log(f"🔎 Cluster→confianza (actualizada): {self.cluster_confidence}")

                    self.save()
        except Exception as e:
            log(f"❌ Error en UModel.update: {e}", logging.ERROR)

# ----------------------
# Supervised Model
# ----------------------
class SupervisedModel:
    def __init__(self, u_model):
        """
        u_model: instancia de UnsupervisedModel, para generar pseudo-etiquetas
        si aún no hay modelo supervisado en disco.
        """
        self.u_model = u_model
        self._warned_unfitted = False

        if os.path.exists(SUPERVISED_FILE):
            # Si ya existe, simplemente cargamos
            self.clf = joblib.load(SUPERVISED_FILE)
            log("🔄 Supervisado cargado.")
        else:
            # Si no existe, inicializamos y bootstrappeamos
            self.clf = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight="balanced"
            )
            self._bootstrap_with_pseudo_labels()

    def log_label(self, ticket_id: str,
                  features: list,
                  label: int,
                  source: str,
                  description: str):
        """
        Registra en JSONL cada etiqueta generada.

        - ticket_id: identificador único (p.ej. filename o ticket)
        - features: lista [return, range] usada para etiquetar.
        - label: 0 o 1.
        - source: 'pseudo' o 'feedback'.
        - description: breve texto de contexto (e.g. "bootstrap KMeans".
        """
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "ticket":    ticket_id,
            "features":  features,
            "label":     label,
            "source":    source,
            "description": description
        }
        path = "labels_log.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _bootstrap_with_pseudo_labels(self, pseudo_threshold: float = 0.0):
        """
        Genera pseudo-etiquetas para todos los feat_*.json
        usando el cluster_signal de u_model y entrena el RF sólo
        si hay ambas clases. Registra todas las etiquetas.
        """
        pseudo_X, pseudo_y = [], []
        um = self.u_model

        for fn in glob.glob("feat_*.json"):
            ticket = os.path.splitext(os.path.basename(fn))[0][5:]
            data = json.load(open(fn, "r", encoding="utf-8"))
            feat = [data['return'], data['range']]
            full = [data.get(name, 0) for name in FEATURE_NAMES_30]
            cid = um.model.predict(um.scaler.transform([full]))[0]
            label = 1 if um.cluster_signal.get(int(cid), 0) > 0 else 0

            # Registrar siempre la etiqueta pseudo
            self.log_label(ticket, feat, label,
                           source="pseudo",
                           description=f"cluster {cid} → label {label}")

            pseudo_X.append(feat)
            pseudo_y.append(label)

        # Contar clases
        counts = {lbl: pseudo_y.count(lbl) for lbl in set(pseudo_y)}
        if len(counts) < 2:
            log(f"⚠️ Bootstrap omitido: solo una clase en pseudo_y = {counts}")
            return

        # Entrenar con pseudo-etiquetas equilibradas
        self.clf.fit(pseudo_X, pseudo_y)
        log(f"🏷️ Supervisado inicial con pseudo-etiquetas: {counts}")
        save_atomic(self.clf, SUPERVISED_FILE, log)

    def predict_proba(self, feat):
        """
        Devuelve la probabilidad de éxito para [return, range].
        Si el clasificador solo vio una clase, devuelve 0.5.
        """
        try:
            # Si solo hay una clase entrenada, probability neutral
            if len(self.clf.classes_) < 2:
                return 0.5
            probs = self.clf.predict_proba([feat])[0]
            return probs[1]
        except NotFittedError:
            log("⚠️ Modelo supervisado sin entrenar, usando probabilidad neutra.", logging.WARNING)
            return 0.5
        except Exception as e:
            log(f"❌ Error predict_proba RF: {e}", logging.ERROR)
            return 0.5

    def train_from_feedback(self, include_pseudo=True, pseudo_threshold=0.8):
        """Re-entrena el RF con feedback humano y opcionalmente pseudo-etiquetas."""
        fb = load_feedback() if os.path.exists(FEEDBACK_FILE) else {}
        X, y = [], []
        processed = set()

        # 1) Feedback humano
        for ticket, label in fb.items():
            fn = f"feat_{ticket}.json"
            if os.path.exists(fn):
                data = json.load(open(fn, "r", encoding="utf-8"))
                X.append([data['return'], data['range']])
                y.append(int(label))
                processed.add(str(ticket))

        # 2) Pseudo-etiquetas
        if include_pseudo:
            um = self.u_model
            for fn in glob.glob("feat_*.json"):
                ticket = os.path.splitext(os.path.basename(fn))[0][5:]
                if ticket in processed:
                    continue
                data = json.load(open(fn, "r", encoding="utf-8"))
                conf = data.get('unsup_conf')
                lbl  = data.get('unsup_label')
                if conf is not None and lbl is not None and conf >= pseudo_threshold:
                    X.append([data['return'], data['range']])
                    y.append(int(lbl))

        if X:
            try:
                self.clf.fit(X, y)
                self._warned_unfitted = False
                log(f"🏋️ Supervisado entrenado con {len(X)} ejemplos.")
                save_atomic(self.clf, SUPERVISED_FILE, log)
            except Exception as e:
                log(f"❌ Error entrenando RF: {e}", logging.ERROR)

    def save(self):
        save_atomic(self.clf, SUPERVISED_FILE, log)

    def predict(self, feat):
        """Predicción binaria (True=éxito)."""
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
            # 1) Si no existe modelo supervisado, hacer initial_training
            if not os.path.exists(SUPERVISED_FILE):
                log("⚠️ Modelo Supervisado no encontrado, ejecutando entrenamiento inicial completo.")
                self.initial_training()
                log("✅ Modelos entrenados. Comenzando trading.")

            # 2) Ahora sí arrancar el loop
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
        if info is None:
            # Saltar este símbolo o usar valores por defecto
            log(f"⏭️ Símbolo de mercado {symbol} no hallado (info None)")
            return np.array([]), df  # o maneja de otra forma
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
        """
        1) Lee todos los CSV mensuales e intradía de data/
        2) Para cada archivo extrae ventanas de tamaño WINDOW
        3) Genera features y entrena unsupervised + bootstrap supervised
        4) Actualiza progress_bar en UI
        """
        WINDOW = load_config().get("window_size", 20)

        # 1) Listado de archivos históricos e intradía
        files_raw   = glob.glob(os.path.join(DATA_DIR, "meses_*.csv"))
        files_intra = glob.glob(os.path.join(DATA_DIR, "intradia_*.csv"))
        total       = len(files_raw) + len(files_intra)
        processed   = 0

        all_feats = []

        for fn in files_raw + files_intra:
            # 2) Leemos el CSV y renombramos columnas para compatibilidad
            df = pd.read_csv(fn, parse_dates=["timestamp"])
            # Alias: 'timestamp' → 'time' y 'volume' → 'tick_volume'
            df = df.rename(columns={
                'timestamp': 'time',
                'volume':    'tick_volume'
            })

            # Alias para mantener compatibilidad con generate_features:
            df = df.rename(columns={"volume": "tick_volume"})

            # 2.1) Extraer el ticker (p.ej. 'AAPL', 'BTCUSD', 'XAUUSD')
            #    nombres: meses_AAPL_2024-06-17_2025-06-17.csv
            #            intradia_BTCUSD_2025-05-19_2025-06-17.csv
            basename = os.path.basename(fn)
            ticker   = basename.split("_")[1]
            symbol = MT5_SYMBOLS.get(ticker, ticker)  # coincide con el símbolo XBTUSD

            # 2.2) Ventanas deslizantes de tamaño WINDOW
            for start in range(0, len(df) - WINDOW + 1, WINDOW):
                win = df.iloc[start : start + WINDOW]
                feats, _ = self.generate_features(win, symbol)
                if feats.size:
                    all_feats.append(feats)

            # 3) Avanzar barra de progreso
            processed += 1
            pct = 100 * processed / total
            if getattr(self, 'ui_panel', None):
                self.ui_panel.progress_bar['value'] = pct

        # 4) Validación: ¿sacamos algo?
        if not all_feats:
            log("❌ No se extrajeron features de CSV locales.")
            return
        X = np.vstack(all_feats)
        # — Entrenamiento no supervisado —
        self.u_model.scaler.fit(X)
        Xs = self.u_model.scaler.transform(X)
        self.u_model.model.partial_fit(Xs)
        self.u_model.is_fitted = True

        # — Recalcular cluster_signal y cluster_confidence —  # <<<
        dfX = pd.DataFrame(X, columns=FEATURE_NAMES_30)
        dfX['cluster'] = self.u_model.model.predict(Xs)
        # señal BUY/SELL
        self.u_model.cluster_signal = {
            int(c): (1 if dfX.loc[dfX.cluster == c, 'return'].mean() > 0 else -1)
            for c in dfX['cluster'].unique()
        }
        # confianza histórica
        counts    = dfX.groupby('cluster').size().to_dict()
        successes = dfX[dfX['return'] > 0].groupby('cluster').size().to_dict()
        self.u_model.cluster_confidence = {
            int(c): successes.get(c, 0) / counts[c]
            for c in counts
        }
        self.u_model.save()
 
        log("✅ Unsupervised entrenado con CSV locales.")

        # supervised bootstrap
        pseudo_X, pseudo_y = [], []
        for row in X:
            cid = int(self.u_model.model.predict([row])[0])
            label = 1 if self.u_model.cluster_signal[cid]>0 else 0
            pseudo_X.append([row[0], row[1]])
            pseudo_y.append(label)
        if pseudo_X:
            self.s_model.clf.fit(pseudo_X, pseudo_y)
            self.s_model.save()
            log(f"🏷️ Supervisado inicial ({len(pseudo_X)} ejemplos).")

        # reset  barra
        if getattr(self, 'ui_panel', None):
            self.ui_panel.progress_bar['value'] = 0

    def descargar_siguiente_task(self, accumulated, tasks, total_slices, month_list):
        """
        Función recursiva que descarga cada slice con delay
        y va actualizando la progress bar y el log.
        - accumulated: dict de listas donde guardamos DataFrames por símbolo.
        - tasks: lista de tuplas (sym, month_index) pendientes.
        - total_slices: número total de slices para calcular pasos.
        """
        # 1) Si ya no quedan tareas, procesamos los datos descargados
        if not tasks:
            self.procesar_slices(accumulated)
            return

        # 2) Sacamos la siguiente tarea
        sym, m = tasks.pop(0)
        # obtenemos YYYY-MM desde month_list (creado previamente)
        month_str = month_list[m-1]  
        url = (
            f"https://www.alphavantage.co/query"
            f"?function=TIME_SERIES_INTRADAY"
            f"&symbol={sym}&interval=1min"
            f"&month={month_str}"
            f"&outputsize=full"
            f"&extended_hours=true"
            f"&datatype=csv"
            f"&apikey={self.API_KEY}"
        )
        resp = requests.get(url)
        if resp.status_code == 200:
            try:
                df_csv = pd.read_csv(
                    StringIO(resp.text),
                    parse_dates=['timestamp']
                ).rename(columns={'timestamp':'time'})
                accumulated[sym].append(df_csv.sort_values('time'))
            except Exception as e:
                log(f"⚠️ Error parseando CSV de {sym} mes {month_str}: {e}", logging.WARNING)
        else:
            log(f"⚠️ Falló mes {month_str} de {sym}: HTTP {resp.status_code}", logging.WARNING)

        # 3) Avanzamos contador y calculamos paso de la barra
        processed = total_slices - len(tasks)
        step = 100.0 / total_slices
        log(f"📥 Slice {month_str} de {sym} descargado ({processed}/{total_slices})")
        if getattr(self, 'ui_panel', None):
            self.ui_panel.progress_bar.step(step)

        # 4) Programamos la siguiente descarga con delay
        delay_ms = 2000  # para pruebas; en producción podrías usar 12000
        if getattr(self, 'ui_panel', None):
            # Agenda la siguiente descarga en el hilo de la UI
            self.ui_panel.after(delay_ms, lambda: self.descargar_siguiente_task(accumulated, tasks, total_slices, month_list))
        else:
            # Sin UI, podemos bloquear el hilo principal
            time.sleep(delay_ms / 1000.0)
            self.descargar_siguiente_task(accumulated, tasks, total_slices, month_list)

    def procesar_slices(self, accumulated):
        """
        Una vez descargados todos los slices, concatena, extrae features
        y entrena el modelo no supervisado y el bootstrap supervisado.
        """
        WINDOW = load_config().get("window_size", 20)
        all_feats = []

        for sym, frames in accumulated.items():
            if not frames:
                log(f"⚠️ Sin datos para {sym}, se omite.")
                continue

            df_sym = pd.concat(frames, ignore_index=True)
            for start in range(0, len(df_sym), WINDOW):
                window = df_sym.iloc[start:start + WINDOW]
                if len(window) < WINDOW:
                    break
                feats, _ = self.generate_features(window, sym)
                if feats.size:
                    all_feats.append(feats)

        if not all_feats:
            log("❌ No se extrajeron features de ningún símbolo.")
            return

        # Entrenamiento unsupervised
        X  = np.vstack(all_feats)
        self.u_model.scaler.fit(X)
        Xs = self.u_model.scaler.transform(X)
        self.u_model.model.partial_fit(Xs)
        self.u_model.is_fitted = True

        dfX = pd.DataFrame(X, columns=FEATURE_NAMES_30)
        dfX['cluster'] = self.u_model.model.predict(Xs)
        self.u_model.cluster_signal = {
            int(c): (1 if dfX.loc[dfX.cluster == c, 'return'].mean() > 0 else -1)
            for c in dfX['cluster'].unique()
        }
        counts = dfX.groupby('cluster').size().to_dict()
        succs  = dfX[dfX['return'] > 0].groupby('cluster').size().to_dict()
        self.u_model.cluster_confidence = {
            c: succs.get(c, 0) / counts[c] for c in counts
        }
        self.u_model.save()
        log("✅ Unsupervised entrenado con AlphaVantage.")

        # Bootstrap supervisado (pseudo-etiquetas)
        pseudo_X, pseudo_y = [], []
        for row in X:
            cid   = int(self.u_model.model.predict([row])[0])
            label = 1 if self.u_model.cluster_signal[cid] > 0 else 0
            pseudo_X.append([row[0], row[1]])
            pseudo_y.append(label)

        if pseudo_X:
            self.s_model.clf.fit(pseudo_X, pseudo_y)
            self.s_model.save()
            log(f"🏷️ Supervisado inicial con {len(pseudo_X)} pseudo-etiquetas.")

        # Reset de la barra al 0 en GUI
        if getattr(self, 'ui_panel', None):
            self.ui_panel.after(0, lambda: self.ui_panel.progress_bar.configure(value=0))

    def backtest_training(self, months: int = None, window_size: int = None):
        """
        Backtest usando AlphaVantage:
         - Descarga datos intradía completos (outputsize=full).
         - Extrae ventanas de `window_size` minutos y genera features.
         - Hace partial_fit incremental en KMeans por batch de ventanas.
        """
        cfg     = load_config()
        API_KEY = cfg.get("alpha_vantage_key")
        WINDOW  = window_size or cfg.get("window_size", 20)
        BATCH   = cfg.get("backtest_batch", 5000)

        log(f"📊 Backtest histórico iniciando via AlphaVantage (intraday full)...")
        buffer = []
        total_processed = 0

        for sym in self.symbols:
            # 1) Descarga completa intradía
            url = (
                f"https://www.alphavantage.co/query"
                f"?function=TIME_SERIES_INTRADAY"
                f"&symbol={sym}&interval=1min"
                f"&outputsize=full"
                f"&datatype=csv"
                f"&apikey={API_KEY}"
            )
            resp = requests.get(url)
            if resp.status_code != 200:
                log(f"⚠️ Falló descarga intraday full de {sym}: HTTP {resp.status_code}", logging.WARNING)
                continue

            try:
                # 2) Parse CSV, renombrar timestamp→time
                df_sym = (
                    pd.read_csv(StringIO(resp.text), parse_dates=['timestamp'])
                      .rename(columns={'timestamp': 'time'})
                      .sort_values('time')
                      .reset_index(drop=True)
                )
            except Exception as e:
                log(f"⚠️ Error parseando CSV intraday full de {sym}: {e}", logging.WARNING)
                continue

            # 3) Ventanas de `WINDOW` filas cada una
            for start in range(0, len(df_sym), WINDOW):
                window_df = df_sym.iloc[start:start + WINDOW]
                if len(window_df) < WINDOW:
                    break
                feats, _ = self.generate_features(window_df, sym)
                if feats.size:
                    buffer.extend(feats)

                # 4) partial_fit por batch
                if len(buffer) >= BATCH:
                    Xb = np.vstack(buffer)
                    Xs = self.u_model.scaler.transform(Xb)
                    self.u_model.model.partial_fit(Xs)
                    total_processed += len(buffer)
                    log(f"🔄 Backtest batch {sym}: +{len(buffer)} muestras (total {total_processed}).")
                    buffer.clear()

        # 5) Procesar remanente final
        if buffer:
            Xb = np.vstack(buffer)
            Xs = self.u_model.scaler.transform(Xb)
            self.u_model.model.partial_fit(Xs)
            total_processed += len(buffer)
            log(f"🔄 Backtest final: +{len(buffer)} muestras (total {total_processed}).")

        # 6) Guardar modelo y notificar GUI
        self.u_model.save()
        log("✅ Backtest completo y modelo guardado.")
        if getattr(self, 'ui_panel', None):
            done_queue.put('backtest_done')

    def train_supervised_from_backtest(self):
        """
        Re-entrena el RandomForest con etiquetas reales de todo el histórico M1:
        etiqueta = 1 si return>0, else 0.
        """
        log("🔨 Re-entrenando supervisado con histórico M1 completo…")

        X_sup, y_sup = [], []
        # Asumimos que hemos guardado antes los feats en disco:
        # e.g. feat_hist_{sym}_{i}.npy
        for fn in glob.glob("feat_hist_*.npy"):
            f = np.load(fn)       # shape (30,)
            ret, ran = f[0], f[1]
            X_sup.append([ret, ran])
            y_sup.append(1 if ret > 0 else 0)

        if not X_sup:
            log("⚠️ No hay feats históricos para supervisado.")
            return

        self.s_model.clf.fit(X_sup, y_sup)
        self.s_model.save()
        log(f"🏋️ RF re-entrenado con {len(X_sup)} ejemplos históricos.")

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

    def calculate_atr(self, symbol, period=14, timeframe=mt5.TIMEFRAME_M1):
        """
        Calcula el ATR sobre un timeframe intradía (por defecto M1).
        - symbol: par a analizar.
        - period: número de velas para ATR.
        - timeframe: unidad de tiempo (mt5.TIMEFRAME_M1, M5, etc.).
        """
        # número de velas mínimo = period * 5
        count = period * 5
        # obtenemos 'count' velas en el timeframe indicado
        rates = mt5.copy_rates_from(symbol, timeframe,
                                    datetime.now(timezone.utc), count)
        df = pd.DataFrame(rates)
        if df.empty:
            return 0.0
        # TA
        atr = AverageTrueRange(high=df['high'],
                               low=df['low'],
                               close=df['close'],
                               window=period)
        return float(atr.average_true_range().iloc[-1])

    def compute_volume(self, info, entry_price, sl_price):
        """
        Calcula lotaje para arriesgar hasta max_risk_pct del balance:
        """
        cfg         = load_config()
        risk_pct    = cfg.get("risk_pct", 0.01)       # % por operación
        max_risk_pct= cfg.get("max_risk_pct", 0.02)   # techo de riesgo
        balance     = mt5.account_info().balance
        risk_amount = balance * risk_pct
        max_risk    = balance * max_risk_pct
        # Limitamos el riesgo
        risk_amount = min(risk_amount, max_risk)

        sl_dist = abs(entry_price - sl_price)
        if sl_dist == 0:
            return info.volume_min

        # valor monetario de un pip
        pip_value = info.point * getattr(info, "trade_contract_size", 1.0)
        raw_lots  = risk_amount / (sl_dist * pip_value)

        # ajustamos a step, min y max
        step = info.volume_step
        lots = max(info.volume_min,
                   min(info.volume_max,
                       round(raw_lots / step) * step))
        return lots

    def open_order(self, sym, otype, feat, unsup_conf=None):
        """
        Abre una orden de compra o venta en MetaTrader 5 y guarda las características de la operación en un archivo JSON.

        :param sym: Símbolo del instrumento (por ejemplo, 'EURUSD').
        :param otype: Tipo de orden (mt5.ORDER_TYPE_BUY o mt5.ORDER_TYPE_SELL).
        :param feat: Características de la operación.
        :param unsup_conf: Confianza no supervisada (opcional).
        :return: ticket de la operación si se ejecutó correctamente, None en caso contrario.
        """
        # Verificar si el símbolo está disponible
        symbol_info = mt5.symbol_info(sym)
        if symbol_info is None:
            log(f"Símbolo {sym} no encontrado.")
            return None

        # Comprobar si el símbolo está visible en el Market Watch
        if not symbol_info.visible:
            log(f"Símbolo {sym} no visible en el Market Watch. Intentando agregarlo...")
            if not mt5.symbol_select(sym, True):
                log(f"Error al agregar el símbolo {sym} al Market Watch.")
                return None

        # Obtener el precio de compra o venta según el tipo de orden
        price = mt5.symbol_info_tick(sym).ask if otype == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(sym).bid
        point = mt5.symbol_info(sym).point
        lot = 0.1  # Tamaño de la operación en lotes
        deviation = 20  # Desviación permitida en puntos

        # Calcular el precio de Stop Loss y Take Profit
        sl = price - 100 * point if otype == mt5.ORDER_TYPE_BUY else price + 100 * point
        tp = price + 100 * point if otype == mt5.ORDER_TYPE_BUY else price - 100 * point

        # Crear la solicitud de la orden
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": sym,
            "volume": lot,
            "type": otype,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": deviation,
            "magic": 234000,
            "comment": "Orden abierta desde Python",
            "type_time": mt5.ORDER_TIME_GTC,  # Good Till Canceled
            "type_filling": mt5.ORDER_FILLING_RETURN,
        }

        # Enviar la solicitud de la orden
        result = mt5.order_send(request)

        # Verificar el resultado de la operación
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            log(f"Error al abrir la orden: {result.retcode}")
            return None

        # Obtener el ticket de la operación
        ticket = result.order
        log(f"Orden abierta con éxito. Ticket: {ticket}")

        # Guardar las características de la operación en un archivo JSON
        row = {name: float(feat[0][i]) for i, name in enumerate(FEATURE_NAMES_30)}
        row.update({
            "unsup_conf": unsup_conf,
            "unsup_label": int(self.u_model.predict(feat) == mt5.ORDER_TYPE_BUY)
        })
        with open(f"feat_{ticket}.json", "w", encoding="utf-8") as f:
            json.dump(row, f, ensure_ascii=False, indent=2)

        return ticket

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
        """
        Cierra posición si:
         - pérdida > 1%, o
         - precio alcanza el target calculado.
        """
        try:
            for pos in mt5.positions_get() or []:
                entry = pos.price_open
                tick  = mt5.symbol_info_tick(pos.symbol)
                if not tick: continue

                current = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
                pnl_pct = ((current - entry) / entry) * 100 * (1 if pos.type == mt5.ORDER_TYPE_BUY else -1)

                # Stop-loss fijo (-1%)
                if pnl_pct < -1.0:
                    log(f"🔻 Cerrando {pos.ticket}: pérdida {pnl_pct:.2f}%")
                    self.close_position(pos)
                    continue

                # Cierre al alcanzar target dinámico
                meta_file = f"meta_{pos.ticket}.json"
                if os.path.exists(meta_file):
                    meta = json.load(open(meta_file, "r", encoding="utf-8"))
                    target = meta.get("target_price")
                    # Para BUY: current >= target; para SELL: current <= target
                    if (pos.type == mt5.ORDER_TYPE_BUY and current >= target) or \
                       (pos.type == mt5.ORDER_TYPE_SELL and current <= target):
                        log(f"✨ Cerrando {pos.ticket}: alcanzado TP {current:.5f} (target {target:.5f})")
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

        # Cargamos config inicial
        cfg = load_config()
        alpha = cfg.get("alpha", 0.7)  # peso para el modelo supervisado en la confianza combinada

        while self.trading_mode:
            try:
                start_time = time.time()

                # 1) Recarga parámetros dinámicos
                cfg = load_config()
                interval = cfg.get("interval", interval)
                max_ops  = cfg.get("max_ops", max_ops)
                alpha    = cfg.get("alpha", alpha)

                # 2) Gestionar stops y cierres forzosos
                self.manage_positions()

                # 3) Comprobar cuántas posiciones hay abiertas
                positions = mt5.positions_get() or []
                if len(positions) < max_ops:
                    candidates = []

                    # 4) Generar lista de candidatos
                    for sym in self.symbols:
                        # 4.1) Datos de precio recientes
                        df = get_historical_data(sym, mt5.TIMEFRAME_M1, 20)
                        feats, _ = self.generate_features(df, sym)
                        if feats.size == 0:
                            continue

                        last_feat   = feats[-1].reshape(1, -1)
                        last_return = float(last_feat[0][0])  # retorno de la última vela

                        # 4.2) Solo movimientos al alza
                        if last_return <= 0:
                            continue

                        # 4.3) Señal y confianza no supervisada
                        # (aunque abriremos siempre BUY)
                        conf_uns = self.u_model.predict_proba(last_feat)

                        # 4.4) Confianza supervisada
                        feat_sup = [last_return, float(last_feat[0][1])]
                        conf_sup = self.s_model.predict_proba(feat_sup)

                        # 4.5) Confianza combinada
                        combined_conf = alpha * conf_sup + (1 - alpha) * conf_uns

                        # 4.6) Filtrar por umbral mínimo
                        if combined_conf >= self.min_confidence:
                            score = last_return * combined_conf
                            candidates.append((sym, score, combined_conf, conf_uns, last_feat))

                    # 5) Ordenar candidatos por score descendente
                    candidates.sort(key=lambda x: x[1], reverse=True)

                    # 6) Abrir hasta completar max_ops
                    slots = max_ops - len(positions)
                    ops_count = 0
                    for sym, score, combined_conf, conf_uns, feat in candidates:
                        if slots <= 0:
                            break
                        # Forzamos BUY al ser solo alzas
                        ticket = self.open_order(sym, mt5.ORDER_TYPE_BUY, feat, conf_uns)
                        if ticket:
                            ops_count += 1
                            slots -= 1
                            log(f"✅ Trade #{ops_count} enviado para {sym} (conf={combined_conf:.2f})")
                else:
                    log(f"⏸️ Ya hay {len(positions)} posiciones abiertas (límite={max_ops}).")

                # 7) Logging post-apertura
                positions = mt5.positions_get() or []
                if positions:
                    for pos in positions:
                        log(f"📌 Posición abierta: {pos.symbol} #{pos.ticket} vol={pos.volume} open={pos.price_open}")
                else:
                    log("📌 No hay posiciones abiertas este ciclo.")

                # 8) Logging de deals del día
                now = datetime.now(timezone.utc)
                deals = mt5.history_deals_get(now.replace(hour=0, minute=0, second=0), now) or []
                if deals:
                    for d in deals:
                        log(f"📑 Deal: ticket={d.ticket}, sym={d.symbol}, price={d.price}, vol={d.volume}")
                else:
                    log("📑 Sin deals hoy.")

                # 9) Esperar hasta el siguiente ciclo
                elapsed = time.time() - start_time
                time.sleep(max(0, interval * 60 - elapsed))

            except Exception as e:
                log(f"❌ Error trading_loop: {e}", logging.ERROR)

        log("⏹️ Trading detenido.")

# Al iniciar el bot instanciamos los modelos y bot (inyección de u_model en SupervisedModel)
u_model = UnsupervisedModel()                               # crea el modelo no supervisado
s_model = SupervisedModel(u_model)                          # le pasas u_model para el bootstrap de pseudo-etiquetas
cfg     = load_config()                                     # carga config (interval, max_ops, min_confidence…)
bot = TradingBot([MT5_SYMBOLS[key] for key in SYMBOL_KEYS], # crea el bot con ambos modelos
                    u_model, s_model,
                    min_confidence=cfg.get("min_confidence", 0.75))
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
        # 1) Leer y validar credenciales de la UI
        try:
            login = int(self.e_login.get())
        except ValueError:
            return messagebox.showerror("Error", "Login debe ser numérico")
        pwd, srv = self.e_pwd.get(), self.e_srv.get()

        # 2) Guardar contraseña en keyring
        keyring.set_password(KEYRING_SERVICE, str(login), pwd)
        log("🔐 Credenciales seguras en keyring.")

        # 3) Actualizar config.json con todos los valores relevantes
        cfg.update({
            "window_size":       cfg.get("window_size", 20),
            "auto":              self.var_auto.get(),
            "login":             login,
            "server":            srv,
            "interval":          cfg.get("interval", 1),
            "max_ops":           cfg.get("max_ops", 5),
            "min_confidence":    cfg.get("min_confidence", 0.75),
            "alpha":             cfg.get("alpha", 0.7),
            "atr_factor":        cfg.get("atr_factor", 2.0),
            "risk_pct":          cfg.get("risk_pct", 0.01),
            "max_risk_pct":      cfg.get("max_risk_pct", 0.02),
            "use_sigmoid":       cfg.get("use_sigmoid", False),
            "sigmoid_a":         cfg.get("sigmoid_a", 10.0),
            "conf_exponent":     cfg.get("conf_exponent", 1.0),
            "months_to_fetch":   cfg.get("months_to_fetch", 12),
            "alpha_vantage_key": cfg.get("alpha_vantage_key", "XN8PA6Y72IUJ81ZA")
        })
        save_config(cfg)

        # 4) Intentar inicializar MT5; si falla, relanzar login
        try:
            initialize_mt5(login, srv)
        except MT5InitError as e:
            messagebox.showerror("Error de conexión", str(e))
            self.destroy()
            LoginWindow().mainloop()
            return

        '''
        # ——————————————————————————————————————————————
        # —— AÑADE ESTO PARA VOLCAR LOS SÍMBOLOS EN EL LOG ——
        # MT5
        for sym in mt5.symbols_get():
            log(f"🔍 MT5 disponible: {sym.name}")
        # Yahoo Finance (tu mapeo)
        for key, yf_ticker in YF_TICKERS.items():
            log(f"🔍 YF mapping: {key} → {yf_ticker}")
        # ——————————————————————————————————————————————
        '''

        # 5) Limpiar posiciones abiertas de sesiones previas
        for pos in mt5.positions_get() or []:
            bot.close_position(pos)

        # 6) Seleccionar símbolos y comprobar mercados abiertos
        open_syms = []
        for key in SYMBOL_KEYS:
            sym_mt5 = MT5_SYMBOLS[key]
            select_symbol(sym_mt5)
            info = mt5.symbol_info(sym_mt5)
            if info and info.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL:
                log(f"🔓 Mercado abierto: {sym_mt5}")
                open_syms.append(sym_mt5)
            else:
                log(f"🔒 Mercado cerrado: {sym_mt5}")

        if not open_syms:
            messagebox.showwarning(
                "Atención",
                "Ningún símbolo disponible. Volviendo al login."
            )
            shutdown_mt5()
            self.destroy()
            LoginWindow().mainloop()
            return

        # 7) Detectar y descargar CSV faltantes antes de mostrar el panel
        from datetime import datetime, timedelta

        for key in SYMBOL_KEYS:
            # raw histórico completo
            raw_path = os.path.join(
                DATA_DIR,
                f"meses_{key}_{START_DATE_STR}_{END_DATE_STR}.csv"
            )
            # intradía últimos INTRADAY_DAYS
            ini = datetime.now() - timedelta(days=INTRADAY_DAYS)
            fin = datetime.now()
            intra_path = os.path.join(
                DATA_DIR,
                f"intradia_{key}_{ini.strftime('%Y-%m-%d')}_{fin.strftime('%Y-%m-%d')}.csv"
            )

            if not validar_csv(raw_path):
                log(f"⬇️ Falta histórico mensual de {key}, descargando…")
                descarga_diaria(YF_TICKERS[key], key)

            if not validar_csv(intra_path):
                log(f"⬇️ Falta intradía de {key}, descargando…")
                descarga_intradia(YF_TICKERS[key], key)

        # 8) Asignar símbolos y abrir panel principal
        bot.symbols = open_syms
        panel = PanelWindow()
        bot.ui_panel = panel
        # cerramos el login y pasamos el control al panel
        self.destroy()
        # 9) Lanzamos la rutina de entrenamiento en segundo plano
        #    que usará todos los CSV locales ya presentes
        self.after(0, bot.initial_training)

class PanelWindow(tk.Tk):
    WINDOW_SIZE = 100

    def __init__(self):
        super().__init__()
        self.view_start = None
        self.view_end = None

        self.title("Panel IA bot MT5")
        self.configure(bg=DarkStyle.bg)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Área de logs
        self.rowconfigure(6, weight=0)
        self.log_widget = ScrolledText(
            self,
            state="disabled",
            bg=DarkStyle.log_bg,
            fg=DarkStyle.log_fg,
            font=("Consolas", 10),
            wrap="none",
            height=8
        )
        self.log_widget.grid(row=6, column=0, columnspan=8,
                             sticky="nsew", padx=5, pady=5)
        self._process_log_queue()

        # Column sizing
        for i in range(8):
            self.columnconfigure(i, weight=1)

        # Variables y carga de config
        cfg = load_config()
        self.var_interval = tk.StringVar(
            master=self, value=str(cfg.get("interval", 1)))
        self.var_maxops = tk.StringVar(
            master=self, value=str(cfg.get("max_ops", 5)))
        self.var_backtest = tk.StringVar(
            master=self, value=str(cfg.get("months_to_fetch", 12)))
        self.var_apikey = tk.StringVar(
            master=self, value=cfg.get("alpha_vantage_key", ""))

        for var in (self.var_interval, self.var_maxops,
                    self.var_backtest, self.var_apikey):
            var.trace_add("write", self._on_params)

        # ----- ROW 0: Barra de progreso -----
        self.progress_bar = ttk.Progressbar(
            self,
            orient="horizontal",
            length=400,
            mode="determinate",
            maximum=100,
            value=0
        )
        self.progress_bar.grid(
            row=0, column=0, columnspan=8,
            padx=5, pady=(5, 0), sticky="ew"
        )

        # ----- ROW 1: Parámetros de configuración -----
        tk.Label(self, text="Ciclos * Minuto:", bg=DarkStyle.bg,
                 fg=DarkStyle.fg).grid(
            row=1, column=0, padx=5, pady=5, sticky="e")
        tk.Entry(self, textvariable=self.var_interval,
                 bg=DarkStyle.entry_bg, fg=DarkStyle.entry_fg).grid(
            row=1, column=1, padx=5, pady=5, sticky="ew")

        tk.Label(self, text="Ops/ciclo:", bg=DarkStyle.bg,
                 fg=DarkStyle.fg).grid(
            row=1, column=2, padx=5, pady=5, sticky="e")
        tk.Entry(self, textvariable=self.var_maxops,
                 bg=DarkStyle.entry_bg, fg=DarkStyle.entry_fg).grid(
            row=1, column=3, padx=5, pady=5, sticky="ew")

        tk.Label(self, text="APIKey AlphaVantage:", bg=DarkStyle.bg,
                 fg=DarkStyle.fg).grid(
            row=1, column=4, padx=5, pady=5, sticky="e")
        tk.Entry(self, textvariable=self.var_apikey,
                 bg=DarkStyle.entry_bg, fg=DarkStyle.entry_fg).grid(
            row=1, column=5, padx=5, pady=5, sticky="ew")

        tk.Label(self, text="Meses de Entrenado:", bg=DarkStyle.bg,
                 fg=DarkStyle.fg).grid(
            row=1, column=6, padx=5, pady=5, sticky="e")
        tk.Entry(self, textvariable=self.var_backtest,
                 bg=DarkStyle.entry_bg, fg=DarkStyle.entry_fg).grid(
            row=1, column=7, padx=5, pady=5, sticky="ew")

        # ----- ROW 2: Gestión de modelos y sesión -----
        tk.Button(self, text="Exportar Modelo",
                  command=self._export_model,
                  bg=DarkStyle.btn_bg, fg=DarkStyle.btn_fg).grid(
            row=2, column=0, padx=5, pady=5, sticky="ew")
        tk.Button(self, text="Importar Modelo",
                  command=self._import_model,
                  bg=DarkStyle.btn_bg, fg=DarkStyle.btn_fg).grid(
            row=2, column=1, padx=5, pady=5, sticky="ew")
        tk.Button(self, text="Logs", command=self.show_logs,
                  bg=DarkStyle.btn_bg, fg=DarkStyle.btn_fg).grid(
            row=2, column=2, padx=5, pady=5, sticky="ew")
        tk.Button(self, text="Logout", command=self._logout,
                  bg=DarkStyle.btn_bg, fg=DarkStyle.btn_fg).grid(
            row=2, column=3, padx=5, pady=5, sticky="ew")
        tk.Button(self, text="Cerrar Operaciones",
                  command=self.force_close_all,
                  bg="#aa5500", fg=DarkStyle.fg).grid(
            row=2, column=4, padx=5, pady=5, sticky="ew")

        # ----- ROW 3: Trading & Training -----
        self.trade_btn = tk.Button(self, text="Iniciar Trading",
                                   command=self._toggle_trading,
                                   bg=DarkStyle.btn_bg, fg=DarkStyle.btn_fg)
        self.trade_btn.grid(row=3, column=0, padx=5, pady=5, sticky="ew")

        self.train_btn = tk.Button(self, text="Entrenamiento",
                                   command=self.toggle_training,
                                   bg=DarkStyle.btn_bg, fg=DarkStyle.btn_fg)
        self.train_btn.grid(row=3, column=1, padx=5, pady=5, sticky="ew")

        # **Aquí guardamos el botón en un atributo para poder referenciarlo luego**
        self.hist_btn = tk.Button(self, text="Entrenamiento Histórico",
                                  command=self._start_backtest,
                                  bg=DarkStyle.btn_bg, fg=DarkStyle.btn_fg)
        self.hist_btn.grid(row=3, column=2, padx=5, pady=5, sticky="ew")

        # ----- ROW 4: TWO LISTBOXES (OPEN / HISTORIC) -----
        self.rowconfigure(4, weight=1, minsize=150)
        frame_hist = tk.Frame(self, bg=DarkStyle.bg)
        frame_hist.grid(row=4, column=0, columnspan=8,
                        sticky="nsew", padx=5, pady=5)
        frame_hist.columnconfigure(0, weight=1)
        frame_hist.columnconfigure(1, weight=0)
        frame_hist.columnconfigure(2, weight=0)
        frame_hist.columnconfigure(3, weight=1)
        frame_hist.columnconfigure(4, weight=0)
        frame_hist.rowconfigure(0, weight=1)

        # Listbox izquierda
        self.lb_open = tk.Listbox(
            frame_hist, bg=DarkStyle.entry_bg, fg=DarkStyle.fg,
            font=("Consolas", 10), borderwidth=0, highlightthickness=0
        )
        self.lb_open.grid(row=0, column=0, sticky="nsew")
        tk.Scrollbar(frame_hist, orient="vertical",
                     command=self.lb_open.yview,
                     troughcolor=DarkStyle.entry_bg,
                     bg=DarkStyle.btn_bg,
                     activebackground=DarkStyle.btn_fg).grid(
            row=0, column=1, sticky="ns")
        self.lb_open.config(yscrollcommand=lambda *args: None)

        # Separador
        tk.Frame(frame_hist, width=10, bg=DarkStyle.bg).grid(
            row=0, column=2, sticky="ns")

        # Listbox derecha
        self.lb_hist = tk.Listbox(
            frame_hist, bg=DarkStyle.entry_bg, fg=DarkStyle.fg,
            font=("Consolas", 10), borderwidth=0, highlightthickness=0
        )
        self.lb_hist.grid(row=0, column=3, sticky="nsew")
        tk.Scrollbar(frame_hist, orient="vertical",
                     command=self.lb_hist.yview,
                     troughcolor=DarkStyle.entry_bg,
                     bg=DarkStyle.btn_bg,
                     activebackground=DarkStyle.btn_fg).grid(
            row=0, column=4, sticky="ns")
        self.lb_hist.config(yscrollcommand=lambda *args: None)
        self.lb_hist.bind("<Double-Button-1>",
                          lambda e: self.on_hist_doubleclick())
        self._last_hist_count = 0

        # ----- ROW 5: CHART -----
        self.rowconfigure(5, weight=1)
        self.fig = plt.Figure(figsize=(6, 3), facecolor=DarkStyle.bg)
        self.ax = self.fig.add_subplot(111, facecolor=DarkStyle.bg)
        self.ax.grid(True, color="#555555")
        self.fig.subplots_adjust(
            left=0.02, right=0.98, top=0.95, bottom=0.05)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        widget = self.canvas.get_tk_widget()
        widget.grid(row=5, column=0, columnspan=8, sticky="nsew")
        widget.configure(bg=DarkStyle.bg)
        tb_frame = tk.Frame(self, bg=DarkStyle.bg)
        tb_frame.grid(row=6, column=0, columnspan=8, sticky="ew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, tb_frame)
        self.toolbar.update()
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.fig.canvas.mpl_connect("pick_event", self._on_pick)
        self.fig.canvas.mpl_connect(
            "motion_notify_event", self._on_hover)

        # ----- START PERIODIC TASKS -----
        self._refresh_positions_list()
        self.after(1000, self._update_chart)
        self.after(1000, self._refresh_positions_list)
        self._poll_backtest_done()
        self.bind("<<BacktestDone>>", self._on_backtest_done)

        # Finalmente: ejecutar preload + entrenamiento histórico
        self.after(0, self._preload_and_train)

    def _preload_and_train(self):
        total_tasks = len(SYMBOL_KEYS) * 2
        step = 100 / total_tasks
        for key in SYMBOL_KEYS:
            # Histórico mensual
            raw_path = os.path.join(
                DATA_DIR,
                f"meses_{key}_{START_DATE_STR}_{END_DATE_STR}.csv"
            )
            if not validar_csv(raw_path):
                log(f"⬇️ Descargando raw {key}")
                descarga_diaria(YF_TICKERS[key], key)
            self.progress_bar.step(step)

            # Intradía reciente
            f_fin = datetime.now()
            f_ini = f_fin - timedelta(days=INTRADAY_DAYS)
            intra_path = os.path.join(
                DATA_DIR,
                f"intradia_{key}_{f_ini:%Y-%m-%d}_{f_fin:%Y-%m-%d}.csv"
            )
            if not validar_csv(intra_path):
                log(f"⬇️ Descargando intradía {key}")
                descarga_intradia(YF_TICKERS[key], key)
            self.progress_bar.step(step)

        bot.initial_training()

    # ------------------------------------------------------------------
    # Backtest management
    # ------------------------------------------------------------------
    def _poll_backtest_done(self):
        try:
            token = done_queue.get_nowait()
        except Empty:
            pass
        else:
            if token == "backtest_done":
                self.event_generate("<<BacktestDone>>")
        finally:
            self.after(100, self._poll_backtest_done)

    def _start_backtest(self):
        cfg = load_config()
        days = cfg.get("months_to_fetch", 12)
        self.hist_btn.config(text="Procesando CSV…", bg="#ffaa00")
        threading.Thread(target=bot.initial_training, daemon=True).start()
        log(f"▶️ Entrenamiento histórico de (con DATA) iniciado.")

    def _on_backtest_done(self, event=None):
        self.hist_btn.config(bg=DarkStyle.btn_bg,
                             text="Estado de IA")
        messagebox.showinfo(f"✅ Entrenamiento histórico de ({load_config().get('months_to_fetch')} meses) finalizado.")

    # ------------------------------------------------------------------
    # Position list handling
    # ------------------------------------------------------------------
    def _refresh_positions_list(self):
        """Refresh open positions and append new historical trades."""
        self.lb_open.delete(0, tk.END)
        for pos in mt5.positions_get() or []:
            tick = mt5.symbol_info_tick(pos.symbol)
            if not tick:
                continue
            current = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
            pnl_pct = ((current - pos.price_open) / pos.price_open) * 100 \
                      * (1 if pos.type == mt5.ORDER_TYPE_BUY else -1)
            text = f"OPEN {pos.symbol} #{pos.ticket}  PnL: {pnl_pct:+.2f}%"
            idx = self.lb_open.size()
            self.lb_open.insert("end", text)
            color = "#00FF00" if pnl_pct > 0 else \
                    "#FF0000" if pnl_pct < 0 else "#888888"
            self.lb_open.itemconfig(idx, fg=color)
            if pnl_pct < -1.0:        # auto close if loss exceeds 1%
                log(f"🔻 Auto-cerrando {pos.ticket}: pérdida {pnl_pct:.2f}%")
                bot.close_position(pos)

        trades = load_history().get("trades", [])
        for ses in trades[self._last_hist_count:]:
            pnl = ses.get("pnl", 0)
            marker = "▲" if pnl > 0 else "▼" if pnl < 0 else "▽"
            text = f"{ses['fecha']}   {marker}  PnL: {pnl:+.2f}"
            idx = self.lb_hist.size()
            self.lb_hist.insert("end", text)
            color = "#00FF00" if pnl > 0 else \
                    "#FF0000" if pnl < 0 else "#888888"
            self.lb_hist.itemconfig(idx, fg=color)
        self._last_hist_count = len(trades)

        self.lb_hist.see(tk.END)
        self.after(1000, self._refresh_positions_list)

    # ------------------------------------------------------------------
    # Chart helpers
    # ------------------------------------------------------------------
    def plot_recent_trades(self):
        now = datetime.now(timezone.utc)
        deals = mt5.history_deals_get(now.replace(hour=0, minute=0, second=0),
                                      now) or []
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
            pnl = (pc - po) / 0.0001 * \
                  (1 if od.type == mt5.ORDER_TYPE_BUY else -1)
            color = "green" if pnl > 0 else "red" if pnl < 0 else "gray"

            marker = self.ax.scatter(t_open, po, marker="v",
                                     s=80, color=color, zorder=5)
            txt = self.ax.text(
                (mdates.date2num(t_open) + mdates.date2num(t_close)) / 2,
                (po + pc) / 2,
                f"{pnl:+.1f} pips", color=color, fontsize=8,
                ha="center", backgroundcolor=DarkStyle.bg
            )

            marker.ticket = cd.ticket
            marker.pnl = pnl
            marker.set_picker(True)

    def add_trade_to_list(self, entry):
        idx = self.lb_hist.size()
        marker = "▲" if entry["pnl"] > 0 else \
                 "▼" if entry["pnl"] < 0 else "▽"
        text = f"{idx + 1:2d}. {entry['fecha']}   {marker}  PnL: {entry['pnl']:+.2f}"
        self.lb_hist.insert("end", text)
        color = "#00FF00" if entry["pnl"] > 0 else \
                "#FF0000" if entry["pnl"] < 0 else "#888888"
        self.lb_hist.itemconfig(idx, fg=color)
        self.lb_hist.see(tk.END)
        self._update_chart()

    # ------------------------------------------------------------------
    # Utility dialogs and logging
    # ------------------------------------------------------------------
    def ask_yes_no_cancel(self, title, prompt):
        dlg = tk.Toplevel(self)
        dlg.title(title)
        dlg.resizable(False, False)
        dlg.configure(bg=DarkStyle.bg)
        dlg.protocol("WM_DELETE_WINDOW", lambda: dlg.destroy())

        lbl = tk.Label(
            dlg, text=prompt, justify="left", wraplength=300,
            bg=DarkStyle.bg, fg=DarkStyle.fg
        )
        lbl.pack(padx=20, pady=(20, 10))

        result = {"value": None}

        def on_choice(val):
            result["value"] = val
            dlg.destroy()

        frm = tk.Frame(dlg, bg=DarkStyle.bg)
        frm.pack(pady=(0, 20))
        for text, val in [("Sí", True), ("No", False), ("Cancelar", None)]:
            tk.Button(
                frm, text=text, width=8,
                command=lambda v=val: on_choice(v),
                bg=DarkStyle.btn_bg, fg=DarkStyle.btn_fg,
                activebackground=DarkStyle.entry_bg,
                activeforeground=DarkStyle.btn_fg,
                relief="flat"
            ).pack(side="left", padx=5)

        dlg.transient(self)
        dlg.grab_set()
        self.wait_window(dlg)
        return result["value"]

    # ------------------------------------------------------------------
    # Trading actions
    # ------------------------------------------------------------------
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
        self.log_widget.configure(state="normal")
        self.log_widget.delete("1.0", tk.END)
        self.log_widget.configure(state="disabled")
        self.lb_open.delete(0, tk.END)
        self.lb_hist.delete(0, tk.END)

    def _toggle_trading(self):
        if bot.trading_mode:
            bot.trading_mode = False
            self.trade_btn.config(text="Iniciar Trading",
                                  bg=DarkStyle.btn_bg)
            log("⏹️ Se detuvo el trading de la IA.")
        else:
            self._start_bal = mt5.account_info().balance
            cfg = load_config()
            interval = cfg.get("interval", 1)
            max_ops = cfg.get("max_ops", 5)
            threading.Thread(
                target=bot.start_trading,
                args=(interval, max_ops),
                daemon=True
            ).start()
            self.trade_btn.config(text="Detener Trading", bg="#aa3333")
            log("▶️ Se comenzó el trading de la IA.")

    def copy_log(self):
        self.log_widget.configure(state="normal")
        txt = self.log_widget.get("1.0", tk.END)
        self.log_widget.configure(state="disabled")
        self.clipboard_clear()
        self.clipboard_append(txt)
        log("📋 Logs copiados al portapapeles.")

    # ------------------------------------------------------------------
    # Chart interaction handlers
    # ------------------------------------------------------------------
    def _on_pick(self, event):
        artista = event.artist
        ticket = getattr(artista, "ticket", None)
        pnl = getattr(artista, "pnl", None)
        log(f"🖱️ Pick en ticket={ticket}, pnl={pnl}")
        try:
            msg = f"Ganancia/Pérdida: {pnl:+.1f} pips" if pnl is not None \
                  else "No hay información de PnL."
            correcto = self.ask_yes_no_cancel(
                "Feedback transacción",
                msg + "\n\n¿Fue esta operación CORRECTA?"
            )
            if ticket is not None:
                fb = load_feedback() if os.path.exists(FEEDBACK_FILE) else {}
                fb[str(ticket)] = int(correcto)
                with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
                    json.dump(fb, f, indent=2)
                log(f"📝 Feedback guardado ticket={ticket}: {correcto}")
        except Exception as e:
            log(f"⚠️ Error en pick_event: {e}", logging.WARNING)

    def _on_hover(self, event):
        for txt in self.ax.texts:
            cont, _ = txt.contains(event)
            if cont:
                ticket = getattr(txt, "ticket", None)
                if ticket:
                    self.tooltip = tk.Toplevel(self)
                    tk.Label(
                        self.tooltip,
                        text=f"Ticket {ticket}: {txt.get_text()}"
                    ).pack()
                    self.tooltip.after(1000, self.tooltip.destroy)
                break

    # ------------------------------------------------------------------
    # Closing and logging
    # ------------------------------------------------------------------
    def _on_close(self):
        bot.training_mode = False
        bot.trading_mode = False
        shutdown_mt5()
        self.destroy()

    def _process_log_queue(self):
        try:
            while True:
                msg = log_queue.get_nowait()
                self.log_widget.configure(state="normal")
                self.log_widget.insert(tk.END, msg + "\n")
                self.log_widget.see(tk.END)
                self.log_widget.configure(state="disabled")
        except Empty:
            pass
        self.after(100, self._process_log_queue)

    def toggle_training(self):
        if bot.training_mode:
            bot.training_mode = False
            self.train_btn.config(text="Entrenamiento",
                                  bg=DarkStyle.btn_bg)
            log("⏹️ Deteniendo entrenamiento en vivo.")
        else:
            bot.start_live_training()
            self.train_btn.config(text="Detener Entrenamiento",
                                  bg="#aa3333")
            log("▶️ Entrenamiento iniciado desde UI.")

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    def _on_params(self, *args):
        try:
            cfg = load_config()
            cfg["interval"] = int(self.var_interval.get())
            cfg["max_ops"] = int(self.var_maxops.get())
            cfg["months_to_fetch"]   = int(self.var_backtest.get())
            cfg["alpha_vantage_key"] = self.var_apikey.get().strip()
            save_config(cfg)
        except ValueError:
            pass

    def _export_model(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".pkl",
            filetypes=[("Pickle", "*.pkl"), ("All", "*.*")],
            title="Exportar Modelo"
        )
        if path:
            joblib.dump(u_model.model, path)
            log(f"📤 KMeans exportado a {path}")

    def _import_model(self):
        path = filedialog.askopenfilename(
            filetypes=[("Pickle", "*.pkl"), ("All", "*.*")],
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

    # ------------------------------------------------------------------
    # Chart scrolling and updating
    # ------------------------------------------------------------------
    def _on_scroll(self, event):
        if self.view_start is None:
            return

        shift = PanelWindow.WINDOW_SIZE // 10 * (1 if event.step > 0 else -1)
        new_start = self.view_start + shift
        new_end = new_start + PanelWindow.WINDOW_SIZE

        max_index = len(self.xnums)
        if new_start < 0:
            new_start = 0
            new_end = min(PanelWindow.WINDOW_SIZE, max_index)
        if new_end > max_index:
            new_end = max_index
            new_start = max(0, max_index - PanelWindow.WINDOW_SIZE)

        self.view_start, self.view_end = new_start, new_end
        self.ax.set_xlim(
            self.xnums[self.view_start], self.xnums[self.view_end - 1]
        )
        self.canvas.draw()

    def _update_chart(self):
        cfg = load_config()
        interval = cfg.get("interval", 1)

        symbol_mt5 = MT5_SYMBOLS["BTCUSD"]
        df = get_historical_data(symbol_mt5, mt5.TIMEFRAME_M1, interval * 60)

        if df.empty:
            self.after(2000, self._update_chart)
            return

        total = len(df)
        N = min(total, PanelWindow.WINDOW_SIZE)
        if total == 0 or N == 0:
            self.after(2000, self._update_chart)
            return

        if self.view_start is None:
            self.view_start = total - N
            self.view_end = total
        data = df.iloc[self.view_start:self.view_end].copy()
        if data.empty:
            self.after(2000, self._update_chart)
            return

        self.xnums = mdates.date2num(data["time"])

        mc = mpf.make_marketcolors(up="cyan", down="magenta", inherit=True)
        style = mpf.make_mpf_style(
            marketcolors=mc,
            gridcolor="#555555",
            facecolor=DarkStyle.bg,
            edgecolor=DarkStyle.fg,
        )
        mpf_df = data.set_index("time")[["open", "high", "low", "close"]]
        if mpf_df.empty:
            self.after(2000, self._update_chart)
            return

        self.ax.clear()
        self.ax.set_facecolor(DarkStyle.bg)
        mpf.plot(
            mpf_df, type="candle", style=style,
            ax=self.ax, datetime_format="%H:%M",
            volume=False, show_nontrading=True
        )
        self.ax.grid(True, color="#555555")
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        self.ax.tick_params(colors=DarkStyle.fg)

        acc = mt5.account_info()
        if acc:
            title = f"Saldo:({acc.balance:.2f}) - Mercados:(AAPL, BTCUSD, XAUUSD) M1"
        else:
            title = "Saldo:(N/D) - Mercados:(AAPL, BTCUSD, XAUUSD) M1"
        self.ax.set_title(title, color=DarkStyle.fg)

        self.plot_recent_trades()

        self.fig.tight_layout()
        self.canvas.draw()
        self.after(2000, self._update_chart)

    # ------------------------------------------------------------------
    # Log viewer & feedback
    # ------------------------------------------------------------------
    def show_logs(self):
        win = tk.Toplevel(self)
        win.title("Logs del sistema")
        win.configure(bg=DarkStyle.bg)
        win.transient(self)
        win.grab_set()
        win.geometry("700x400")

        txt = ScrolledText(
            win, state="normal", bg=DarkStyle.log_bg, fg=DarkStyle.log_fg,
            font=("Consolas", 10), wrap="none"
        )
        txt.pack(fill="both", expand=True, padx=10, pady=10)

        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            content = f"⚠️ Error leyendo {LOG_FILE}: {e}"

        txt.insert("1.0", content)
        txt.configure(state="disabled")
        txt.see(tk.END)

        xsb = tk.Scrollbar(win, orient="horizontal", command=txt.xview)
        txt.configure(xscrollcommand=xsb.set)
        xsb.pack(fill="x", side="bottom")

        def poll_queue():
            try:
                while True:
                    line = log_queue.get_nowait()
                    txt.configure(state="normal")
                    txt.insert(tk.END, line + "\n")
                    txt.configure(state="disabled")
                    txt.see(tk.END)
            except Empty:
                pass
            win.after(100, poll_queue)

        win.after(100, poll_queue)

    def _record_history(self, start_balance, end_balance):
        if os.path.exists(HISTORY_FILE):
            history = json.load(open(HISTORY_FILE, "r", encoding="utf-8"))
        else:
            history = []
        entry = {
            "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "start_balance": start_balance,
            "end_balance": end_balance,
            "pnl": end_balance - start_balance,
        }
        history.append(entry)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    def on_hist_doubleclick(self):
        sel = self.lb_hist.curselection()
        if not sel:
            return
        idx = sel[0]
        trades = load_history().get("trades", [])
        ses = trades[idx]
        respuesta = self.ask_yes_no_cancel(
            "Feedback Sesión",
            f"Sesión {ses['fecha']}\nPnL = {ses['pnl']:+.2f}\n\n¿Fue satisfactoria?"
        )
        log(f"📝 Feedback sesión {ses['fecha']}: {respuesta}")
        fb = load_feedback()
        fb_key = f"session_{ses['fecha']}_{idx}"
        fb[fb_key] = int(respuesta) if respuesta is not None else None
        with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
            json.dump(fb, f, indent=2)
            bot.u_model.save()
            bot.s_model.train_from_feedback()

if __name__ == "__main__":
    LoginWindow().mainloop()
