import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from backtesting import Backtest, Strategy

TICKER = "AMD"
START = "2019-01-01"
INTERVAL = "1d"

CASH0 = 100_000
COMMISSION = 0.001

TRAIN_END = "2022-12-30"
VAL_START = "2023-01-02"
VAL_END = "2023-12-29"
TEST_START = "2024-01-02"

THETA_GRID = [0.0, 0.0005, 0.0010, 0.0015, 0.0020, 0.0030]

#pobieranie danych
def download_ohlcv(ticker: str, start: str, interval: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=start,
        interval=interval,
        auto_adjust=False,
        progress=False
    )

    if df is None or df.empty:
        raise RuntimeError("yfinance zwrócił pusty DataFrame.")

    #czasem MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.rename(columns={"Adj Close": "AdjClose"})

    required = {"Open", "High", "Low", "Close", "AdjClose", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Brakuje kolumn: {missing}. Dostępne: {df.columns.tolist()}")

    df = df.copy()
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()

    df["log_ret"] = np.log(df["AdjClose"]).diff()
    df["ret"] = df["AdjClose"].pct_change()

    return df


#feature engineering
def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger(series: pd.Series, window: int = 20, n_std: float = 2.0):
    ma = series.rolling(window).mean()
    sd = series.rolling(window).std(ddof=0)
    upper = ma + n_std * sd
    lower = ma - n_std * sd
    width = (upper - lower) / (ma + 1e-12)
    return ma, upper, lower, width


def build_features(df: pd.DataFrame):
    df = df.copy()

    price = df["AdjClose"]
    log_price = np.log(price)

    # LAGI log-ret
    for k in [1, 2, 3, 5, 10]:
        df[f"log_ret_lag_{k}"] = df["log_ret"].shift(k)

    #rolling staty log-ret
    for w in [5, 10, 20]:
        df[f"roll_logret_sum_{w}"] = df["log_ret"].rolling(w).sum()
        df[f"roll_logret_mean_{w}"] = df["log_ret"].rolling(w).mean()
        df[f"roll_logret_std_{w}"] = df["log_ret"].rolling(w).std(ddof=0)

    #SMA/EMA + ratios
    for w in [10, 20, 50]:
        df[f"sma_{w}"] = price.rolling(w).mean()
        df[f"ema_{w}"] = price.ewm(span=w, adjust=False).mean()
        df[f"price_sma_ratio_{w}"] = price / (df[f"sma_{w}"] + 1e-12)
        df[f"price_ema_ratio_{w}"] = price / (df[f"ema_{w}"] + 1e-12)

    #RSI
    df["rsi_14"] = rsi(price, 14)

    #MACD
    macd_line, signal_line, hist = macd(price)
    df["macd_line"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = hist

    #Bollinger
    bb_ma, bb_up, bb_low, bb_width = bollinger(price, 20, 2)
    df["bb_ma_20"] = bb_ma
    df["bb_up_20"] = bb_up
    df["bb_low_20"] = bb_low
    df["bb_width_20"] = bb_width
    df["price_bb_ma_ratio_20"] = price / (df["bb_ma_20"] + 1e-12)

    #Volume
    df["vol_log"] = np.log(df["Volume"].replace(0, np.nan))
    df["vol_chg_1"] = df["Volume"].pct_change()
    df["vol_roll_mean_20"] = df["Volume"].rolling(20).mean()
    df["vol_roll_std_20"] = df["Volume"].rolling(20).std(ddof=0)

    #Range / intraday
    df["hl_range"] = (df["High"] - df["Low"]) / (df["Close"] + 1e-12)
    df["oc_return"] = (df["Close"] - df["Open"]) / (df["Open"] + 1e-12)

    # TARGET D+1 (log-return)
    df["y_next"] = log_price.shift(-1) - log_price

    feature_cols = [
        "log_ret_lag_1", "log_ret_lag_2", "log_ret_lag_3", "log_ret_lag_5", "log_ret_lag_10",
        "roll_logret_sum_5", "roll_logret_sum_10", "roll_logret_sum_20",
        "roll_logret_mean_5", "roll_logret_mean_10", "roll_logret_mean_20",
        "roll_logret_std_5", "roll_logret_std_10", "roll_logret_std_20",
        "price_sma_ratio_10", "price_sma_ratio_20", "price_sma_ratio_50",
        "price_ema_ratio_10", "price_ema_ratio_20", "price_ema_ratio_50",
        "rsi_14", "macd_line", "macd_signal", "macd_hist",
        "bb_width_20", "price_bb_ma_ratio_20",
        "vol_log", "vol_chg_1", "vol_roll_mean_20", "vol_roll_std_20",
        "hl_range", "oc_return",
    ]

    data = df[feature_cols + ["y_next"]].dropna()
    X = data[feature_cols]
    y = data["y_next"]

    return df, data, X, y, feature_cols

#time split
def time_split(data: pd.DataFrame, X: pd.DataFrame, y: pd.Series):
    idx = data.index

    train_mask = idx <= pd.to_datetime(TRAIN_END)
    val_mask = (idx >= pd.to_datetime(VAL_START)) & (idx <= pd.to_datetime(VAL_END))
    test_mask = idx >= pd.to_datetime(TEST_START)

    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_val, y_val = X.loc[val_mask], y.loc[val_mask]
    X_test, y_test = X.loc[test_mask], y.loc[test_mask]

    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        raise ValueError("Jeden z podzbiorów jest pusty. Sprawdź daty splitu.")

    return (train_mask, val_mask, test_mask,
            X_train, y_train, X_val, y_val, X_test, y_test)


#xgboost + fit
def fit_xgb_model(X_train, y_train, X_val, y_val, X_test, y_test):
    try:
        from xgboost import XGBRegressor
    except Exception as e:
        raise ImportError("Brak xgboost w środowisku. Zainstaluj: pip install xgboost") from e

    base_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )

    tscv = TimeSeriesSplit(n_splits=4)

    param_dist = {
        "n_estimators": [200, 400, 600, 800, 1000],
        "max_depth": [2, 3, 4, 5, 6],
        "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5, 7],
        "reg_alpha": [0.0, 0.01, 0.1, 1.0],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0],
    }

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=40,
        scoring="neg_mean_absolute_error",
        cv=tscv,
        verbose=0,
        random_state=42,
        n_jobs=-1,
        refit=True
    )

    search.fit(X_train, y_train)

    best_params = search.best_params_
    best_cv_mae = -search.best_score_

    # walidacja
    val_pred = search.best_estimator_.predict(X_val)
    val_mae = mean_absolute_error(y_val, val_pred)
    val_rmse = mean_squared_error(y_val, val_pred) ** 0.5

    # final fit na train+val
    X_train_val = pd.concat([X_train, X_val], axis=0)
    y_train_val = pd.concat([y_train, y_val], axis=0)

    final_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        **best_params
    )
    final_model.fit(X_train_val, y_train_val)

    # test
    test_pred = final_model.predict(X_test)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_rmse = mean_squared_error(y_test, test_pred) ** 0.5

    metrics = pd.DataFrame([{
        "best_cv_mae": best_cv_mae,
        "val_mae": val_mae,
        "val_rmse": val_rmse,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        **best_params
    }])

    return final_model, metrics


#theta
def simple_strategy_eval(df_: pd.DataFrame, theta: float):
    tmp = df_.copy()
    tmp["signal_raw"] = (tmp["y_pred_xgb"] > theta).astype(int)
    tmp["position"] = tmp["signal_raw"].shift(1).fillna(0).astype(int)  # brak look-ahead
    tmp["strat_logret"] = tmp["position"] * tmp["y_next"]

    exposure = tmp["position"].mean()
    hit_rate = (
        (tmp.loc[tmp["position"] == 1, "y_next"] > 0).mean()
        if (tmp["position"] == 1).any() else np.nan
    )

    cum_log = tmp["strat_logret"].sum()
    cum_simple = float(np.exp(cum_log) - 1)

    return {
        "theta": theta,
        "n_obs": len(tmp),
        "exposure_%": round(100 * exposure, 2),
        "hit_rate_in_pos_%": round(100 * hit_rate, 2) if hit_rate == hit_rate else np.nan,
        "cum_return_%": round(100 * cum_simple, 2),
        "avg_daily_logret": tmp["strat_logret"].mean()
    }


def select_theta_and_build_signal(data: pd.DataFrame, val_mask: pd.Series):
    val_df = data.loc[val_mask].copy()

    results = [simple_strategy_eval(val_df, th) for th in THETA_GRID]
    val_table = pd.DataFrame(results).sort_values("cum_return_%", ascending=False)

    best_theta = float(val_table.iloc[0]["theta"])

    data = data.copy()
    data["signal_raw"] = (data["y_pred_xgb"] > best_theta).astype(int)
    data["Signal"] = data["signal_raw"].shift(1).fillna(0).astype(int)

    return data, val_table, best_theta


#backtesting
class MLAllInLongCash(Strategy):
    def init(self):
        pass

    def next(self):
        sig = int(self.data.Signal[-1])
        if sig == 1 and not self.position:
            self.buy()
        elif sig == 0 and self.position:
            self.position.close()


class BuyHold(Strategy):
    def init(self):
        self.bought = False

    def next(self):
        if not self.bought:
            self.buy()
            self.bought = True


class SanityBuyOnce(Strategy):
    def init(self):
        self.bought = False

    def next(self):
        if not self.bought:
            self.buy()
            self.bought = True


def run_backtests(df_bt_test: pd.DataFrame):
    bt_ml = Backtest(
        df_bt_test,
        MLAllInLongCash,
        cash=CASH0,
        commission=COMMISSION,
        trade_on_close=False,
        exclusive_orders=True,
        finalize_trades=True
    )

    bt_bh = Backtest(
        df_bt_test,
        BuyHold,
        cash=CASH0,
        commission=COMMISSION,
        trade_on_close=False,
        exclusive_orders=True,
        finalize_trades=True
    )

    stats_ml = bt_ml.run()
    stats_bh = bt_bh.run()

    # sanity
    df_slim = df_bt_test[["Open", "High", "Low", "Close"]].copy()
    bt_sanity = Backtest(
        df_slim,
        SanityBuyOnce,
        cash=CASH0,
        commission=COMMISSION,
        trade_on_close=False,
        exclusive_orders=False,
        finalize_trades=True
    )
    stats_sanity = bt_sanity.run()

    return bt_ml, bt_bh, bt_sanity, stats_ml, stats_bh, stats_sanity


def stats_to_csv(stats, filename: str):
    s = stats.copy()
    if hasattr(s, "to_frame"):
        out = s.to_frame(name="value")
    else:
        out = pd.DataFrame({"value": s})
    out.to_csv(filename)


#pipeline
def main():

    #data
    df = download_ohlcv(TICKER, START, INTERVAL)
    print(f"Data OK | rows={len(df)} | range={df.index.min().date()} -> {df.index.max().date()}")

    #features
    df_feat, data, X, y, feature_cols = build_features(df)
    print(f"Features OK | n_features={len(feature_cols)} | X={X.shape} | y={y.shape}")

    #split
    (train_mask, val_mask, test_mask,
     X_train, y_train, X_val, y_val, X_test, y_test) = time_split(data, X, y)

    print(f"[3] Split OK | "
          f"train={len(X_train)} ({X_train.index.min().date()}->{X_train.index.max().date()}) | "
          f"val={len(X_val)} ({X_val.index.min().date()}->{X_val.index.max().date()}) | "
          f"test={len(X_test)} ({X_test.index.min().date()}->{X_test.index.max().date()})")

    #model
    final_model, metrics = fit_xgb_model(X_train, y_train, X_val, y_val, X_test, y_test)
    metrics.to_csv("model_metrics.csv", index=False)
    print("XGB OK | saved: model_metrics.csv")

    #predictions
    data = data.copy()
    data["y_pred_xgb"] = final_model.predict(X)

    #theta+signal
    data, val_table, best_theta = select_theta_and_build_signal(data, val_mask)
    val_table.to_csv("theta_validation_table.csv", index=False)
    print(f"Theta OK | best_theta={best_theta} | saved: theta_validation_table.csv")

    #exposition report
    exp_train = round(100 * data.loc[train_mask, "Signal"].mean(), 2)
    exp_val = round(100 * data.loc[val_mask, "Signal"].mean(), 2)
    exp_test = round(100 * data.loc[test_mask, "Signal"].mean(), 2)
    print(f"    Exposure % | train={exp_train} | val={exp_val} | test={exp_test}")

    #backtest feed
    df_bt = df.loc[data.index, ["Open", "High", "Low", "Close", "Volume"]].copy()
    df_bt["Signal"] = data["Signal"].astype(int)

    df_bt_test = df_bt.loc[test_mask].copy()
    if df_bt_test.empty:
        raise ValueError("df_bt_test jest pusty. Sprawdź maski dat.")

    print(f"Backtest data OK | test_rows={len(df_bt_test)} | "
          f"range={df_bt_test.index.min().date()}->{df_bt_test.index.max().date()}")

    #run backtest
    bt_ml, bt_bh, bt_sanity, stats_ml, stats_bh, stats_sanity = run_backtests(df_bt_test)

    print("Backtests OK")
    print("    ML Return [%]:", round(float(stats_ml["Return [%]"]), 2),
          "| BH Return [%]:", round(float(stats_bh["Return [%]"]), 2))

    #export stats
    stats_to_csv(stats_ml, "backtest_stats_ml.csv")
    stats_to_csv(stats_bh, "backtest_stats_bh.csv")
    stats_to_csv(stats_sanity, "backtest_stats_sanity.csv")
    print("Saved: backtest_stats_ml.csv, backtest_stats_bh.csv, backtest_stats_sanity.csv")

    #plots
    bt_ml.plot(filename="MLAllInLongCash.html", open_browser=False)
    bt_bh.plot(filename="BuyHold.html", open_browser=False)
    print("Saved HTML reports: MLAllInLongCash.html, BuyHold.html")

main()
