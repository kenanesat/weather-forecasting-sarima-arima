import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

BASE_DIR = Path(__file__).resolve().parent

TEMP_FILE = BASE_DIR / "temperature.csv"
WIND_FILE = BASE_DIR / "wind.csv"
HUMIDITY_FILE = BASE_DIR / "humidity.csv"
RAIN_FILE = BASE_DIR / "rain.csv"

DATA_FREQ = "6h"
STEP_HOURS = 6
STEPS_PER_DAY = 24 // STEP_HOURS

FORECAST_DAYS = 365
FORECAST_STEPS = STEPS_PER_DAY * FORECAST_DAYS

# SARIMA on residuals
# d is NOT fixed here. It is selected by ADF, but minimum d is forced to 1.
RESIDUAL_P = 1
RESIDUAL_Q = 1
RESIDUAL_SEASONAL_ORDER = (1, 0, 1, 7 * STEPS_PER_DAY)   # 7 days in 6-hour steps

# ADF settings
ADF_ALPHA = 0.05
MIN_ADF_D = 1
MAX_ADF_D = 2
MIN_ADF_OBS = 40

MAX_ITER = 60
SHORT_GAP_LIMIT = 4

TEMP_LOWER_MARGIN = 3.0
TEMP_UPPER_MARGIN = 3.0
WIND_UPPER_MARGIN = 5.0


def find_matching_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    lower_map = {col.lower(): col for col in df.columns}

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    raise ValueError(
        f"No suitable {label} column found. "
        f"Expected one of: {', '.join(candidates)}"
    )


def load_csv_or_fail(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"{name} is empty.")

    return df


def prepare_base_columns(df: pd.DataFrame, value_col: str, output_col: str) -> pd.DataFrame:
    if not {"datetime", "city", value_col}.issubset(df.columns):
        raise ValueError(f"Required columns missing for {output_col}: datetime, city, {value_col}")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    df = df.dropna(subset=["datetime", "city", value_col]).copy()
    df = df.sort_values(["city", "datetime"])

    return df.rename(columns={value_col: output_col})


def resample_city_series(df: pd.DataFrame, value_col: str, how: str) -> pd.DataFrame:
    grouped = df.set_index("datetime").groupby("city")[value_col].resample(DATA_FREQ)

    if how == "mean":
        result = grouped.mean()
    elif how == "sum":
        result = grouped.sum()
    else:
        raise ValueError(f"Unsupported resample method: {how}")

    return result.reset_index()


def load_and_prepare() -> pd.DataFrame:
    temp_df = load_csv_or_fail(TEMP_FILE, "temperature.csv")
    wind_df = load_csv_or_fail(WIND_FILE, "wind.csv")
    humidity_df = load_csv_or_fail(HUMIDITY_FILE, "humidity.csv")
    rain_df = load_csv_or_fail(RAIN_FILE, "rain.csv")

    wind_col = find_matching_column(
        wind_df,
        ["wind", "windspeed", "wind_speed", "windspeed_10m", "wind_speed_10m"],
        "wind"
    )

    humidity_col = find_matching_column(
        humidity_df,
        ["humidity", "relativehumidity", "relative_humidity", "relative_humidity_2m", "relativehumidity_2m"],
        "humidity"
    )

    rain_col = find_matching_column(
        rain_df,
        ["rain", "precipitation", "precipitation_sum", "rain_sum"],
        "rain"
    )

    temp_df = prepare_base_columns(temp_df, "temperature", "temp")
    wind_df = prepare_base_columns(wind_df, wind_col, "wind")
    humidity_df = prepare_base_columns(humidity_df, humidity_col, "humidity")
    rain_df = prepare_base_columns(rain_df, rain_col, "rain")

    temp_df = resample_city_series(temp_df, "temp", "mean")
    wind_df = resample_city_series(wind_df, "wind", "mean")
    humidity_df = resample_city_series(humidity_df, "humidity", "mean")
    rain_df = resample_city_series(rain_df, "rain", "sum")

    df = pd.merge(temp_df, wind_df, on=["datetime", "city"], how="inner")
    df = pd.merge(df, humidity_df, on=["datetime", "city"], how="inner")
    df = pd.merge(df, rain_df, on=["datetime", "city"], how="inner")

    if df.empty:
        raise ValueError("Merged dataframe is empty. Check datetime/city matches.")

    return df.sort_values(["city", "datetime"]).reset_index(drop=True)


def longest_valid_block(mask: pd.Series) -> pd.Index:
    if mask.empty or not mask.any():
        return mask.index[:0]

    group_id = (mask != mask.shift(fill_value=False)).cumsum()
    valid_groups = []

    for gid, part in mask.groupby(group_id):
        if bool(part.iloc[0]):
            valid_groups.append((len(part), gid))

    if not valid_groups:
        return mask.index[:0]

    best_gid = max(valid_groups)[1]
    return mask[group_id == best_gid].index


def prepare_city_block(city_df: pd.DataFrame) -> pd.DataFrame:
    city_df = city_df.sort_values("datetime").set_index("datetime")[
        ["temp", "wind", "humidity", "rain"]
    ]

    full_index = pd.date_range(
        start=city_df.index.min(),
        end=city_df.index.max(),
        freq=DATA_FREQ
    )

    city_df = city_df.reindex(full_index)

    for col in ["temp", "wind", "humidity", "rain"]:
        city_df[col] = city_df[col].interpolate(
            method="time",
            limit=SHORT_GAP_LIMIT,
            limit_direction="both"
        )

    valid_mask = city_df[["temp", "wind", "humidity", "rain"]].notna().all(axis=1)
    keep_index = longest_valid_block(valid_mask)
    city_df = city_df.loc[keep_index].copy()

    if city_df.empty:
        raise ValueError("no continuous valid data block")

    city_df.index.name = "datetime"
    return city_df.astype(float)


def slot_key(index: pd.DatetimeIndex) -> pd.Index:
    return pd.Index(
        [f"{ts.month:02d}-{ts.day:02d} {ts.hour:02d}:00" for ts in index],
        dtype="object"
    )


def fallback_key(index: pd.DatetimeIndex) -> pd.Index:
    return pd.Index(
        [f"{ts.month:02d} {ts.hour:02d}:00" for ts in index],
        dtype="object"
    )


def build_climatology(series: pd.Series):
    s = series.dropna().copy()

    frame = pd.DataFrame({
        "value": s.values,
        "main_key": slot_key(s.index),
        "fallback_key": fallback_key(s.index),
    })

    main_climatology = frame.groupby("main_key")["value"].mean().to_dict()
    fallback_climatology = frame.groupby("fallback_key")["value"].mean().to_dict()
    global_mean = float(s.mean())

    return main_climatology, fallback_climatology, global_mean


def make_baseline(index: pd.DatetimeIndex, main_climatology: dict, fallback_climatology: dict, global_mean: float) -> pd.Series:
    main_keys = slot_key(index)
    fallback_keys = fallback_key(index)

    values = []
    for mk, fk in zip(main_keys, fallback_keys):
        if mk in main_climatology:
            values.append(main_climatology[mk])
        elif fk in fallback_climatology:
            values.append(fallback_climatology[fk])
        else:
            values.append(global_mean)

    return pd.Series(values, index=index, dtype=float)


def clean_series(series: pd.Series) -> pd.Series:
    return series.replace([np.inf, -np.inf], np.nan).dropna().astype(float)


def difference_series(series: pd.Series, d: int) -> pd.Series:
    result = clean_series(series)

    for _ in range(d):
        result = result.diff()

    return clean_series(result)


def get_adf_p_value(series: pd.Series) -> float:
    s = clean_series(series)

    if len(s) < MIN_ADF_OBS:
        return 1.0

    if s.nunique() <= 1:
        return 0.0

    try:
        return float(adfuller(s, autolag="AIC")[1])
    except Exception:
        return 1.0


def choose_d_with_adf_min_d1(residual_series: pd.Series, label: str) -> int:
    """
    ADF is used, but d=0 is not allowed.
    The model tries d=1 first. If d=1 is stationary, it uses d=1.
    If d=1 is still not stationary, it tries d=2 and uses d=2.
    """
    for d in range(MIN_ADF_D, MAX_ADF_D + 1):
        test_series = difference_series(residual_series, d)
        p_value = get_adf_p_value(test_series)

        if p_value < ADF_ALPHA:
            print(f"ADF: {label} -> d={d}, p-value={p_value:.6g}, stationary")
            return d

        print(f"ADF: {label} -> d={d}, p-value={p_value:.6g}, not stationary")

    print(f"ADF: {label} -> using maximum d={MAX_ADF_D}")
    return MAX_ADF_D


def fit_residual_sarima(residual_series: pd.Series, label: str):
    # IMPORTANT: No 3-year cut here. The full available history is used.
    residual_series = clean_series(residual_series)

    d_value = choose_d_with_adf_min_d1(residual_series, label)
    order = (RESIDUAL_P, d_value, RESIDUAL_Q)

    # With differencing, no constant trend is safer for long forecasts.
    trend_value = "n"

    model = SARIMAX(
        residual_series,
        order=order,
        seasonal_order=RESIDUAL_SEASONAL_ORDER,
        trend=trend_value,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    return model.fit(disp=False, maxiter=MAX_ITER)


def clip_forecast(series: pd.Series, original: pd.Series, variable_name: str) -> pd.Series:
    observed_min = float(original.min())
    observed_max = float(original.max())

    if variable_name == "temp":
        lower_bound = observed_min - TEMP_LOWER_MARGIN
        upper_bound = observed_max + TEMP_UPPER_MARGIN
        return series.clip(lower=lower_bound, upper=upper_bound)

    if variable_name == "wind":
        upper_bound = observed_max + WIND_UPPER_MARGIN
        return series.clip(lower=0.0, upper=upper_bound)

    if variable_name == "humidity":
        return series.clip(lower=0.0, upper=100.0)

    if variable_name == "rain":
        upper_bound = max(observed_max * 1.5, observed_max + 5.0, 5.0)
        return series.clip(lower=0.0, upper=upper_bound)

    return series


def forecast_variable(series: pd.Series, steps: int, future_index: pd.DatetimeIndex, variable_name: str, city_name: str) -> pd.Series:
    main_climatology, fallback_climatology, global_mean = build_climatology(series)

    baseline_hist = make_baseline(
        series.index,
        main_climatology,
        fallback_climatology,
        global_mean
    )

    residual = series - baseline_hist
    label = f"{city_name} {variable_name} residual"

    try:
        fit = fit_residual_sarima(residual, label)
        residual_forecast = pd.Series(
            fit.forecast(steps=steps),
            index=future_index,
            dtype=float
        )
    except Exception as e:
        print(f"SARIMA fallback: {label} -> {e}")
        residual_forecast = pd.Series(
            np.zeros(steps, dtype=float),
            index=future_index
        )

    baseline_future = make_baseline(
        future_index,
        main_climatology,
        fallback_climatology,
        global_mean
    )

    forecast = baseline_future + residual_forecast
    forecast = clip_forecast(forecast, series, variable_name)

    return forecast


def train_and_forecast(city_df: pd.DataFrame, city_name: str) -> pd.DataFrame:
    prepared = prepare_city_block(city_df)

    if len(prepared) < 365:
        raise ValueError("not enough regular data")

    future_index = pd.date_range(
        start=prepared.index[-1] + pd.Timedelta(hours=STEP_HOURS),
        periods=FORECAST_STEPS,
        freq=DATA_FREQ
    )

    temp_forecast = forecast_variable(
        prepared["temp"],
        steps=FORECAST_STEPS,
        future_index=future_index,
        variable_name="temp",
        city_name=city_name
    )

    wind_forecast = forecast_variable(
        prepared["wind"],
        steps=FORECAST_STEPS,
        future_index=future_index,
        variable_name="wind",
        city_name=city_name
    )

    humidity_forecast = forecast_variable(
        prepared["humidity"],
        steps=FORECAST_STEPS,
        future_index=future_index,
        variable_name="humidity",
        city_name=city_name
    )

    rain_forecast = forecast_variable(
        prepared["rain"],
        steps=FORECAST_STEPS,
        future_index=future_index,
        variable_name="rain",
        city_name=city_name
    )

    return pd.DataFrame({
        "datetime": future_index,
        "city": city_name,
        "predicted_temperature": temp_forecast.values,
        "predicted_wind": wind_forecast.values,
        "predicted_humidity": humidity_forecast.values,
        "predicted_rain": rain_forecast.values
    })


def main():
    df = load_and_prepare()

    print("Number of cities:", df["city"].nunique())
    print("Data frequency:", DATA_FREQ)
    print("Forecast horizon (days):", FORECAST_DAYS)
    print("Model: baseline + SARIMA residual")
    print("ADF: enabled")
    print("Minimum d:", MIN_ADF_D)
    print("Maximum d:", MAX_ADF_D)
    print("Training history: full available history")
    print("Variables: temperature, wind, humidity, rain")

    all_results = []

    for city in df["city"].unique():
        print(f"Training: {city}")
        city_df = df[df["city"] == city][
            ["datetime", "temp", "wind", "humidity", "rain"]
        ].copy()

        try:
            result = train_and_forecast(city_df, city)
            all_results.append(result)
            print(f"OK: {city}")
        except MemoryError:
            print(f"FAIL: {city} -> memory error")
        except Exception as e:
            print(f"FAIL: {city} -> {e}")

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
    else:
        final_df = pd.DataFrame(columns=[
            "datetime",
            "city",
            "predicted_temperature",
            "predicted_wind",
            "predicted_humidity",
            "predicted_rain"
        ])

    output_path = BASE_DIR / "final_forecast_1year_6hour_sarima_adf_min_d1_full_history_all.csv"
    final_df.to_csv(output_path, index=False)

    print("\nDONE")
    print("Forecast rows:", len(final_df))
    print("Forecast city count:", final_df["city"].nunique() if not final_df.empty else 0)
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
