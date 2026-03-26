import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================

# SEITE KONFIGURIEREN

# ============================================================

st.set_page_config(

    page_title="Ventilanalyse Dashboard",

    page_icon="wrench",

    layout="wide"

)

# ============================================================

# ZUORDNUNG VENTIL -> ANLAGENART

# ============================================================

ZUORDNUNG = {

    "F2251151TV6017": "Lufterhitzer",

    "F2251151TV6027": "Lufterhitzer",

    "F1311221TV6177": "Lufterhitzer",

    "F1311211TV5807": "WRG",

    "F1311221TV6137": "Lufterhitzer",

    "F1311221TV6147": "Lufterhitzer",

    "F1311221TV6157": "Lufterhitzer",

    "F1311221TV6167": "Lufterhitzer",

    "F1311221TV6117": "Lufterhitzer",

    "F1311211TV6017": "Luftkuehler",

    "F1311211TV6007": "Lufterhitzer",

    "F1311212TV6007": "Lufterhitzer",

    "F1311213TV6017": "Luftkuehler",

    "F1311212TV6017": "Luftkuehler",

    "F1311213TV6007": "Lufterhitzer",

    "F1311212TV5807": "WRG",

    "F1311213TV5807": "WRG",

    "F1311221TV6057": "Lufterhitzer",

    "F1311221TV6067": "Lufterhitzer",

    "F1311221TV6077": "Lufterhitzer",

    "F1311221TV6087": "Lufterhitzer",

    "F1311221TV6017": "Lufterhitzer",

    "F1311221TV6027": "Lufterhitzer",

    "F1313622TV6607": "Lufterhitzer",

    "F1313622TV6617": "Lufterhitzer",

    "F1311221TV6007": "Lufterhitzer",

    "F2252111TV6037": "Lufterhitzer",

    "F2251155TV6027": "Lufterhitzer",

    "F2251155TV6017": "Lufterhitzer",

    "F2251155TV6007": "Lufterhitzer",

    "F2251235TV6047": "Lufterhitzer",

    "F2251155TV6057": "Lufterhitzer",

    "F2251155TV6047": "Lufterhitzer",

    "F2251221TV6077": "Lufterhitzer",

    "F2251221TV6067": "Lufterhitzer",

    "F2821187TV2707": "WRG",

    "F2821186TV5807": "WRG",

    "F2821184TV2707": "WRG",

    "F2821183TV2707": "WRG",

    "F2821182TV2707": "WRG",

    "F2821181TV2707": "WRG",

    "F2821166TV5807": "WRG",

    "F2821165TV2707": "WRG",

    "F2821161TV6027": "WRG",

    "F2821151TV6004": "WRG",

    "F2821135TV6027": "WRG",

    "F2821131TV6008": "WRG",

    "F2821112TV5807": "WRG",

    "F2823662TV6157": "Lufterhitzer",

    "F2823662TV6147": "Lufterhitzer",

    "F2823662TV6137": "Lufterhitzer",

    "F2823662TV6127": "Lufterhitzer",

    "F2823662TV6117": "Lufterhitzer",

    "F2823662TV6107": "Lufterhitzer",

    "F2823662TV6177": "Lufterhitzer",

    "F2823662TV6187": "Lufterhitzer",

    "F2821187TV6007": "Lufterhitzer",

    "F2821186TV6007": "Lufterhitzer",

    "F2821184TV6007": "Lufterhitzer",

    "F2821184TV6008": "Lufterhitzer",

    "F2821183TV6007": "Lufterhitzer",

    "F2821182TV6007": "Lufterhitzer",

    "F2821181TV6007": "Lufterhitzer",

    "F2821166TV6007": "Lufterhitzer",

    "F2821165TV6007": "Lufterhitzer",

    "F2821161TV6007": "Lufterhitzer",

    "F2821151TV2106": "Lufterhitzer",

    "F2821151TV2116": "Lufterhitzer",

    "F2821151TV2126": "Lufterhitzer",

    "F2821136TV6007": "Lufterhitzer",

    "F2821135TV6007": "Lufterhitzer",

    "F2821131TV6007": "Lufterhitzer",

    "F2821125TV6007": "Lufterhitzer",

    "F2821112TV6007": "Lufterhitzer",

    "F2821112TV6017": "Lufterhitzer",

    "F2821187TV6017": "Luftkuehler",

    "F2821184TV6017": "Luftkuehler",

    "F2821183TV6017": "Luftkuehler",

    "F2821182TV6017": "Luftkuehler",

    "F2821181TV6017": "Luftkuehler",

    "F2821165TV6017": "Luftkuehler",

    "F2821161TV6017": "Luftkuehler",

    "F2821135TV6017": "Luftkuehler",

    "F2821131TV6017": "Luftkuehler",

    "F2821131TV6047": "Luftkuehler",

    "F2821131TV6057": "Luftkuehler",

    "F2821112TV6027": "Luftkuehler",

}

# ============================================================

# CONFIG

# ============================================================

CONFIG = {

    "time_col": "time",

    "temp_col": "temperature",

    "threshold": 1,

    "interval_minutes": 15

}

VALVE_IDS = list(ZUORDNUNG.keys())


# ============================================================

# HILFSFUNKTIONEN

# ============================================================


def get_valve_columns(df, mode="all", selected=None):
    valve_cols = [col for col in df.columns if "TV" in col]

    if mode == "all":

        return valve_cols

    elif mode == "select":

        if selected is None:
            raise ValueError("selected_valves darf nicht None sein")

        return [col for col in valve_cols if col in selected]

    else:

        raise ValueError("mode muss 'all' oder 'select' sein")


def filter_valves_by_anlagenart(valve_list, anlagenart_filter):
    """Filtert eine Liste von Ventilnamen nach Anlagenart(en)."""

    if not anlagenart_filter or "Alle" in anlagenart_filter:
        return valve_list

    return [v for v in valve_list if ZUORDNUNG.get(v, "Unbekannt") in anlagenart_filter]


def apply_time_filter(df, start=None, end=None, config=CONFIG):
    df = df.copy()

    if start:
        df = df[df[config["time_col"]] >= pd.to_datetime(start)]

    if end:
        df = df[df[config["time_col"]] <= pd.to_datetime(end)]

    return df


def prepare_data_for_kpi(df, interval="15min", start=None, end=None, time_col="time"):
    if start:
        df = df[df[time_col] >= pd.to_datetime(start)]

    if end:
        df = df[df[time_col] <= pd.to_datetime(end)]

    if interval != "15min":
        df = df.set_index(time_col).resample(interval).mean().reset_index()

    return df


# ============================================================

# KPI COMPUTE FUNKTIONEN

# ============================================================


def compute_valve_kpi(df, valve_col, config=CONFIG):
    threshold = config["threshold"]

    dt_hours = config["interval_minutes"] / 60

    active = df[valve_col] > threshold

    active_time = active.sum() * dt_hours

    total_time = len(df) * dt_hours

    if total_time == 0:
        return np.nan

    return active_time / total_time * 100


def compute_valve_rest_time_kpi(df, valve_col, config=CONFIG):
    dt_hours = config["interval_minutes"] / 60

    inactive = df[valve_col] == 0

    rest_time = inactive.sum() * dt_hours

    total_time = len(df) * dt_hours

    if total_time == 0:
        return np.nan

    return rest_time / total_time * 100


def compute_valve_total_travel(df, valve_col):
    delta = df[valve_col].diff().abs()

    return delta[1:].sum()


def compute_valve_direction_changes(df, valve_col):
    delta = df[valve_col].diff()

    sign = np.sign(delta)

    direction_changes = (sign[1:] * sign[:-1].values < 0).sum()

    positive_changes = (delta[1:] > 0).sum()

    negative_changes = (delta[1:] < 0).sum()

    return {

        "ventilrichtungswechsel_total": direction_changes,

        "ventilpositiv_total": positive_changes,

        "ventilnegativ_total": negative_changes

    }


def compute_valve_stability_on(df, valve_col, threshold=1):
    active_values = df.loc[df[valve_col] > threshold, valve_col]

    if len(active_values) == 0:
        return np.nan

    return active_values.std()


def compute_valve_reaction_rate(df, valve_col):
    delta = df[valve_col].diff()

    return (delta != 0).sum()


def compute_valve_temp_correlation(df, valve_col, temp_col="temperature"):
    if temp_col not in df.columns:
        return np.nan

    corr = df[[valve_col, temp_col]].corr().iloc[0, 1]

    return corr


def compute_valve_temp_correlation_on(df, valve_col, temp_col="temperature", threshold_on=1):
    if temp_col not in df.columns:
        return np.nan

    df_active = df[df[valve_col] > threshold_on][[valve_col, temp_col]].dropna()

    if len(df_active) < 2:
        return np.nan

    if df_active[valve_col].nunique() < 2 or df_active[temp_col].nunique() < 2:
        return np.nan

    return df_active[valve_col].corr(df_active[temp_col])


def compute_valve_statistics(df, valve_col):
    series = df[valve_col].dropna()

    if series.empty:
        return {k: np.nan for k in [

            "Ventil_q5", "Ventil_q25", "Ventil_q50", "Ventil_q75",

            "Ventil_q95", "Ventil_mean", "Ventil_std", "Ventil_distance", "Ventil_count"

        ]}

    return {

        "Ventil_q5": series.quantile(0.05),

        "Ventil_q25": series.quantile(0.25),

        "Ventil_q50": series.quantile(0.50),

        "Ventil_q75": series.quantile(0.75),

        "Ventil_q95": series.quantile(0.95),

        "Ventil_mean": series.mean(),

        "Ventil_std": series.std(),

        "Ventil_distance": series.max() - series.min(),

        "Ventil_count": series.count()

    }


def compute_valve_mean_on(df, valve_col):
    series = df[valve_col].dropna()

    series_on = series[series > 1]

    if series_on.empty:
        return np.nan

    return series_on.mean()


def compute_valve_error_warm(df, valve_col, temp_col="temperature", valve_threshold=50, temp_threshold=18):
    if temp_col not in df.columns:
        return np.nan

    total_count = len(df)

    if total_count == 0:
        return np.nan

    condition = (df[valve_col] > valve_threshold) & (df[temp_col] > temp_threshold)

    return (condition.sum() / total_count) * 100


def compute_valve_error_cold(df, valve_col, temp_col="temperature", valve_threshold=50, temp_threshold=10):
    if temp_col not in df.columns:
        return np.nan

    total_count = len(df)

    if total_count == 0:
        return np.nan

    condition = (df[valve_col] > valve_threshold) & (df[temp_col] < temp_threshold)

    return (condition.sum() / total_count) * 100


def compute_valve_temp_correlation_cold(df, valve_col, temp_col="temperature", threshold_temp=10):
    if temp_col not in df.columns:
        return np.nan

    df_cold = df[df[temp_col] < threshold_temp]

    if df_cold.empty or df_cold[valve_col].nunique() < 2:
        return np.nan

    return df_cold[[valve_col, temp_col]].corr().iloc[0, 1]


def compute_valve_temp_correlation_mild(df, valve_col, temp_col="temperature", threshold_low=10, threshold_high=18):
    if temp_col not in df.columns:
        return np.nan

    df_mild = df[(df[temp_col] >= threshold_low) & (df[temp_col] <= threshold_high)]

    if df_mild.empty or df_mild[valve_col].nunique() < 2:
        return np.nan

    return df_mild[[valve_col, temp_col]].corr().iloc[0, 1]


def compute_valve_temp_correlation_warm(df, valve_col, temp_col="temperature", threshold_temp=18):
    if temp_col not in df.columns:
        return np.nan

    df_warm = df[df[temp_col] > threshold_temp]

    if df_warm.empty or df_warm[valve_col].nunique() < 2:
        return np.nan

    return df_warm[[valve_col, temp_col]].corr().iloc[0, 1]


def compute_valve_mean_daytime(df, valve_col, time_col="time", start_hour=6, end_hour=23):
    df = df.copy()

    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')

    day_mask = (df[time_col].dt.hour >= start_hour) & (df[time_col].dt.hour <= end_hour)

    df_day = df[day_mask]

    if df_day.empty:
        return np.nan

    return df_day[valve_col].mean()


def compute_valve_mean_night(df, valve_col, time_col="time", night_start_hour=23, night_end_hour=5):
    df = df.copy()

    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')

    night_mask = (df[time_col].dt.hour >= night_start_hour) | (df[time_col].dt.hour <= night_end_hour)

    df_night = df[night_mask]

    if df_night.empty:
        return np.nan

    return df_night[valve_col].mean()


def compute_valve_utilization_continuous(df, valve_col, time_col="time", threshold=90, min_consecutive=92):
    df = df.copy()

    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')

    if df.empty:
        return np.nan, False

    over_thresh = df[valve_col] >= threshold

    percent = over_thresh.sum() / len(df) * 100

    max_consecutive = 0

    current = 0

    for val in over_thresh:

        if val:

            current += 1

            if current > max_consecutive:
                max_consecutive = current

        else:

            current = 0

    alert_flag = max_consecutive >= min_consecutive

    return percent, alert_flag


def compute_valve_distance_day(df, valve_col, time_col="time", date=None):
    df = df.copy()

    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')

    if date is not None:

        date = pd.to_datetime(date)

        df_day = df[(df[time_col] >= date) & (df[time_col] < date + pd.Timedelta(days=1))]

    else:

        df_day = df.copy()

    if df_day.empty:
        return {"valve_span_1day_min": np.nan, "valve_span_1day_max": np.nan, "valve_span_1day_distance": np.nan}

    return {

        "valve_span_1day_min": df_day[valve_col].min(),

        "valve_span_1day_max": df_day[valve_col].max(),

        "valve_span_1day_distance": df_day[valve_col].max() - df_day[valve_col].min()

    }


def compute_valve_distance_month(df, valve_col, time_col="time", date=None):
    df = df.copy()

    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')

    if date is not None:

        date = pd.to_datetime(date)

        start_month = date.replace(day=1)

        if start_month.month == 12:

            end_month = start_month.replace(year=start_month.year + 1, month=1)

        else:

            end_month = start_month.replace(month=start_month.month + 1)

        df_month = df[(df[time_col] >= start_month) & (df[time_col] < end_month)]

    else:

        df_month = df.copy()

    if df_month.empty:
        return {"valve_span_1month_min": np.nan, "valve_span_1month_max": np.nan, "valve_span_1month_distance": np.nan}

    return {

        "valve_span_1month_min": df_month[valve_col].min(),

        "valve_span_1month_max": df_month[valve_col].max(),

        "valve_span_1month_distance": df_month[valve_col].max() - df_month[valve_col].min()

    }


def compute_valve_stability_on_day(df, valve_col, time_col="time", date=None, threshold=1):
    df = df.copy()

    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')

    if date is not None:

        date = pd.to_datetime(date)

        df_day = df[(df[time_col] >= date) & (df[time_col] < date + pd.Timedelta(days=1))]

    else:

        df_day = df.copy()

    active_values = df_day[df_day[valve_col] > threshold][valve_col]

    if active_values.empty:
        return np.nan

    return active_values.std()


def compute_valve_reaction_rate_day(df, valve_col, time_col="time", date=None):
    df = df.copy()

    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')

    if date is not None:

        date = pd.to_datetime(date)

        df_day = df[(df[time_col] >= date) & (df[time_col] < date + pd.Timedelta(days=1))]

    else:

        df_day = df.copy()

    delta = df_day[valve_col].diff()

    return (delta != 0).sum()


# ============================================================

# ANALYSE-FUNKTIONEN (geben DataFrames zurueck)

# ============================================================


def analyze_valves(df, valves="all", selected_valves=None, start=None, end=None, config=CONFIG):
    df_filtered = apply_time_filter(df, start, end, config)

    valve_cols = get_valve_columns(df_filtered, mode=valves, selected=selected_valves)

    results = {}

    for col in valve_cols:
        results[col] = compute_valve_kpi(df_filtered, col, config)

    return pd.DataFrame.from_dict(results, orient="index", columns=["ventilbetriebszeit_%"]).sort_values(
        "ventilbetriebszeit_%", ascending=False)


def analyze_valves_rest_time(df, valves="all", selected_valves=None, start=None, end=None):
    df_filtered = apply_time_filter(df, start, end)

    valve_cols = get_valve_columns(df_filtered, mode=valves, selected=selected_valves)

    results = {}

    for col in valve_cols:
        results[col] = compute_valve_rest_time_kpi(df_filtered, col)

    return pd.DataFrame.from_dict(results, orient="index", columns=["ventilruhezeit_%"]).sort_values("ventilruhezeit_%",
                                                                                                     ascending=False)


def analyze_valves_total_travel(df, valves="all", selected_valves=None, start=None, end=None):
    df_filtered = apply_time_filter(df, start, end)

    valve_cols = get_valve_columns(df_filtered, mode=valves, selected=selected_valves)

    results = {}

    for col in valve_cols:
        results[col] = compute_valve_total_travel(df_filtered, col)

    return pd.DataFrame.from_dict(results, orient="index", columns=["ventilstrecke_total"]).sort_values(
        "ventilstrecke_total", ascending=False)


def analyze_valves_direction_changes(df, valves="all", selected_valves=None, start=None, end=None):
    df_filtered = apply_time_filter(df, start, end)

    valve_cols = get_valve_columns(df_filtered, mode=valves, selected=selected_valves)

    results = {}

    for col in valve_cols:
        results[col] = compute_valve_direction_changes(df_filtered, col)

    return pd.DataFrame.from_dict(results, orient="index").sort_values("ventilrichtungswechsel_total", ascending=False)


def analyze_valves_stability_on_auto(df, valves="all", selected_valves=None, start=None, end=None, threshold=1):
    df_filtered = apply_time_filter(df, start, end)

    valve_cols = get_valve_columns(df_filtered, mode=valves, selected=selected_valves)

    results = {}

    for col in valve_cols:
        active_values = df_filtered[df_filtered[col] > threshold][col]

        results[col] = active_values.std() if not active_values.empty else np.nan

    return pd.DataFrame.from_dict(results, orient="index", columns=["ventilstabilitaet_on"]).sort_values(
        "ventilstabilitaet_on", ascending=False)


def analyze_valves_reaction_rate(df, valves="all", selected_valves=None, start=None, end=None):
    df_filtered = apply_time_filter(df, start, end)

    valve_cols = get_valve_columns(df_filtered, mode=valves, selected=selected_valves)

    results = {}

    for col in valve_cols:
        results[col] = compute_valve_reaction_rate(df_filtered, col)

    return pd.DataFrame.from_dict(results, orient="index", columns=["Ventilreaktionen"]).sort_values("Ventilreaktionen",
                                                                                                     ascending=False)


def analyze_valves_temp_correlation(df, valves="all", selected_valves=None, start=None, end=None):
    df_filtered = apply_time_filter(df, start, end)

    valve_cols = get_valve_columns(df_filtered, mode=valves, selected=selected_valves)

    results = {}

    for col in valve_cols:
        results[col] = compute_valve_temp_correlation(df_filtered, col)

    return pd.DataFrame.from_dict(results, orient="index", columns=["Ventil_Korrelation_AUL"]).sort_values(
        "Ventil_Korrelation_AUL", ascending=False)


def analyze_valves_temp_correlation_on(df, valves="all", selected_valves=None, start=None, end=None, threshold_on=1):
    df_filtered = apply_time_filter(df, start, end)

    valve_cols = get_valve_columns(df_filtered, mode=valves, selected=selected_valves)

    results = {}

    for col in valve_cols:
        results[col] = compute_valve_temp_correlation_on(df_filtered, col, threshold_on=threshold_on)

    return pd.DataFrame.from_dict(results, orient="index", columns=["Ventil_Korrelation_AUL_on"]).sort_values(
        "Ventil_Korrelation_AUL_on", ascending=False)


def analyze_valve_statistics(df, valves="all", selected_valves=None, start=None, end=None, sort_by="Ventil_mean",
                             ascending=False):
    df_filtered = apply_time_filter(df, start, end)

    valve_cols = get_valve_columns(df_filtered, mode=valves, selected=selected_valves)

    results = {}

    for col in valve_cols:
        results[col] = compute_valve_statistics(df_filtered, col)

    result_df = pd.DataFrame.from_dict(results, orient="index")

    if sort_by in result_df.columns:
        result_df = result_df.sort_values(by=sort_by, ascending=ascending)

    return result_df


def analyze_valve_mean_on(df, valves="all", selected_valves=None, start=None, end=None):
    df_filtered = apply_time_filter(df, start, end)

    valve_cols = get_valve_columns(df_filtered, mode=valves, selected=selected_valves)

    results = {}

    for col in valve_cols:
        results[col] = compute_valve_mean_on(df_filtered, col)

    return pd.DataFrame.from_dict(results, orient="index", columns=["Ventil_mean_on"]).sort_values("Ventil_mean_on",
                                                                                                   ascending=False)


def analyze_temp_difference(df, start=None, end=None, valves="all", selected_valves=None, temp_col="temperature",
                            room_temp=21):
    df_filtered = apply_time_filter(df, start, end)

    if temp_col not in df_filtered.columns:
        return pd.DataFrame()

    delta_t = (room_temp - df_filtered[temp_col]).mean()

    valve_cols = get_valve_columns(df_filtered, mode=valves, selected=selected_valves)

    return pd.DataFrame({"Temperaturdifferenz_AUL_RAL": [delta_t] * len(valve_cols)}, index=valve_cols)


def analyze_valve_error_warm(df, valves="all", selected_valves=None, start=None, end=None, valve_threshold=50,
                             temp_threshold=18):
    df_filtered = apply_time_filter(df, start, end)

    valve_cols = get_valve_columns(df_filtered, mode=valves, selected=selected_valves)

    results = {}

    for col in valve_cols:
        results[col] = compute_valve_error_warm(df_filtered, col, valve_threshold=valve_threshold,
                                                temp_threshold=temp_threshold)

    return pd.DataFrame.from_dict(results, orient="index", columns=["Ventilfehler_AUL_Warm"]).sort_values(
        "Ventilfehler_AUL_Warm", ascending=False)


def analyze_valve_error_cold(df, valves="all", selected_valves=None, start=None, end=None, valve_threshold=50,
                             temp_threshold=10):
    df_filtered = apply_time_filter(df, start, end)

    valve_cols = get_valve_columns(df_filtered, mode=valves, selected=selected_valves)

    results = {}

    for col in valve_cols:
        results[col] = compute_valve_error_cold(df_filtered, col, valve_threshold=valve_threshold,
                                                temp_threshold=temp_threshold)

    return pd.DataFrame.from_dict(results, orient="index", columns=["Ventilfehler_AUL_Kalt"]).sort_values(
        "Ventilfehler_AUL_Kalt", ascending=False)


def analyze_valves_temp_correlation_cold(df, valves="all", selected_valves=None, start=None, end=None,
                                         threshold_temp=10):
    df_filtered = apply_time_filter(df, start, end)

    valve_cols = get_valve_columns(df_filtered, mode=valves, selected=selected_valves)

    results = {}

    for col in valve_cols:
        results[col] = compute_valve_temp_correlation_cold(df_filtered, col, threshold_temp=threshold_temp)

    return pd.DataFrame.from_dict(results, orient="index", columns=["Ventil_Korrelation_AUL_Kalt"]).sort_values(
        "Ventil_Korrelation_AUL_Kalt", ascending=False)


def analyze_valves_temp_correlation_mild(df, valves="all", selected_valves=None, start=None, end=None, threshold_low=10,
                                         threshold_high=18):
    df_filtered = apply_time_filter(df, start, end)

    valve_cols = get_valve_columns(df_filtered, mode=valves, selected=selected_valves)

    results = {}

    for col in valve_cols:
        results[col] = compute_valve_temp_correlation_mild(df_filtered, col, threshold_low=threshold_low,
                                                           threshold_high=threshold_high)

    return pd.DataFrame.from_dict(results, orient="index", columns=["Ventil_Korrelation_AUL_Mild"]).sort_values(
        "Ventil_Korrelation_AUL_Mild", ascending=False)


def analyze_valves_temp_correlation_warm(df, valves="all", selected_valves=None, start=None, end=None,
                                         threshold_temp=18):
    df_filtered = apply_time_filter(df, start, end)

    valve_cols = get_valve_columns(df_filtered, mode=valves, selected=selected_valves)

    results = {}

    for col in valve_cols:
        results[col] = compute_valve_temp_correlation_warm(df_filtered, col, threshold_temp=threshold_temp)

    return pd.DataFrame.from_dict(results, orient="index", columns=["Ventil_Korrelation_AUL_Warm"]).sort_values(
        "Ventil_Korrelation_AUL_Warm", ascending=False)


def analyze_valves_mean_daytime(df, valves="all", selected_valves=None, start=None, end=None, start_hour=6,
                                end_hour=23):
    df_filtered = apply_time_filter(df, start, end)

    if selected_valves is not None:

        valve_cols = get_valve_columns(df_filtered, mode="select", selected=selected_valves)

    else:

        valve_cols = get_valve_columns(df_filtered, mode="all")

    results = {}

    for col in valve_cols:
        results[col] = compute_valve_mean_daytime(df_filtered, col, start_hour=start_hour, end_hour=end_hour)

    return pd.DataFrame.from_dict(results, orient="index", columns=["Ventil_mean_Tag"]).sort_values("Ventil_mean_Tag",
                                                                                                    ascending=False)


def analyze_valves_mean_night(df, valves="all", selected_valves=None, start=None, end=None, night_start_hour=23,
                              night_end_hour=5):
    df_filtered = apply_time_filter(df, start, end)

    if selected_valves is not None:

        valve_cols = get_valve_columns(df_filtered, mode="select", selected=selected_valves)

    else:

        valve_cols = get_valve_columns(df_filtered, mode="all")

    results = {}

    for col in valve_cols:
        results[col] = compute_valve_mean_night(df_filtered, col, night_start_hour=night_start_hour,
                                                night_end_hour=night_end_hour)

    return pd.DataFrame.from_dict(results, orient="index", columns=["Ventil_mean_Nacht"]).sort_values(
        "Ventil_mean_Nacht", ascending=False)


def analyze_valves_utilization_continuous(df, start=None, end=None, valves="all", selected_valves=None, threshold=90,
                                          min_consecutive=50):
    df_filtered = apply_time_filter(df, start, end)

    valve_cols = get_valve_columns(df_filtered, mode=valves, selected=selected_valves)

    results = {}

    for col in valve_cols:
        percent, alert_flag = compute_valve_utilization_continuous(df_filtered, col, threshold=threshold,
                                                                   min_consecutive=min_consecutive)

        results[col] = percent

    return pd.DataFrame.from_dict(results, orient="index", columns=["Ventil_Auslastungsgrenze"]).sort_values(
        "Ventil_Auslastungsgrenze", ascending=False)


def analyze_valve_distance_day(df, valves="all", selected_valves=None, date=None):
    valve_cols = get_valve_columns(df, mode=valves, selected=selected_valves)

    results = {}

    for col in valve_cols:
        results[col] = compute_valve_distance_day(df, col, date=date)

    return pd.DataFrame.from_dict(results, orient="index").sort_values("valve_span_1day_distance", ascending=False)


def analyze_valve_distance_month(df, valves="all", selected_valves=None, date=None):
    valve_cols = get_valve_columns(df, mode=valves, selected=selected_valves)

    results = {}

    for col in valve_cols:
        results[col] = compute_valve_distance_month(df, col, date=date)

    return pd.DataFrame.from_dict(results, orient="index").sort_values("valve_span_1month_distance", ascending=False)


def analyze_valves_stability_on_day(df, valves="all", selected_valves=None, date=None, threshold=1):
    valve_cols = get_valve_columns(df, mode=valves, selected=selected_valves)

    results = {}

    for col in valve_cols:
        results[col] = compute_valve_stability_on_day(df, col, date=date, threshold=threshold)

    return pd.DataFrame.from_dict(results, orient="index", columns=["ventilstabilitaet_on_1Tag"]).sort_values(
        "ventilstabilitaet_on_1Tag", ascending=False)


def analyze_valves_reaction_rate_day(df, valves="all", selected_valves=None, date=None):
    valve_cols = get_valve_columns(df, mode=valves, selected=selected_valves)

    results = {}

    for col in valve_cols:
        results[col] = compute_valve_reaction_rate_day(df, col, date=date)

    return pd.DataFrame.from_dict(results, orient="index", columns=["ventilreaktionen_1Tag"]).sort_values(
        "ventilreaktionen_1Tag", ascending=False)


# ============================================================

# RUN ALL KPIs

# ============================================================


def run_all_kpis(df, start=None, end=None, day=None, valves="all", selected_valves=None,

                 threshold=90, min_consecutive=50, threshold_on=1):
    dfs = []

    if start and end:
        dfs.append(analyze_valves(df, start=start, end=end, valves=valves, selected_valves=selected_valves))

        dfs.append(analyze_valves_rest_time(df, start=start, end=end, valves=valves, selected_valves=selected_valves))

        dfs.append(
            analyze_valves_total_travel(df, start=start, end=end, valves=valves, selected_valves=selected_valves))

        dfs.append(
            analyze_valves_direction_changes(df, start=start, end=end, valves=valves, selected_valves=selected_valves))

        dfs.append(
            analyze_valves_stability_on_auto(df, start=start, end=end, valves=valves, selected_valves=selected_valves))

        dfs.append(
            analyze_valves_reaction_rate(df, start=start, end=end, valves=valves, selected_valves=selected_valves))

        dfs.append(
            analyze_valves_temp_correlation(df, start=start, end=end, valves=valves, selected_valves=selected_valves))

        dfs.append(
            analyze_valves_temp_correlation_on(df, start=start, end=end, valves=valves, selected_valves=selected_valves,
                                               threshold_on=threshold_on))

        dfs.append(analyze_valve_mean_on(df, start=start, end=end, valves=valves, selected_valves=selected_valves))

        dfs.append(analyze_valve_statistics(df, start=start, end=end, valves=valves, selected_valves=selected_valves))

        dfs.append(analyze_temp_difference(df, start=start, end=end, valves=valves, selected_valves=selected_valves))

        dfs.append(analyze_valve_error_warm(df, start=start, end=end, valves=valves, selected_valves=selected_valves))

        dfs.append(analyze_valve_error_cold(df, start=start, end=end, valves=valves, selected_valves=selected_valves))

        dfs.append(analyze_valves_temp_correlation_cold(df, start=start, end=end, valves=valves,
                                                        selected_valves=selected_valves))

        dfs.append(analyze_valves_temp_correlation_mild(df, start=start, end=end, valves=valves,
                                                        selected_valves=selected_valves))

        dfs.append(analyze_valves_temp_correlation_warm(df, start=start, end=end, valves=valves,
                                                        selected_valves=selected_valves))

        dfs.append(analyze_valves_utilization_continuous(

            df, start=start, end=end, threshold=threshold, valves=valves,

            selected_valves=selected_valves, min_consecutive=min_consecutive

        ))

        dfs.append(
            analyze_valves_mean_daytime(df, start=start, end=end, valves=valves, selected_valves=selected_valves))

        dfs.append(analyze_valves_mean_night(df, start=start, end=end, valves=valves, selected_valves=selected_valves))

    if day:
        dfs.append(analyze_valve_distance_day(df, date=day, valves=valves, selected_valves=selected_valves))

        dfs.append(analyze_valve_distance_month(df, date=day, valves=valves, selected_valves=selected_valves))

        dfs.append(analyze_valves_stability_on_day(df, date=day, valves=valves, selected_valves=selected_valves))

        dfs.append(analyze_valves_reaction_rate_day(df, date=day, valves=valves, selected_valves=selected_valves))

    final_df = None

    for d in dfs:

        if d is not None and not d.empty:

            if final_df is None:

                final_df = d.copy()

            else:

                final_df = final_df.join(d, how="outer")

    if final_df is not None:
        final_df = final_df.drop(index="temperature", errors="ignore")

        # Anlagenart-Spalte hinzufuegen

        final_df.insert(0, "Anlagenart", final_df.index.map(lambda v: ZUORDNUNG.get(v, "Unbekannt")))

    return final_df


# ============================================================

# PLOT-FUNKTIONEN FUER DASHBOARD

# ============================================================


def plot_valve_onoff(df, valve_cols, start=None, end=None, config=CONFIG):
    df_filtered = apply_time_filter(df, start, end, config)

    df_filtered = df_filtered.copy()

    df_filtered['time'] = pd.to_datetime(df_filtered['time'], errors='coerce')

    fig, ax = plt.subplots(figsize=(14, 4))

    for col in valve_cols:
        threshold = config["threshold"]

        status = (df_filtered[col] > threshold).astype(int)

        active_pct = status.mean() * 100

        label_text = f"{col} [{ZUORDNUNG.get(col, '?')}] ({active_pct:.1f}%)"

        ax.step(df_filtered['time'], status, where='mid', label=label_text, linewidth=2)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))

    plt.xticks(rotation=45)

    ax.set_ylim(-0.1, 1.1)

    ax.set_yticks([0, 1])

    ax.set_yticklabels(["Aus", "Ein"])

    ax.set_xlabel("Zeit")

    ax.set_ylabel("Status")

    ax.set_title("Ventil Ein/Aus Verlauf")

    ax.grid(axis='y', linestyle='--', alpha=0.5)

    ax.legend(title="Ventil [Typ] (Betriebszeit %)", fontsize=7)

    plt.tight_layout()

    return fig


def plot_total_travel(result_df):
    fig, ax = plt.subplots(figsize=(12, 5))

    colors = [_anlagenart_color(idx) for idx in result_df.index]

    ax.bar(result_df.index, result_df['ventilstrecke_total'], color=colors, alpha=0.8)

    ax.set_ylabel("Summe Ventilstrecke")

    ax.set_title("Ventilstrecke Total pro Ventil")

    plt.xticks(rotation=45, ha='right')

    ax.grid(axis='y', linestyle='--', alpha=0.5)

    for i, v in enumerate(result_df['ventilstrecke_total']):
        ax.text(i, v + max(result_df['ventilstrecke_total']) * 0.01, f"{v:.0f}", ha='center', va='bottom', fontsize=7)

    _add_anlagenart_legend(ax)

    plt.tight_layout()

    return fig


def plot_direction_changes(result_df):
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.bar(result_df.index, result_df['ventilpositiv_total'], color='green', label='Positiv (Auf)')

    ax.bar(result_df.index, result_df['ventilnegativ_total'],

           bottom=result_df['ventilpositiv_total'], color='red', label='Negativ (Zu)')

    for i, (pos, neg) in enumerate(zip(result_df['ventilpositiv_total'], result_df['ventilnegativ_total'])):
        total = pos + neg

        ax.text(i, total + max(result_df['ventilrichtungswechsel_total']) * 0.01, f"{total:.0f}",

                ha='center', va='bottom', fontsize=7)

    ax.set_ylabel("Anzahl Richtungswechsel")

    ax.set_title("Ventilrichtungswechsel pro Ventil")

    plt.xticks(rotation=45, ha='right')

    ax.grid(axis='y', linestyle='--', alpha=0.5)

    ax.legend()

    plt.tight_layout()

    return fig


def plot_direction_trend(df, valve_cols, start=None, end=None):
    df_filtered = apply_time_filter(df, start, end)

    df_filtered = df_filtered.copy()

    df_filtered['time'] = pd.to_datetime(df_filtered['time'], errors='coerce')

    fig, ax = plt.subplots(figsize=(14, 5))

    for col in valve_cols:

        delta = df_filtered[col].diff()

        prev = df_filtered[col].shift(1)

        times = df_filtered['time'].values

        vals = df_filtered[col].values

        for i in range(1, len(df_filtered)):

            if pd.isna(delta.iloc[i]):
                continue

            color = 'green' if delta.iloc[i] > 0 else 'red'

            ax.plot([times[i - 1], times[i]], [prev.iloc[i], vals[i]], color=color, linewidth=1.5)

    ax.set_xlabel("Zeit")

    ax.set_ylabel("Stellwert [%]")

    ax.set_title("Ventilbewegungen ueber Zeit (gruen = Auf, rot = Zu)")

    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()

    return fig


def plot_stability_band(df, valve_cols, start=None, end=None, threshold=1):
    df_filtered = apply_time_filter(df, start, end)

    df_filtered = df_filtered.copy()

    df_filtered['time'] = pd.to_datetime(df_filtered['time'], errors='coerce')

    fig, ax = plt.subplots(figsize=(14, 5))

    time = df_filtered['time']

    for col in valve_cols:

        series = df_filtered[col]

        ax.plot(time, series, label=f"{col}", linewidth=2)

        active_mask = series > threshold

        if active_mask.any():
            std_val = series[active_mask].std()

            upper = series.copy()

            lower = series.copy()

            upper[~active_mask] = np.nan

            lower[~active_mask] = np.nan

            ax.fill_between(time, lower - std_val, upper + std_val, color='gray', alpha=0.2)

    ax.set_xlabel("Zeit")

    ax.set_ylabel("Stellwert [%]")

    ax.set_title(f"Ventilstabilitaet mit +/-Std-Band ({start} bis {end})")

    ax.legend(fontsize=8)

    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()

    return fig


def plot_reaction_rate(result_df, start=None, end=None):
    fig, ax = plt.subplots(figsize=(12, 5))

    colors = [_anlagenart_color(idx) for idx in result_df.index]

    ax.bar(result_df.index, result_df["Ventilreaktionen"], color=colors)

    ax.set_xlabel("Ventil")

    ax.set_ylabel("Anzahl der Aenderungen")

    ax.set_title(f"Ventilreaktionsrate ({start} bis {end})")

    plt.xticks(rotation=45, ha='right')

    ax.grid(axis='y', linestyle='--', alpha=0.5)

    _add_anlagenart_legend(ax)

    plt.tight_layout()

    return fig


def plot_correlation_bar(result_df, col_name, title):
    fig, ax = plt.subplots(figsize=(12, 5))

    colors = ['green' if v > 0 else 'red' for v in result_df[col_name]]

    ax.bar(result_df.index, result_df[col_name], color=colors, alpha=0.8)

    ax.set_xlabel("Ventil")

    ax.set_ylabel("Korrelationskoeffizient r")

    ax.set_title(title)

    plt.xticks(rotation=45, ha='right')

    ax.set_ylim(-1, 1)

    ax.axhline(y=0, color='black', linewidth=0.5)

    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()

    return fig


def plot_error_bar(result_df, col_name, title):
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.bar(result_df.index, result_df[col_name], color='coral', alpha=0.8)

    ax.set_xlabel("Ventil")

    ax.set_ylabel("Anteil [%]")

    ax.set_title(title)

    plt.xticks(rotation=45, ha='right')

    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()

    return fig


def plot_raw_timeseries(df, valve_cols, start=None, end=None):
    df_filtered = apply_time_filter(df, start, end)

    df_filtered = df_filtered.copy()

    df_filtered['time'] = pd.to_datetime(df_filtered['time'], errors='coerce')

    fig, ax = plt.subplots(figsize=(14, 5))

    for col in valve_cols:
        ax.plot(df_filtered['time'], df_filtered[col], label=col, linewidth=1.5)

    ax.set_xlabel("Zeit")

    ax.set_ylabel("Stellwert [%]")

    ax.set_title("Ventilstellwerte ueber Zeit")

    ax.legend(fontsize=8)

    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()

    return fig


# ============================================================

# FARB-HILFSFUNKTIONEN FUER ANLAGENART

# ============================================================


ANLAGENART_COLORS = {

    "Lufterhitzer": "#e07b39",

    "Luftkuehler": "#3998e0",

    "WRG": "#5cb85c",

    "Unbekannt": "#999999"

}


def _anlagenart_color(valve_name):
    art = ZUORDNUNG.get(valve_name, "Unbekannt")

    return ANLAGENART_COLORS.get(art, "#999999")


def _add_anlagenart_legend(ax):
    from matplotlib.patches import Patch

    handles = [Patch(facecolor=c, label=a) for a, c in ANLAGENART_COLORS.items() if a != "Unbekannt"]

    ax.legend(handles=handles, title="Anlagenart", fontsize=8, loc="upper right")


# ============================================================

# STREAMLIT DASHBOARD

# ============================================================


st.title("Ventilanalyse Dashboard")

st.markdown("---")

# --- SIDEBAR: DATEN LADEN ---

st.sidebar.header("Daten laden")

uploaded_file = st.sidebar.file_uploader(

    "CSV-Datei hochladen (mit 'time' Spalte + Ventilspalten)",

    type=["csv"]

)

if uploaded_file is not None:

    pdf = pd.read_csv(uploaded_file)

    pdf['time'] = pd.to_datetime(pdf['time'], errors='coerce')

    # temperature-Spalte umbenennen falls vorhanden

    if "F1311211TE6161" in pdf.columns:
        pdf = pdf.rename(columns={"F1311211TE6161": "temperature"})

    # NA auffuellen

    pdf.fillna(method='ffill', inplace=True)

    pdf.fillna(method='bfill', inplace=True)

    st.sidebar.success(f"Daten geladen: {len(pdf)} Zeilen, {len(pdf.columns)} Spalten")

    # --- Verfuegbare Ventile ermitteln ---

    all_valves = [col for col in pdf.columns if "TV" in col]

    # --- SIDEBAR: ANLAGENART-FILTER ---

    st.sidebar.header("Anlagenart")

    available_arts = sorted(set(ZUORDNUNG.get(v, "Unbekannt") for v in all_valves))

    if not available_arts:
        available_arts = ["Alle"]

    anlagenart_filter = st.sidebar.multiselect(

        "Anlagenart filtern",

        options=["Alle"] + available_arts,

        default=["Alle"]

    )

    # Ventile nach Anlagenart filtern

    if "Alle" in anlagenart_filter or not anlagenart_filter:

        valves_by_art = all_valves

    else:

        valves_by_art = filter_valves_by_anlagenart(all_valves, anlagenart_filter)

    if not valves_by_art:
        st.sidebar.warning("Keine Ventile fuer die gewaehlte Anlagenart in den Daten vorhanden.")

        st.stop()

    st.sidebar.info(f"{len(valves_by_art)} Ventile nach Anlagenart-Filter verfuegbar")

    # --- SIDEBAR: FILTER ---

    st.sidebar.header("Filter")

    # Zeitraum

    min_date = pdf['time'].min().date()

    max_date = pdf['time'].max().date()

    col1, col2 = st.sidebar.columns(2)

    with col1:

        start_date = st.date_input("Start", value=min_date, min_value=min_date, max_value=max_date)

    with col2:

        end_date = st.date_input("Ende", value=max_date, min_value=min_date, max_value=max_date)

    start_str = str(start_date)

    end_str = str(end_date)

    # Tag fuer Tages-KPIs

    day_date = st.sidebar.date_input("Tag (fuer Tages-KPIs)", value=start_date, min_value=min_date, max_value=max_date)

    day_str = str(day_date)

    # Ventilauswahl

    st.sidebar.header("Ventilauswahl")

    valve_mode = st.sidebar.radio("Modus", ["all", "select"], index=0)

    selected_valves = None

    if valve_mode == "select":

        selected_valves = st.sidebar.multiselect(

            "Ventile auswaehlen",

            options=valves_by_art,

            default=valves_by_art[:2] if len(valves_by_art) >= 2 else valves_by_art,

            format_func=lambda v: f"{v} [{ZUORDNUNG.get(v, '?')}]"

        )

        if not selected_valves:
            st.sidebar.warning("Bitte mindestens ein Ventil auswaehlen!")

            st.stop()

    else:

        # Im "all"-Modus: alle nach Anlagenart gefilterten Ventile verwenden

        selected_valves = valves_by_art

    # Erweiterte Parameter

    st.sidebar.header("Erweiterte Parameter")

    threshold_utilization = st.sidebar.slider("Auslastungs-Schwelle [%]", 50, 100, 90)

    min_consecutive = st.sidebar.slider("Min. konsekutive Intervalle", 1, 200, 50)

    threshold_on = st.sidebar.slider("Betriebsschwelle [%]", 0, 10, 1)

    # Zeitaufloesung

    interval = st.sidebar.selectbox("Zeitaufloesung", ["15min", "30min", "1h", "2h", "4h", "1D"], index=0)

    # --- Daten vorbereiten ---

    if interval != "15min":

        pdf_prepared = prepare_data_for_kpi(pdf, interval=interval, start=start_str, end=end_str)

    else:

        pdf_prepared = pdf.copy()

    # Aktive Ventile fuer Plots

    active_valve_cols = selected_valves if selected_valves else valves_by_art

    # ============================================================

    # ANALYSE STARTEN

    # ============================================================

    st.sidebar.markdown("---")

    run_button = st.sidebar.button("Analyse starten", type="primary", use_container_width=True)

    if run_button:

        with st.spinner("Berechne alle KPIs..."):

            # --- Gesamttabelle ---

            df_kpis = run_all_kpis(

                pdf_prepared,

                start=start_str,

                end=end_str,

                day=day_str,

                valves="select",

                selected_valves=active_valve_cols,

                threshold=threshold_utilization,

                min_consecutive=min_consecutive,

                threshold_on=threshold_on

            )

        if df_kpis is not None and not df_kpis.empty:

            # === TAB LAYOUT ===

            tab_overview, tab_time, tab_travel, tab_direction, tab_stability, tab_correlation, tab_errors, tab_stats, tab_day, tab_doku = st.tabs(
                [

                    "Uebersicht",

                    "Betriebszeit",

                    "Ventilstrecke",

                    "Richtungswechsel",

                    "Stabilitaet",

                    "Korrelation",

                    "Fehleranalyse",

                    "Statistik",

                    "Tages-KPIs",

                    "KPI Dokumentation"

                ])

            # ==========================================

            # TAB: UEBERSICHT

            # ==========================================

            with tab_overview:

                st.header("Gesamtuebersicht aller KPIs")

                art_text = ", ".join(anlagenart_filter) if anlagenart_filter else "Alle"

                st.markdown(

                    f"**Zeitraum:** {start_str} bis {end_str} | **Tag:** {day_str} | "

                    f"**Anlagenart:** {art_text} | **Ventile:** {len(active_valve_cols)}"

                )

                # Kennzahlen-Karten

                col1, col2, col3, col4 = st.columns(4)

                with col1:

                    avg_betrieb = df_kpis[
                        "ventilbetriebszeit_%"].mean() if "ventilbetriebszeit_%" in df_kpis.columns else 0

                    st.metric("Durchschn. Betriebszeit", f"{avg_betrieb:.1f}%")

                with col2:

                    avg_travel = df_kpis[
                        "ventilstrecke_total"].mean() if "ventilstrecke_total" in df_kpis.columns else 0

                    st.metric("Durchschn. Ventilstrecke", f"{avg_travel:.0f}")

                with col3:

                    avg_reactions = df_kpis["Ventilreaktionen"].mean() if "Ventilreaktionen" in df_kpis.columns else 0

                    st.metric("Durchschn. Reaktionen", f"{avg_reactions:.0f}")

                with col4:

                    if "Temperaturdifferenz_AUL_RAL" in df_kpis.columns:
                        delta_t = df_kpis["Temperaturdifferenz_AUL_RAL"].iloc[0]

                        st.metric("Delta-T (Raum-Aussen)", f"{delta_t:.1f} C")

                st.markdown("---")

                # Anlagenart-Zusammenfassung

                st.subheader("Zusammenfassung nach Anlagenart")

                if "Anlagenart" in df_kpis.columns:

                    numeric_cols = df_kpis.select_dtypes(include=[np.number]).columns.tolist()

                    if numeric_cols:
                        summary = df_kpis.groupby("Anlagenart")[numeric_cols].mean()

                        st.dataframe(summary.style.format("{:.2f}", na_rep="--"), use_container_width=True)

                st.markdown("---")

                st.subheader("Komplette KPI-Tabelle")

                st.dataframe(df_kpis.style.format(

                    {c: "{:.2f}" for c in df_kpis.select_dtypes(include=[np.number]).columns},

                    na_rep="--"

                ), use_container_width=True, height=500)

                # Download

                csv = df_kpis.to_csv()

                st.download_button(

                    label="KPI-Tabelle als CSV herunterladen",

                    data=csv,

                    file_name=f"ventil_kpis_{start_str}_{end_str}.csv",

                    mime="text/csv"

                )

            # ==========================================

            # TAB: BETRIEBSZEIT

            # ==========================================

            with tab_time:

                st.header("Ventilbetriebszeit und Ruhezeit")

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Betriebszeit [%]")

                    df_betrieb = analyze_valves(pdf_prepared, valves="select", selected_valves=active_valve_cols,
                                                start=start_str, end=end_str)

                    df_betrieb.insert(0, "Anlagenart", df_betrieb.index.map(lambda v: ZUORDNUNG.get(v, "Unbekannt")))

                    st.dataframe(df_betrieb.style.format({"ventilbetriebszeit_%": "{:.2f}%"}, na_rep="--"),
                                 use_container_width=True)

                with col2:
                    st.subheader("Ruhezeit [%]")

                    df_ruhe = analyze_valves_rest_time(pdf_prepared, valves="select", selected_valves=active_valve_cols,
                                                       start=start_str, end=end_str)

                    df_ruhe.insert(0, "Anlagenart", df_ruhe.index.map(lambda v: ZUORDNUNG.get(v, "Unbekannt")))

                    st.dataframe(df_ruhe.style.format({"ventilruhezeit_%": "{:.2f}%"}, na_rep="--"),
                                 use_container_width=True)

                st.markdown("---")

                st.subheader("Ein/Aus Verlauf")

                if len(active_valve_cols) > 10:
                    st.warning(
                        "Viele Ventile ausgewaehlt - Plot kann unuebersichtlich werden. Empfehlung: max. 10 Ventile.")

                fig_onoff = plot_valve_onoff(pdf_prepared, active_valve_cols, start=start_str, end=end_str)

                st.pyplot(fig_onoff)

                plt.close(fig_onoff)

                st.markdown("---")

                st.subheader("Rohe Stellwerte ueber Zeit")

                fig_raw = plot_raw_timeseries(pdf_prepared, active_valve_cols, start=start_str, end=end_str)

                st.pyplot(fig_raw)

                plt.close(fig_raw)

                st.markdown("---")

                st.subheader("Mittelwert Tag vs. Nacht")

                col_day, col_night = st.columns(2)

                with col_day:
                    st.markdown("**Tag (06:00-23:00)**")

                    df_mean_day = analyze_valves_mean_daytime(pdf_prepared, valves="select",
                                                              selected_valves=active_valve_cols, start=start_str,
                                                              end=end_str)

                    st.dataframe(df_mean_day.style.format("{:.2f}"), use_container_width=True)

                with col_night:
                    st.markdown("**Nacht (23:00-06:00)**")

                    df_mean_night = analyze_valves_mean_night(pdf_prepared, valves="select",
                                                              selected_valves=active_valve_cols, start=start_str,
                                                              end=end_str)

                    st.dataframe(df_mean_night.style.format("{:.2f}"), use_container_width=True)

                st.markdown("---")

                st.subheader("Auslastungsgrenze (>= Schwelle)")

                st.caption(f"Schwelle: {threshold_utilization}% | Min. konsekutive Intervalle: {min_consecutive}")

                df_util = analyze_valves_utilization_continuous(

                    pdf_prepared, start=start_str, end=end_str,

                    valves="select", selected_valves=active_valve_cols,

                    threshold=threshold_utilization, min_consecutive=min_consecutive

                )

                st.dataframe(df_util.style.format("{:.2f}%"), use_container_width=True)

            # ==========================================

            # TAB: VENTILSTRECKE

            # ==========================================

            with tab_travel:

                st.header("Ventilstrecke (Summe aller Stellwertaenderungen)")

                df_travel = analyze_valves_total_travel(pdf_prepared, valves="select",
                                                        selected_valves=active_valve_cols, start=start_str, end=end_str)

                col1, col2 = st.columns([1, 2])

                with col1:
                    st.subheader("Tabelle")

                    df_travel_display = df_travel.copy()

                    df_travel_display.insert(0, "Anlagenart",
                                             df_travel_display.index.map(lambda v: ZUORDNUNG.get(v, "Unbekannt")))

                    st.dataframe(df_travel_display.style.format({"ventilstrecke_total": "{:.1f}"}, na_rep="--"),
                                 use_container_width=True)

                with col2:
                    st.subheader("Diagramm")

                    fig_travel = plot_total_travel(df_travel)

                    st.pyplot(fig_travel)

                    plt.close(fig_travel)

            # ==========================================

            # TAB: RICHTUNGSWECHSEL

            # ==========================================

            with tab_direction:

                st.header("Ventilrichtungswechsel")

                df_dir = analyze_valves_direction_changes(pdf_prepared, valves="select",
                                                          selected_valves=active_valve_cols, start=start_str,
                                                          end=end_str)

                col1, col2 = st.columns([1, 2])

                with col1:

                    st.subheader("Tabelle")

                    st.dataframe(df_dir.style.format("{:.0f}"), use_container_width=True)

                with col2:

                    st.subheader("Gestapeltes Balkendiagramm")

                    fig_dir = plot_direction_changes(df_dir)

                    st.pyplot(fig_dir)

                    plt.close(fig_dir)

                st.markdown("---")

                st.subheader("Trend-Plot (gruen = Auf, rot = Zu)")

                if len(active_valve_cols) > 5:
                    st.warning(
                        "Bei vielen Ventilen kann der Trend-Plot langsam werden. Empfehlung: max. 5 Ventile im 'select' Modus.")

                if len(active_valve_cols) <= 5:

                    fig_trend = plot_direction_trend(pdf_prepared, active_valve_cols, start=start_str, end=end_str)

                    st.pyplot(fig_trend)

                    plt.close(fig_trend)

                else:

                    st.info("Trend-Plot wird nur bei <= 5 Ventilen angezeigt. Bitte 'select' Modus verwenden.")

            # ==========================================

            # TAB: STABILITAET

            # ==========================================

            with tab_stability:

                st.header("Ventilstabilitaet (Standardabweichung waehrend Betrieb)")

                df_stab = analyze_valves_stability_on_auto(pdf_prepared, valves="select",
                                                           selected_valves=active_valve_cols, start=start_str,
                                                           end=end_str)

                col1, col2 = st.columns([1, 2])

                with col1:

                    st.subheader("Tabelle")

                    st.dataframe(df_stab.style.format("{:.3f}"), use_container_width=True)

                with col2:

                    st.subheader("Zeitreihe mit +/-Std-Band")

                    if len(active_valve_cols) <= 8:

                        fig_stab = plot_stability_band(pdf_prepared, active_valve_cols, start=start_str, end=end_str)

                        st.pyplot(fig_stab)

                        plt.close(fig_stab)

                    else:

                        st.info("Stabilitaets-Band wird nur bei <= 8 Ventilen angezeigt.")

                st.markdown("---")

                st.subheader("Ventilreaktionsrate (Aenderungen pro Intervall)")

                df_react = analyze_valves_reaction_rate(pdf_prepared, valves="select",
                                                        selected_valves=active_valve_cols, start=start_str, end=end_str)

                col1, col2 = st.columns([1, 2])

                with col1:

                    st.dataframe(df_react.style.format("{:.0f}"), use_container_width=True)

                with col2:

                    fig_react = plot_reaction_rate(df_react, start=start_str, end=end_str)

                    st.pyplot(fig_react)

                    plt.close(fig_react)

                st.markdown("---")

                st.subheader("Mittelwert nur waehrend Betrieb (Ventil > 1%)")

                df_mean_on = analyze_valve_mean_on(pdf_prepared, valves="select", selected_valves=active_valve_cols,
                                                   start=start_str, end=end_str)

                st.dataframe(df_mean_on.style.format("{:.2f}"), use_container_width=True)

            # ==========================================

            # TAB: KORRELATION

            # ==========================================

            with tab_correlation:

                st.header("Korrelation Ventil <-> Aussentemperatur")

                # Gesamt-Korrelation

                st.subheader("Gesamt-Korrelation")

                df_corr = analyze_valves_temp_correlation(pdf_prepared, valves="select",
                                                          selected_valves=active_valve_cols, start=start_str,
                                                          end=end_str)

                col1, col2 = st.columns([1, 2])

                with col1:

                    st.dataframe(df_corr.style.format("{:.4f}"), use_container_width=True)

                with col2:

                    if not df_corr.empty:
                        fig_corr = plot_correlation_bar(df_corr, "Ventil_Korrelation_AUL",
                                                        f"Ventil-Korrelation mit Aussentemperatur ({start_str} bis {end_str})")

                        st.pyplot(fig_corr)

                        plt.close(fig_corr)

                st.markdown("---")

                # Korrelation nur bei Betrieb

                st.subheader("Korrelation nur waehrend Betrieb")

                df_corr_on = analyze_valves_temp_correlation_on(pdf_prepared, valves="select",
                                                                selected_valves=active_valve_cols, start=start_str,
                                                                end=end_str, threshold_on=threshold_on)

                col1, col2 = st.columns([1, 2])

                with col1:

                    st.dataframe(df_corr_on.style.format("{:.4f}"), use_container_width=True)

                with col2:

                    if not df_corr_on.empty:
                        fig_corr_on = plot_correlation_bar(df_corr_on, "Ventil_Korrelation_AUL_on",
                                                           "Korrelation nur waehrend Betrieb")

                        st.pyplot(fig_corr_on)

                        plt.close(fig_corr_on)

                st.markdown("---")

                # Korrelation nach Temperaturbereich

                st.subheader("Korrelation nach Temperaturbereich")

                col_cold, col_mild, col_warm = st.columns(3)

                with col_cold:

                    st.markdown("**Kalt (< 10 C)**")

                    df_corr_cold = analyze_valves_temp_correlation_cold(pdf_prepared, valves="select",
                                                                        selected_valves=active_valve_cols,
                                                                        start=start_str, end=end_str)

                    st.dataframe(df_corr_cold.style.format("{:.4f}"), use_container_width=True)

                with col_mild:

                    st.markdown("**Mild (10-18 C)**")

                    df_corr_mild = analyze_valves_temp_correlation_mild(pdf_prepared, valves="select",
                                                                        selected_valves=active_valve_cols,
                                                                        start=start_str, end=end_str)

                    st.dataframe(df_corr_mild.style.format("{:.4f}"), use_container_width=True)

                with col_warm:

                    st.markdown("**Warm (> 18 C)**")

                    df_corr_warm = analyze_valves_temp_correlation_warm(pdf_prepared, valves="select",
                                                                        selected_valves=active_valve_cols,
                                                                        start=start_str, end=end_str)

                    st.dataframe(df_corr_warm.style.format("{:.4f}"), use_container_width=True)

            # ==========================================

            # TAB: FEHLERANALYSE

            # ==========================================

            with tab_errors:

                st.header("Fehleranalyse")

                st.subheader("Temperaturdifferenz (Raum 21 C - Aussen)")

                df_temp_diff = analyze_temp_difference(pdf_prepared, start=start_str, end=end_str, valves="select",
                                                       selected_valves=active_valve_cols)

                if not df_temp_diff.empty:
                    delta_val = df_temp_diff["Temperaturdifferenz_AUL_RAL"].iloc[0]

                    st.metric("Durchschn. Delta-T (Raum - Aussen)", f"{delta_val:.2f} C")

                st.markdown("---")

                col1, col2 = st.columns(2)

                with col1:

                    st.subheader("Ventilfehler bei Waerme")

                    st.caption("Ventil > 50% UND Aussentemperatur > 18 C")

                    df_err_warm = analyze_valve_error_warm(pdf_prepared, valves="select",
                                                           selected_valves=active_valve_cols, start=start_str,
                                                           end=end_str)

                    st.dataframe(df_err_warm.style.format("{:.2f}%"), use_container_width=True)

                    if not df_err_warm.empty:
                        fig_err_warm = plot_error_bar(df_err_warm, "Ventilfehler_AUL_Warm",
                                                      f"Ventilfehler bei Waerme ({start_str} bis {end_str})")

                        st.pyplot(fig_err_warm)

                        plt.close(fig_err_warm)

                with col2:

                    st.subheader("Ventilfehler bei Kaelte")

                    st.caption("Ventil > 50% UND Aussentemperatur < 10 C")

                    df_err_cold = analyze_valve_error_cold(pdf_prepared, valves="select",
                                                           selected_valves=active_valve_cols, start=start_str,
                                                           end=end_str)

                    st.dataframe(df_err_cold.style.format("{:.2f}%"), use_container_width=True)

                    if not df_err_cold.empty:
                        fig_err_cold = plot_error_bar(df_err_cold, "Ventilfehler_AUL_Kalt",
                                                      f"Ventilfehler bei Kaelte ({start_str} bis {end_str})")

                        st.pyplot(fig_err_cold)

                        plt.close(fig_err_cold)

            # ==========================================

            # TAB: STATISTIK

            # ==========================================

            with tab_stats:

                st.header("Statistische Kennzahlen")

                df_stats = analyze_valve_statistics(pdf_prepared, valves="select", selected_valves=active_valve_cols,
                                                    start=start_str, end=end_str)

                st.subheader("Quantile, Mittelwert, Std, Spannweite")

                df_stats_display = df_stats.copy()

                df_stats_display.insert(0, "Anlagenart",
                                        df_stats_display.index.map(lambda v: ZUORDNUNG.get(v, "Unbekannt")))

                st.dataframe(df_stats_display.style.format(

                    {c: "{:.2f}" for c in df_stats_display.select_dtypes(include=[np.number]).columns},

                    na_rep="--"

                ), use_container_width=True, height=400)

                st.markdown("---")

                # Boxplot

                st.subheader("Boxplot der Stellwerte")

                df_filtered_plot = apply_time_filter(pdf_prepared, start_str, end_str)

                fig_box, ax_box = plt.subplots(figsize=(14, 6))

                box_data = []

                box_labels = []

                box_colors = []

                for col in active_valve_cols:

                    if col in df_filtered_plot.columns:

                        data = df_filtered_plot[col].dropna().values

                        if len(data) > 0:
                            box_data.append(data)

                            box_labels.append(col)

                            box_colors.append(_anlagenart_color(col))

                if box_data:

                    bp = ax_box.boxplot(box_data, labels=box_labels, patch_artist=True)

                    for patch, color in zip(bp['boxes'], box_colors):
                        patch.set_facecolor(color)

                        patch.set_alpha(0.7)

                    ax_box.set_ylabel("Stellwert [%]")

                    ax_box.set_title(f"Verteilung der Stellwerte ({start_str} bis {end_str})")

                    plt.xticks(rotation=45, ha='right')

                    ax_box.grid(axis='y', linestyle='--', alpha=0.5)

                    _add_anlagenart_legend(ax_box)

                    plt.tight_layout()

                    st.pyplot(fig_box)

                    plt.close(fig_box)

                st.markdown("---")

                # Heatmap

                st.subheader("Korrelations-Heatmap zwischen Ventilen")

                df_filtered_hm = apply_time_filter(pdf_prepared, start_str, end_str)

                hm_cols = [c for c in active_valve_cols if c in df_filtered_hm.columns]

                valve_data = df_filtered_hm[hm_cols].dropna()

                if not valve_data.empty and len(hm_cols) >= 2:

                    corr_matrix = valve_data.corr()

                    fig_hm, ax_hm = plt.subplots(figsize=(max(10, len(hm_cols) * 0.6), max(8, len(hm_cols) * 0.5)))

                    sns.heatmap(corr_matrix, annot=True if len(hm_cols) <= 15 else False,

                                fmt=".2f", cmap="RdBu_r", center=0, ax=ax_hm,

                                xticklabels=True, yticklabels=True)

                    ax_hm.set_title("Korrelation zwischen Ventilen")

                    plt.xticks(rotation=45, ha='right', fontsize=7)

                    plt.yticks(fontsize=7)

                    plt.tight_layout()

                    st.pyplot(fig_hm)

                    plt.close(fig_hm)

                else:

                    st.info("Heatmap benoetigt mindestens 2 Ventile.")

            # ==========================================

            # TAB: TAGES-KPIs

            # ==========================================

            with tab_day:

                st.header(f"Tages-KPIs fuer {day_str}")

                col1, col2 = st.columns(2)

                with col1:

                    st.subheader("Spannweite pro Tag")

                    df_span_day = analyze_valve_distance_day(

                        pdf_prepared,

                        valves="select",

                        selected_valves=active_valve_cols,

                        date=day_str

                    )

                    st.dataframe(

                        df_span_day.style.format("{:.2f}"),

                        use_container_width=True

                    )

                with col2:

                    st.subheader("Spannweite pro Monat")

                    df_span_month = analyze_valve_distance_month(

                        pdf_prepared,

                        valves="select",

                        selected_valves=active_valve_cols,

                        date=day_str

                    )

                    st.dataframe(

                        df_span_month.style.format("{:.2f}"),

                        use_container_width=True

                    )

                st.markdown("---")

                col3, col4 = st.columns(2)

                with col3:

                    st.subheader("Stabilitaet am Tag")

                    st.caption("Standardabweichung waehrend Betrieb an diesem Tag")

                    df_stab_day = analyze_valves_stability_on_day(

                        pdf_prepared,

                        valves="select",

                        selected_valves=active_valve_cols,

                        date=day_str

                    )

                    st.dataframe(

                        df_stab_day.style.format("{:.3f}"),

                        use_container_width=True

                    )

                with col4:

                    st.subheader("Reaktionen am Tag")

                    st.caption("Anzahl Stellwertaenderungen an diesem Tag")

                    df_react_day = analyze_valves_reaction_rate_day(

                        pdf_prepared,

                        valves="select",

                        selected_valves=active_valve_cols,

                        date=day_str

                    )

                    st.dataframe(

                        df_react_day.style.format("{:.0f}"),

                        use_container_width=True

                    )

                st.markdown("---")

                # Tagesverlauf als Plot

                st.subheader(f"Stellwertverlauf am {day_str}")

                day_start = day_str

                day_end = str(day_date + timedelta(days=1))

                df_day_plot = apply_time_filter(

                    pdf_prepared, start=day_start, end=day_end

                )

                df_day_plot = df_day_plot.copy()

                df_day_plot['time'] = pd.to_datetime(

                    df_day_plot['time'], errors='coerce'

                )

                if not df_day_plot.empty:

                    fig_day, ax_day = plt.subplots(figsize=(14, 5))

                    for vc in active_valve_cols:

                        if vc in df_day_plot.columns:
                            ax_day.plot(

                                df_day_plot['time'],

                                df_day_plot[vc],

                                label=f"{vc} [{ZUORDNUNG.get(vc, '?')}]",

                                linewidth=1.5

                            )

                    # Temperatur auf zweiter Y-Achse

                    if 'temperature' in df_day_plot.columns:
                        ax_temp = ax_day.twinx()

                        ax_temp.plot(

                            df_day_plot['time'],

                            df_day_plot['temperature'],

                            color='black',

                            linestyle='--',

                            linewidth=1,

                            alpha=0.6,

                            label='Temperatur'

                        )

                        ax_temp.set_ylabel("Temperatur [C]", color='black')

                        ax_temp.legend(loc='upper right', fontsize=8)

                    ax_day.set_xlabel("Zeit")

                    ax_day.set_ylabel("Stellwert [%]")

                    ax_day.set_title(f"Ventilstellwerte am {day_str}")

                    ax_day.legend(fontsize=8, loc='upper left')

                    ax_day.grid(True, linestyle='--', alpha=0.5)

                    ax_day.xaxis.set_major_formatter(

                        mdates.DateFormatter('%H:%M')

                    )

                    plt.xticks(rotation=45)

                    plt.tight_layout()

                    st.pyplot(fig_day)

                    plt.close(fig_day)

                else:

                    st.warning(

                        f"Keine Daten fuer den {day_str} vorhanden."

                    )

                st.markdown("---")

                # Vergleich Spannweite Tag vs Monat

                st.subheader("Vergleich: Spannweite Tag vs. Monat")

                if not df_span_day.empty and not df_span_month.empty:

                    fig_compare, ax_compare = plt.subplots(figsize=(12, 5))

                    valves_list = df_span_day.index.tolist()

                    x = np.arange(len(valves_list))

                    width = 0.35

                    day_vals = []

                    month_vals = []

                    for v in valves_list:

                        if v in df_span_day.index:

                            day_vals.append(

                                df_span_day.loc[v, "valve_span_1day_distance"]

                            )

                        else:

                            day_vals.append(0)

                        if v in df_span_month.index:

                            month_vals.append(

                                df_span_month.loc[v, "valve_span_1month_distance"]

                            )

                        else:

                            month_vals.append(0)

                    ax_compare.bar(

                        x - width / 2, day_vals, width,

                        label=f'Tag ({day_str})', color='steelblue'

                    )

                    ax_compare.bar(

                        x + width / 2, month_vals, width,

                        label='Monat', color='coral'

                    )

                    ax_compare.set_xlabel("Ventil")

                    ax_compare.set_ylabel("Spannweite [%]")

                    ax_compare.set_title("Spannweite: Tag vs. Monat")

                    ax_compare.set_xticks(x)

                    ax_compare.set_xticklabels(

                        valves_list, rotation=45, ha='right'

                    )

                    ax_compare.legend()

                    ax_compare.grid(axis='y', linestyle='--', alpha=0.5)

                    plt.tight_layout()

                    st.pyplot(fig_compare)

                    plt.close(fig_compare)

            # ==========================================

            # TAB: KPI DOKUMENTATION

            # ==========================================

            with tab_doku:

                st.header("KPI Dokumentation")

                st.markdown("### Übersicht aller berechneten Kennzahlen mit Formeln und Beschreibungen")

                # --- KPI-Dokumentationstabelle ---

                kpi_docs = [

                    {

                        "Nr": 1,

                        "KPI": "Ventilbetriebszeit",

                        "Beschreibung": "Anteil der Zeit, in der das Ventil aktiv ist (Stellwert > 1 %)",

                        "Nutzen": "Zeigt, wie oft das Ventil aktiv (und geöffnet) ist",

                        "Formel": "Ventilbetriebszeit [%] = (Zeit mit Stellwert > 1 %) / Gesamtzeit × 100",

                        "Zugehörigkeit": "Komponente - Ventil",

                        "Priorität": "Prio"

                    },

                    {

                        "Nr": 2,

                        "KPI": "Ventilruhezeit",

                        "Beschreibung": "Zeitanteil mit Stellwert = 0 %",

                        "Nutzen": "Gibt Hinweis auf Auslastung im Kontext mit Aussentemperatur",

                        "Formel": "Ventilruhezeit [%] = (Zeit mit Stellwert = 0 %) / Gesamtzeit × 100",

                        "Zugehörigkeit": "Komponente - Ventil",

                        "Priorität": "Prio"

                    },

                    {

                        "Nr": 3,

                        "KPI": "Ventilstrecke_Total",

                        "Beschreibung": "Summe der total getätigten Ventiländerungen",

                        "Nutzen": "Hinweis auf Verschleiss und Instandhaltungsaufwand",

                        "Formel": "Ventilstrecke = ∑ |Stellwert(t) - Stellwert(t-1)|",

                        "Zugehörigkeit": "Komponente - Ventil",

                        "Priorität": "Prio"

                    },

                    {

                        "Nr": 4,

                        "KPI": "Ventilrichtungswechsel_Total",

                        "Beschreibung": "Summe der total getätigten Richtungswechsel (auf/zu)",

                        "Nutzen": "Hinweis auf Verschleiss und Instandhaltungsaufwand",

                        "Formel": "Anzahl der Richtungswechsel = ∑ Wechsel von steigend zu fallend oder umgekehrt",

                        "Zugehörigkeit": "Komponente - Ventil",

                        "Priorität": "Prio"

                    },

                    {

                        "Nr": 5,

                        "KPI": "Ventilstabilität_on",

                        "Beschreibung": "Standardabweichung des Stellwerts über die Zeit während Betrieb",

                        "Nutzen": "Hohe Schwankung deutet auf instabile Regelung hin",

                        "Formel": "σ = √(∑(xᵢ - x̄)² / n) wenn x > 1",

                        "Zugehörigkeit": "Komponente - Ventil",

                        "Priorität": "Prio"

                    },

                    {

                        "Nr": 6,

                        "KPI": "Ventilreaktionsrate",

                        "Beschreibung": "Anzahl der Stellwertänderungen pro Zeiteinheit",

                        "Nutzen": "Hohe Anzahl deutet auf ein nervöses System hin",

                        "Formel": "Reaktionsrate = Anzahl Stellwertänderungen / Δt",

                        "Zugehörigkeit": "Komponente - Ventil",

                        "Priorität": "Prio"

                    },

                    {

                        "Nr": 7,

                        "KPI": "Ventil_Korrelation_AUL",

                        "Beschreibung": "Korrelation zwischen Stellwert des Ventils und der Aussentemperatur",

                        "Nutzen": "Zeigt, ob das Ventil temperaturabhängig arbeitet",

                        "Formel": "r = ∑((xᵢ - x̄)(yᵢ - ȳ)) / √(∑(xᵢ - x̄)² · ∑(yᵢ - ȳ)²)",

                        "Zugehörigkeit": "Komponente - Ventil",

                        "Priorität": "Modell notwendig"

                    },

                    {

                        "Nr": 8,

                        "KPI": "Ventil_q25",

                        "Beschreibung": "25%-Quantil des Stellsignals pro Zeitintervall",

                        "Nutzen": "Hinweis auf Verteilung der Ventilstellungen",

                        "Formel": "q25 = 25%-Quantil(x)",

                        "Zugehörigkeit": "Komponente - Ventil",

                        "Priorität": "Prio"

                    },

                    {

                        "Nr": 9,

                        "KPI": "Ventil_q50",

                        "Beschreibung": "50%-Quantil (Median) des Stellsignals pro Zeitintervall",

                        "Nutzen": "Hinweis auf Verteilung der Ventilstellungen",

                        "Formel": "q50 = Median(x)",

                        "Zugehörigkeit": "Komponente - Ventil",

                        "Priorität": "Prio"

                    },

                    {

                        "Nr": 10,

                        "KPI": "Ventil_q75",

                        "Beschreibung": "75%-Quantil des Stellsignals pro Zeitintervall",

                        "Nutzen": "Hinweis auf Verteilung der Ventilstellungen",

                        "Formel": "q75 = 75%-Quantil(x)",

                        "Zugehörigkeit": "Komponente - Ventil",

                        "Priorität": "Prio"

                    },

                    {

                        "Nr": 11,

                        "KPI": "Ventil_q5",

                        "Beschreibung": "5%-Quantil des Stellsignals pro Zeitintervall",

                        "Nutzen": "Hinweis auf Verteilung der Ventilstellungen - unterer Bereich",

                        "Formel": "q5 = 5%-Quantil(x)",

                        "Zugehörigkeit": "Komponente - Ventil",

                        "Priorität": "Prio"

                    },

                    {

                        "Nr": 12,

                        "KPI": "Ventil_q95",

                        "Beschreibung": "95%-Quantil des Stellsignals pro Zeitintervall",

                        "Nutzen": "Hinweis auf Verteilung der Ventilstellungen - oberer Bereich",

                        "Formel": "q95 = 95%-Quantil(x)",

                        "Zugehörigkeit": "Komponente - Ventil",

                        "Priorität": "Prio"

                    },

                    {

                        "Nr": 13,

                        "KPI": "Ventil_mean",

                        "Beschreibung": "Mittelwert des Stellsignals pro Zeitintervall",

                        "Nutzen": "Hinweis auf Verteilung der Ventilstellungen",

                        "Formel": "x̄ = ∑xᵢ / n",

                        "Zugehörigkeit": "Komponente - Ventil",

                        "Priorität": "Prio"

                    },

                    {

                        "Nr": 14,

                        "KPI": "Ventil_std",

                        "Beschreibung": "Standardabweichung des Stellsignals pro Zeitintervall",

                        "Nutzen": "Hinweis auf Verteilung der Ventilstellungen",

                        "Formel": "σ = √(∑(xᵢ - x̄)² / n)",

                        "Zugehörigkeit": "Komponente - Ventil",

                        "Priorität": "Prio"

                    },

                    {

                        "Nr": 15,

                        "KPI": "Ventil_distance",

                        "Beschreibung": "Spannweite zwischen Min und Max des Stellsignals pro Zeiteinheit",

                        "Nutzen": "Hinweis auf Verteilung der Ventilstellungen",

                        "Formel": "Spannweite = max(x) - min(x)",

                        "Zugehörigkeit": "Komponente - Ventil",

                        "Priorität": "Prio"

                    },

                    {

                        "Nr": 16,

                        "KPI": "Ventil_count",

                        "Beschreibung": "Anzahl der Werte pro Zeitintervall",

                        "Nutzen": "Kontext für Zeitraum der Datenaufnahme",

                        "Formel": "count = Anzahl der Messwerte",

                        "Zugehörigkeit": "Komponente - Ventil",

                        "Priorität": "—"

                    },

                    {

                        "Nr": 17,

                        "KPI": "Temperaturdifferenz_AUL_RAL",

                        "Beschreibung": "Differenz zwischen Raumlufttemperatur (Annahme: 21 °C) und Aussentemperatur",

                        "Nutzen": "Kontext für Heiz- und Kühlbedarf",

                        "Formel": "ΔT = T_Raum - T_Aussen (T_Raum = 21 °C konstant)",

                        "Zugehörigkeit": "Komponente - Ventil",

                        "Priorität": "Prio"

                    },

                    {

                        "Nr": 18,

                        "KPI": "Ventilfehler_AUL_Warm",

                        "Beschreibung": "Anteil der Zeit, in der Ventil > 50 % bei Aussentemperatur > 18 °C",

                        "Nutzen": "Erkennt Fehlverhalten (z.B. Heizen bei warmem Wetter) und Normverhalten (z.B. Kühlventil sollte stärker geöffnet sein bei hoher Aussentemperatur)",

                        "Formel": "Anteil = (Zeit mit Ventil > 50 % und T_Aussen > 18 °C) / Gesamtzeit × 100",

                        "Zugehörigkeit": "Komponente - Ventil",

                        "Priorität": "⚠️ Problematisch (Heiz-/Kühlventil)"

                    },

                    {

                        "Nr": 19,

                        "KPI": "Ventilfehler_AUL_Kalt",

                        "Beschreibung": "Anteil der Zeit, in der Ventil > 50 % bei Aussentemperatur < 10 °C",

                        "Nutzen": "Erkennt Fehlverhalten (z.B. Kühlen bei kaltem Wetter) und Normverhalten (z.B. Heizventil sollte stärker geöffnet sein bei tiefer Aussentemperatur)",

                        "Formel": "Anteil = (Zeit mit Ventil > 50 % und T_Aussen < 10 °C) / Gesamtzeit × 100",

                        "Zugehörigkeit": "Komponente - Ventil",

                        "Priorität": "⚠️ Problematisch (Heiz-/Kühlventil)"

                    },

                    {

                        "Nr": 22,

                        "KPI": "Ventil_distance_1Tag",

                        "Beschreibung": "Spannweite zwischen Min und Max des Stellsignals pro Tag",

                        "Nutzen": "Zeigt, ob das Ventil den gesamten Stellbereich nutzt",

                        "Formel": "Spannweite = max(Stellwert) - min(Stellwert)",

                        "Zugehörigkeit": "Komponente - Ventil",

                        "Priorität": "Prio"

                    },

                    {

                        "Nr": 23,

                        "KPI": "Ventil_distance_1Monat",

                        "Beschreibung": "Spannweite zwischen Min und Max des Stellsignals pro Monat",

                        "Nutzen": "Zeigt, ob das Ventil den gesamten Stellbereich nutzt",

                        "Formel": "Spannweite = max(Stellwert) - min(Stellwert)",

                        "Zugehörigkeit": "Komponente - Ventil",

                        "Priorität": "Prio"

                    },

                    {

                        "Nr": 24,

                        "KPI": "Ventilstabilität_on_1Tag",

                        "Beschreibung": "Standardabweichung des Stellwerts über 1 Tag während Betrieb",

                        "Nutzen": "Hohe Schwankung deutet auf instabile Regelung hin",

                        "Formel": "σ = √(∑(xᵢ - x̄)² / n) wenn x > 1",

                        "Zugehörigkeit": "Komponente - Ventil",

                        "Priorität": "Prio"

                    },

                    {

                        "Nr": 25,

                        "KPI": "Ventilreaktionsrate_1Tag",

                        "Beschreibung": "Anzahl der Stellwertänderungen pro Tag",

                        "Nutzen": "Hohe Anzahl deutet auf ein nervöses System hin",

                        "Formel": "Reaktionsrate = Anzahl Stellwertänderungen / Δt",

                        "Zugehörigkeit": "Komponente - Ventil",

                        "Priorität": "Prio"

                    },

                    {

                        "Nr": 26,

                        "KPI": "Ventil_Korrelation_AUL_Kalt",

                        "Beschreibung": "Korrelation zwischen Stellwert und Aussenluft bei T < 10 °C",

                        "Nutzen": "Erkennt Fehlverhalten (z.B. Heizventil übermässig geschlossen bei Kälte, Kühlventil übermässig offen bei Kälte)",

                        "Formel": "r_kalt = Pearson-Korrelation wenn T_AUL < 10 °C",

                        "Zugehörigkeit": "Komponente - Ventil",

                        "Priorität": "Prio"

                    },

                    {

                        "Nr": 27,

                        "KPI": "Ventil_Korrelation_AUL_Mild",

                        "Beschreibung": "Korrelation zwischen Stellwert und Aussenluft bei 10 ≤ T ≤ 18 °C",

                        "Nutzen": "Erkennt Fehlverhalten bei milden Temperaturen",

                        "Formel": "r_mild = Pearson-Korrelation wenn 10 ≤ T_AUL ≤ 18 °C",

                        "Zugehörigkeit": "Komponente - Ventil",

                        "Priorität": "Prio"

                    },

                    {

                        "Nr": 28,

                        "KPI": "Ventil_Korrelation_AUL_Warm",

                        "Beschreibung": "Korrelation zwischen Stellwert und Aussenluft bei T > 18 °C",

                        "Nutzen": "Erkennt Fehlverhalten (z.B. Heizventil übermässig offen bei Wärme, Kühlventil übermässig geschlossen bei Wärme)",

                        "Formel": "r_warm = Pearson-Korrelation wenn T_AUL > 18 °C",

                        "Zugehörigkeit": "Komponente - Ventil",

                        "Priorität": "Prio"

                    },

                    {

                        "Nr": 29,

                        "KPI": "Ventil_mean_Tag",

                        "Beschreibung": "Mittelwert des Stellsignals in den Tagesstunden von 06–23 Uhr",

                        "Nutzen": "Hinweis auf Verteilung der Ventilstellungen und Zeitprogramm für Absenkung – Vergleich mit Ventil_mean_Nacht",

                        "Formel": "x̄ = ∑xᵢ / n wenn Stunde ∈ [6, 23]",

                        "Zugehörigkeit": "Komponente - Ventil",

                        "Priorität": "Prio"

                    },

                    {

                        "Nr": 30,

                        "KPI": "Ventil_mean_Nacht",

                        "Beschreibung": "Mittelwert des Stellsignals in den Nachtstunden von 23:01–05:59 Uhr",

                        "Nutzen": "Hinweis auf Verteilung der Ventilstellungen und Zeitprogramm für Absenkung – Vergleich mit Ventil_mean_Tag",

                        "Formel": "x̄ = ∑xᵢ / n wenn Stunde ∉ [6, 23]",

                        "Zugehörigkeit": "Komponente - Ventil",

                        "Priorität": "Prio"

                    },

                    {

                        "Nr": 31,

                        "KPI": "Ventil_Auslastungsgrenze_90",

                        "Beschreibung": "Prozentualer Zeitanteil mit Stellsignal ≥ 90 %",

                        "Nutzen": "Hinweis auf Unterdimensionierung oder Überlast, die nicht über mehrere Tage am Stück andauern sollte",

                        "Formel": "Anteil = (Zeit mit Stellwert ≥ 90 %) / Gesamtzeit × 100",

                        "Zugehörigkeit": "Komponente - Ventil",

                        "Priorität": "Prio"

                    },

                ]

                df_doku = pd.DataFrame(kpi_docs)

                # --- Suchfilter ---

                st.markdown("---")

                col_search, col_prio = st.columns([2, 1])

                with col_search:

                    search_term = st.text_input("KPI suchen (Name oder Beschreibung)", "")

                with col_prio:

                    prio_filter = st.selectbox("Priorität filtern",
                                               ["Alle", "Prio", "Problematisch", "Modell notwendig", "—"])

                # Filter anwenden

                df_doku_filtered = df_doku.copy()

                if search_term:
                    mask = (

                            df_doku_filtered["KPI"].str.contains(search_term, case=False, na=False) |

                            df_doku_filtered["Beschreibung"].str.contains(search_term, case=False, na=False) |

                            df_doku_filtered["Nutzen"].str.contains(search_term, case=False, na=False)

                    )

                    df_doku_filtered = df_doku_filtered[mask]

                if prio_filter != "Alle":
                    df_doku_filtered = df_doku_filtered[
                        df_doku_filtered["Priorität"].str.contains(prio_filter, case=False, na=False)]

                st.markdown(f"**{len(df_doku_filtered)} von {len(df_doku)} KPIs angezeigt**")

                # --- Gesamttabelle ---

                st.markdown("---")

                st.subheader("KPI-Übersichtstabelle")

                st.dataframe(

                    df_doku_filtered[["Nr", "KPI", "Beschreibung", "Formel", "Priorität"]],

                    use_container_width=True,

                    height=400,

                    hide_index=True

                )

                # --- Detailkarten ---

                st.markdown("---")

                st.subheader("Detailansicht")

                for _, row in df_doku_filtered.iterrows():
                    with st.expander(f"**KPI #{row['Nr']}** — {row['KPI']}  ({row['Priorität']})"):
                        col_left, col_right = st.columns([1, 1])

                        with col_left:
                            st.markdown(f"**Beschreibung:**")

                            st.info(row["Beschreibung"])

                            st.markdown(f"**Nutzen:**")

                            st.success(row["Nutzen"])

                        with col_right:
                            st.markdown(f"**Formel:**")

                            st.code(row["Formel"], language=None)

                            st.markdown(f"**Zugehörigkeit:** {row['Zugehörigkeit']}")

                            st.markdown(f"**Priorität:** {row['Priorität']}")

                # --- Kategorisierte Ansicht ---

                st.markdown("---")

                st.subheader("KPIs nach Kategorie")

                categories = {

                    "Betriebszeit & Auslastung": [1, 2, 31],

                    "Verschleiss & Bewegung": [3, 4, 6, 25],

                    "Stabilität": [5, 24],

                    "Statistische Verteilung": [8, 9, 10, 11, 12, 13, 14, 15, 16],

                    "Temperatur-Korrelation": [7, 17, 26, 27, 28],

                    "Fehleranalyse": [18, 19],

                    "Spannweite": [22, 23],

                    "Tag/Nacht-Vergleich": [29, 30],

                }

                for cat_name, cat_nrs in categories.items():

                    cat_df = df_doku[df_doku["Nr"].isin(cat_nrs)]

                    if not cat_df.empty:
                        st.markdown(f"#### {cat_name}")

                        st.dataframe(

                            cat_df[["Nr", "KPI", "Beschreibung", "Formel"]],

                            use_container_width=True,

                            hide_index=True

                        )

                # --- Download der Dokumentation ---

                st.markdown("---")

                csv_doku = df_doku.to_csv(index=False)

                st.download_button(

                    label="KPI-Dokumentation als CSV herunterladen",

                    data=csv_doku,

                    file_name="kpi_dokumentation.csv",

                    mime="text/csv"

                )

        else:

            st.error(

                "Keine KPI-Ergebnisse. "

                "Bitte Zeitraum und Ventilauswahl pruefen."

            )



    else:

        # Vor dem Klick auf "Analyse starten"

        st.info(

            "Bitte Filter in der Sidebar einstellen "

            "und 'Analyse starten' klicken."

        )

        # Vorschau der Daten

        st.subheader("Datenvorschau")

        st.dataframe(pdf.head(20), use_container_width=True)

        st.subheader("Verfuegbare Ventile nach Anlagenart")

        # Gruppierte Anzeige

        valve_art_df = pd.DataFrame({

            "Ventil": valves_by_art,

            "Anlagenart": [ZUORDNUNG.get(v, "Unbekannt") for v in valves_by_art]

        }).sort_values("Anlagenart")

        for art_name, group in valve_art_df.groupby("Anlagenart"):

            st.markdown(f"**{art_name}** ({len(group)} Ventile)")

            num_cols = 4

            cols_display = st.columns(num_cols)

            for i, valve in enumerate(group["Ventil"].tolist()):
                with cols_display[i % num_cols]:
                    st.code(valve)

        if 'temperature' in pdf.columns:
            st.subheader("Temperaturuebersicht")

            tc1, tc2, tc3 = st.columns(3)

            with tc1:
                st.metric(

                    "Min Temperatur",

                    f"{pdf['temperature'].min():.1f} C"

                )

            with tc2:
                st.metric(

                    "Durchschn. Temperatur",

                    f"{pdf['temperature'].mean():.1f} C"

                )

            with tc3:
                st.metric(

                    "Max Temperatur",

                    f"{pdf['temperature'].max():.1f} C"

                )



else:

    # Kein File hochgeladen

    st.markdown("""

    ## Willkommen zum Ventilanalyse Dashboard



    Dieses Dashboard analysiert Ventilstellsignale und berechnet verschiedene KPIs.



    ### So geht's:

    1. **CSV-Datei hochladen** (Sidebar links)

    2. **Anlagenart waehlen** (Lufterhitzer, Luftkuehler, WRG)

    3. **Filter einstellen** (Zeitraum, Ventile)

    4. **Analyse starten** klicken



    ### Erwartetes Datenformat:

    - Spalte `time` mit Zeitstempel

    - Ventilspalten mit `TV` im Namen (z.B. `F1311211TV5807`)

    - Optional: Temperaturspalte `F1311211TE6161` (wird zu `temperature`)



    ### Anlagenarten:

    | Anlagenart | Beschreibung |

    |------------|-------------|

    | Lufterhitzer | Heizventile |

    | Luftkuehler | Kuehlventile |

    | WRG | Waermerueckgewinnung |



    ### Verfuegbare Analysen:

    | # | Analyse | Beschreibung |

    |---|---------|-------------|

    | 1 | Betriebszeit | Anteil aktiv (> 1%) |

    | 2 | Ruhezeit | Anteil bei 0% |

    | 3 | Ventilstrecke | Summe aller Aenderungen |

    | 4 | Richtungswechsel | Auf/Zu Wechsel |

    | 5 | Stabilitaet | Std waehrend Betrieb |

    | 6 | Reaktionsrate | Aenderungen pro Intervall |

    | 7 | Korrelation | Ventil <-> Temperatur |

    | 8-16 | Statistik | Quantile, Mean, Std |

    | 17 | Delta-T | Raum vs. Aussen |

    | 18-19 | Fehleranalyse | Warm/Kalt Fehler |

    | 22-23 | Spannweite | Tag/Monat Min-Max |

    | 24-25 | Tages-KPIs | Stabilitaet/Reaktionen pro Tag |

    | 26-28 | Korrelation | Kalt/Mild/Warm Bereich |

    | 29-30 | Tag/Nacht | Mittelwert Tag vs. Nacht |

    | 31 | Auslastung | Zeit >= 90% |

    """)

    st.markdown("---")

    st.caption(

        "Dashboard erstellt fuer die Ventilanalyse MUV2 | "

        "Powered by Streamlit"

    )