import os
import time
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="DataFrame.mean and DataFrame.median with numeric_only=None")


# =========================================================
# 1. Basic configuration
# =========================================================
# Use environment variables instead of hard-coded local paths.
# Example:
#   START_DATE=2024-07-23 END_DATE=2024-07-24 \
#   PATH_CONFIG=data_path_config.csv OUTPUT_DIR=outputs/created_data \
#   python datacreate_optimized.py

START_DATE = pd.to_datetime(os.getenv("START_DATE", "2024-07-23"))
END_DATE = pd.to_datetime(os.getenv("END_DATE", "2024-07-24"))

PATH_CONFIG = Path(os.getenv("PATH_CONFIG", "data_path_config.csv"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs/created_data"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = os.getenv("TARGET_COL", "predict_T2_0.5")
TIME_COL = "start_time"

# The first timestamp generated for each day.
# Your original code used 07:10, but hard-coded the date as 2024-07-23.
DAY_START_HOUR = int(os.getenv("DAY_START_HOUR", "7"))
DAY_START_MINUTE = int(os.getenv("DAY_START_MINUTE", "10"))

# Whether to generate future target columns predict_T2_* and predict_T3_*.
# For model training/testing data construction, keep it enabled.
# For real online prediction, set GENERATE_TARGETS=0 to avoid using future information.
GENERATE_TARGETS = os.getenv("GENERATE_TARGETS", "1") == "1"

# If enabled, local raw paths will be printed. Keep disabled for public GitHub code.
PRINT_RAW_PATHS = os.getenv("PRINT_RAW_PATHS", "0") == "1"


# =========================================================
# 2. Window definitions
# =========================================================
SHORT_WINDOWS = [
    ("5", timedelta(minutes=5)),
    ("10", timedelta(minutes=10)),
    ("15", timedelta(minutes=15)),
    ("20", timedelta(minutes=20)),
    ("25", timedelta(minutes=25)),
    ("30", timedelta(minutes=30)),
    ("35", timedelta(minutes=35)),
    ("40", timedelta(minutes=40)),
    ("60", timedelta(minutes=60)),
    ("90", timedelta(minutes=90)),
    ("120", timedelta(minutes=120)),
]

HOUR_WINDOWS = [
    (f"{hour}h", timedelta(hours=hour), timedelta(hours=hour - 1))
    for hour in range(3, 25)
]

TIME_WINDOWS = [label for label, _ in SHORT_WINDOWS] + [label for label, _, _ in HOUR_WINDOWS]

# Future windows for target construction.
# label, start offset, end offset
FUTURE_TARGET_WINDOWS = [
    ("0.5", timedelta(minutes=0), timedelta(minutes=30)),
    ("1", timedelta(minutes=30), timedelta(minutes=60)),
    ("1.5", timedelta(minutes=60), timedelta(minutes=90)),
    ("2", timedelta(minutes=90), timedelta(hours=2)),
] + [
    (str(hour), timedelta(hours=hour - 1), timedelta(hours=hour))
    for hour in range(3, 25)
]

# Future flight feature windows.
# The original code uses cumulative windows for 30/60/90/120 minutes,
# and hourly intervals from 2-3h to 11-12h.
PLANE_WINDOWS = [
    ("30", timedelta(minutes=0), timedelta(minutes=30)),
    ("60", timedelta(minutes=0), timedelta(minutes=60)),
    ("90", timedelta(minutes=0), timedelta(minutes=90)),
    ("120", timedelta(minutes=0), timedelta(minutes=120)),
] + [
    (str(hour), timedelta(hours=hour - 1), timedelta(hours=hour))
    for hour in range(3, 13)
]

WAIT_POINT_WINDOWS = [
    ("5", 5),
    ("10", 10),
    ("15", 15),
    ("20", 20),
    ("25", 25),
    ("30", 30),
    ("35", 35),
    ("40", 40),
]


# =========================================================
# 3. Path configuration
# =========================================================
def load_path_config(path_config: Path) -> pd.DataFrame:
    """
    Load path configuration.

    Required columns:
        source,start_date,end_date,path

    source must be one of:
        queue, traff_t2, traff_t3, check, plane

    The path column may contain:
        {date:%Y%m%d}
    """

    if not path_config.exists():
        raise FileNotFoundError(
            f"Path configuration file not found: {path_config}\n"
            "Please create data_path_config.csv before running this script."
        )

    config = pd.read_csv(path_config)

    required_cols = {"source", "start_date", "end_date", "path"}
    missing_cols = required_cols - set(config.columns)

    if missing_cols:
        raise ValueError(f"Path configuration is missing columns: {missing_cols}")

    config["start_date"] = pd.to_datetime(config["start_date"])
    config["end_date"] = pd.to_datetime(config["end_date"])

    return config


def resolve_source_path(path_config: pd.DataFrame, source: str, current_date: pd.Timestamp) -> Path:
    matched = path_config[
        (path_config["source"] == source)
        & (path_config["start_date"] <= current_date)
        & (current_date < path_config["end_date"])
    ]

    if matched.empty:
        raise FileNotFoundError(
            f"No path rule found for source='{source}' and date={current_date.date()}."
        )

    if len(matched) > 1:
        print(
            f"Warning: multiple path rules matched source='{source}', date={current_date.date()}. "
            "The first one will be used."
        )

    path_template = str(matched.iloc[0]["path"])
    resolved = path_template.format(date=current_date)

    return Path(resolved)


def get_file_paths(path_config: pd.DataFrame, current_date: pd.Timestamp) -> dict:
    return {
        "queue": resolve_source_path(path_config, "queue", current_date),
        "traff_t2": resolve_source_path(path_config, "traff_t2", current_date),
        "traff_t3": resolve_source_path(path_config, "traff_t3", current_date),
        "check": resolve_source_path(path_config, "check", current_date),
        "plane": resolve_source_path(path_config, "plane", current_date),
    }


# =========================================================
# 4. Loading and preprocessing
# =========================================================
def read_excel_safe(path: Path, source_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{source_name} file not found: {path}")

    return pd.read_excel(path)


def load_sources_for_day(path_config: pd.DataFrame, day: pd.Timestamp) -> dict:
    """
    Load raw data for day-1, day, and day+1.

    This keeps enough historical and future information for:
    - past 24-hour statistics
    - future target and flight windows
    """

    combined = {
        "queue": [],
        "traff_t2": [],
        "traff_t3": [],
        "check": [],
        "plane": [],
    }

    loaded_files = set()

    for offset in [-1, 0, 1]:
        current_date = day + timedelta(days=offset)
        paths = get_file_paths(path_config, current_date)

        print(f"Loading source files for {current_date.strftime('%Y%m%d')}")

        if PRINT_RAW_PATHS:
            for source_name, source_path in paths.items():
                print(f"  {source_name}: {source_path}")

        for source_name, source_path in paths.items():
            file_key = (source_name, str(source_path))

            # Avoid reading the same multi-day Excel file repeatedly.
            if file_key in loaded_files:
                continue

            df = read_excel_safe(source_path, source_name)
            combined[source_name].append(df)
            loaded_files.add(file_key)

    return {
        source_name: pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()
        for source_name, df_list in combined.items()
    }


def preprocess_sources(raw_sources: dict) -> dict:
    queue = raw_sources["queue"].copy()
    check = raw_sources["check"].copy()
    traff_t2 = raw_sources["traff_t2"].copy()
    traff_t3 = raw_sources["traff_t3"].copy()
    plane = raw_sources["plane"].copy()

    # Queue data
    queue["date"] = pd.to_datetime(queue["date"], errors="coerce")
    queue["wait_time"] = pd.to_numeric(queue["wait_time"], errors="coerce").fillna(0)
    queue["time"] = queue["date"] - pd.to_timedelta(queue["wait_time"], unit="s")
    queue = queue.drop_duplicates().fillna(0)

    # Check data
    check["check_time"] = pd.to_datetime(check["check_time"], errors="coerce")
    check["time"] = check["check_time"]
    check = check[check["security_check_point"].astype(str).str.contains("DC", na=False)]
    check = check.drop_duplicates().fillna(0)

    # Traffic data
    traff_t2["time"] = pd.to_datetime(traff_t2["time"], errors="coerce")
    traff_t3["time"] = pd.to_datetime(traff_t3["time"], errors="coerce")

    traff_t2 = traff_t2.drop_duplicates().fillna(0)
    traff_t3 = traff_t3.drop_duplicates().fillna(0)

    # Flight data
    plane["selected_est_out_at"] = pd.to_datetime(plane["selected_est_out_at"], errors="coerce")
    plane = plane.drop_duplicates().fillna(0)

    # Precompute check-to-flight waiting time once.
    check_wait = build_check_wait_data(check, plane)

    return {
        "queue": queue,
        "check": check,
        "traff_t2": traff_t2,
        "traff_t3": traff_t3,
        "plane": plane,
        "check_wait": check_wait,
    }


def build_check_wait_data(check: pd.DataFrame, plane: pd.DataFrame) -> pd.DataFrame:
    """
    Match each check record with the next corresponding flight departure time.

    This replaces the repeated merge_asof inside every minute loop in the original code.
    """

    if check.empty or plane.empty:
        return pd.DataFrame(columns=["check_time", "wait_time"])

    if "flight_no" not in check.columns or "flight_no" not in plane.columns:
        return pd.DataFrame(columns=["check_time", "wait_time"])

    selected_plane = plane[["flight_no", "selected_est_out_at"]].dropna().copy()

    merged = pd.merge_asof(
        check.sort_values("check_time"),
        selected_plane.sort_values("selected_est_out_at"),
        by="flight_no",
        left_on="check_time",
        right_on="selected_est_out_at",
        direction="forward",
    )

    merged["wait_time"] = merged["selected_est_out_at"] - merged["check_time"]

    return merged


# =========================================================
# 5. Generic utilities
# =========================================================
def slice_time_window(df: pd.DataFrame, time_col: str, start_time: pd.Timestamp, end_time: pd.Timestamp) -> pd.DataFrame:
    if df.empty or time_col not in df.columns:
        return df.iloc[0:0].copy()

    return df[(start_time < df[time_col]) & (df[time_col] < end_time)]


def slice_time_window_closed_left(df: pd.DataFrame, time_col: str, start_time: pd.Timestamp, end_time: pd.Timestamp) -> pd.DataFrame:
    if df.empty or time_col not in df.columns:
        return df.iloc[0:0].copy()

    return df[(start_time <= df[time_col]) & (df[time_col] < end_time)]


def safe_sum(series) -> float:
    if series is None or len(series) == 0:
        return 0.0
    return float(pd.to_numeric(series, errors="coerce").fillna(0).sum())


def safe_max(series) -> float:
    series = pd.to_numeric(series, errors="coerce").dropna()
    return float(series.max()) if len(series) > 0 else 0.0


def safe_min(series) -> float:
    series = pd.to_numeric(series, errors="coerce").dropna()
    return float(series.min()) if len(series) > 0 else 0.0


def safe_mean(series) -> float:
    series = pd.to_numeric(series, errors="coerce").dropna()
    return float(series.mean()) if len(series) > 0 else 0.0


def safe_timedelta_max_seconds(series) -> float:
    series = pd.to_timedelta(series, errors="coerce").dropna()
    return float(series.max().total_seconds()) if len(series) > 0 else 0.0


def safe_timedelta_min_seconds(series) -> float:
    series = pd.to_timedelta(series, errors="coerce").dropna()
    return float(series.min().total_seconds()) if len(series) > 0 else 0.0


def safe_timedelta_mean_seconds(series) -> float:
    series = pd.to_timedelta(series, errors="coerce").dropna()
    return float(series.mean().total_seconds()) if len(series) > 0 else 0.0


def queue_id_count_stats(df: pd.DataFrame):
    if df.empty or "queue_id" not in df.columns:
        return 0, 0, 0

    counts = df["queue_id"].value_counts()

    if counts.empty:
        return 0, 0, 0

    return int(len(counts)), int(counts.max()), int(counts.min())


def terminal_is_t2(series: pd.Series) -> pd.Series:
    return series.astype(str).str.split("-").str[0].eq("T2")


# =========================================================
# 6. Feature construction
# =========================================================
def get_past_window_data(df: pd.DataFrame, time_col: str, current_time: pd.Timestamp) -> dict:
    windows = {}

    for label, delta in SHORT_WINDOWS:
        windows[label] = slice_time_window(
            df,
            time_col,
            current_time - delta,
            current_time,
        )

    for label, start_delta, end_delta in HOUR_WINDOWS:
        windows[label] = slice_time_window(
            df,
            time_col,
            current_time - start_delta,
            current_time - end_delta,
        )

    return windows


def build_queue_features(queue: pd.DataFrame, current_time: pd.Timestamp) -> dict:
    features = {}

    queue_windows = get_past_window_data(queue, "time", current_time)

    for label, window_df in queue_windows.items():
        queue_count, queue_max, queue_min = queue_id_count_stats(window_df)

        features[f"queue_countpassed{label}"] = queue_count
        features[f"queue_maxpassed{label}"] = queue_max
        features[f"queue_minpassed{label}"] = queue_min
        features[f"count_passed{label}"] = safe_sum(window_df.get("person_counts", pd.Series(dtype=float)))
        features[f"wait_max{label}"] = safe_max(window_df.get("wait_time", pd.Series(dtype=float)))
        features[f"wait_min{label}"] = safe_min(window_df.get("wait_time", pd.Series(dtype=float)))
        features[f"wait_avr{label}"] = safe_mean(window_df.get("wait_time", pd.Series(dtype=float)))

    # Original point-window features:
    # t-5 to t-4, t-10 to t-9, ..., t-40 to t-39 based on queue['date'].
    for label, minutes in WAIT_POINT_WINDOWS:
        point_df = slice_time_window(
            queue,
            "date",
            current_time - timedelta(minutes=minutes),
            current_time - timedelta(minutes=minutes - 1),
        )

        features[f"wait_passed{label}"] = safe_sum(point_df.get("person_counts", pd.Series(dtype=float)))

    return features


def build_check_features(check: pd.DataFrame, check_wait: pd.DataFrame, current_time: pd.Timestamp) -> dict:
    features = {}

    check_windows = get_past_window_data(check, "time", current_time)
    check_wait_windows = get_past_window_data(check_wait, "check_time", current_time)

    for label in TIME_WINDOWS:
        window_df = check_windows[label]
        wait_df = check_wait_windows[label]

        if "security_check_point" in window_df.columns:
            is_t2 = window_df["security_check_point"].astype(str).str.contains("T2", na=False)
        else:
            is_t2 = pd.Series(False, index=window_df.index)

        gender_counts = window_df["gender"].value_counts() if "gender" in window_df.columns else pd.Series(dtype=int)

        features[f"check_countpassed{label}"] = int(len(window_df))
        features[f"check_T2passed{label}"] = int(is_t2.sum())
        features[f"check_T3passed{label}"] = int(len(window_df) - is_t2.sum())
        features[f"check_manpassed{label}"] = int(gender_counts.get(1, 0))
        features[f"check_womanpassed{label}"] = int(gender_counts.get(2, 0))
        features[f"check_childpassed{label}"] = int(gender_counts.get(3, 0))

        features[f"check_waitmax{label}"] = safe_timedelta_max_seconds(wait_df.get("wait_time", pd.Series(dtype="timedelta64[ns]")))
        features[f"check_waitmin{label}"] = safe_timedelta_min_seconds(wait_df.get("wait_time", pd.Series(dtype="timedelta64[ns]")))
        features[f"check_waitavr{label}"] = safe_timedelta_mean_seconds(wait_df.get("wait_time", pd.Series(dtype="timedelta64[ns]")))

    return features


def build_traffic_features(traff_t2: pd.DataFrame, traff_t3: pd.DataFrame, current_time: pd.Timestamp) -> dict:
    features = {}

    t2_windows = get_past_window_data(traff_t2, "time", current_time)
    t3_windows = get_past_window_data(traff_t3, "time", current_time)

    for label in TIME_WINDOWS:
        t2_df = t2_windows[label]
        t3_df = t3_windows[label]

        features[f"traff_t2enterpassed{label}"] = safe_sum(t2_df.get("enter", pd.Series(dtype=float)))
        features[f"traff_t2exitpassed{label}"] = safe_sum(t2_df.get("exit", pd.Series(dtype=float)))
        features[f"traff_t3enterpassed{label}"] = safe_sum(t3_df.get("enter", pd.Series(dtype=float)))
        features[f"traff_t3exitpassed{label}"] = safe_sum(t3_df.get("exit", pd.Series(dtype=float)))

    return features


def build_plane_features(plane: pd.DataFrame, current_time: pd.Timestamp) -> dict:
    features = {}

    for label, start_delta, end_delta in PLANE_WINDOWS:
        window_df = slice_time_window_closed_left(
            plane,
            "selected_est_out_at",
            current_time + start_delta,
            current_time + end_delta,
        )

        di_counts = window_df["di"].value_counts() if "di" in window_df.columns else pd.Series(dtype=int)
        site_counts = window_df["site"].value_counts() if "site" in window_df.columns else pd.Series(dtype=int)

        features[f"plane_in{label}"] = int(di_counts.get("domestic", 0))
        features[f"plane_mix{label}"] = int(di_counts.get("mix", 0))
        features[f"plane_t2_{label}"] = int(site_counts.get("T2", 0))
        features[f"plane_t3_{label}"] = int(site_counts.get("T3", 0))
        features[f"plane_num_{label}"] = safe_sum(window_df.get("ordered_num", pd.Series(dtype=float)))

    return features


def build_target_features(check: pd.DataFrame, current_time: pd.Timestamp) -> dict:
    """
    Build future prediction targets.

    Warning:
    These columns use future check records. They are suitable for dataset construction,
    but must not be generated for real online prediction.
    """

    targets = {}

    for label, start_delta, end_delta in FUTURE_TARGET_WINDOWS:
        window_df = slice_time_window_closed_left(
            check,
            "time",
            current_time + start_delta,
            current_time + end_delta,
        )

        if "security_check_point" in window_df.columns:
            is_t2 = terminal_is_t2(window_df["security_check_point"])
            t2_count = int(is_t2.sum())
        else:
            t2_count = 0

        total_count = int(len(window_df))
        t3_count = total_count - t2_count

        targets[f"predict_T2_{label}"] = t2_count
        targets[f"predict_T3_{label}"] = t3_count

    return targets


def build_time_features(current_time: pd.Timestamp) -> dict:
    return {
        "time_index": current_time.hour * 60 + current_time.minute + 1,
        "weekday_index": current_time.weekday() + 1,
    }


def build_one_row(sources: dict, current_time: pd.Timestamp) -> dict:
    row = {
        TIME_COL: current_time,
    }

    row.update(build_queue_features(sources["queue"], current_time))
    row.update(build_check_features(sources["check"], sources["check_wait"], current_time))
    row.update(build_traffic_features(sources["traff_t2"], sources["traff_t3"], current_time))
    row.update(build_plane_features(sources["plane"], current_time))
    row.update(build_time_features(current_time))

    if GENERATE_TARGETS:
        row.update(build_target_features(sources["check"], current_time))

    return row


# =========================================================
# 7. Day-level generation
# =========================================================
def build_one_day(path_config: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    start_timer = time.time()

    print("\n" + "=" * 80)
    print(f"Building features for {day.strftime('%Y%m%d')}")
    print("=" * 80)

    raw_sources = load_sources_for_day(path_config, day)
    sources = preprocess_sources(raw_sources)

    if not sources["check"].empty:
        print(f"Check data time range: {sources['check']['time'].min()} - {sources['check']['time'].max()}")

    current_time = day + timedelta(hours=DAY_START_HOUR, minutes=DAY_START_MINUTE)
    end_time = day + timedelta(days=1) - timedelta(minutes=1)

    records = []

    while current_time <= end_time:
        row_start = time.time()

        row = build_one_row(sources, current_time)
        records.append(row)

        print(f"{current_time} completed in {time.time() - row_start:.2f} seconds")

        current_time += timedelta(minutes=1)

    output_df = pd.DataFrame(records).fillna(0)

    output_path = OUTPUT_DIR / f"{day.strftime('%Y%m%d')}.csv"
    output_df.to_csv(output_path, index=False)

    print(f"{day.strftime('%Y%m%d')} saved to: {output_path}")
    print(f"Day runtime: {time.time() - start_timer:.2f} seconds")

    return output_df


# =========================================================
# 8. Main process
# =========================================================
def main():
    path_config = load_path_config(PATH_CONFIG)

    current_day = START_DATE

    while current_day <= END_DATE:
        build_one_day(path_config, current_day)
        current_day += timedelta(days=1)

    print("\nAll feature files have been created.")


if __name__ == "__main__":
    main()