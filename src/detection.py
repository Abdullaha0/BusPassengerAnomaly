"""
Anomaly detection on APC bus passenger data.

Two families of detectors:
  - Rule-based: deterministic flags from physical/sensor constraints.
  - ML-based: any scikit-learn estimator with fit() and score_samples().

Each detector adds its own flag column to the DataFrame so we can
compare which trips each detector finds.
"""

import pandas as pd


# ---------------------------------------------------------------
# Rule-based detectors
# ---------------------------------------------------------------

def add_rule_flags(df):
    """
        Add boolean flag columns for all rule-based detectors.
        Each detector lives in its own helper so we can add or remove
        rules without touching the others.
    """
    df = flag_negative_load(df)
    df = flag_capacity_exceeded(df)
    df = flag_gps_zero(df)
    df = flag_gps_outside_halland(df)
    return df


def flag_negative_load(df):
    """
        Flag trips where load went below zero at any stop.
        A negative load is physically impossible (a bus cannot carry
        a negative number of passengers), so this is a clear sensor
        error: more alightings than boardings were registered.
    """
    df['flag_negative_load'] = df['trip_min_load'] < 0
    return df


def flag_capacity_exceeded(df):
    """
        Flag trips where peak load exceeded a realistic bus capacity.
        Set at 100 — beyond crush capacity for a standard regional bus.
    """
    df['flag_capacity_exceeded'] = df['trip_max_load'] > 100
    return df


def flag_gps_zero(df):
    """
        Flag rows with (latitude, longitude) == (0, 0).
        GPS sensor reported origin coordinates in the Atlantic ocean —
        always a sensor failure or unmapped position.
    """
    df['flag_gps_zero'] = (df['latitude'] == 0) & (df['longitude'] == 0)
    return df


def flag_gps_outside_halland(df):
    """
        Flag rows with coordinates outside the Halland region bounding box.
        
        Bounds chosen empirically: 99% of non-(0,0) rows in the January 2025
        APC data fell within latitude (56.47, 57.31) and longitude (12.13, 13.24).
        We use a slightly wider box (56.1-57.5, 12.0-13.7) for safety margin.
        See detectionTest.ipynb for the percentile analysis.
    """
    in_halland = (
        df['latitude'].between(56.1, 57.5) &
        df['longitude'].between(12.0, 13.7)
    )
    df['flag_gps_outside_halland'] = ~in_halland
    return df


# ---------------------------------------------------------------
# ML-based detector
# ---------------------------------------------------------------

# Trip-level features used as input to ML detectors.
# Kept short on purpose: fewer features = more interpretable model.
ML_FEATURES = [
    'trip_total_boardings',
    'trip_total_alightings',
    'trip_imbalance',
    'trip_n_stops',
    'trip_min_load',
    'trip_max_load',
    'trip_final_load',
    'trip_imbalance_z_score',
]


def add_ml_flags(df, model, name='ml', threshold_pct=5):
    """
        Run a scikit-learn anomaly model on trip-level features and add
        score + flag columns to the DataFrame.

        Works with any sklearn estimator that has fit() and score_samples().
        To switch algorithm, just pass a different model instance.

        Args:
            df: DataFrame with trip-level features broadcast to row level.
            model: sklearn estimator (e.g. IsolationForest, LocalOutlierFactor
                   with novelty=True).
            name: short identifier for the output columns ('if', 'lof', etc).
            threshold_pct: percentage of trips with the lowest scores to flag.

        Adds:
            score_<name>: anomaly score per row (lower = more anomalous).
            flag_<name>:  boolean, True for the bottom threshold_pct% of trips.
    """
    from sklearn.preprocessing import StandardScaler

    # ---- Build trip-level table (one row per trip) ----
    trips = (df.dropna(subset=['trip'])
               .drop_duplicates(['vehicleCode', 'trip', 'date'])
               [['vehicleCode', 'trip', 'date'] + ML_FEATURES]
               .reset_index(drop=True))

    # ---- Scale features ----
    # Some models (LOF, OneClassSVM) are sensitive to feature scale.
    # IF is not, but scaling never hurts and keeps the function generic.
    X = StandardScaler().fit_transform(trips[ML_FEATURES])

    # ---- Fit and score ----
    model.fit(X)
    trips[f'score_{name}'] = model.score_samples(X)

    # ---- Flag bottom threshold_pct% (most anomalous) ----
    cutoff = trips[f'score_{name}'].quantile(threshold_pct / 100)
    trips[f'flag_{name}'] = trips[f'score_{name}'] <= cutoff

    # ---- Broadcast back to row level ----
    df = df.merge(
        trips[['vehicleCode', 'trip', 'date', f'score_{name}', f'flag_{name}']],
        on=['vehicleCode', 'trip', 'date'],
        how='left'
    )
    df[f'flag_{name}'] = df[f'flag_{name}'].fillna(False)

    return df