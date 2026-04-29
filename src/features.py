import pandas as pd

def build_features(apc_clean):
    """
        Add derived features to a cleaned APC DataFrame.
        Each feature is added by a separate helper function.
    """

    df = apc_clean.copy()
    df = add_pism_features(df)
    df = add_trip_features(df)
    df = add_context_features(df)

    return df


def  add_pism_features(df):
    """
        Extract pointCode, line, trip, block, tripType from the
        pismInformation string into their own columns.
    """

    df['pointCode'] = df['pismInformation'].str.extract(r'pointCode:(\d+)').astype(float)
    df['block'] = df['pismInformation'].str.extract(r'block:(\d+)').astype(float)
    df['trip'] = df['pismInformation'].str.extract(r'trip:(\d+)').astype(float)
    df['line'] = df['pismInformation'].str.extract(r'line:(\d+)').astype(float)

    return df

def add_trip_features(df):
    """
        Add trip-level features by grouping on (vehicleCode, trip).
        Each feature is broadcast to every row in the same trip.
    """

    # Make sure rows are in chronological order within each trip
    df = df.sort_values(['vehicleCode', 'trip', 'arrival']).reset_index(drop=True)
    
    # Per-row load = running sum of (boardings - alightings) within each trip
    grp = df.groupby(['vehicleCode', 'trip'])

    df['load'] = grp.apply(
        lambda g: (g['boardings'] - g['alightings']).cumsum()
    ).reset_index(level=[0,1], drop=True)
    
    # Trip-level aggregates (broadcast to all rows in the trip)
    df['trip_total_boardings']  = grp['boardings'].transform('sum')
    df['trip_total_alightings'] = grp['alightings'].transform('sum')
    df['trip_imbalance']        = df['trip_total_boardings'] - df['trip_total_alightings']
    df['trip_n_stops']          = grp['arrival'].transform('count')
    df['trip_min_load']         = grp['load'].transform('min')
    df['trip_max_load']         = grp['load'].transform('max')
    df['trip_final_load']       = grp['load'].transform('last')
    
    return df

def add_context_features(df):
    """
    Add contextual features that compare each row/trip to what is "normal"
    in its context (stop, hour, line, weekday).
    """
    
    # ----- Helper columns -----
    df['arrival_dt'] = pd.to_datetime(df['arrival'], format='%d/%m/%Y %H:%M:%S')
    df['hour']       = df['arrival_dt'].dt.hour
    df['weekday']    = df['arrival_dt'].dt.dayofweek      # 0=Monday, 6=Sunday
    
    # ----- Feature 1: boardings z-score for (pointCode, hour, weekday) -----
    grp = df.groupby(['pointCode', 'hour', 'weekday'])
    df['boardings_mean']    = grp['boardings'].transform('mean')
    df['boardings_std']     = grp['boardings'].transform('std')
    df['boardings_z_score'] = (df['boardings'] - df['boardings_mean']) / df['boardings_std']
    df['boardings_z_score'] = df['boardings_z_score'].fillna(0).replace([float('inf'), float('-inf')], 0)
    
    # ----- Feature 2: trip_imbalance z-score per line -----
    # For each line, compute mean and std of trip_imbalance across all its trips.
    # Express each row's trip_imbalance as how many std-devs from its line's mean.
    
    grp_line = df.groupby('line')
    df['line_imbalance_mean']      = grp_line['trip_imbalance'].transform('mean')
    df['line_imbalance_std']       = grp_line['trip_imbalance'].transform('std')
    df['trip_imbalance_z_score']   = (df['trip_imbalance'] - df['line_imbalance_mean']) / df['line_imbalance_std']
    df['trip_imbalance_z_score']   = df['trip_imbalance_z_score'].fillna(0).replace([float('inf'), float('-inf')], 0)
    
    return df