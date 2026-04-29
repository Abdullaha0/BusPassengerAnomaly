import pandas as pd

def build_features(apc_clean):
    """
        Add derived features to a cleaned APC DataFrame.
        Each feature is added by a separate helper function.
    """

    df = apc_clean.copy()

    # Each helper adds one group of features
    df = add_pism_features(df)

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

