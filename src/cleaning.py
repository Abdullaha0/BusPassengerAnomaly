import pandas as pd

"""
############### How to use - Example ##############

import sys
sys.path.append('..')

from src.cleaning import clean_apc

apc = clean_apc('../data/raw/apc.csv')

"""

def clean_apc(filepath):
    apc = pd.read_csv(filepath, sep=";", encoding='latin-1')

    columns_to_drop = [
    'fileName',                   # Doesn't help
    'stationName',                # always empty
    'stationCode',                # always empty
    'gpsDeviation',               # always '0 m'
    'isGpsMapped',                # always False
    'wheelchairRampAccessCount',  # always 0
    'isBicycleRackAccessed',      # always False
    'area',                       # always empty
    'netstop_latitude',           # always empty
    'netstop_longitude',          # always empty
    ]

    # Drop chosen columns in 'columns_to_drop'
    apc_clean = apc.drop(columns= columns_to_drop)
    # Drop duplicated rows
    apc_clean = apc_clean.drop_duplicates().reset_index(drop=True)
    # Drop empty rows
    apc_clean = apc_clean.dropna(how='all').reset_index(drop=True)

    print("before", apc.shape)
    print("after", apc_clean.shape)

    apc_clean.to_csv('../data/processed/apc_clean.csv', sep=';', index=False, encoding='latin-1')

    print("Saved to ../data/processed/")

    return apc_clean
    