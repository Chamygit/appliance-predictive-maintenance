# -----------------------------------------------------------
# Script used for extracting features from time series data measured from home appliances
#
# (C) 2022 Tiago Fonseca, Porto, Portugal
# This work was supported by project SMART-PDM https://smart-pdm.eu/
# Released under GNU Public License (GPL)
# email calof@isep.ipp.pt
# -----------------------------------------------------------

import pandas as pd
# Alias the import to avoid shadowing the local function name
from tsfresh import extract_features as tsfresh_extract_features

# Features selected for extraction.
# See https://tsfresh.readthedocs.io/en/latest/text/introduction.html
fc_parameters = {
    "variance_larger_than_standard_deviation": None,
    "sum_values": None,
    "mean_abs_change": None,
    "mean_change": None,
    "median": None,
    "mean": None,
    "length": None,
    "standard_deviation": None,
    "variation_coefficient": None,
    "variance": None,
    "skewness": None,
    "kurtosis": None,
    "root_mean_square": None,
    "count_above_mean": None,
    "count_below_mean": None,
    "maximum": None,
    "absolute_maximum": None,
    "minimum": None,
    "number_peaks": [{"n": 5}],
    "fft_aggregated": [
        {"aggtype": "centroid"},
        {"aggtype": "variance"},
        {"aggtype": "skew"},
        {"aggtype": "kurtosis"},
    ],
    "fourier_entropy": [
        {"bins": 5},
        {"bins": 10},
    ],
}


def extract_cycle_features(df):
    """
    Extracts statistical and frequency-domain features for each cycle using tsfresh.

    :param df: time-series dataframe with columns:
        - "Id" : cycle identifier (e.g. begin_end timestamp string)
        - "Ts" : datetime of the measurement
        - additional sensor columns (any name)
    :type df: pandas.DataFrame
    :return: dataframe with one row per cycle Id and one column per extracted feature
    :rtype: pandas.DataFrame
    """
    return tsfresh_extract_features(
        df,
        column_id="Id",
        column_sort="Ts",
        default_fc_parameters=fc_parameters,
    )
