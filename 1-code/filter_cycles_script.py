# -----------------------------------------------------------
# Script used for identifying the begin and end time of home appliances working cycles
# Please contact the team to request raw files
#
# (C) 2022 Tiago Fonseca, Porto, Portugal
# This work was supported by project SMART-PDM https://smart-pdm.eu/
# Released under GNU Public License (GPL)
# email calof@isep.ipp.pt
# -----------------------------------------------------------

import pandas as pd

MINIMUM_MARGIN = 720  # 12 minutes minimum cycle duration (in seconds)


def identify_cycles_from_raw_slow_stream(df):
    """
    Identifies the begin and end times of cycles given a dataframe containing
    the raw slow stream.

    :param df: the slow stream to identify the cycles of. A time series with columns:
        - "Ts"   : datetime of the measurement
        - "ActP" : Active Power measurement (Watts)
    :type df: pandas.DataFrame
    :return: list of (begin_ts, end_ts) pairs for each detected cycle
    :rtype: list[tuple]
    """
    lst_cycles = []

    flag = 0
    ignore_counter = 0
    increase_threshold = 10   # W — power must exceed this to start a cycle
    decrease_threshold = 3    # W — power must stay below this for `debounce` seconds to end
    aux_begin_date = None
    aux_end_date = None

    for _, row in df.iterrows():
        if row["ActP"] > increase_threshold and flag == 0:
            flag = 1
            ignore_counter = 0
            aux_begin_date = row["Ts"]
        elif flag == 1:
            if row["ActP"] < decrease_threshold:
                if ignore_counter >= 30:
                    flag = 0
                    aux_end_date = row["Ts"]
                    if (aux_end_date - aux_begin_date) >= MINIMUM_MARGIN:
                        lst_cycles.append((aux_begin_date, aux_end_date))
                else:
                    ignore_counter += 1
            else:
                # Power recovered — reset the debounce counter
                ignore_counter = 0

    return lst_cycles
