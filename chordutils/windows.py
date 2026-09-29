"""Look-up of timeline events in a window around an index event (legacy helpers).

``find_stagings``, ``find_ps`` and ``find_markers`` summarise imaging, ECOG and tumour-marker
records before, after or during an event (surgery, radiation, therapy episode). They return
one-row DataFrames with colon-joined strings and are kept for backwards compatibility with
the original PDAC analysis notebook.
"""

import re
import numbers
import numpy as np
import pandas as pd
from typing import Union, Sequence
from collections.abc import Iterable
from collections import defaultdict

def find_stagings(stagings:pd.DataFrame,
                  event:Union[int, Sequence[int]],  
                  offset:int=0, 
                  staging_time_col:str = "START_DATE", 
                  cancer_col:str="HAS_CANCER", 
                  progression_col:str = "PROGRESSION",
                  procedure_col:str = "PROCEDURE_TYPE",
                  site_col:str = "TUMOR_SITE"):

    """The indended use is to find all staging examinations for a given treatment event, be it surgery, radiation or medical therapy. 

    Keyword arguments:
    stagings -- A DataFrame containing pertinent staging examinations
    event -- Integer or sequence of integery specifying the time point or time range of treatment (or surgery or radiation in case of time point)
    offset -- Integer specifying the amount of offset in either direction (negative = before, positive = after) of the event.
    staging_time_col -- String to access the staging event time column (default START_DATE)
    cancer_col -- String to access the column that carries info on whether cancer was detected (default HAS_CANCER)
    progression_col -- String to access the column that carries info on whether progression was evident (default PROGRESSION)
    procedure_col -- String to access the staging procedure (default PROCEDURE_TYPE)
    site_col -- String to access the staging tumor sites (default TUMOR_SITE)
    """

    time_range = gen_time_range(event, offset)

    off_type = determine_offset_combo(event, offset)

    col_names = [f"n_stagings_{off_type}", f"n_hascancer_{off_type}", f"n_progression_{off_type}", 
                f"stagings_{off_type}", f'stagings_hascancer_{off_type}', f"hascancer_dates_{off_type}",
                f"stagings_progression_{off_type}", f"progression_dates_{off_type}", f"staging_tumor_sites_{off_type}",  
                f"staging_tumor_sites_dates_{off_type}"]

    suitable = stagings[np.isin(stagings[staging_time_col], time_range)]

    has_cancer = 0
    progression = 0
    all_stagings = []
    has_cancer_stagings = []
    has_cancer_dates = []
    progression_stagings = []
    progression_dates = []
    staging_sites = []
    staging_dates = []

    for tp in suitable.itertuples():

        modality = getattr(tp, procedure_col)
        all_stagings.append(modality)

        if getattr(tp, cancer_col) == "Y":
            has_cancer += 1
            has_cancer_stagings.append(modality)
            has_cancer_dates.append(str(getattr(tp, staging_time_col)))

        if getattr(tp, progression_col) == "Y":
            progression += 1
            progression_stagings.append(modality)
            progression_dates.append(str(getattr(tp, staging_time_col)))

        if pd.notnull( (sites:=getattr(tp, site_col)) ):
            staging_sites.append(sites)
            staging_dates.append(str(getattr(tp, staging_time_col)))


    staging_infos = []

    for st in [all_stagings, has_cancer_stagings, has_cancer_dates, progression_stagings, progression_dates, staging_sites, staging_dates]:

        if len(st)>0:
            staging_infos.append(":".join(st))

        else:
            staging_infos.append("NONE")

    out = [len(suitable), has_cancer, progression, *staging_infos]

    return pd.DataFrame(out, index=col_names).T


def find_ps(ps:pd.DataFrame,
                  event:Union[int, Sequence[int]],
                  offset:int=0, 
                  time_col:str = "START_DATE",
                  data_col:str = "ECOG"):

    """The indended use is to find all performance status values for given time event (usually treatment, surgery or radiation) 

    Keyword arguments:
    stagings -- A DataFrame containing pertinent performance status recordings
    event -- Integer or sequence of integery specifying the time point or time range of treatment (or surgery or radiation in case of time point)
    offset -- Integer specifying the amount of offset in either direction (negative = before, positive = after) of the event. If the event
              is a sequence of integers, the offset will be subtracted from the first value and added to the second of value
    time_col -- String to access the staging event time column (default START_DATE)
    data_col -- String to access the performances statu column (default ECOG)
    
    """

    time_range = gen_time_range(event, offset)

    off_type = determine_offset_combo(event, offset)

    col_names = [f"n_ps_{off_type}", f"ps_values_{off_type}", f"ps_dates_{off_type}"]

    suitable = ps[np.isin(ps[time_col], time_range)]

    ps_list = []

    ps_dates = []

    for tp in suitable.itertuples():

        ps_list.append(str(getattr(tp, data_col)))

        ps_dates.append(str(getattr(tp, time_col)))
  
    ps_infos = []

    for psi in [ps_list, ps_dates]:

        if len(psi)>0:
            ps_infos.append(":".join(psi))

        else:
            ps_infos.append("NONE")

    out = [len(suitable), *ps_infos]

    return pd.DataFrame(out, index=col_names).T

def find_markers(tm:pd.DataFrame,
                 event:Union[int, Sequence[int]],
                 *args,
                 offset:int = 0, 
                 time_col:str = "START_DATE",
                 marker_name_col:str = "TEST",
                 data_col:str = "RESULT"):

    """The intended use is to find all performance status values for given time event (usually treatment, surgery or radiation) 

    Keyword arguments:
    tm -- A DataFrame containing pertinent tumor marker recordings
    event -- Integer or sequence of integers specifying the time point or time range of treatment (or surgery or radiation in case of time point)
    *args -- Variable strings of tumor marker names (e.g. CA19-9 (U/mL), CEA (ng/mL), ...).
    offset -- Integer specifying the amount of offset in either direction (negative = before, positive = after) of the event. If the event
              is a sequence of integers, the offset will be subtracted from the first value and added to the second of value
    marker_name_col -- String to access the column (default TEST)
    time_col -- String to access the staging event time column (default START_DATE_tumor_marker)
    data_col -- String to access the numeric values column (default RESULT)
    """

    time_range = gen_time_range(event, offset)


    off_type = determine_offset_combo(event, offset)

    suitable = tm[np.isin(tm[time_col], time_range)]


    marker_values = defaultdict(list)

    replacements = {"(":"", ")":"", "/":"", " ":"_"} 

    if not args:
        
        raise ValueError("We need strings reflecting tumor marker names here!")

    for arg in args:

        if not arg in suitable[marker_name_col].to_list():

            print(f"{arg} is not a valid tumor marker name")

        else:

            suitable_sub = suitable.query(f"{marker_name_col}=='{arg}'")

            for tp in suitable_sub.itertuples():

                val = getattr(tp, data_col)
                date = getattr(tp, time_col)

                arg_new = multiple_replace(replacements, arg)

                marker_values[arg_new].append(f"{val}|{date}")

    col_names = []

    for k in marker_values:

        for col_name in ['n', 'values|dates']:

            col_names.append(f"{k}_{col_name}_{off_type}")

    marker_infos = []

    for k in marker_values:

        vals = marker_values.get(k)

        marker_infos.append(len(vals))

        if len(vals)>0:
            marker_infos.append(":".join(vals))
        else:
            marker_infos.append("NONE")

    return pd.DataFrame(marker_infos, index=col_names).T



def gen_time_range(event, offset):

    """Helper function to generate an array of integers, i.e. a time range, which can be checked for overlap with other time stamps or ranges.""" 

    if not validate_event(event):

        raise ValueError(f"Event input is not suitable. Need integer or sequence of two integers, got {event!r}")

    if _is_int(event):
        vals = np.array(sorted([event + offset, event]))
    else: 
        # if event is a sequence of integers, offset will subtracted from value 1 (start) and added to value 2 (stop)
        start, stop = event
        offset = np.abs(offset) #in case some passes a negative value which is not feasible here
        vals = np.array([start - offset, stop + offset])
    
    vals[1] = vals[1] + 1 #last day should be included

    return np.arange(*vals)


def determine_offset_combo(event, offset):

    """Helper function to determine the type of event interval (prior, after) for single integers, (during) for sequence of two intergers"""

    if not validate_event(event):

        raise ValueError(f"Event input is not suitable. Need integer or sequence of two integers, got {event!r}")

    if _is_int(event):
        offset_type = 'prior' if np.sign(offset)==-1 else 'after'
        offset = np.abs(offset)
        offset_str = f"{offset}d_{offset_type}"
    else:
        offset_type = 'during'
        offset = np.abs(offset)
        offset_str = f"during_{event[0]:d}_to_{event[1]:d}_with_{offset}_offset"
  
    return offset_str


def _is_int(x) -> bool:

    """True for Python and NumPy integers (e.g. values taken from a DataFrame are np.int64), False for bools."""

    return isinstance(x, numbers.Integral) and not isinstance(x, (bool, np.bool_))


def validate_event(event: Union[int, Sequence[int]]):

    """Helper to validate input for finding stagings before, after or during an event or time range, respectively"""

    if isinstance(event, Iterable):
        if len(event)==2:
            if all(_is_int(i) for i in event):
                return True
            else:
                return False
        else:
            return False    
    else:
        if _is_int(event):
            return True
        else:
            return False


def multiple_replace(replacements, text):

    """Helper function from Python cookbook for multiple replacements using a mapping dictinonary"""
    # Create a regular expression from the dictionary keys
    regex = re.compile(f"{'|'.join(map(re.escape, replacements.keys()))}")
    # For each match, look-up corresponding value in dictionary
    return regex.sub(lambda mo: replacements[mo.group()], text)
