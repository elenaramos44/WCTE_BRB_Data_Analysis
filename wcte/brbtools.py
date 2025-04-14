import re
import glob
import uproot

from tqdm import tqdm

import numpy   as np
import pandas  as pd
import awkward as ak


def extract_p_number(filename):
    """
    Inputs a file path to the part file want to analyze:
    Returns part file number.
    Used to sort run_files in ascendent part_file number.
    Used to created the "parts_file" list of all the part files in the run.
    With that, you can then use select_good_parts function to keep just the parts with data.
    """
    match = re.search(r"P(\d+)", filename)
    if match:
        return int(match.group(1))
    return -1  

def sort_run_files(path):
    """
    Inputs a string with a path to a specific run
    Path example: f"/eos/experiment/wcte/data/2025_commissioning/offline_data/{run}/WCTE_offline_R{run}S*P*.root"
    Returns a list with the sorted by part_file number of files for the run.
    """
    run_files = glob.glob(path)
    return sorted(run_files, key=extract_p_number)

def get_part_files(run_files):
    """
    Inputs the run_files list created with sort_run_files.
    This creates the list with all the part_file numbers.
    Returns that list.
    """
    return [extract_p_number(file) for file in run_files]

def selectec_good_events(part_file, run_files):
    """
    DEPRECATED
    Input a part_file number.
    Used in the past to get only those events with the same number of hits than waveforms.
    Returns the event_id of those events.
    """
    tree = uproot.open(run_files[part_file]+":WCTEReadoutWindows")

    file_hit_card_ids      = tree["hit_mpmt_card_ids"].array()
    file_waveform_card_ids = tree["pmt_waveform_mpmt_card_ids"].array()

    num_hits = np.array([len(h) for h in file_hit_card_ids])
    num_wave = np.array([len(w) for w in file_waveform_card_ids])
    good_mask = (num_hits == num_wave) & (num_hits != 0)

    good_events = np.where(good_mask)[0]

    return good_events

def select_good_parts(parts, run_files):
    """
    Input the part_files numbers list created with get_part_files.
    Selects the part files that are not empty.
    Returns an updated list of usable part_files.
    """
    return [part_file for part_file in tqdm(parts, total=len(parts), desc="Selecting Good Parts") if np.any(uproot.open(run_files[part_file]+":WCTEReadoutWindows")["event_number"].array())]

def get_files_from_part(part_file, run_files):
    """
    Input a part_file number and the run_files list created with sort_run_files.
    Extracts all the card_id, channel_id, charge and time information from the hits.
    Returns the arrays for the variables for the part_file.
    """
    tree = uproot.open(run_files[part_file]+":WCTEReadoutWindows")
    file_hit_card_ids    = tree["hit_mpmt_card_ids"].array()
    file_hit_channel_ids = tree["hit_pmt_channel_ids"].array()
    file_hit_charges     = tree["hit_pmt_charges"].array()
    file_hit_times       = tree["hit_pmt_times"].array()
    file_window_times    = tree["window_time"].array()
    file_event_number    = tree["event_number"].array()

    return file_hit_card_ids, file_hit_channel_ids, file_hit_charges, file_hit_times, file_window_times, file_event_number
 
def create_df_from_file(files, part):
    """
    Input the information created with get_files_from_part.
    Creates a Pandas DataFrame with the event, card, channel, charge and time information.
    Returns the DataFrame.
    """
    cards        = files[0]
    channels     = files[1]
    charges      = files[2]
    hit_times    = files[3]
    window_times = files[4]
    event_number = files[5]

    nevents = len(cards)
    evts    = np.arange(nevents)
    nhits   = [len(cards[ievt]) for ievt in evts]
    evt_column = np.repeat(evts, nhits)
    window_time_column = np.repeat(window_times, nhits)
    event_number_column = np.repeat(event_number, nhits)
    part_number_column = np.repeat(part, np.sum(nhits))

    xcards    = ak.flatten(cards)
    xchannels = ak.flatten(channels)
    xcharges  = ak.flatten(charges)
    xtimes    = ak.flatten(hit_times)
    df = pd.DataFrame({'evt':evt_column, "part_number": part_number_column,"event_number":event_number_column, "window_time":window_time_column, 
                       'card':xcards, 'channel':xchannels, 'charge':xcharges, "time":xtimes})
    return df, len(event_number)

def df_event_summary(df, map):
    """ 
    returns a DataFrame with one entry per event and the number of hits and total charge
    for the ids (each id is a pair (card ID, channel ID)).
    inputs:
        df   : DataFrame with columns ‘event’, ‘card’, ‘channel’, ‘charge’ and "time".
        ids  : list of pairs (card, channel) i.e [(0, 0), (0, 1)].
        map  : a dictionary that maps every card channel to its detector name. Example: {(130, 0): "ACT0-L", (130, 1): "ACT0-R"}
    returns:
        DataFrame with columnes ‘event’ ‘name_of_card_channel_nhits’, ‘name_of_card_channel_charge’, ‘name_of_card_channel_time’
    
    """
    nevts = np.max(df.evt) + 1

    xdf   = {"evt" : np.arange(nevts),  
             "event_number":np.unique(df["event_number"]), 
             "window_time":np.unique(df["window_time"])}
    
    for id in map.keys():
        card, channel = id
        sel   = (df.card == card) & (df.channel == channel)
        ievts = df[sel].evt.unique()

        _groups =  df[sel].groupby("evt")
        
        nhits         = _groups.count()["channel"]
        xhits         = np.empty(nevts)
        xhits[:]      = np.nan
        xhits[ievts]  = nhits
        qtots         = _groups.sum()["charge"]
        xqtots        = np.empty(nevts)
        xqtots[:]     = np.nan
        xqtots[ievts] = qtots
        ttots         = _groups.mean()["time"]
        xttots        = np.empty(nevts)
        xttots[:]     = np.nan
        xttots[ievts] = ttots
        """WARNING!
        Now all zeroes generated in the DataFrame due to a hit non-existing in that event for that channel
        is converted into a nan value. This COULD cause some issues that can be easily solved if you follow
        instructions at the end of df_extend.
        """

        name = map[(card, channel)]
        xdf[name+"_nhits"]  = xhits
        xdf[name+"_charge"] = xqtots
        xdf[name+"_time"]   = xttots

    return pd.DataFrame(xdf)

def concat_dfs(parts, run_files):
    dfs = []
    evt_offset = 0
    event_number_offset = 0
    parts_length = []

    for ipar in tqdm(parts, total=len(parts), desc="Creating DataFrames For Each Part"):
        files            = get_files_from_part(ipar, run_files)
        df, part_length  = create_df_from_file(files, ipar)

        df['evt'] += evt_offset
        df["event_number"] += event_number_offset
        evt_offset = df['evt'].max() + 1  
        event_number_offset = df["event_number"].max() +1
        parts_length.append(part_length)
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True), parts_length

def create_big_df(parts):
    df_concat, parts_length = concat_dfs(parts)
    parts_id = np.repeat(parts, parts_length)
    # Create final df
    df_all = df_event_summary(df_concat)

    # Add Column
    df_all["part_id"] = parts_id

    # Relocate Column
    cols = df_all.columns.tolist()
    cols.insert(3, cols.pop(cols.index('part_id')))
    df_all = df_all[cols]

    return df_all

def df_extend(df, numACTs):
    """ input a BRB-Beam ntuple and extend it: 
        it computes the ACT charge in each box and in the to
        it computes the time in T0, T1 and the T1-T0 time difference
    return extended dt
    """

    def _operate(sens, vars, oper = np.nansum):
        labs = [sen + var for var in vars for sen in sens]
        vv = oper([df[lab].values for lab in labs], axis = 0)
        return vv
    
    df['T0_time'] = _operate(['T0-0','T0-1'], ['L_time','R_time'], np.nanmean)
    df['T1_time'] = _operate(['T1-0','T1-1'], ['L_time','R_time'], np.nanmean)
    df['T1-T0_time'] = df['T1_time'] - df['T0_time']

    for i in range(numACTs):
        df['ACT'+str(i)+'_charge'] = _operate(['ACT'+str(i)+'-',], ['L_charge', 'R_charge'], np.nansum)

    df['ACT_g1_charge'] = np.sum([df['ACT'+str(i)+'_charge'] for i in (0, 1, 2)], axis = 0)
    df['ACT_g2_charge'] = np.sum([df['ACT'+str(i)+'_charge'] for i in (3, 4, 5)], axis = 0)

    tof_keys = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'A', 'B', 'C', 'D', 'E', 'F']
    df['TOF_nhits'] = _operate(['TOF-'+str(k) for k in tof_keys], ['_nhits', ], np.nansum)
    """ WARNING!
    Now that all the zeros have been converted into nan values, when we perform a numpy nanOperation
    in a full nan array, it returns a nan value, so then when you filter one of these windows and try
    to plot it with hist2d you will get an error.
    You can check wether your filtered variables have nan values doing:
    print("variable with NaN:", np.isnan(variable).sum())
    Then you can mask your NaN values doing:
    mask = np.isfinite(variable1) & np.isfinite(variable2)
    masked_variable1 = variable1[mask]
    masked_variable2 = variable2[mask]
    And then plot.
    """

    return df
