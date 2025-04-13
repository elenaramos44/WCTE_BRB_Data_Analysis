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
    return [part_file for part_file in tqdm(parts, total=len(parts)) if np.any(uproot.open(run_files[part_file]+":WCTEReadoutWindows")["event_number"].array())]

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

    return file_hit_card_ids, file_hit_channel_ids, file_hit_charges, file_hit_times

def create_df_from_file(files):
    """
    Input the information created with get_files_from_part.
    Creates a Pandas DataFrame with the event, card, channel, charge and time information.
    Returns the DataFrame.
    """
    cards    = files[0]
    channels = files[1]
    charges  = files[2]
    times    = files[3]

    nevents = len(cards)
    evts    = np.arange(nevents)
    nhits   = [len(cards[ievt]) for ievt in evts]
    evt_column = np.repeat(evts, nhits)

    xcards    = ak.flatten(cards)
    xchannels = ak.flatten(channels)
    xcharges  = ak.flatten(charges)
    xtimes    = ak.flatten(times)
    df = pd.DataFrame({'evt':evt_column, 'card':xcards, 'channel':xchannels, 'charge':xcharges, "time":xtimes})
    return df

def df_event_summary(df, ids, map):
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
    xdf   = {"evt" : np.arange(nevts)}

    for  id in ids:
        card, channel = id
        sel   = (df.card == card) & (df.channel == channel)
        ievts = df[sel].evt.unique()

        _groups =  df[sel].groupby("evt")
        
        nhits         = _groups.count()["channel"]
        xhits         = np.zeros(nevts, int)
        xhits[ievts]  = nhits
        qtots         = _groups.sum()["charge"]
        xqtots        = np.zeros(nevts, float)
        xqtots[ievts] = qtots
        ttots         = _groups.mean()["time"]
        xttots        = np.zeros(nevts, float)
        xttots[ievts] = ttots

        name = map[(card, channel)]
        xdf[name+"_nhits"]  = xhits
        xdf[name+"_charge"] = xqtots
        xdf[name+"_time"]   = xttots
    
    return pd.DataFrame(xdf)

def concat_dfs(good_parts):
    """
    Inputs the good_parts list.
    Creates the DataFrame for every part_file using create_df_from_file and get_files_from_part.
    Concatenates every histogram updating the event_id so if we have N events event_id goes up to N.
    Returns the concatenated DataFrame.
    """
    dfs = []
    evt_offset = 0

    for ipar in tqdm(good_parts, total=len(good_parts)):
        df = create_df_from_file(get_files_from_part(ipar))
        df['evt'] += evt_offset
        evt_offset = df['evt'].max() + 1  
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)

def create_df_all(df_concat, map):
    """
    Inputs the DataFrame with all concatenated part_file information and the channel mapping.
    Returns a big DataFrame with all the info from all the channels.
    """
    return df_event_summary(df_concat, map.keys())

def df_extend(df):
    """ input a BRB-Beam ntuple and extend it: 
        it computes the ACT charge in each box and in the to
        it computes the time in T0, T1 and the T1-T0 time difference
    return extended dt
    """

    def _operate(sens, vars, oper = np.sum):
        labs = [sen + var for var in vars for sen in sens]
        vv = oper([df[lab].values for lab in labs], axis = 0)
        return vv
    
    df['T0_time'] = _operate(['T0-0','T0-1'], ['L_time',], np.mean) ## ??? WHY THERE is not R_time ???
    df['T1_time'] = _operate(['T1-0','T1-1'], ['L_time','R_time'], np.mean)
    df['T1-T0_time'] = df['T1_time'] - df['T0_time']

    for i in range(6):
        df['ACT'+str(i)+'_charge'] = _operate(['ACT'+str(i)+'-',], ['L_charge', 'R_charge'], np.mean)

    df['ACT_g1_charge'] = np.sum([df['ACT'+str(i)+'_charge'] for i in (0, 1, 2)], axis = 0)
    df['ACT_g2_charge'] = np.sum([df['ACT'+str(i)+'_charge'] for i in (3, 4, 5)], axis = 0)

    tof_keys = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'A', 'B', 'C', 'D', 'E', 'F']
    df['TOF_nhits'] = _operate(['TOF-'+str(k) for k in tof_keys], ['_nhits', ], np.sum)

    df['T4-RL'] = ((df['T4-R_nhits'] == 1) & (df['T4-L_nhits'] == 1))

    return df
