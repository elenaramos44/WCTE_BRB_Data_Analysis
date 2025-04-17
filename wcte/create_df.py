
import sys
import os
import argparse

# Añade el directorio padre al sys.path
sys.path.append(os.path.abspath("/eos/home-d/dcostasr/SWAN_projects/2025_data"))
sys.path.append('/eos/home-d/dcostasr/SWAN_projects/software/hipy')

from wcte.brbtools import sort_run_files, get_part_files, select_good_parts
from wcte.brbtools import concat_dfs, df_extend, full_df_mPMT, df_mpmt_sumCharge
from wcte.mapping  import id_names


parser = argparse.ArgumentParser()
parser.add_argument('run', type=int, help='Run number. Required.')
args = parser.parse_args()

run        = args.run
map        = id_names
run_files  = sort_run_files(f"/eos/experiment/wcte/data/2025_commissioning/offline_data/{run}/WCTE_offline_R{run}S*P*.root")
part_files = get_part_files(run_files)
good_parts = select_good_parts(part_files, run_files)

# Beam Monitors DF and Info
df_all = concat_dfs(good_parts, run_files, map)

cols = df_all.columns.tolist()
cols.insert(1, cols.pop(cols.index('evt_number')))
cols.insert(2, cols.pop(cols.index('window_time')))
cols.insert(3, cols.pop(cols.index('part_number')))
df_all = df_all[cols]

f = df_all[(df_all["T0-0L_time"].values != 0) &
           (df_all["T0-0R_time"].values != 0) &
           (df_all["T0-1L_time"].values != 0) &
           (df_all["T0-1R_time"].values != 0) &
           (df_all["T1-0L_time"].values != 0) &
           (df_all["T1-0R_time"].values != 0) &
           (df_all["T1-1L_time"].values != 0) &
           (df_all["T1-1R_time"].values != 0)].copy()

df_extended = df_extend(f, 6)
output_dir = f"/eos/home-d/dcostasr/SWAN_projects/2025_data/data/{run}"
os.makedirs(output_dir, exist_ok=True)
df_extended.to_parquet(path=f"{output_dir}/df_extended_goodTime.parquet", index=None)

# mPMTs DF
# DataFrame With Cards, Channels, Charge and Time. This is also important to have.
df_concat = full_df_mPMT(good_parts, run_files)
output_dir = f"/eos/home-d/dcostasr/SWAN_projects/2025_data/data/{run}"
os.makedirs(output_dir, exist_ok=True)
df_concat.to_parquet(path=f"{output_dir}/df_concat.parquet", index=None)

# DataFrame With mPMTs information.
df_mpts = df_mpmt_sumCharge(df_concat)
output_dir = f"/eos/home-d/dcostasr/SWAN_projects/2025_data/data/{run}"
os.makedirs(output_dir, exist_ok=True)
df_mpts.to_parquet(path=f"{output_dir}/df_mPMTs.parquet", index=None)