print("Some necessart imports...")
import sys
import os
import argparse
import gc

# Añade el directorio padre al sys.path
sys.path.append(os.path.abspath("/eos/home-d/dcostasr/SWAN_projects/2025_data"))
sys.path.append('/eos/home-d/dcostasr/SWAN_projects/software/hipy')

from wcte.brbtools import sort_run_files, get_part_files, select_good_parts
from wcte.brbtools import concat_dfs, df_extend, filter_times_and_relocate_columns, df_mpmt_sumCharge
from wcte.mapping  import id_names

print("Creating argparser...")
parser = argparse.ArgumentParser()
parser.add_argument('run', type=int, help='Run number. Required.')
args = parser.parse_args()

run        = args.run
map        = id_names
run_files  = sort_run_files(f"/eos/experiment/wcte/data/2025_commissioning/offline_data/{run}/WCTE_offline_R{run}S*P*.root")
part_files = get_part_files(run_files)
good_parts = select_good_parts(part_files, run_files)

# Beam Monitors DF and Info
print("1")
df_all, hits_df = concat_dfs(good_parts, run_files, map)
print("2")
df_good_times, good_events = filter_times_and_relocate_columns(df_all)

# Free memory
del df_all
gc.collect()

print("3")
df_extended = df_extend(df_good_times, 6)
del df_good_times
gc.collect()

output_dir = f"/eos/home-d/dcostasr/SWAN_projects/2025_data/data/{run}"
os.makedirs(output_dir, exist_ok=True)
df_extended.to_parquet(path=f"{output_dir}/df_beam.parquet", index=None)
del df_extended
gc.collect()

# mPMTs DF
# DataFrame With Cards, Channels, Charge and Time. This is also important to have.
print("4")
hits_df_good_events = hits_df[hits_df["evt"].isin(good_events)]
del hits_df
gc.collect()

output_dir = f"/eos/home-d/dcostasr/SWAN_projects/2025_data/data/{run}"
os.makedirs(output_dir, exist_ok=True)
hits_df_good_events.to_parquet(path=f"{output_dir}/df_hits.parquet", index=None)

# DataFrame With mPMTs information.
print("5")
df_mpts = df_mpmt_sumCharge(hits_df_good_events)
hits_df_good_events = None  
gc.collect()

output_dir = f"/eos/home-d/dcostasr/SWAN_projects/2025_data/data/{run}"
os.makedirs(output_dir, exist_ok=True)
df_mpts.to_parquet(path=f"{output_dir}/df_mPMTs.parquet", index=None)