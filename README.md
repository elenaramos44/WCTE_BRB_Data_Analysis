# WCTE_BRB_Data_Analysis

## To Do
- [ ] Update mapping so it includes the rest of the cards that are mPMTs
- [ ] Update the creation of the part_file DataFrame (and hence the df_all big DataFrame) so it includes all charge (summed) seen by the mPMTs per event
- [ ] Add plotting functions. For example, plotting the charge of given channel, group, etc, same for time and 2D Histograms
- [X] Optimize appending window_time, event_number and part_file. Now it is done generating a event_summary DataFrame for every part and the concatenating the DFs. This is very memory consuming since you need to create the "compressed" event_summary DataFrame for every part. The goal would be to add those 3 variables to df_from_file for every part, concatenate and then run event_summary just once: TRIED AND IT'S SLOWER (Maybe I'm doing something wrong...
- [ ] Tools that create the same kind of DataFrame but for the rest of the mPMTs. 

