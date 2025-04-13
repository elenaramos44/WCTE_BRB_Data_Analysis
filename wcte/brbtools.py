import numpy  as np
#import pandas as pd


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
