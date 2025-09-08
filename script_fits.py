import numpy as np
import os
from scipy.optimize import minimize
from tqdm import tqdm 

# --- Directories for signal and control runs ---
signal_dir = "/scratch/elena/waveform_npz/run2521/"
control_dir = "/scratch/elena/waveform_npz/run2519/"

signal_files = [f for f in os.listdir(signal_dir) if f.endswith(".npz")]
control_files = [f for f in os.listdir(control_dir) if f.endswith(".npz")]

signal_pmts = set(f.replace(".npz", "") for f in signal_files)
control_pmts = set(f.replace(".npz", "") for f in control_files)

pmts_both = sorted(list(signal_pmts & control_pmts))
print(f"PMTs found in both runs: {len(pmts_both)}")

# --- Parameters for integration windows ---
pre_peak   = 4   # bins before max (for SPE)
post_peak  = 2   # bins after max (for SPE)
ped_window = 4   # symmetric window width (for pedestal)

# --- Functions ---

def load_waveforms(npz_file):
    data = np.load(npz_file)
    return data["waveforms"]

def integrate_waveform_signal(wf, pre_peak=pre_peak, post_peak=post_peak):
    """Integrate waveform around max sample with asymmetric window (signal run)."""
    peak_idx = np.argmax(wf)
    start = max(0, peak_idx - pre_peak)
    end   = min(len(wf), peak_idx + post_peak + 1)
    return np.sum(wf[start:end])

def integrate_waveform_control(wf, window=ped_window):
    """Integrate waveform with symmetric window around max sample (control run)."""
    peak_idx = np.argmax(wf)  # in control this is random noise, but ok
    half_w = window // 2
    start = max(0, peak_idx - half_w)
    end   = min(len(wf), peak_idx + half_w)
    return np.sum(wf[start:end])

def compute_charges_signal(waveforms, pre_peak=pre_peak, post_peak=post_peak):
    return np.array([integrate_waveform_signal(wf, pre_peak=pre_peak, post_peak=post_peak) for wf in waveforms])

def compute_charges_control(waveforms, window=ped_window):
    return np.array([integrate_waveform_control(wf, window=window) for wf in waveforms])

def find_modal_window(charges, adc_range=None, bins=200, threshold_frac=0.2, min_width=20):
    if adc_range is not None:
        lo_cut, hi_cut = adc_range
        charges = np.array(charges)
        charges = charges[(charges > lo_cut) & (charges < hi_cut)]
    
    hist, edges = np.histogram(charges, bins=bins)
    bin_centers = 0.5*(edges[1:] + edges[:-1])
    
    max_bin = np.argmax(hist)
    mode_est = bin_centers[max_bin]
    
    threshold = threshold_frac * hist[max_bin]
    left = max_bin
    right = max_bin
    while left > 0 and hist[left] > threshold:
        left -= 1
    while right < len(hist)-1 and hist[right] > threshold:
        right += 1
        
    lo = bin_centers[left]
    hi = bin_centers[right]
    if hi - lo < min_width:
        hi = lo + min_width
    
    selected = charges[(charges > lo) & (charges < hi)]
    return mode_est, lo, hi, selected

def fit_gaussian_unbinned(data):
    data = np.array(data)
    if len(data) < 10:
        return np.nan, np.nan, False
    
    def nll(params):
        mu, sigma = params
        if sigma <= 0:
            return np.inf
        return 0.5 * np.sum(((data - mu)/sigma)**2 + np.log(2*np.pi*sigma**2))
    
    mu0, sigma0 = np.mean(data), np.std(data)
    result = minimize(nll, x0=[mu0, sigma0], method="Nelder-Mead")
    
    if result.success:
        mu_fit, sigma_fit = result.x
        return mu_fit, sigma_fit, True
    else:
        return mu0, sigma0, False

# --- Loop over PMTs and calculate gain ---
results_list = []

for pmt_label in tqdm(pmts_both, desc="Processing PMTs"):
    # Extract card, slot, channel
    card_id = int(pmt_label.split("_")[0][4:])
    slot_id = int(pmt_label.split("_")[1][4:])
    channel_id = int(pmt_label.split("_")[2][2:])
    
    signal_npz = os.path.join(signal_dir, pmt_label + ".npz")
    control_npz = os.path.join(control_dir, pmt_label + ".npz")
    
    signal_waveforms = load_waveforms(signal_npz)
    control_waveforms = load_waveforms(control_npz)
    
    # Compute charges
    charges_signal = compute_charges_signal(signal_waveforms, pre_peak=pre_peak, post_peak=post_peak)
    charges_control = compute_charges_control(control_waveforms, window=ped_window)
    
    # Pedestal
    ped_mode, ped_lo, ped_hi, ped_sel = find_modal_window(charges_control, adc_range=(-50,50))
    mu_ped, sigma_ped, ok_ped = fit_gaussian_unbinned(ped_sel)
    
    # SPE
    spe_mode, spe_lo, spe_hi, spe_sel = find_modal_window(charges_signal, adc_range=(80,220))
    mu_spe, sigma_spe, ok_spe = fit_gaussian_unbinned(spe_sel)
    
    # Gain & error
    gain = mu_spe - mu_ped
    err_gain = np.sqrt( (sigma_ped/np.sqrt(len(ped_sel)))**2 + (sigma_spe/np.sqrt(len(spe_sel)))**2 )
    
    results_list.append((
        card_id, slot_id, channel_id,
        mu_ped, sigma_ped, len(ped_sel),
        mu_spe, sigma_spe, len(spe_sel),
        gain, err_gain
    ))

# --- Save results as structured npz ---
dtype = np.dtype([
    ('card_id', 'i4'), ('slot_id', 'i4'), ('channel_id', 'i4'),
    ('pedestal_mean', 'f8'), ('pedestal_sigma', 'f8'), ('N_pedestal', 'i4'),
    ('spe_mean', 'f8'), ('spe_sigma', 'f8'), ('N_spe', 'i4'),
    ('gain', 'f8'), ('gain_error', 'f8')
])

results_array = np.array(results_list, dtype=dtype)

npz_file_out = "/scratch/elena/WCTE_2025_commissioning/2025_data/WCTE_BRB_Data_Analysis/pmt_charge_fit_results_enhanced_[-4,+2]_pedestalSym4.npz"
np.savez(npz_file_out, results=results_array)
print(f"Saved results to {npz_file_out}")
