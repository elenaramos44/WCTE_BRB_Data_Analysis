import numpy as np

fpath = "./pmt_arrays_test/card76_slot86_ch0.npz"
data = np.load(fpath)
waveforms = data["waveforms"]
print(f"Loaded {waveforms.shape[0]} waveforms, each of length {waveforms.shape[1]}")
print("Ejemplo de la primera waveform:", waveforms[0])

