import matplotlib.pyplot as plt
import numpy as np

def plot_spectrogram(spec):
    spec_I, spec_Q = spec[0], spec[1]

    plt.figure(figsize=(12,4))

    plt.subplot(1,2,1)
    plt.imshow(20*np.log10(spec_I+1e-8), aspect='auto', origin='lower')
    plt.title("I-Channel Spectrogram")
    plt.colorbar()

    plt.subplot(1,2,2)
    plt.imshow(20*np.log10(spec_Q+1e-8), aspect='auto', origin='lower')
    plt.title("Q-Channel Spectrogram")
    plt.colorbar()

    plt.tight_layout()
    plt.show()
