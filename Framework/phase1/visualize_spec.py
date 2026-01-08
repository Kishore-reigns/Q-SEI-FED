import matplotlib.pyplot as plt
import numpy as np

def plot_spectrogram(spec):
    fig, ax = plt.subplots(1, 2, figsize=(10,4))

    ax[0].imshow(20*np.log10(spec[0] + 1e-8), aspect='auto', origin='lower')
    ax[0].set_title("I-Spectrogram")

    ax[1].imshow(20*np.log10(spec[1] + 1e-8), aspect='auto', origin='lower')
    ax[1].set_title("Q-Spectrogram")

    for a in ax:
        a.set_xlabel("Time bins")
        a.set_ylabel("Frequency bins")

    plt.tight_layout()
    plt.show()


def plot_combined_spectrogram(spec):
    plt.figure(figsize=(7,4))
    plt.imshow(20*np.log10(spec + 1e-8),
               aspect='auto',
               origin='lower')
    plt.colorbar(label="dB")
    plt.title("Combined RF Spectrogram")
    plt.xlabel("Time bins")
    plt.ylabel("Frequency bins")
    plt.tight_layout()
    plt.show()