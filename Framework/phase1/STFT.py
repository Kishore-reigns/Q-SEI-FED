import numpy as np
from scipy.signal import stft

def iq_to_spectrogram(iq, fs=1e6, nperseg=256, noverlap=128):
    I = iq[:,0]
    Q = iq[:,1]

    _, _, ZI = stft(I, fs=fs, nperseg=nperseg, noverlap=noverlap)
    _, _, ZQ = stft(Q, fs=fs, nperseg=nperseg, noverlap=noverlap)

    spec_I = np.abs(ZI)
    spec_Q = np.abs(ZQ)

    # Stack as channels → (2, F, T)
    spec = np.stack([spec_I, spec_Q], axis=0)
    return spec


def iq_to_combined_spectrogram(iq, fs=1e6, nperseg=256, noverlap=128):
    """
    Returns a single combined spectrogram for visualization.
    """
    # build complex signal again
    complex_signal = iq[:,0] + 1j * iq[:,1]

    _, _, Z = stft(complex_signal,
                   fs=fs,
                   nperseg=nperseg,
                   noverlap=noverlap)

    spec = np.abs(Z)   # magnitude spectrogram
    return spec        # (F, T)