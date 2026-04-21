from pathlib import Path
import numpy as np
from scipy.signal import find_peaks, fftconvolve

def read_pcm_mono(path: Path, dtype: np.dtype, channels: int):
    """Read raw PCM, return (full_data, mono_float32)."""
    raw = np.fromfile(path, dtype=dtype)
    if channels > 1:
        raw = raw[: raw.size - (raw.size % channels)]
        data = raw.reshape(-1, channels)
    else:
        data = raw.reshape(-1, 1)
    mono = data.mean(axis=1).astype(np.float32)
    return data, mono

def read_pcm_stereo(path: Path, dtype: np.dtype, channels: int):
    """Read raw PCM, return (full_data, left_float32, right_float32)."""
    raw = np.fromfile(path, dtype=dtype)

    # Trim to full frames
    frames = raw.size // channels
    raw = raw[:frames * channels]

    # Reshape to (frames, channels)
    data = raw.reshape(frames, channels)

    # Extract channels
    left = data[:, 0].astype(np.float32)
    right = data[:, 1].astype(np.float32) if channels > 1 else left.copy()

    # Normalize if integer PCM
    if np.issubdtype(dtype, np.integer):
        max_val = np.iinfo(dtype).max
        left /= max_val
        right /= max_val

    return data, left, right

def plot_spectrogram(signal, sample_rate, channels=2, title="Spectrogram"):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    if channels > 1:
        for ch in range(channels):
            plt.subplot(channels, 1, ch + 1)
            plt.specgram(signal[:, ch], Fs=sample_rate)
            plt.title(f"{title} - Channel {ch + 1}")
            plt.xlabel("Time [s]")
            plt.ylabel("Frequency [Hz]")
    else:
        plt.specgram(signal, Fs=sample_rate)
        plt.title(title)
        plt.xlabel("Time [s]")
        plt.ylabel("Frequency [Hz]")
    plt.tight_layout()
    plt.show()
    
def plot_correlation(ref, sig, sample_rate, prominence=0.5):
    import matplotlib.pyplot as plt

    # Normalize
    ref = ref / (np.linalg.norm(ref) + 1e-12)
    sig = sig / (np.max(np.abs(sig)) + 1e-12)

    # Cross-correlation via FFT (equivalent to np.correlate 'valid', much faster)
    corr = fftconvolve(sig, ref[::-1], mode='valid') / len(ref)
    corr = np.abs(corr)
    corr /= (np.max(corr) + 1e-12)

    peak_indices, _ = find_peaks(
        corr,
        distance=max(1, len(ref)),
        prominence=prominence,
    )

    # Time axis
    time_axis = np.arange(len(corr)) / sample_rate

    # Plot
    plt.figure(figsize=(14, 4))
    plt.plot(time_axis, corr, linewidth=0.8)
    if len(peak_indices) > 0:
        plt.scatter(
            time_axis[peak_indices],
            corr[peak_indices],
            color='r',
            s=24,
            label='Detected Peaks',
            zorder=3,
        )
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized Correlation")
    plt.title("Cross-Correlation: Reference Chirp vs sample_lowerA")
    plt.legend()
    plt.tight_layout()
    plt.show()
