"""
src/preprocessing.py
Signal processing utilities for ECG data:
- Butterworth Bandpass filtering (baseline wander & high frequency noise removal)
- Notch filtering (50 Hz powerline interference removal)
- Normalization (Z-Score)
- Length standardization (Pad / Crop)
"""

from typing import Union
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

from src.config import (
    BANDPASS_HIGHCUT,
    BANDPASS_LOWCUT,
    BANDPASS_ORDER,
    NOTCH_FREQ,
    NOTCH_Q,
    SAMPLING_RATE,
    TARGET_LENGTH,
)


def butter_bandpass_filter(
    signal: np.ndarray,
    lowcut: float = BANDPASS_LOWCUT,
    highcut: float = BANDPASS_HIGHCUT,
    fs: int = SAMPLING_RATE,
    order: int = BANDPASS_ORDER,
) -> np.ndarray:
    """
    Applies a zero-phase Butterworth bandpass filter.
    
    Args:
        signal: 1D numpy array of raw ECG values.
        lowcut: Lower cutoff frequency (Hz).
        highcut: Upper cutoff frequency (Hz).
        fs: Sampling frequency (Hz).
        order: Filter order.
        
    Returns:
        Filtered 1D numpy array.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


def notch_filter(
    signal: np.ndarray,
    freq: float = NOTCH_FREQ,
    q: float = NOTCH_Q,
    fs: int = SAMPLING_RATE,
) -> np.ndarray:
    """
    Applies a zero-phase Notch (band-stop) filter to suppress powerline noise.
    
    Args:
        signal: 1D numpy array.
        freq: Frequency to notch out (e.g., 50.0 Hz).
        q: Quality factor (higher means narrower rejection band).
        fs: Sampling frequency (Hz).
        
    Returns:
        Notch-filtered 1D numpy array.
    """
    b, a = iirnotch(freq, q, fs)
    return filtfilt(b, a, signal)


def z_score_normalize(signal: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Standardizes signal to zero mean and unit variance: (x - mean) / (std + eps).
    """
    mean = float(np.mean(signal))
    std = float(np.std(signal))
    if std < eps:
        return signal - mean
    return (signal - mean) / (std + eps)


def adjust_signal_length(
    signal: np.ndarray,
    target_length: int = TARGET_LENGTH,
    is_train: bool = True,
) -> np.ndarray:
    """
    Standardizes signal length to `target_length`:
    - Shorter signals: Zero-padded at the end.
    - Longer signals: Random crop (Train) or Center crop (Val/Test).
    
    Args:
        signal: 1D numpy array.
        target_length: Desired number of samples (default: 9000).
        is_train: If True, uses random cropping; otherwise center crop.
        
    Returns:
        1D numpy array of length `target_length`.
    """
    sig_len = len(signal)

    if sig_len == target_length:
        return signal
    elif sig_len < target_length:
        pad_width = target_length - sig_len
        return np.pad(signal, (0, pad_width), mode="constant", constant_values=0.0)
    else:
        if is_train:
            start_idx = np.random.randint(0, sig_len - target_length + 1)
        else:
            start_idx = (sig_len - target_length) // 2
        return signal[start_idx : start_idx + target_length]


def clean_ecg_pipeline(
    signal: np.ndarray,
    apply_filter: bool = True,
    normalize: bool = True,
    target_length: Union[int, None] = TARGET_LENGTH,
    is_train: bool = False,
) -> np.ndarray:
    """
    Full end-to-end preprocessing pipeline for a single ECG recording.
    """
    processed = signal.copy().astype(np.float32)

    if apply_filter:
        processed = butter_bandpass_filter(processed)
        processed = notch_filter(processed)

    if normalize:
        processed = z_score_normalize(processed)

    if target_length is not None:
        processed = adjust_signal_length(
            processed, target_length=target_length, is_train=is_train
        )

    return processed


def segment_signal_sliding_windows(
    signal: np.ndarray,
    window_length: int = 9000,
    stride: int = 4500  # 50% overlap (15 seconds)
) -> np.ndarray:
    """
    Splits long signals into overlapping windows without losing any segment.
    Returns array of shape: [num_windows, window_length]
    """
    sig_len = len(signal)
    
    # 1. If signal is shorter than window_length -> Pad
    if sig_len <= window_length:
        pad_width = window_length - sig_len
        padded = np.pad(signal, (0, pad_width), mode='constant', constant_values=0.0)
        return np.expand_dims(padded, axis=0) # Shape: [1, 9000]
    
    # 2. If longer -> Extract sliding windows
    windows = []
    for start in range(0, sig_len - window_length + 1, stride):
        windows.append(signal[start : start + window_length])
        
    # Ensure the very last part of the signal is not missed
    if (sig_len - window_length) % stride != 0:
        windows.append(signal[-window_length:])
        
    return np.array(windows) # Shape: [N_windows, 9000]
