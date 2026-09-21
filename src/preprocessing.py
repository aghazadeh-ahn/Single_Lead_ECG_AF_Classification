"""
src/preprocessing.py
Signal processing utilities for ECG data:
- Butterworth Bandpass filtering (baseline wander & high frequency noise removal)
- Notch filtering (50 Hz powerline interference removal)
- Normalization (Z-Score)
- Length standardization (Pad / Crop)
- Training-time augmentation
- Sliding-window segmentation (TTA)
"""

from typing import Tuple, Union
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

from src.config import (
    AUG_BASELINE_AMP,
    AUG_NOISE_STD,
    AUG_PROB,
    AUG_SCALE_MAX,
    AUG_SCALE_MIN,
    AUG_TIME_STRETCH_MAX,
    AUG_TIME_STRETCH_MIN,
    BANDPASS_HIGHCUT,
    BANDPASS_LOWCUT,
    BANDPASS_ORDER,
    NOTCH_FREQ,
    NOTCH_Q,
    SAMPLING_RATE,
    TARGET_LENGTH,
    TTA_STRIDE,
    TTA_WINDOW_LENGTH,
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
    - Shorter signals: Zero-padded (random left/right split in train, end-pad otherwise).
    - Longer signals: Random crop (Train) or Center crop (Val/Test).
    
    Args:
        signal: 1D numpy array.
        target_length: Desired number of samples (default: 9000).
        is_train: If True, uses random cropping / random pad position.
        
    Returns:
        1D numpy array of length `target_length`.
    """
    sig_len = len(signal)

    if sig_len == target_length:
        return signal
    elif sig_len < target_length:
        pad_width = target_length - sig_len
        if is_train:
            left = int(np.random.randint(0, pad_width + 1))
            right = pad_width - left
            return np.pad(signal, (left, right), mode="constant", constant_values=0.0)
        return np.pad(signal, (0, pad_width), mode="constant", constant_values=0.0)
    else:
        if is_train:
            start_idx = np.random.randint(0, sig_len - target_length + 1)
        else:
            start_idx = (sig_len - target_length) // 2
        return signal[start_idx : start_idx + target_length]


def augment_ecg(
    signal: np.ndarray,
    noise_std: float = AUG_NOISE_STD,
    scale_range: Tuple[float, float] = (AUG_SCALE_MIN, AUG_SCALE_MAX),
    baseline_amp: float = AUG_BASELINE_AMP,
    time_stretch_range: Tuple[float, float] = (AUG_TIME_STRETCH_MIN, AUG_TIME_STRETCH_MAX),
    p: float = AUG_PROB,
    fs: int = SAMPLING_RATE,
) -> np.ndarray:
    """
    Light ECG augmentations (each applied independently with probability `p`):
    amplitude scaling, Gaussian noise, low-frequency baseline wander, time stretch.
    """
    x = signal.astype(np.float32).copy()
    n = len(x)

    if np.random.rand() < p:
        scale = float(np.random.uniform(scale_range[0], scale_range[1]))
        x = x * scale

    if np.random.rand() < p and noise_std > 0:
        x = x + np.random.normal(0.0, noise_std, size=n).astype(np.float32)

    if np.random.rand() < p and baseline_amp > 0:
        t = np.arange(n, dtype=np.float32)
        freq = float(np.random.uniform(0.05, 0.5))
        amp = float(np.random.uniform(0.0, baseline_amp))
        phase = float(np.random.uniform(0.0, 2.0 * np.pi))
        x = x + amp * np.sin(2.0 * np.pi * freq * t / fs + phase)

    if np.random.rand() < p:
        stretch = float(np.random.uniform(time_stretch_range[0], time_stretch_range[1]))
        new_len = max(1, int(round(n * stretch)))
        old_idx = np.linspace(0.0, 1.0, n, dtype=np.float32)
        new_idx = np.linspace(0.0, 1.0, new_len, dtype=np.float32)
        stretched = np.interp(new_idx, old_idx, x).astype(np.float32)
        if new_len > n:
            start = int(np.random.randint(0, new_len - n + 1))
            x = stretched[start : start + n]
        elif new_len < n:
            pad = n - new_len
            left = int(np.random.randint(0, pad + 1))
            x = np.pad(stretched, (left, pad - left), mode="constant", constant_values=0.0)
        else:
            x = stretched

    return x.astype(np.float32)


def clean_ecg_pipeline(
    signal: np.ndarray,
    apply_filter: bool = True,
    normalize: bool = True,
    target_length: Union[int, None] = TARGET_LENGTH,
    is_train: bool = False,
    use_augmentation: bool = False,
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

    if use_augmentation and is_train:
        processed = augment_ecg(processed)

    return processed


def segment_signal_sliding_windows(
    signal: np.ndarray,
    window_length: int = TTA_WINDOW_LENGTH,
    stride: int = TTA_STRIDE,
) -> np.ndarray:
    """
    Splits long signals into overlapping windows without losing any segment.
    Returns array of shape: [num_windows, window_length]
    """
    sig_len = len(signal)

    # 1. If signal is shorter than window_length -> Pad
    if sig_len <= window_length:
        pad_width = window_length - sig_len
        padded = np.pad(signal, (0, pad_width), mode="constant", constant_values=0.0)
        return np.expand_dims(padded, axis=0)  # Shape: [1, 9000]

    # 2. If longer -> Extract sliding windows
    windows = []
    for start in range(0, sig_len - window_length + 1, stride):
        windows.append(signal[start : start + window_length])

    # Ensure the very last part of the signal is not missed
    if (sig_len - window_length) % stride != 0:
        windows.append(signal[-window_length:])

    return np.asarray(windows, dtype=np.float32)  # Shape: [N_windows, 9000]


def prepare_tta_windows(
    signal: np.ndarray,
    apply_filter: bool = False,
    window_length: int = TTA_WINDOW_LENGTH,
    stride: int = TTA_STRIDE,
) -> np.ndarray:
    """
    Filter + normalize the full recording, then cut overlapping windows for TTA.
    Does not crop to a single center window.
    """
    processed = signal.copy().astype(np.float32)
    if apply_filter:
        processed = butter_bandpass_filter(processed)
        processed = notch_filter(processed)
    processed = z_score_normalize(processed)
    return segment_signal_sliding_windows(
        processed, window_length=window_length, stride=stride
    )
