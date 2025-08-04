import numpy as np
from scipy.signal import butter, filtfilt
from scipy.fftpack import dctn, idctn


def lpf(data, cutoff_freq, fs=1.0, order=4):
    """
    Zero-phase Butterworth low-pass filter along last axis of 2D data or 1D.
    """
    nyq = 0.5 * fs
    norm_cutoff = cutoff_freq / nyq
    if norm_cutoff <= 0 or norm_cutoff >= 1:
        return data
    b, a = butter(order, norm_cutoff, btype='lowpass', analog=False)
    return filtfilt(b, a, data, axis=-1)

def dct2d_flat(vector, shape):
    """2D DCT flatten"""
    return dctn(vector.reshape(shape), norm='ortho').ravel()

def idct2d(coeffs, shape):
    """Inverse 2D DCT"""
    return idctn(coeffs.reshape(shape), norm='ortho')

def construct_measurement_indices(chunk_shape, n_measurements, signal_chunk):
    """
    Choose 70% energy-based, 30% random indices from flattened 2D block.
    """
    C, L = chunk_shape
    fft2d = np.fft.fft2(signal_chunk)
    energy = np.abs(fft2d)**2
    prob = energy / energy.sum()
    prob_flat = prob.ravel()
    n_energy = int(0.7 * n_measurements)
    n_random = n_measurements - n_energy
    energy_idx = np.random.choice(prob_flat.size, size=n_energy, p=prob_flat, replace=False)
    remaining = np.setdiff1d(np.arange(prob_flat.size), energy_idx)
    random_idx = np.random.choice(remaining, size=n_random, replace=False)
    return np.concatenate([energy_idx, random_idx])

def plot_signal(signal, title="Signal"):
    """
    Plot a 1D or 2D signal.
    """
    import matplotlib.pyplot as plt

    plt.figure()
    if signal.ndim == 1 or signal.shape[0] == 1:
        plt.plot(signal.flatten())
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
    elif signal.ndim == 2:
        im = plt.imshow(signal, aspect='auto', cmap='seismic', origin='lower')
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Channel")
        plt.colorbar(im)  # only add colorbar when imshow is used
    else:
        raise ValueError("Signal must be 1D or 2D.")

    plt.tight_layout()
    plt.show()

def plot_signal_3d(signal, title="3D Signal"):
    """
    Plot a 3D signal using matplotlib.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(signal.shape[1])
    y = np.arange(signal.shape[0])
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, signal, cmap='viridis')
    
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Channel")
    ax.set_zlabel("Amplitude")
    
    plt.show()

def rmse(signal1, signal2):
    """
    Calculate Root Mean Squared Error between two signals.
    """
    if signal1.shape != signal2.shape:
        raise ValueError("Signals must have the same shape for RMSE calculation.")
    return np.sqrt(np.mean((signal1 - signal2) ** 2))

def bandpass_rmse(signal1, signal2, low_freq, high_freq, fs):
    """
    Calculate RMSE after applying a bandpass filter to both signals.
    """
    from scipy.signal import butter, filtfilt

    nyq = 0.5 * fs
    low = low_freq / nyq
    high = high_freq / nyq
    b, a = butter(4, [low, high], btype='band')

    filtered1 = filtfilt(b, a, signal1)
    filtered2 = filtfilt(b, a, signal2)

    return rmse(filtered1, filtered2)

def snr(clean, recon):
    """
    Calculate Signal-to-Noise Ratio (SNR) between clean and reconstructed signals.
    """
    signal_power = np.mean(clean**2)
    noise_power  = np.mean((clean - recon)**2)
    return 10 * np.log10(signal_power / noise_power)

def psnr(clean, recon):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between clean and reconstructed signals.
    """
    mse = np.mean((clean - recon) ** 2)
    if mse == 0:
        return float('inf')
    peak = np.max(np.abs(clean))
    return 20 * np.log10(peak / np.sqrt(mse))