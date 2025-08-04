import numpy as np
from utils import lpf, plot_signal, plot_signal_3d

class LPFCompress:
    def __init__(self, data, cutoff_freq, fs, downsample_factor=1, order=4):
        '''
        data: np.ndarray of shape (C, T) or (T,)
        cutoff_freq: low-pass filter cutoff frequency (Hz)
        fs: Sampling frequency (Hz)
        downsample_factor: integer factor by which to downsample the signal
        order: order of the Butterworth filter
        '''
        self.data = np.array(data)
        self.cutoff_freq = cutoff_freq
        self.fs = fs
        if not isinstance(downsample_factor, int) or downsample_factor < 1:
            raise ValueError("downsample_factor must be a positive integer")
        self.downsample_factor = downsample_factor
        self.order = order

        self.filtered = None       # after lpf but before downsampling
        self.compressed = None     # after decimation
        self.reconstructed = None  # after interpolation
        self.original_shape = self.data.shape

    def compress(self):
        """
        Apply zero-phase LPF via `lpf` and downsample by the init factor.
        """
        # 1) filter
        self.filtered = lpf(
            self.data,
            cutoff_freq=self.cutoff_freq,
            fs=self.fs,
            order=self.order
        )
        # 2) downsample by slicing every D-th sample
        D = self.downsample_factor
        self.compressed = self.filtered[..., ::D]
        return self.compressed

    def reconstruct(self, final_cutoff: float = None):
        """
        Upsample back to original rate by linear interpolation,
        then optional smoothing LPF.
        """
        if self.compressed is None:
            raise ValueError("Call compress() before reconstruct()")

        D = self.downsample_factor
        arr = np.atleast_2d(self.compressed)
        C, T_ds = arr.shape
        T = T_ds * D

        # sample indices
        t_ds = np.arange(T_ds) * D
        t_orig = np.arange(T)

        # allocate result
        recon = np.zeros((C, T), dtype=arr.dtype)
        for ch in range(C):
            recon[ch] = np.interp(t_orig, t_ds, arr[ch])

        # if original was 1D, squeeze
        if self.data.ndim == 1:
            recon = recon.ravel()

        # optional smoothing LPF
        if final_cutoff is not None:
            from scipy.signal import butter, filtfilt
            nyq = 0.5 * self.fs
            norm_cut = final_cutoff / nyq
            if 0 < norm_cut < 1:
                b, a = butter(self.order, norm_cut, btype='low')
                recon = filtfilt(b, a, recon, axis=-1)

        recon = recon[..., :self.original_shape[-1]]
        self.reconstructed = recon
        return recon

    def plot_filtered(self):
        """
        Plot the reconstructed (filtered + down/up-sampled) signal.
        """
        if self.reconstructed is None:
            raise ValueError("Reconstructed signal is not available. Call reconstruct() first.")
        plot_signal(self.reconstructed, title="Filtered Signal")

    def plot_filtered_3d(self):
        """
        Plot the 3D representation of the filtered signal.
        """
        if self.reconstructed is None:
            raise ValueError("Reconstructed signal is not available. Call reconstruct() first.")
        plot_signal_3d(self.reconstructed, title="Filtered Signal 3D")
