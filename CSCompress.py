import sys
import numpy as np
from scipy.signal import get_window
from sklearn.linear_model import Lasso
from utils import lpf, dct2d_flat, idct2d, construct_measurement_indices, plot_signal, plot_signal_3d
from tqdm import tqdm


class CSCompress:
    def __init__(self, data, fs, chunk_size, subsample_ratio, alpha, max_iter=1000,
                 cutoff_freq=None, order=4, overlap=0.5):
        """
        data: np.ndarray of shape (C, T)
        fs: Sampling frequency (Hz)
        chunk_size: number of time samples per block
        subsample_ratio: fraction of block elements to measure
        alpha: Lasso regularization weight
        max_iter: maximum number of iterations for Lasso
        cutoff_freq: optional pre-filter cutoff (Hz)
        overlap: fraction overlap between consecutive blocks
        """
        self.data = np.array(data)
        if self.data.ndim == 1:
            self.data = self.data[np.newaxis, :]
        self.fs = fs
        self.C, self.T = self.data.shape
        self.chunk_size = chunk_size
        self.hop = int(chunk_size * (1 - overlap))
        self.n_measurements = int(subsample_ratio * (self.C * chunk_size))
        self.alpha = alpha
        self.max_iter = max_iter
        self.cutoff = cutoff_freq
        self.order = order
        self.filtered = None
        self.chunks = []  # list of dicts with 'start','indices','y'
        self.reconstructed = None

    def load_chunks(self, chunks):
        """
        Load pre-computed chunks for reconstruction.
        Each chunk should be a dict with 'start', 'indices', 'y'.
        """
        self.chunks = chunks
        self.reconstructed = None

    def compress(self):
        """
        Perform optional low-pass filter and then measure each overlapping block.
        Stores measurement vectors but does not reconstruct full signal.

        Returns:
            List[np.ndarray]: measurement vectors for each block.
        """
        data = self.data
        if self.cutoff is not None:
            data = lpf(data, self.cutoff, fs=self.fs, order=self.order)
        self.filtered = data
        self.chunks = []

        for start in range(0, self.T - self.chunk_size + 1, self.hop):
            block = data[:, start:start+self.chunk_size]
            idx = construct_measurement_indices(block.shape, self.n_measurements, block)
            flat = block.ravel()
            y = flat[idx]
            self.chunks.append({'start': start, 'indices': idx, 'y': y})

        return np.array(self.chunks)

    def reconstruct(self):
        """
        Reconstruct full-resolution signal from stored measurements.
        Must call compress() first.

        Returns:
            np.ndarray: reconstructed data of shape (C, T).
        """
        if not self.chunks or self.filtered is None:
            raise ValueError("Must call compress() before reconstruct() or load_chunks().")

        recon = np.zeros_like(self.filtered)
        win = get_window('hann', self.chunk_size)
        win2d = np.tile(win, (self.C, 1))
        win_sum = np.zeros_like(self.filtered)

        for chunk in tqdm(self.chunks, desc="Reconstructing"):
            start = chunk['start']
            idx   = chunk['indices']
            y     = chunk['y']
            N2    = self.C * self.chunk_size

            # build sensing matrix
            A = np.zeros((idx.size, N2))
            basis = np.zeros(N2)
            for k, i in enumerate(idx):
                basis[i] = 1.0
                A[k, :] = dct2d_flat(basis, (self.C, self.chunk_size))
                basis[i] = 0.0

            # solve for sparse DCT coefficients
            lasso = Lasso(alpha=self.alpha, fit_intercept=False,
                          max_iter=self.max_iter, tol=1e-5)
            lasso.fit(A, y)
            coeffs = lasso.coef_

            # reconstruct block
            rec_block = idct2d(coeffs, (self.C, self.chunk_size))
            recon[:, start:start+self.chunk_size] += rec_block * win2d
            win_sum[:, start:start+self.chunk_size] += win2d

        mask = win_sum > 1e-8
        recon[mask] /= win_sum[mask]
        self.reconstructed = recon
        return recon

    def plot_original(self):
        """Plot the raw multi-channel signals."""
        plot_signal(self.data, title="Original Signal")

    def plot_compressed(self):
        """Plot the reconstructed compressed multi-channel signals."""
        if self.reconstructed is None:
            raise ValueError("Call reconstruct() before plotting.")
        plot_signal(self.reconstructed, title="Compressed Signal")

    def plot_original_3d(self):
        """Plot the original multi-channel signals in 3D."""
        plot_signal_3d(self.data, title="Original Signal")

    def plot_compressed_3d(self):
        """Plot the reconstructed compressed multi-channel signals in 3D."""
        if self.reconstructed is None:
            raise ValueError("Call reconstruct() before plotting.")
        plot_signal_3d(self.reconstructed, title="Compressed Signal")

    def get_compressed_size(self):
        """Get the size of the compressed data in bytes, including list overhead."""
        size = sys.getsizeof(self.chunks)
        for chunk in self.chunks:
            size += chunk['y'].nbytes
            size += chunk['indices'].nbytes
            size += sys.getsizeof(chunk['start'])
        return size