from CSCompress import CSCompress
from LPFCompress import LPFCompress
from DAS import DAS
from utils import plot_signal, plot_signal_3d, rmse, snr, psnr
import argparse

def main():
    parser = argparse.ArgumentParser(description="Compress and reconstruct DAS signals.")
    parser.add_argument('--n_channels', type=int, default=20, help='Number of DAS channels')
    parser.add_argument('--time', type=int, default=5, help='Duration of the signal in seconds')
    parser.add_argument('--fs', type=int, default=1000, help='Sampling frequency in Hz')
    parser.add_argument('--noise_std', type=float, default=0.05, help='Standard deviation of additive Gaussian noise')
    parser.add_argument('--frequencies', type=float, nargs='+', default=[0.01, 0.1, 1, 10], help='Frequencies to use in the signal')
    parser.add_argument('--chunk_size', type=int, default=50, help='Number of time samples per block (should be divisible by fs * time)')
    parser.add_argument('--overlap', type=float, default=0.5, help='Fraction overlap between consecutive blocks')
    parser.add_argument('--subsample_ratio', type=float, default=0.1, help='Fraction of block elements to measure')
    parser.add_argument('--alpha', type=float, default=5e-4, help='Lasso regularization weight')
    parser.add_argument('--max_iter', type=int, default=600, help='Maximum number of iterations for Lasso')
    parser.add_argument('--cs_cutoff_freq', type=float, default=10, help='Optional pre-filter cutoff (Hz)')
    parser.add_argument('--cs_order', type=int, default=4, help='Order of the Butterworth filter for CSCompress')
    parser.add_argument('--lpf_cutoff_freq', type=float, default=1, help='Low-pass filter cutoff frequency (Hz)')
    parser.add_argument('--lpf_order', type=int, default=4, help='Order of the Butterworth filter for LPFCompress')
    parser.add_argument('--downsample_factor', type=int, default=4, help='Integer factor by which to downsample the LPF signal')
    parser.add_argument('--plot_3d', action='store_true', help='Plot signals in 3D if set')
    args = parser.parse_args()

    # Extract arguments
    n_channels = args.n_channels
    time = args.time
    fs = args.fs
    noise_std = args.noise_std
    frequencies = args.frequencies
    chunk_size = args.chunk_size
    overlap = args.overlap
    subsample_ratio = args.subsample_ratio
    alpha = args.alpha
    max_iter = args.max_iter
    cs_cutoff_freq = args.cs_cutoff_freq
    cs_order = args.cs_order
    lpf_cutoff_freq = args.lpf_cutoff_freq
    downsample_factor = args.downsample_factor
    lpf_order = args.lpf_order

    # Create a DAS instance
    das_signal = DAS(
        n_channels=n_channels,
        time=time,
        fs=fs,
        noise_std=noise_std,
        frequencies=frequencies
    )
    clean_data, data = das_signal.generate_signal(velocity_type='same')
    data_size = data.nbytes
    print(f"Original data size: {data_size} bytes")

    # Compress the DAS signal data
    cs_compressor = CSCompress(
        data=data,
        fs=fs,
        chunk_size=chunk_size,
        overlap=overlap,
        subsample_ratio=subsample_ratio,
        alpha=alpha,
        max_iter=max_iter,
        cutoff_freq=cs_cutoff_freq,
        order=cs_order
    )

    lpf_compressor = LPFCompress(
        data=data,
        cutoff_freq=lpf_cutoff_freq,
        fs=fs,
        downsample_factor=downsample_factor,
        order=lpf_order
    ) 
    filtered_data = lpf_compressor.compress()
    filtered_size = filtered_data.nbytes
    print(f"Filtered data size: {filtered_size} bytes")

    compressed_data = cs_compressor.compress()
    compressed_size = cs_compressor.get_compressed_size()
    print(f"Compressed data size: {compressed_size} bytes")

    filtered_reconstructed = lpf_compressor.reconstruct()
    cs_reconstructed = cs_compressor.reconstruct()
    if n_channels == 1:
        filtered_reconstructed = filtered_reconstructed.flatten()
        cs_reconstructed = cs_reconstructed.flatten()

    if not args.plot_3d:
        plot_signal(clean_data, title="Clean Signal")
        cs_compressor.plot_original()
        lpf_compressor.plot_filtered()
        cs_compressor.plot_compressed()
    else:
        plot_signal_3d(clean_data, title="Clean Signal 3D")
        cs_compressor.plot_original_3d()
        lpf_compressor.plot_filtered_3d()
        cs_compressor.plot_compressed_3d()

    print("RMSE between clean and filtered data:", rmse(clean_data, filtered_reconstructed))
    print("RMSE between clean and compressed data:", rmse(clean_data, cs_reconstructed))

    print("RMSE between noisy and filtered data:", rmse(data, filtered_reconstructed))
    print("RMSE between noisy and compressed data:", rmse(data, cs_reconstructed))

    print("SNR of filtered data:", snr(clean_data, filtered_reconstructed))
    print("SNR of compressed data:", snr(clean_data, cs_reconstructed))

    print("PSNR of filtered data:", psnr(clean_data, filtered_reconstructed))
    print("PSNR of compressed data:", psnr(clean_data, cs_reconstructed))

if __name__ == "__main__":
    main()