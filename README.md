# CS Compress

A compressed sensing approach to compressing multi-channel DAS signals. This repository contains modular implementations of both low-pass filter (LPF) compression and compressed sensing (CS)-based compression, along with signal simulation and utility functions.

---

## 📁 Files

### `LPFCompress.py`

Implements a Butterworth low-pass filter compression algorithm. The data is filtered and then optionally downsampled. This is a simple yet effective method that is sensitive to aliasing, so the cutoff frequency should be chosen carefully.

**Parameters:**
- `data`: Data to be compressed (`np.ndarray` of shape `(C, T)` or `(T,)`)
- `cutoff_freq`: Cutoff frequency in Hz
- `fs`: Sampling frequency in Hz
- `downsample_factor`: Factor by which to downsample the signal (default: `1`)
- `order`: Order of the Butterworth filter (default: `4`)

---

### `CSCompress.py`

Implements a compressed sensing pipeline. Data is pre-filtered using a Butterworth low-pass filter (optional), chunked into overlapping windows, and randomly subsampled. Reconstruction is done via Lasso regression.

**Parameters:**
- `data`: Data to be compressed (`np.ndarray` of shape `(C, T)`)
- `fs`: Sampling frequency in Hz
- `chunk_size`: Number of time samples per block
- `subsample_ratio`: Fraction of elements in each block to measure
- `alpha`: Lasso regularization weight
- `max_iter`: Maximum number of iterations for Lasso
- `cutoff_freq`: Optional pre-filter cutoff frequency (Hz)
- `overlap`: Fractional overlap between consecutive blocks

---

### `DAS.py`

Simulates Distributed Acoustic Sensing (DAS) signal data. Four signal modes are supported:
- 1D signals
- 2D signals with the same velocity across channels
- 2D signals with varying velocities across channels
- 2D signals with different frequencies per channel

---

### `Utils.py`

Provides utility functions shared across modules. Includes data chunking, reconstruction metrics, visualization helpers, and matrix generation tools.

---

### `Main.py`

The main entry point. Use this script to simulate data, compress it using CS or LPF methods, and evaluate the results.

To run with default settings:
```bash
python -m main
```

To customize the run, you may pass optional arguments such as:
```bash
python -m main --frequencies 0.01 0.1 1 10 100 --time 100 --fs 100 --noise_std 0.15 --n_channels 20 --subsample_ratio 0.1 --alpha 1e-4 --downsample_factor 4 --lpf_order 4 --lpf_cutoff_freq 10 --max_iter 1500 --chunk_size 500
```

---

## ⚙️ Command-Line Arguments

| Argument                | Description                                                        | Default       |
|-------------------------|--------------------------------------------------------------------|---------------|
| `--n_channels`          | Number of DAS channels                                             | `20`          |
| `--time`                | Duration of the signal in seconds                                  | `5`           |
| `--fs`                  | Sampling frequency in Hz                                           | `1000`        |
| `--noise_std`           | Standard deviation of Gaussian noise                               | `0.05`        |
| `--frequencies`         | Frequencies in the signal (space-separated list)                   | `0.01 0.1 1 10`|
| `--chunk_size`          | Number of samples per compressed block                             | `50`          |
| `--overlap`             | Fractional overlap between blocks (0 to 1)                         | `0.5`         |
| `--subsample_ratio`     | Fraction of measurements to keep in CS                             | `0.1`         |
| `--alpha`               | Lasso regularization strength                                      | `5e-4`        |
| `--max_iter`            | Max number of iterations for Lasso                                 | `600`         |
| `--cs_cutoff_freq`      | Pre-filter cutoff frequency for CS (Hz)                            | `10`          |
| `--cs_order`            | Filter order for CS pre-filter                                     | `4`           |
| `--lpf_cutoff_freq`     | LPF cutoff frequency (Hz)                                          | `1`           |
| `--lpf_order`           | LPF filter order                                                   | `4`           |
| `--downsample_factor`   | Downsampling factor after LPF                                      | `4`           |
| `--plot_3d`             | Enable 3D plotting of signals (if set)                             | `False`       |

---

## ✅ Example Use Cases

- Compare LPF and CS-based compression methods on clean or noisy data.
- Simulate different signal profiles to evaluate robustness.
- Visualize compressed vs. original signal performance in time or frequency domain.

---

## 🧪 Notes

- Ensure `chunk_size` divides evenly into `fs * time` for clean segmentation.
- Always set cutoff frequencies below Nyquist (`fs/2`) to avoid aliasing.
- For rapid prototyping, start with low `fs`, low `n_channels`, and short `time`.

---
