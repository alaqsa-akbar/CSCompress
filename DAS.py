import numpy as np

class DAS:
    def __init__(self, n_channels, time, fs, frequencies, noise_std=0.01):
        """
        n_channels: Number of channels.
        time: Duration of the signal (in seconds).
        fs: Sampling frequency (Hz).
        frequencies: List of frequencies to use (either superposed in 1D or for propagation).
        noise_std: Standard deviation of additive Gaussian noise.
        """
        if n_channels < 1:
            raise ValueError("At least one channel is required.")
        self.n_channels = n_channels
        self.time = time
        self.fs = fs
        self.frequencies = frequencies
        self.noise_std = noise_std

    def generate_signal(self, velocity_type='same', **kwargs):
        """
        Generate a synthetic DAS signal based on the specified velocity type.
        velocity_type:
            - 'same': same velocity across channels
            - 'different': increasing delay per channel
            - 'nonstationary': frequency varies with channel index
        kwargs:
            - for 'different': delay_step (sec per channel)
            - for 'nonstationary': velocity (channels per second)
        Returns:
            - clean_signal: [channels x time]
            - noisy_signal: clean_signal + Gaussian noise
        """
        if self.n_channels == 1:
            return self._generate_1d_signal()

        if velocity_type == 'same':
            return self._generate_signal_same_velocity()
        elif velocity_type == 'different':
            return self._generate_signal_different_velocity(
                delay_step=kwargs.get('delay_step', 0.002)
            )
        elif velocity_type == 'nonstationary':
            return self._generate_signal_nonstationary(
                velocity=kwargs.get('velocity', 100)
            )
        else:
            raise ValueError("velocity_type must be 'same', 'different', or 'nonstationary'")

    def _generate_1d_signal(self):
        """
        Generate a 1D superposed frequency signal with noise.
        """
        t = np.arange(0, self.time, 1 / self.fs)
        clean_signal = np.zeros_like(t)
        for freq in self.frequencies:
            clean_signal += np.sin(2 * np.pi * freq * t)

        noisy_signal = clean_signal + np.random.normal(
            0.0, self.noise_std, size=t.shape
        )
        return clean_signal, noisy_signal

    def _generate_signal_same_velocity(self):
        """
        Signal arrives at all channels at the same time, superposing all frequencies.
        """
        t = np.arange(0, self.time, 1 / self.fs)
        clean_signal_1d = np.zeros_like(t)
        for freq in self.frequencies:
            clean_signal_1d += np.sin(2 * np.pi * freq * t)

        clean_signal = np.tile(clean_signal_1d, (self.n_channels, 1))
        noisy_signal = clean_signal + np.random.normal(
            0.0, self.noise_std, clean_signal.shape
        )
        return clean_signal, noisy_signal

    def _generate_signal_different_velocity(self, delay_step=0.002):
        """
        Signal travels channel-by-channel, simulating propagation delay per channel.
        delay_step: time delay per channel in seconds.
        """
        t = np.arange(0, self.time, 1 / self.fs)
        clean_signal = np.zeros((self.n_channels, len(t)))

        for i in range(self.n_channels):
            delay = i * delay_step
            t_shifted = t - delay
            t_shifted[t_shifted < 0] = 0
            # superpose all frequencies on shifted time
            for freq in self.frequencies:
                clean_signal[i] += np.sin(2 * np.pi * freq * t_shifted)

        noisy_signal = clean_signal + np.random.normal(
            0.0, self.noise_std, clean_signal.shape
        )
        return clean_signal, noisy_signal

    def _generate_signal_nonstationary(self, velocity=100):
        """
        Signal with frequency varying linearly from uphole (channel 0) to downhole (channel n-1).
        velocity: propagation velocity in channels/sec (optional).
        """
        t = np.arange(0, self.time, 1 / self.fs)
        clean_signal = np.zeros((self.n_channels, len(t)))

        f_min, f_max = min(self.frequencies), max(self.frequencies)
        x = np.arange(self.n_channels)
        # linear frequency profile across channels
        f_x = f_min + (f_max - f_min) * (x / (self.n_channels - 1))

        for i in range(self.n_channels):
            delay = i / velocity
            t_shifted = t - delay
            t_shifted[t_shifted < 0] = 0
            clean_signal[i] = np.sin(2 * np.pi * f_x[i] * t_shifted)

        noisy_signal = clean_signal + np.random.normal(
            0.0, self.noise_std, clean_signal.shape
        )
        return clean_signal, noisy_signal
