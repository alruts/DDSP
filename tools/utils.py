# import numpy as np
from numpy import (
    abs,
    angle,
    argmax,
    array,
    correlate,
    hanning,
    log10,
    pad,
    rad2deg,
    unwrap,
    pi,
    cos,
    arange,
    exp,
)
from numpy.fft import fft, fftfreq, ifft


def zero_pad(arr, num_zeros_start, num_zeros_end):
    """
    Adds a specified number of zeros to both ends of a NumPy array.
    """
    padded_arr = pad(
        arr, (num_zeros_start, num_zeros_end), "constant", constant_values=(0, 0)
    )
    return padded_arr


def detect_sample_delay(signal1, signal2):
    """
    Calculates the sample delay between two signals using cross-correlation.

    Args:
        signal1 (array-like): First signal.
        signal2 (array-like): Second signal (reference).

    Returns:
        detect_sample_delay (int): sample delay in samples between the two signals.
    """
    # Convert signals to numpy arrays
    signal1 = array(signal1)
    signal2 = array(signal2)

    # Calculate the cross-correlation between the two signals
    cross_corr = correlate(signal1, signal2, mode="full")

    # Find the index of the maximum value in the cross-correlation
    max_index = argmax(cross_corr)

    # Calculate the sample delay in samples
    sample_delay = max_index - len(signal2) + 1

    return sample_delay


def deconvolve(excitation_signal, recorded_signal):
    """
    Returns the impulse response of a system using deconvolution (time-domain).

        Inputs:
        -excitation_signal: signal that is used to excite the system (input)
        -recorded_signal: output of system excited with 'excitation_signal'.
    """

    # Compute the FFT of excitation_signal
    excitation_signal_freq_domain = fft(excitation_signal)

    # Compute the FFT of recorded_signal
    recorded_signal_freq_domain = fft(recorded_signal)

    # Multiply the spectra of the recorded and excitation signals
    freq_response = (1 / excitation_signal_freq_domain) * recorded_signal_freq_domain
    impulse_response = ifft(freq_response).real

    return impulse_response


def hanning_fade(signal, sampling_rate, fade_length_ms):
    """
    Applies a Hanning window to taper the ends of a signal.

    Inputs:
        signal (ndarray): The input signal.
        sampling_rate (int): The sampling rate of the signal.
        fade_length_ms (float): The fade length in milliseconds.

    Returns:
        ndarray: The signal with the Hanning window applied.
    """
    # Convert fade length to samples
    fade_length = int((fade_length_ms / 1000) * sampling_rate)
    window = hanning(2 * fade_length)
    # Apply Hanning window to the beginning
    signal[:fade_length] *= window[:fade_length]
    # Apply Hanning window to the end
    signal[-fade_length:] *= window[fade_length:]
    return signal


def get_spectrum(signal, samplerate):
    """
    Returns:
        ndarray: Complex spectrum (one-sided).
        ndarray: Frequency axis in Hz for plotting (one-sided).
    """
    # Compute the FFT
    spectrum = fft(signal)

    # Compute the frequency axis
    frequency = fftfreq(len(signal), d=1 / samplerate)

    # Compute the one-sided spectrum
    spectrum_one_sided = spectrum[: len(signal) // 2 + 1]
    frequency_one_sided = abs(frequency[: len(signal) // 2 + 1])

    return spectrum_one_sided, frequency_one_sided


def get_phase_deg(complex_spectrum):
    """Computes unwrapped phase spectrum in degrees."""
    return rad2deg(unwrap(angle(complex_spectrum)))


def get_mag(complex_spectrum, mode="db"):
    """Computes magnitude spectrum.

    Parameters:
    complex_spectrum (np.array): Array of complex spectrum values.
    mode (str, optional): Computation mode ('db' for dB, 'linear' for linear magnitude). Defaults to 'db'.

    Returns:
    np.array: Magnitude spectrum (in dB if mode is 'log', linear otherwise).
    """
    mode = mode.lower()
    if mode == "db":
        return 20 * log10(abs(complex_spectrum))
    elif mode == "linear":
        return abs(complex_spectrum)
    else:
        raise ValueError("Invalid mode. Supported modes: 'db', 'linear'.")


def get_magphase(complex_spectrum, mode="db"):
    """Computes magnitude and phase spectrum.

    Parameters:
    complex_spectrum (np.array): Array of complex spectrum values.
    mode (str, optional): Computation mode ('db' for dB, 'linear' for linear magnitude). Defaults to 'db'.

    Returns:
    tuple: A tuple containing magnitude and phase spectrum.
    """
    mag = get_mag(complex_spectrum, mode)
    phase = get_phase_deg(complex_spectrum)
    return mag, phase


def gamma_tone_ir(duration_sec, fc_hz, fs_hz, impairment_factor=0):
    n = 4  # filter order
    phi_rad = 0  # initial phase

    ERB = 24.7 + 0.108 * fc_hz + impairment_factor
    b = 1.018 * ERB
    a = 6 / (-2 * pi * b) ** 4
    t_sec = arange(duration_sec * fs_hz) / fs_hz
    return (
        a**-1
        * t_sec ** (n - 1)
        * cos(2 * pi * fc_hz * t_sec + phi_rad)
        * exp(-2 * pi * b * t_sec)
    )
