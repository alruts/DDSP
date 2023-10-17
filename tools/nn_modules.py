import torch
import torch.nn as nn
import torch.nn.functional as F
from math import pi


class FIRFilter1D(nn.Module):
    """Implements an arbitrary phase FIR filter"""

    def __init__(self, num_taps):
        super(FIRFilter1D, self).__init__()
        self.num_taps = num_taps
        # Initialize filter taps as learnable parameters
        self.filter_taps = nn.Parameter(torch.randn(num_taps, 1), requires_grad=True)

    def forward(self, x):
        # FIR filter implemented with a conv1d layer
        filtered_signal = F.conv1d(
            x, self.filter_taps.view(1, 1, -1), padding=self.num_taps - 1
        )
        return filtered_signal


class FIRFilter1DLinearPhaseI(nn.Module):
    """Implements a linear phase FIR type I filter as a torch module."""

    def __init__(self, num_taps):
        super(FIRFilter1DLinearPhaseI, self).__init__()
        self.num_taps = num_taps

        # Only allow odd number of taps
        assert num_taps % 2 != 0, "Even number of taps not allowed in type I filter"

        # Initialize parameters
        self.learnable_taps = nn.Parameter(
            torch.randn(num_taps // 2 + 1, 1)
        )  # Taps from mid-index and up are learnable
        self.filter_taps = []

    def forward(self, x):
        mirrored_taps = torch.flip(
            self.learnable_taps[1:], dims=[0]
        )  # Mirror learnable taps for symmetry
        self.filter_taps = torch.cat([mirrored_taps, self.learnable_taps])
        # Apply the FIR filter operation with a conv1d layer
        filtered_signal = torch.conv1d(
            x, self.filter_taps.view(1, 1, -1), padding=self.num_taps - 1
        )

        return filtered_signal


class GammaToneFilter(nn.Module):
    """Generates a gammatone filter as a torch module (non-trainable)."""

    def __init__(self, duration, fc_hz, fs_hz, impairment_factor=0):
        """
        Args:
            fc_hz (any): Center frequency
            fs_hz (any): Sampling frequency
            impairment_factor (int, optional): numerical value to widen ERB. Defaults to 0.
        """
        super(GammaToneFilter, self).__init__()

        self.duration = duration
        self.fc_hz = fc_hz
        self.fs_hz = fs_hz
        self.impairment_factor = impairment_factor

        ERB = (
            24.7 + 0.108 * fc_hz + impairment_factor
        )  # ? is this a good way to 'impair'
        b = 1.018 * ERB
        a = 6 / (-2 * pi * b) ** 4

        t = torch.arange(0, duration, 1 / fs_hz)
        self.impulse_response = (
            a**-1
            * t**3
            * torch.cos(2 * pi * fc_hz * t)
            * torch.exp(-2 * pi * b * t)
        )

    def forward(self, x):
        # Apply the gamma-tone filter to the input signal
        x = torch.flip(x, dims=[-1])
        filtered_signal = F.conv1d(
            x,
            self.impulse_response.view(1, 1, -1),
            padding=len(self.impulse_response) - 1,
        )
        return torch.flip(filtered_signal, dims=[-1])


class GammaToneFilterbank(nn.Module):
    """Generates a gammatone filterbank with given center frequencies
    as a torch module (non-trainable).
    """

    def __init__(self, duration, center_frequencies, fs_hz, impairment_factor=0):
        """_summary_

        Args:
            center_frequencies (list[int,...]): list of center frequencies
            fs_hz (_type_): sampling frequency
            impairment_factor (int, optional): numerical value that widens ERBs. Defaults to 0.
        """
        super(GammaToneFilterbank, self).__init__()
        self.filters = nn.ModuleList(
            [
                GammaToneFilter(duration, fc, fs_hz, impairment_factor)
                for fc in center_frequencies
            ]
        )

    def forward(self, x):
        # Apply the entire filterbank to the input signal
        outputs = [filter(x) for filter in self.filters]
        return outputs


class ImpairedModel(nn.Module):
    """Impaired model"""

    def __init__(self, num_taps, samplerate, center_frequencies, impairment_factor=1.0):
        super(ImpairedModel, self).__init__()
        self.num_taps = num_taps
        self.samplerate = samplerate
        self.center_frequencies = center_frequencies
        self.impairent_factor = impairment_factor
        self.duration = 0.25  # fixed at 1

        self.gamma_bank = GammaToneFilterbank(
            self.duration,
            self.center_frequencies,
            self.samplerate,
            self.impairent_factor,
        )

        self.gain = FIRFilter1D(self.num_taps)

    def forward(self, x):
        x = self.gain(x)
        signals = self.gamma_bank(x)
        out = torch.zeros_like(signals[0])
        for signal in signals:
            out += signal  # Sum outputs of filterbank
        # Normalize
        return torch.div(out, torch.max(torch.abs(out)))


class NormalModel(nn.Module):
    """Normal hearing model"""

    def __init__(self, samplerate, center_frequencies):
        super(NormalModel, self).__init__()
        self.samplerate = samplerate
        self.center_frequencies = center_frequencies
        self.duration = 0.25  # fixed

        self.gamma_bank = GammaToneFilterbank(
            self.duration, self.center_frequencies, self.samplerate
        )

    def forward(self, x):
        signals = self.gamma_bank(x)
        out = torch.zeros_like(signals[0])
        for signal in signals:
            out += signal  # Sum outputs of filterbank
        return torch.div(out, torch.max(torch.abs(out)))


class MyModel_v1(nn.Module):
    """My model"""

    def __init__(self, num_taps, samplerate, center_frequencies, impairment_factor=1.0):
        super(MyModel_v1, self).__init__()
        self.num_taps = num_taps
        self.samplerate = samplerate
        self.center_frequencies = center_frequencies
        self.impairent_factor = impairment_factor
        self.duration = 0.25  # fixed at 1

        self.normal_model = NormalModel(
            samplerate=samplerate, center_frequencies=self.center_frequencies
        )

        self.impaired_model = ImpairedModel(
            num_taps=self.num_taps,
            samplerate=self.samplerate,
            center_frequencies=self.center_frequencies,
            impairment_factor=self.impairent_factor,
        )

    def forward(self, x):
        out_HI = self.impaired_model(x)
        out_NH = self.normal_model(x)
        # Zero pad normal hearing to match HI
        out_NH = F.pad(
            input=out_NH, pad=[0, self.num_taps - 1], mode="constant", value=0
        )
        return out_NH, out_HI
