import torch
import torch.nn as nn
import torch.nn.functional as F
from math import pi

import numpy as np
import operator
from scipy._lib._util import float_factorial


def _hz_to_erb(hz):
    """
    Utility for converting from frequency (Hz) to the
    Equivalent Rectangular Bandwidth (ERB) scale
    ERB = frequency / EarQ + minBW
    """
    EarQ = 9.26449
    minBW = 24.7
    return hz / EarQ + minBW

def fft_conv(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """FFT conv in PyTorch. This impementation is fully equivalent to linear conv

    Args:
        x (torch.tensor): signal to convolve with filter kernel of shape (Batch, Channels, N)
        h (torch.tensor): filter kernel to convolve with signal of shape (Batch, Channels, L)

    Returns:
        torch.tensor: convolution of x and h of shape (Batch, Channels, M = N + L - 1)
    """

    _, _, N = x.shape
    _, _, L = h.shape
    M = N + L - 1

    x_pad = F.pad(x, (0, M - N), mode="constant")
    h_pad = F.pad(h, (0, M - L), mode="constant")

    X = torch.fft.fft(x_pad)
    H = torch.fft.fft(h_pad)

    Y = X * H
    y = torch.fft.ifft(Y).real

    return y

def design_gammatone(
    freq, samplerate, ftype, order=4, numtaps=32, band_width_factor=1.0
):
    """
    Adapted from scipy.signal.gammatone  https://github.com/scipy/scipy/blob/v1.11.3/scipy/signal/_filter_design.py#L5393-L5574
    """
    # Converts freq to float
    freq = float(freq)

    # Set sampling rate if not passed
    if samplerate is None:
        samplerate = 2
    samplerate = float(samplerate)

    # Check for invalid cutoff frequency or filter type
    ftype = ftype.lower()
    filter_types = ["fir", "iir"]
    if not 0 < freq < samplerate / 2:
        raise ValueError(
            "The frequency must be between 0 and {}"
            " (nyquist), but given {}.".format(samplerate / 2, freq)
        )
    if ftype not in filter_types:
        raise ValueError("ftype must be either fir or iir.")

    # Calculate FIR gammatone filter
    if ftype == "fir":
        # Set order and numtaps if not passed
        if order is None:
            order = 4
        order = operator.index(order)

        if numtaps is None:
            numtaps = max(int(samplerate * 0.015), 15)
        numtaps = operator.index(numtaps)

        # Check for invalid order
        if not 0 < order <= 24:
            raise ValueError("Invalid order: order must be > 0 and <= 24.")

        # Gammatone impulse response settings
        t = np.arange(numtaps) / samplerate
        bw = 1.019 * _hz_to_erb(freq) * band_width_factor

        # Calculate the FIR gammatone filter
        b = (t ** (order - 1)) * np.exp(-2 * np.pi * bw * t)
        b *= np.cos(2 * np.pi * freq * t)

        # Scale the FIR filter so the frequency response is 1 at cutoff
        scale_factor = 2 * (2 * np.pi * bw) ** (order)
        scale_factor /= float_factorial(order - 1)
        scale_factor /= samplerate
        b *= scale_factor
        a = [1.0]
        return b, a

    elif ftype == "iir":
        raise NotImplementedError()
    else:
        raise ValueError("Invalid filter type")


class FIRFilter1D(nn.Module):
    """Implements an arbitrary phase FIR filter"""

    def __init__(self, num_taps):
        super(FIRFilter1D, self).__init__()
        self.num_taps = num_taps
        # Initialize filter taps as learnable parameters
        self.filter_taps = nn.Parameter(torch.randn(num_taps, 1), requires_grad=True)

    def forward(self, x):
        return fft_conv(x,self.filter_taps.view(1, 1, -1))


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
        return fft_conv(x,self.filter_taps.view(1, 1, -1))


class GammatoneFilter(nn.Module):
    """Generates a gammatone filter as a torch module (non-trainable)."""

    def __init__(
        self,
        center_frequency,
        samplerate,
        ftype="fir",
        order=4,
        numtaps=128,
        band_width_factor=1.0,
    ):
        super(GammatoneFilter, self).__init__()

        self.center_frequency = center_frequency
        self.samplerate = samplerate
        self.band_width_factor = band_width_factor
        self.order = order
        self.numtaps = numtaps
        self.ftype = ftype

        b, a = design_gammatone(
            freq=self.center_frequency,
            samplerate=self.samplerate,
            ftype=self.ftype,
            numtaps=self.numtaps,
            order=self.order,
            band_width_factor=self.band_width_factor,
        )

        self.b = torch.Tensor(b)
        self.a = torch.Tensor(a)

    def forward(self, x):
        # Apply the gamma-tone filter to the input signal
        x = torch.flip(x, dims=[-1])
        filtered_signal = F.conv1d(
            x,
            self.b.view(1, 1, -1),
            padding=self.numtaps - 1,
        )
        return fft_conv(x,self.b.view(1, 1, -1))


class GammatoneFilterbank(nn.Module):
    """Generates a gammatone filterbank with given center frequencies
    as a torch module (non-trainable).
    """

    def __init__(
        self, center_frequencies, samplerate, ftype, numtaps, band_width_factor=1.0
    ):
        """_summary_

        Args:
            center_frequencies (list[int,...]): list of center frequencies
            samplerate (float): sampling frequency
            band_width_factor (int, optional): scalar value that multiplies  ERBs. Defaults to 1.0.
        """
        super(GammatoneFilterbank, self).__init__()
        self.center_frequencies = center_frequencies
        self.samplerate = samplerate
        self.ftype = ftype
        self.numtaps = numtaps
        self.band_width_factor = band_width_factor

        self.filters = nn.ModuleList(
            [
                GammatoneFilter(
                    center_frequency=fc,
                    samplerate=self.samplerate,
                    ftype=self.ftype,
                    numtaps=self.numtaps,
                    band_width_factor=self.band_width_factor,
                )
                for fc in self.center_frequencies
            ]
        )

    def forward(self, x):
        # Apply the entire filterbank to the input signal
        outputs = [filter(x) for filter in self.filters]
        return outputs


class HearingModel(nn.Module):
    """Hearing model"""

    def __init__(
        self, center_frequencies, samplerate, ftype, gamma_numtaps, band_width_factor=1.0
    ):
        """
        Args:
            samplerate (int):
            center_frequencies (list[float,..]): list of center frequencies
            band_width_factor (float): scalar for ERB modification.
        """
        super(HearingModel, self).__init__()
        self.samplerate = samplerate
        self.center_frequencies = center_frequencies
        self.band_width_factor = band_width_factor
        self.ftype = ftype
        self.numtaps = gamma_numtaps

        self.gamma_bank = GammatoneFilterbank(
            center_frequencies=self.center_frequencies,
            samplerate=self.samplerate,
            ftype=self.ftype,
            numtaps=self.numtaps,
            band_width_factor=self.band_width_factor,
        )

        # add future processing here

    def forward(self, x):
        signals = self.gamma_bank(x)
        out = torch.zeros_like(signals[0])
        for signal in signals:
            out += signal  # Sum outputs of filterbank
        return torch.div(out, len(signals))  # ? how to fix this?


class MyModel_v2(nn.Module):
    """My model"""

    def __init__(self, fir_numtaps, samplerate, center_frequencies, gamma_numtaps,band_width_factor=5.0):
        """This model uses
        Args:
            num_taps (int): number of taps for the FIR filter
            samplerate (int): samplerate
            center_frequencies (list[int,...]): list of gammatone filter center frequencies
        """
        super(MyModel_v2, self).__init__()
        self.fir_numtaps = fir_numtaps
        self.samplerate = samplerate
        self.center_frequencies = center_frequencies
        self.band_width_factor = band_width_factor
        self.gamma_numtaps = gamma_numtaps

        self.normal_model = HearingModel(
            center_frequencies=self.center_frequencies,
            samplerate=self.samplerate,
            ftype="fir",
            gamma_numtaps=self.gamma_numtaps,
            band_width_factor=1.0,
        )

        self.impaired_model = HearingModel(
            center_frequencies=self.center_frequencies,
            samplerate=self.samplerate,
            ftype="fir",
            gamma_numtaps=self.gamma_numtaps,
            band_width_factor=self.band_width_factor,
        )

        self.hearing_aid_model = FIRFilter1D(self.fir_numtaps)

    def forward(self, x):
        out_HI = self.hearing_aid_model(self.impaired_model(x))
        out_NH = self.normal_model(x)
        # Zero pad normal hearing to match HI
        out_NH = F.pad(
            input=out_NH, pad=[0, self.fir_numtaps - 1], mode="constant", value=0
        )
        return out_NH, out_HI
