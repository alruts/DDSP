import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as AF
from math import pi

import numpy as np
import pyfilterbank
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
            x,
            self.filter_taps.view(1, 1, -1),
            padding=self.num_taps - 1,  # bias=False
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
            x,
            self.filter_taps.view(1, 1, -1),
            padding=self.num_taps - 1,  # bias=False
        )

        return filtered_signal


class GammaToneFIRFilter(nn.Module):
    """Generates a gammatone filter as a torch module (non-trainable)."""

    def __init__(self, center_frequency, samplerate, band_width_factor=1.0):
        super(GammaToneFIRFilter, self).__init__()

        self.center_frequency = center_frequency
        self.samplerate = samplerate
        self.delta_q = band_width_factor


    def forward(self, x):
        # Apply the gamma-tone filter to the input signal
        x = torch.flip(x, dims=[-1])
        filtered_signal = F.conv1d(
            x,
            self.impulse_response.view(1, 1, -1),
            padding=len(self.impulse_response) - 1,
            # bias=False,
        )
        return torch.flip(filtered_signal, dims=[-1])
    
def design_gammatone(freq, samplerate, ftype, order=4, numtaps=32, band_width_factor=1.0):
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
    filter_types = ['fir', 'iir']
    if not 0 < freq < samplerate / 2:
        raise ValueError("The frequency must be between 0 and {}"
                         " (nyquist), but given {}.".format(samplerate / 2, freq))
    if ftype not in filter_types:
        raise ValueError('ftype must be either fir or iir.')

    # Calculate FIR gammatone filter
    if ftype == 'fir':
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
    
    elif ftype == 'iir':
        raise NotImplementedError()
    else:
        raise ValueError('Invalid filter type')


class GammaToneIIRFilter(nn.Module):
    """Generates a gammatone filter as a torch module (non-trainable)."""

    def __init__(self, center_frequency, samplerate, band_width_factor=1.0):
        super(GammaToneIIRFilter, self).__init__()
        self.center_frequency = center_frequency
        self.samplerate = samplerate
        self.band_width_factor = band_width_factor

        b, a = pyfilterbank.gammatone.design_filter(
            sample_rate=self.samplerate,
            order=32,
            centerfrequency=self.center_frequency,
            band_width=None,
            band_width_factor=self.band_width_factor,
            attenuation_half_bandwidth_db=-3,
        )
        
        self.b = torch.complex(torch.Tensor([b[0], 0.0]), torch.Tensor([0,0]))
        self.a = torch.complex(torch.Tensor(np.real(a)), torch.Tensor(np.imag(a)))

    def forward(self, x):
        # Apply the gamma-tone filter to the input signal
        x_real = AF.lfilter(x, torch.real(self.a), torch.real(self.b))
        x_imag = AF.lfilter(x, torch.imag(self.a), torch.imag(self.b))
        
        print(x_imag)
        x = torch.sqrt(x_real**2)
        return x


class GammaToneFIRFilterbank(nn.Module):
    """Generates a gammatone filterbank with given center frequencies
    as a torch module (non-trainable).
    """

    def __init__(self, duration, center_frequencies, samplerate, delta_q=0):
        """_summary_

        Args:
            center_frequencies (list[int,...]): list of center frequencies
            samplerate (float): sampling frequency
            delta_q (int, optional): numerical value that widens ERBs. Defaults to 0.
        """
        super(GammaToneFIRFilterbank, self).__init__()
        self.filters = nn.ModuleList(
            [
                GammaToneFIRFilter(duration, fc, samplerate, delta_q)
                for fc in center_frequencies
            ]
        )

    def forward(self, x):
        # Apply the entire filterbank to the input signal
        outputs = [filter(x) for filter in self.filters]
        return outputs


class GammaToneIIRFilterbank(nn.Module):
    """Generates a gammatone filterbank with given center frequencies
    as a torch module (non-trainable).
    """

    def __init__(self, center_frequencies, samplerate, band_width_factor=1.0):
        """_summary_

        Args:
            center_frequencies (list[int,...]): list of center frequencies
            samplerate (_type_): sampling frequency
            delta_q (int, optional): numerical value that widens ERBs. Defaults to 0.
        """
        super(GammaToneIIRFilterbank, self).__init__()
        self.filters = nn.ModuleList(
            [
                GammaToneIIRFilter(fc, samplerate, band_width_factor)
                for fc in center_frequencies
            ]
        )

    def forward(self, x):
        # Apply the entire filterbank to the input signal
        outputs = [filter(x) for filter in self.filters]
        return outputs


class HearingModel(nn.Module):
    """Hearing model"""

    def __init__(self, samplerate, center_frequencies, delta_q=0):
        """
        Args:
            samplerate (int):
            center_frequencies (list[float,..]): list of center frequencies
            delta_q (float): delta Q factor to be added to normal base
        """
        super(HearingModel, self).__init__()
        self.samplerate = samplerate
        self.center_frequencies = center_frequencies
        self.delta_q = delta_q
        self.duration = 0.25  # fixed

        self.gfb = GammaToneFIRFilterbank(
            self.duration, self.center_frequencies, self.samplerate, self.delta_q
        )

    def forward(self, x):
        signals = self.gfb(x)
        out = torch.zeros_like(signals[0])
        for signal in signals:
            out += signal  # Sum outputs of filterbank
        return torch.div(out, len(signals))  # ? how to fix this?


class HearingModel_v2(nn.Module):
    """Hearing model"""

    def __init__(self, center_frequencies, samplerate, band_width_factor=1.0):
        """
        Args:
            samplerate (int):
            center_frequencies (list[float,..]): list of center frequencies
            delta_q (float): delta Q factor to be added to normal base
        """
        super(HearingModel_v2, self).__init__()
        self.samplerate = samplerate
        self.center_frequencies = center_frequencies
        self.band_width_factor = band_width_factor

        self.gfb = GammaToneIIRFilterbank(
            self.center_frequencies, self.samplerate, self.band_width_factor
        )

    def forward(self, x):
        signals = self.gfb(x)
        out = torch.zeros_like(signals[0])
        for signal in signals:
            out += signal  # Sum outputs of filterbank
        return torch.div(out, len(signals))  # ? how to fix this?


class MyModel_v1(nn.Module):
    """My model"""

    def __init__(self, num_taps, samplerate, center_frequencies, delta_q=1.0):
        """_summary_

        Args:
            num_taps (int): number of taps for the FIR filter
            samplerate (int):
            center_frequencies (list[int,...]): list of gammatone filter fcs
        """
        super(MyModel_v1, self).__init__()
        self.num_taps = num_taps
        self.samplerate = samplerate
        self.center_frequencies = center_frequencies
        self.delta_q = delta_q
        self.duration = 0.25  # fixed at 1

        self.normal_model = HearingModel(
            samplerate=samplerate, center_frequencies=self.center_frequencies
        )

        self.impaired_model = HearingModel(
            samplerate=self.samplerate,
            center_frequencies=self.center_frequencies,
            delta_q=self.delta_q,
        )

        self.hearing_aid_model = FIRFilter1D(self.num_taps)

    def forward(self, x):
        out_HI = self.hearing_aid_model(self.impaired_model(x))
        out_NH = self.normal_model(x)
        # Zero pad normal hearing to match HI
        out_NH = F.pad(
            input=out_NH, pad=[0, self.num_taps - 1], mode="constant", value=0
        )
        return out_NH, out_HI


class MyModel_v2(nn.Module):
    """My model"""

    def __init__(self, num_taps, samplerate, center_frequencies, band_width_factor=5.0):
        """This model uses
        Args:
            num_taps (int): number of taps for the FIR filter
            samplerate (int):
            center_frequencies (list[int,...]): list of gammatone filter fcs
        """
        super(MyModel_v2, self).__init__()
        self.num_taps = num_taps
        self.samplerate = samplerate
        self.center_frequencies = center_frequencies
        self.band_width_factor = band_width_factor

        self.normal_model = HearingModel_v2(self.center_frequencies, self.samplerate)

        self.impaired_model = HearingModel_v2(
            self.center_frequencies, self.samplerate, self.band_width_factor
        )

        self.hearing_aid_model = FIRFilter1D(self.num_taps)

    def forward(self, x):
        out_HI = self.hearing_aid_model(self.impaired_model(x))
        out_NH = self.normal_model(x)
        # Zero pad normal hearing to match HI
        out_NH = F.pad(
            input=out_NH, pad=[0, self.num_taps * 2 - 2], mode="constant", value=0
        )
        return out_NH, out_HI
