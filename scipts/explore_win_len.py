# %%
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from scipy import signal
from scipy.linalg import toeplitz
from tqdm import tqdm

print(torch.__version__)

import pickle

sys.path.insert(0, os.path.abspath(os.path.join("..")))

from tools import plot, utils
from tools.nn_modules import *

seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def save_dict_to_json(data, filename):
    """
    Save a dictionary to a JSON file.
    """
    with open(filename, "w") as json_file:
        json.dump(data, json_file, indent=4)


def save_list_as_pickle(filename, data):
    """
    Save a list as a pickle file.

    :param filename: The name of the pickle file (including the .pkl extension).
    :param data: The list to be saved.
    """
    with open(filename, "wb") as file:
        pickle.dump(data, file)


def abs_stft_loss(target, estimate, n_fft, overlap):
    """https://static1.squarespace.com/static/5554d97de4b0ee3b50a3ad52/t/5fb1e9031c7089551a30c2e4/1605495044128/DMRN15__auraloss__Audio_focused_loss_functions_in_PyTorch.pdf"""
    batch_size, _, n_samples = estimate.shape

    # reshape audio signal for stft function
    target = target.reshape((batch_size, n_samples))
    estimate = estimate.reshape((batch_size, n_samples))

    hop_length = int((1 - overlap) * n_fft)

    # compute stfts for each batch
    spec_target = torch.stft(
        target, n_fft=n_fft, hop_length=hop_length, return_complex=True
    )
    spec_estimate = torch.stft(
        estimate, n_fft=n_fft, hop_length=hop_length, return_complex=True
    )

    # convert to dB
    spec_target_dB = torch.abs(spec_target)
    spec_estimate_dB = torch.abs(spec_estimate)

    loss = torch.mean(torch.abs(spec_target_dB - spec_estimate_dB))

    return loss

def xcorr(x, y, k):
    N = min(len(x), len(y))
    r_xy = (1 / N) * signal.correlate(
        x, y, "full"
    )  # reference implementation is unscaled
    return r_xy[N - k - 1 : N + k]

def batch_2_np(batch):
    return batch[0][0].detach().numpy()


# signal params
samplerate = 22050
center_frequencies = [1000]
n_batch = 1
n_samples = 22050
n_channels = 1

signal_type = "noise_pulse"

signal_params = {
    "samplerate": samplerate,
    "center_frequencies": center_frequencies,
    "n_batch": n_batch,
    "n_samples": n_samples,
    "n_channels": n_channels,
    "signal_type": signal_type,
}

if signal_type == "speech":
    audio, samplerate = torchaudio.load(r"..\audio\examples\Human_voice\1.wav")
    n_channels, n_samples = audio.shape
    audio = audio.reshape((n_batch, n_channels, n_samples))
    print("Training using single speech signal")
elif signal_type == "noise_pulse":
    x_noise = torch.randn(n_batch, n_channels, n_samples)
    x_noise[:, :, 10000:] = 0.0
    print("Training using single noise pulse signal")

# model params
delta_q = 5
taps_it = [4, 8, 16, 32, 64, 128, 256, 512]

# training params
epochs = 1200
n_window_it = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
overlap = 0.5

models_to_train = len(n_window_it) * len(taps_it)
curr_model = 1

for taps in taps_it:
    for n_window in n_window_it:
        print(f"Training model {curr_model} / {models_to_train}")
        curr_model += 1

        model = MyModel_v1(
            num_taps=taps,
            samplerate=samplerate,
            center_frequencies=center_frequencies,
            delta_q=delta_q,
        )

        model_params = {
            "delta_q": delta_q,
            "taps": taps,
            "epochs": epochs,
            "window_size": n_window,
            "overlap": overlap,
        }

        # % Create folders to save data
        data_path = "data"
        f_path = os.path.join(data_path, f"taps_{taps}_winsize_{n_window}")
        audio_path = os.path.join(f_path, "01_audio")
        figs_path = os.path.join(f_path, "02_figs")

        paths = [
            data_path,
            f_path,
            audio_path,
            figs_path,
        ]

        for path in paths:
            # Check if the directory exists, and if not, create it
            if not os.path.exists(path):
                os.makedirs(path)
                print(f"Directory created: {path}")

        # save all the metadata
        save_dict_to_json(model_params, os.path.join(f_path, "model_params.json"))
        save_dict_to_json(signal_params, os.path.join(f_path, "signal_params.json"))

        # % Train model with specific loss and data
        input_data = x_noise  # use noise

        optimizer = optim.Adam(model.parameters(), lr=0.001)

        loss_curve = []

        for epoch in tqdm(
            range(epochs),
            desc=f"Training (taps {taps}, win size {n_window})",
            unit="epoch",
        ):
            # Forward pass and compute loss
            out_NH, out_HI = model(input_data)
            loss = abs_stft_loss(out_NH, out_HI, n_fft=n_window, overlap=overlap)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # append to loss
            loss_curve.append(loss.item())

        # inference on model one last time
        out_NH, out_HI = model(input_data)

        # save audio data in /audio
        audio = batch_2_np(out_HI) # convert to np
        audio = audio / max(np.abs(audio)) # normalize
        sf.write(
            file=os.path.join(audio_path, "HI_out.wav"),
            data=audio,
            samplerate=samplerate,
        )
        audio = batch_2_np(out_NH) # convert to np
        audio = audio / max(np.abs(audio)) # normalize
        sf.write(
            file=os.path.join(audio_path, "NH_out.wav"),
            data=audio,
            samplerate=samplerate,
        )

        # save plots in /plots

        freq_axes = []
        spec_axes = []
        time_signals = []

        for filter in model.normal_model.gamma_bank.filters:
            h, f = utils.get_spectrum(filter.impulse_response, samplerate=samplerate)
            freq_axes.append(f)
            spec_axes.append(h)
            time_signals.append(filter.impulse_response)
        for filter in model.impaired_model.gamma_bank.filters:
            h, f = utils.get_spectrum(filter.impulse_response, samplerate=samplerate)
            freq_axes.append(f)
            spec_axes.append(h)
            time_signals.append(filter.impulse_response)

        for filternh, filterhi in zip(
            model.normal_model.gamma_bank.filters,
            model.impaired_model.gamma_bank.filters,
        ):
            hnh, f = utils.get_spectrum(
                filternh.impulse_response, samplerate=samplerate
            )
            hhi, f = utils.get_spectrum(
                filterhi.impulse_response, samplerate=samplerate
            )
            h = hnh / hhi
            freq_axes.append(f)
            spec_axes.append(h)
            time_signals.append(filter.impulse_response)

        # get learned FIR coefficients
        coeffs = model.hearing_aid_model.filter_taps.detach().numpy()
        w, h = signal.freqz(coeffs)
        freq_axes.append(w / (2 * pi) * samplerate)
        spec_axes.append(h / max(h))

        # Get Wiener filter for comparison
        L = taps
        x_ = batch_2_np(model.impaired_model.gamma_bank(input_data)[0])
        s = batch_2_np(model.normal_model.gamma_bank(input_data)[0])
        r_xx = xcorr(x_, x_, L - 1)
        R_xx = toeplitz(r_xx[L - 1 :])
        r_sx = xcorr(s, x_, L - 1)
        theta = np.linalg.solve(R_xx, r_sx[L - 1 :])
        w, h = signal.freqz(theta)
        
        freq_axes.append(w / (2 * pi) * samplerate)
        spec_axes.append(h)

        labels = [
            "Normal Hearing",
            "Impaired Hearing",
            "Ideal",
            "Compensation filter",
            "Wiener filter",
        ]

        # Plot IR in time-domain and magnitude repsonse
        plot.magspec(
            freq_axes=freq_axes,
            spec_axes=spec_axes,
            labels=labels,
            out_path=os.path.join(figs_path, "freq_responses.png"),
        )
        # save data as pickle
        save_list_as_pickle(
            filename=os.path.join(f_path, "freq_axes.pkl"), data=freq_axes
        )
        save_list_as_pickle(
            filename=os.path.join(f_path, "spec_axes.pkl"), data=spec_axes
        )

        # plot loss_curve
        plot.timeseries(
            amplitude_axes=[loss_curve],
            out_path=os.path.join(figs_path, "loss_curve.png"),
            units="Loss",
        )
        # save data as pickle
        save_list_as_pickle(
            filename=os.path.join(f_path, "loss_curve.pkl"), data=loss_curve
        )
