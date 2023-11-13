# %%
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch
import torch.optim as optim
import torchaudio
from scipy import signal
from tqdm import tqdm

print(torch.__version__)

import pickle

sys.path.insert(0, os.path.abspath(os.path.join("..")))

from tools import plot, utils
from tools.nn_modules import *

# fix random seed
seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from tools.losses import spectral_loss as criterion


# Helper functions
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


def batch_2_np(batch):
    return batch[0][0].detach().numpy()


# *signal params
samplerate = int(8e3)
center_frequencies = [1000]
n_batch = 1
n_samples = int(8e3)
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
    x = audio
    print("Training using single speech signal")
elif signal_type == "noise_pulse":
    x_noise = torch.randn(n_batch, n_channels, n_samples)
    x_noise[:, :, n_samples // 2 :] = 0.0
    x = x_noise
    print("Training using single noise pulse signal")

# *model params
band_width_factor = 5
taps_it = [16, 32, 64, 128, 256, 512, 1024]
# taps_it = [128]

# *Change destination path as needed
data_path = "logl2_multiscale_noise"

# training params
epochs = 40_000
overlap = 0.5 # 50% overlap

models_to_train = len(taps_it)
model_idx = 1 # init counter
# %%
for taps in taps_it:
    print(f"Training model {model_idx} / {models_to_train}")
    model_idx += 1 # increment counter

    model = MyModel_v2(
        fir_numtaps=taps,
        gamma_numtaps=128,
        samplerate=samplerate,
        center_frequencies=center_frequencies,
        band_width_factor=band_width_factor,
    )

    model_params = {
        "band_width_factor": band_width_factor,
        "taps": taps,
        "epochs": epochs,
        "overlap": overlap,
    }

    # % Create folders to save data
    f_path = os.path.join(data_path, f"taps_{taps}_multiscale")
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
    input_data = x  # use noise

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_curve = []

    for epoch in tqdm(
        range(epochs),
        desc=f"Training (taps {taps})",
        unit="epoch",
    ):
        # Forward pass and compute loss
        out_NH, out_HI = model(input_data)
        loss = criterion(out_NH, out_HI, overlap=overlap, loss_type='L2', logmag_weight=1.0, mag_weight=0.0)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update loss curve
        loss_curve.append(loss.item())

    # Inference model one last time
    out_NH, out_HI = model(input_data)

    # save audio data in /audio
    audio = batch_2_np(out_HI)  # convert to np
    audio = audio / max(np.abs(audio))  # normalize
    
    sf.write(
        file=os.path.join(audio_path, "HI_out.wav"),
        data=audio,
        samplerate=samplerate,
    )
    audio = batch_2_np(out_NH)  # convert to np
    audio = audio / max(np.abs(audio))  # normalize
    sf.write(
        file=os.path.join(audio_path, "NH_out.wav"),
        data=audio,
        samplerate=samplerate,
    )

    # * save plots in /plots
    freq_axes = []
    spec_axes = []
    time_signals = []

    for filter in model.normal_model.gamma_bank.filters:
        h, f = utils.get_spectrum(filter.b, samplerate=samplerate)
        freq_axes.append(f)
        spec_axes.append(h)
        time_signals.append(filter.b)

    for filter in model.impaired_model.gamma_bank.filters:
        h, f = utils.get_spectrum(filter.b, samplerate=samplerate)
        freq_axes.append(f)
        spec_axes.append(h)
        time_signals.append(filter.b)

    n_fft = len(f)

    # get learned FIR coefficients
    coeffs = model.hearing_aid_model.filter_taps.detach().numpy()
    w, h = signal.freqz(coeffs, worN=n_fft)
    freq_axes.append(w / (2 * pi) * samplerate)
    spec_axes.append(h / max(h))

    spec_axes.append(
        spec_axes[1] * spec_axes[2]
    )  # compensation spectrum multiplied with impaired spectrum
    freq_axes.append(f)

    labels = [
        "Normal Hearing",
        "Impaired Hearing",
        "Compensation filter",
        "Result",
    ]

    # Plot IR in time-domain and magnitude repsonse
    plot.magspec(
        freq_axes=freq_axes,
        spec_axes=spec_axes,
        labels=labels,
        out_path=os.path.join(figs_path, "freq_responses.png"),
        mode="dB",
    )
    plot.filter_taps(
        amplitude_axes=[model.hearing_aid_model.filter_taps.detach().numpy()],
        out_path=os.path.join(figs_path, "filter_taps.png"),
        # units="Loss (dB)",
    )
    # * save metadata as pickle
    save_list_as_pickle(
        filename=os.path.join(f_path, "freq_axes.pkl"), data=freq_axes
    )
    save_list_as_pickle(
        filename=os.path.join(f_path, "spec_axes.pkl"), data=spec_axes
    )

    # plot loss_curve
    plot.timeseries(
        amplitude_axes=[20 * np.log10(loss_curve)],
        out_path=os.path.join(figs_path, "loss_curve.png"),
        units="Loss (dB)",
    )
    # save data as pickle
    save_list_as_pickle(
        filename=os.path.join(f_path, "loss_curve.pkl"), data=loss_curve
    )