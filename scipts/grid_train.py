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

from tools.losses import l2_stft_loss as criterion


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
data_path = "data2_speech"

# training params
epochs = 20_000
n_window_it = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
# n_window_it = [64]
overlap = 0.5

models_to_train = len(n_window_it) * len(taps_it)
model_idx = 1 # init counter
# %%
for taps in taps_it:
    for n_window in n_window_it:
        print(f"Training model {model_idx} / {models_to_train}")
        model_idx += 1

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
            "window_size": n_window,
            "overlap": overlap,
        }

        # % Create folders to save data
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
        input_data = x  # use noise

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_curve = []

        for epoch in tqdm(
            range(epochs),
            desc=f"Training (taps {taps}, win size {n_window})",
            unit="epoch",
        ):
            # Forward pass and compute loss
            out_NH, out_HI = model(input_data)
            loss = criterion(out_NH, out_HI, n_fft=n_window, overlap=overlap)

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

# %% make animations
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join("..")))
from tools import plot, utils

# Fixed number of taps
import os
import pickle

# Define the root directory where your folder structure is located
root_directory = data_path

# Create a dictionary to organize the loaded data
data_dict = {}

# Recursively traverse the folder structure
for root, dirs, files in os.walk(root_directory):
    for file in files:
        if file == "freq_axes.pkl":
            # Extract taps and winsizes from the folder name
            folder_name = os.path.basename(root)
            taps, winsize = folder_name.split("_")[1], folder_name.split("_")[3]
            taps = int(taps)
            winsize = int(winsize)

            if taps not in data_dict:
                data_dict[taps] = {}

            if winsize not in data_dict[taps]:
                data_dict[taps][winsize] = {}

            # Load freq_axes.pkl
            with open(os.path.join(root, file), "rb") as f:
                data_dict[taps][winsize]["freq_axes"] = pickle.load(f)

        elif file == "spec_axes.pkl":
            # Load spec_axes.pkl
            with open(os.path.join(root, file), "rb") as f:
                data_dict[taps][winsize]["spec_axes"] = pickle.load(f)

for taps in taps_it:
    # List of win_sizes for which you want to create the animation
    win_sizes = n_window_it

    # Create a figure and axes outside the update function
    fig, ax = plt.subplots(1, 1)

    # Create a function to update the plot for each frame of the animation
    def update(frame):
        # Clear the current plot
        try:
            ax.clear()
        except: pass
        # Get the data for the current win_size
        win_size = win_sizes[frame]
        freq_axes_data = data_dict[taps][win_size]["freq_axes"]
        spec_axes_data = data_dict[taps][win_size]["spec_axes"]

        # reset axes
        freq_axes = []
        spec_axes = []

        # Axes order: [0] "Normal Hearing",
        # [1] "Impaired Hearing",
        # [2] "Ideal",
        # [3] "Compensation filter",
        # [4] "Wiener filter",

        freq_axes.append(freq_axes_data[0])
        freq_axes.append(freq_axes_data[1])
        freq_axes.append(freq_axes_data[3])
        freq_axes.append(freq_axes_data[0])

        spec_axes.append(spec_axes_data[0])
        spec_axes.append(spec_axes_data[1])
        spec_axes.append(spec_axes_data[3])
        spec_axes.append(
            utils.linear_interpolate(spec_axes_data[3], spec_axes_data[1])
            * spec_axes_data[1]
        )  # compensation filter multiplied with impaired filter

        labels = [
            "Normal Hearing",
            "Impaired Hearing",
            "Compensation filter",
            "Result",
        ]

        # Plot the data (customize this according to your data structure)
        plot.spec_anim(
            ax=ax,
            freq_axes=freq_axes,
            spec_axes=spec_axes,
            plot_phase=False,
            units="dB",
            ylim=[-100, 0],
            title=f"Windows: {win_size / samplerate * (1-overlap)**-1 * 1000} msec, Taps: {taps}",
            labels=labels,
        )

    # Create a figure and an animation
    ani = animation.FuncAnimation(
        fig, update, frames=len(win_sizes), repeat=True, blit=False
    )
    # Define the filename and format for saving the animation
    save_filename = f"taps_{taps}.mp4"  # Change the filename and format as needed
    save_filename = os.path.join(data_path, f"taps_{taps}.mp4")

    # Save the animation as a video file (e.g., MP4)
    print(f"Saving {save_filename} with ffmpeg")
    ani.save(
        save_filename, writer="ffmpeg"
    )  # You may need to install FFmpeg for this to work
