from matplotlib import pyplot as plt
import seaborn as sns
from .utils import *
import numpy as np

# Enable LaTeX rendering if available
try:
    plt.rcParams["text.usetex"] = True
except:
    pass


COLOR_PALETTE = "hls"
FONT_SIZE = 12


def magspec(
    freq_axes,
    spec_axes,
    units="dB",
    title=None,
    labels=None,
    out_path=None,
    mode="db",
    colors=None,
    format="png",
    font_size=FONT_SIZE,
    xlim=[20, 20e3],
):
    plt.rcParams.update({"font.size": font_size})

    # number of signals to plot
    n_signals = len(freq_axes)

    # fill with empty strings
    if labels is None:
        labels = []
        for _ in range(n_signals):
            labels.append("")

    # fill with auto color palette
    if colors is None:
        colors = sns.color_palette(COLOR_PALETTE, n_colors=n_signals)

    fig, ax = plt.subplots(figsize=(10, 6))

    lines = []

    for ii, (frq, spec) in enumerate(zip(freq_axes, spec_axes)):
        # Magnitude axis
        ax.set_xscale("log")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(units)
        ax.xaxis.grid(True, which="both", ls="--")
        ax.yaxis.grid(True)
        ax.margins(x=0)

        # Calculate magnitude and phase
        magnitude = get_mag(spec, mode=mode)

        # Plot the magnitude spectrum on the left y-axis
        lines += ax.plot(frq, magnitude, color=colors[ii], label=labels[ii], alpha=1.0)

        try:
            # Set legend
            plt.legend(loc="best")
        except: pass

    plt.xticks(
        [20, 50, 100, 1e3, 10e3, 20e3],
        ["20", "50", "100", "1k", "10k", "20k"],
    )
    plt.xlim(xlim)

    if title is not None:
        plt.title(title)

    if out_path == None:
        plt.show()
    else:
        print(f"saving figure to {out_path} as {format}")
        plt.savefig(f"{out_path}", format=format, bbox_inches="tight", dpi=300)

def timeseries(
    amplitude_axes,
    samplerate = None,
    units="Amplitude",
    title=None,
    labels=None,
    out_path=None,
    colors=None,
    xlim=None,
    format="png",
    font_size=FONT_SIZE,
):
    plt.rcParams.update({"font.size": font_size})

    # number of signals to plot
    n_signals = len(amplitude_axes)
    time_axes = []
    
    # plot with indices
    if samplerate is None:
        for signal in amplitude_axes:
            time_axes.append(list(range(len(signal))))
            xlab = "Index (n)"

    else:
        for signal in amplitude_axes:
            time_axes.append(np.array(range(len(signal)))/samplerate)
            xlab = "Time (s)"


    # fill with empty strings
    if labels is None:
        labels = []
        for _ in range(n_signals):
            labels.append("")

    # fill with auto color palette
    if colors is None:
        colors = sns.color_palette(COLOR_PALETTE, n_colors=n_signals)

    fig, ax = plt.subplots(figsize=(10, 6))

    lines = []

    for ii, (t, amplitude) in enumerate(zip(time_axes, amplitude_axes)):
        # Magnitude axis
        ax.set_xscale("linear")
        ax.set_xlabel(xlab)
        ax.set_ylabel(units)
        ax.xaxis.grid(True, which="both", ls="--")
        ax.yaxis.grid(True)
        ax.margins(x=0)

        # Plot the time signal
        lines += ax.plot(t, amplitude, color=colors[ii], label=labels[ii], alpha=1.0)

        try:
            # Set legend
            plt.legend(loc="best")
        except: pass

    plt.xlim(xlim)

    if title is not None:
        plt.title(title)

    if out_path == None:
        plt.show()
    else:
        print(f"saving figure to {out_path} as {format}")
        plt.savefig(f"{out_path}", format=format, bbox_inches="tight", dpi=300)

def bode_digital(w,h):
    fig, ax1 = plt.subplots()
    ax1.set_title('Digital filter frequency response')
    ax1.plot(w, 20 * np.log10(abs(h)), 'b')
    ax1.set_ylabel('Amplitude [dB]', color='b')
    ax1.set_xlabel('Frequency [rad/sample]')
    ax2 = ax1.twinx()
    angles = np.unwrap(np.angle(h))
    ax2.plot(w, angles, 'g')
    ax2.set_ylabel('Angle (radians)', color='g')
    ax2.grid(True)
    ax2.axis('tight')
    plt.show()

def filter_coeffs(filter_coeffs):
    """
    Plots the FIR filter coefficients.

    Parameters:
        filter_coeffs (list): List of filter coefficients.
    """
    num_taps = len(filter_coeffs)
    x_ticks = np.arange(num_taps)  # Whole number ticks for each tap
    plt.stem(x_ticks, filter_coeffs, basefmt='b', use_line_collection=True)
    plt.xlabel('Tap')
    plt.ylabel('Coefficient Value')
    plt.xticks(x_ticks)  # Set the x-axis ticks to whole numbers
    plt.title('FIR Filter Coefficients')
    plt.show()

