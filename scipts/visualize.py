# %%
import os
import pickle

# Define the root directory where your folder structure is located
root_directory = './data'

# Create a dictionary to organize the loaded data
data_dict = {}

# Recursively traverse the folder structure
for root, dirs, files in os.walk(root_directory):
    for file in files:
        if file == 'freq_axes.pkl':
            # Extract taps and winsizes from the folder name
            folder_name = os.path.basename(root)
            taps, winsize = folder_name.split('_')[1], folder_name.split('_')[3]
            taps = int(taps)
            winsize = int(winsize)

            if taps not in data_dict:
                data_dict[taps] = {}

            if winsize not in data_dict[taps]:
                data_dict[taps][winsize] = {}

            # Load freq_axes.pkl
            with open(os.path.join(root, file), 'rb') as f:
                data_dict[taps][winsize]['freq_axes'] = pickle.load(f)

        elif file == 'spec_axes.pkl':
            # Load spec_axes.pkl
            with open(os.path.join(root, file), 'rb') as f:
                data_dict[taps][winsize]['spec_axes'] = pickle.load(f)

# Now, data_dict contains the organized data separated by taps and winsizes
# You can access the loaded data using the taps and winsizes as keys
# For example, to access freq_axes for taps=128 and winsize=1024:



freq_axes_data = data_dict[128][1024]['freq_axes']
spec_axes_data = data_dict[128][1024]['spec_axes']

# You can repeat the above lines to access data for other taps and winsizes as needed.

# delta_q = 5
# taps_it = [4, 8, 16, 32, 64, 128, 256, 512]
# n_window_it = 
# %%
# %matplotlib notebook
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys 
sys.path.insert(0, os.path.abspath(os.path.join("..")))

from tools import plot, utils


# Fixed number of taps
taps = 128

# List of win_sizes for which you want to create the animation
win_sizes = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]  # Add more win_sizes as needed

# Create a function to update the plot for each frame of the animation
def update(frame):
    # Clear the current plot
    plt.clf()
    # Get the data for the current win_size
    win_size = win_sizes[frame]
    freq_axes_data = data_dict[taps][win_size]['freq_axes']
    spec_axes_data = data_dict[taps][win_size]['spec_axes']
    
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
    freq_axes.append(freq_axes_data[3])
    
    spec_axes.append(spec_axes_data[0])
    spec_axes.append(spec_axes_data[1])
    spec_axes.append(spec_axes_data[3])
    spec_axes.append(spec_axes_data[3] * spec_axes_data[1]) # compensation filter multiplied with impaired filter
    
    labels = [
        "Normal Hearing",
        "Impaired Hearing",
        "Compensation filter",
        "Result",
    ]


    # Plot the data (customize this according to your data structure)
    ax, fig = plt.subplots(1,1)
    plot.magspec_anim(
        ax=ax,
        freq_axes=freq_axes,
        spec_axes=spec_axes,
        units="dB",
        title=f"{win_size}",
        labels=labels,
    )
    

# Create a figure and an animation
fig = plt.figure()
ani = animation.FuncAnimation(fig, update, frames=len(win_sizes), repeat=False, blit=True)

# Display the animation
plt.show()


# %%
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import display, clear_output

# Fixed number of taps
taps = 128

# List of win_sizes for which you want to create the animation
win_sizes = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]  # Add more win_sizes as needed

# Create a function to update the plot for each frame of the animation
def update(frame):
    # Clear the current plot
    plt.clf()
    # Get the data for the current win_size
    win_size = win_sizes[frame]
    freq_axes_data = data_dict[taps][win_size]['freq_axes']
    spec_axes_data = data_dict[taps][win_size]['spec_axes']
    
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
    freq_axes.append(freq_axes_data[3])
    
    spec_axes.append(spec_axes_data[0])
    spec_axes.append(spec_axes_data[1])
    spec_axes.append(spec_axes_data[3])
    spec_axes.append(spec_axes_data[3] * spec_axes_data[1]) # compensation filter multiplied with impaired filter
    
    labels = [
        "Normal Hearing",
        "Impaired Hearing",
        "Compensation filter",
        "Result",
    ]

    # Plot the data (customize this according to your data structure)
    fig, ax = plt.subplots(1,1)
    return plot.magspec_anim(
        ax=ax,
        freq_axes=freq_axes,
        spec_axes=spec_axes,
        units="dB",
        title=f"{win_size}",
        labels=labels,
    )

# Create a figure and an animation
fig = plt.figure()
ani = FuncAnimation(fig, update, frames=len(win_sizes), repeat=False, blit=True)

# Display the animation in the notebook
%matplotlib notebook
display(fig)
