import numpy as np
from matplotlib import animation, rc

rc("animation", html="html5")
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from scipy.fft import fft, fftfreq
import os
from matplotlib.widgets import Slider, Button, CheckButtons

mpl.rcParams["text.usetex"] = False

# folderName = "RPG_d2000_initfx6_00_window400_max_p800"

folderName = "RPG_d2000_initfx6_00_window400_max_p800recording"
# folderName = "RPG_d2000_initfx6"
pathName = f"./data/{folderName}"
# fig, ax = plt.subplots(nrows=2, ncols=2)
fig = plt.figure()
ax = []
gs = gridspec.GridSpec(2, 2)
ax.append(fig.add_subplot(gs[0, :]))
ax.append(fig.add_subplot(gs[1, 0]))
ax.append(fig.add_subplot(gs[1, 1]))

ax[0].set_ylim(0, 0.64)

# The parametrized function to be plotted
def f(frequency):
    return frequency  # ``amplitude * np.sin(2 * np.pi * frequency * t)


def thresh(grouped, coords, coord_add=0):
    # temp = grouped[:,coords[2]: coords[3]]
    coord_add = int(coord_add)
    temp = grouped[:, : coords[3] - coord_add]

    if temp.size == 0:
        return np.zeros(coords[3])
    temp = grouped[
        np.where((temp[:, -1] >= coords[0]) & (temp[:, -1] <= coords[1]))
    ]  # [:,coords[2]:coords[3]]
    return temp


grouped = []


def g(frequency, amplitude):
    return frequency + amplitude  # ``amplitude * np.sin(2 * np.pi * frequency * t)


text_var_1 = plt.figtext(0, 0.01, s="window mode", fontsize=20)
text_var_2 = plt.figtext(0, 0.01, s="free mode", fontsize=20)
text_var_2.set_visible(False)

free_move = False
change = False
plot_all = False
files = os.listdir(pathName)
files = [name for name in files if ".npz" in name]
# files = files[:int(len(files)/4)]
data = np.load(f"{pathName}/{files[0]}")
print(f"Showing file with {data['init_flux']} amplitude")
t = np.arange(0, data["duration"], data["stepsize"])
t = t[int(data["learning_start"] / data["stepsize"]) :]
for name in files:
    data = np.load(f"{pathName}/{name}")
    grouped.append(
        data["running_average"][int(data["learning_start"] / data["stepsize"]) :]
    )
grouped = np.array(grouped)
average = np.mean(grouped, axis=0)


def window(size):
    return np.ones(size) / float(size)


ax[0].plot(t, average, c="k", label="Mean average performance across all trials")

# Define initial parameters
init_y_0 = 0
init_y_1 = 0.65
init_x_0 = 0
init_x_1 = data["duration"]
# Create the figure and the line that we will manipulate
# line, = plt.plot(t, f(t, init_amplitude, init_frequency), lw=2)
begin_y = f(init_y_0)
add_y = g(init_y_0, init_y_1)
begin_x = f(init_x_0)
add_x = g(init_x_0, init_x_1)

arr = thresh(
    grouped,
    (begin_y, add_y, begin_x, add_x),
    coord_add=data["learning_start"] / data["stepsize"],
)
line_1 = ax[0].axhline(begin_y)
line_2 = ax[0].axhline(add_y)
line_4 = ax[0].axvline(add_x)
thresh_line = ax[1].axhline(init_x_0, ls="dotted", c="k")
(line_6,) = ax[0].plot(
    t,
    np.mean(arr, axis=0) + arr.std(axis=0),
    color="y",
    ls="dotted",
    label=r"mean $\pm$ std",
)
(line_7,) = ax[0].plot(
    t, np.mean(arr, axis=0) - arr.std(axis=0), color="y", ls="dotted"
)
(line_5,) = ax[0].plot(t, np.mean(arr, axis=0), color="orange", label="mean of subset")


def powerSpecCleanPlot(mean_arr, threshold=5, dt=0.1, n=20000):
    fhat = np.fft.fft(mean_arr, n)
    PSD = fhat * np.conj(fhat) / n
    freq = (1 / (dt * n)) * np.arange(n)
    L = np.arange(1, np.floor(n / 2), dtype="int")
    power = (freq[L], PSD[L])
    indices = PSD > threshold
    PSDclean = PSD * indices
    print(PSDclean.shape)
    fhat = indices * fhat
    ffilt = np.fft.ifft(fhat)
    clean = ffilt
    return (power, clean, threshold)


def powerSpecExtend(mean_arr, threshold=5, dt=0.1, n=20000, extend=8000):
    fhat = np.fft.fft(mean_arr, n)
    PSD = fhat * np.conj(fhat) / n
    freq = (1 / (dt * n)) * np.arange(n)
    L = np.arange(1, np.floor(n / 2), dtype="int")
    t = np.arange(0, n * dt, dt)
    power = (freq[L], PSD[L])
    indices = PSD > threshold
    PSDclean = PSD * indices
    print(PSDclean.shape)
    fhat = indices * fhat
    ffilt = np.fft.ifft(fhat)
    plt.plot(t, ffilt, color="k")
    ffilt = np.fft.ifft(fhat, n=int((extend - 400) / dt))
    clean = ffilt
    t = np.arange(400, extend, dt)
    plt.plot(t, ffilt, ls="dotted", lw=0.5)
    plt.show()


power, clean, threshold = powerSpecCleanPlot(
    mean_arr=arr.mean(axis=0),
    threshold=begin_x,
    n=(data["duration"] - data["learning_start"]) / data["stepsize"],
)
(power_plot,) = ax[1].plot(power[0], power[1])
ax[2].set_xlim((0, 0.04))
ax[0].title.set_text(
    f"averaged 'running Averages' of {arr.shape[0]} trials\n where trial at x_1:{init_x_1}   in:[{np.round(init_y_0,2)},{np.round(init_y_1, 2)}]\n init_flux:{data['init_flux']}"
)
ax[1].set_xlim((0, 0.04))
ax[1].set_ylim((0, 1))
ax[1].title.set_text("Power Spectrum")
(clean_plot,) = ax[0].plot(
    t, clean, color="r", label=f"filtered for power above {threshold}"
)
power, clean, threshold = powerSpecCleanPlot(
    clean,
    threshold=threshold,
    n=(data["duration"] - data["learning_start"]) / data["stepsize"],
)
(cleaned_power,) = ax[2].plot(power[0], power[1])
# ax[2].set_xlim((0,0.01))
ax[2].set_ylim((0, 1))

# line_6, = ax[0].plot(t, np.mean(arr, axis=0)+arr.std(axis=0), color='y', ls="dotted", label=r"mean $\pm$ std of subset")
# line_7, = ax[0].plot(t, np.mean(arr, axis=0)-arr.std(axis=0), color='y', ls="dotted")

# adjust the main plot to make room for the sliders
plt.subplots_adjust(left=0.25, bottom=0.25)

# Make a horizontal slider to control the frequency.
y_place = plt.axes([0.1, 0.25, 0.0225, 0.63])
y_0_slider = Slider(
    ax=y_place,
    label="y_0",
    valmin=0.0,
    valmax=0.64,
    valinit=init_y_0,
    orientation="vertical",
)

# Make a vertically oriented slider to control the amplitude
y_add = plt.axes([0.15, 0.25, 0.0225, 0.63])
y_1_slider = Slider(
    ax=y_add,
    label="y_1",
    valmin=0,
    valmax=0.64,
    valinit=init_y_1,
    orientation="vertical",
)

x_place = plt.axes([0.25, 0.1, 0.65, 0.03])
x_add = plt.axes([0.25, 0.05, 0.65, 0.03])

x_0_slider = Slider(
    ax=x_place,
    label="thresh",
    valmin=0.00000,
    valmax=1,
    valinit=init_x_0,
    orientation="horizontal",
)
x_1_slider = Slider(
    ax=x_add,
    label="x_1",
    valmin=1,
    valmax=data["duration"],
    valstep=0.1,
    valinit=init_x_1,
    orientation="horizontal",
)
# The function to be called anytime a slider's value changes
def update(val):
    global PSD, threshold, change, power, clean
    line_1.set_ydata(f(y_0_slider.val))
    line_4.set_xdata(f(x_1_slider.val))
    # line_3.set_xdata(f(x_0_slider.val))
    if change == True:
        if not free_move:
            y_1_slider.set_val(y_1_slider.val - y_0_slider.val)
        else:
            y_1_slider.set_val(y_0_slider.val + y_1_slider.val)
        change = False

    if not free_move:
        line_2.set_ydata(g(y_0_slider.val, y_1_slider.val))
    else:
        line_2.set_ydata(f(y_1_slider.val))
    y_0 = line_1.get_ydata()
    y_1 = line_2.get_ydata()
    # x_0 = int(line_3.get_xdata())
    x_1 = int(line_4.get_xdata())
    arr = thresh(
        grouped,
        (y_0, y_1, 0, x_1 * 10),
        coord_add=data["learning_start"] / data["stepsize"],
    )

    #    if plot_all:
    #        for a in arr:
    #            ax[0].plot(t, a, c='k')
    power, clean, threshold = powerSpecCleanPlot(
        mean_arr=arr.mean(axis=0),
        threshold=f(x_0_slider.val),
        n=(data["duration"] - data["learning_start"]) / data["stepsize"],
    )
    thresh_line.set_ydata(f(threshold))
    power_plot.set_xdata(power[0])
    power_plot.set_ydata(power[1])
    clean_plot.set_ydata(clean)
    clean_plot.set_label(f"filtered above power:{np.round(threshold,2)}")

    power, clean, threshold = powerSpecCleanPlot(
        clean,
        threshold=threshold,
        n=(data["duration"] - data["learning_start"]) / data["stepsize"],
    )
    cleaned_power.set_ydata(power[1])
    ax[2].title.set_text(f"cleaned spectrum past {np.round(threshold, 2)}")
    print(arr.shape)
    # line_5.set_xdata(t[x_0*10: x_1*10])
    # line_5.set_xdata(t)
    line_5.set_ydata(np.mean(arr, axis=0))
    line_6.set_ydata(np.mean(arr, axis=0) - arr.std(axis=0))
    line_7.set_ydata(np.mean(arr, axis=0) + arr.std(axis=0))

    # line_6.set_xdata(t[x_0*10: x_1*10])
    # line_6.set_xdata(t)

    #    line_7.set_xdata(t[x_0*10: x_1*10])
    # line_7.set_xdata(t)
    ax[0].title.set_text(
        f"averaged 'running Averages' of {arr.shape[0]} trials\n where trial at x_1:{x_1}   in:[{np.round(y_0,2)},{np.round(y_1, 2)}]"
    )
    ax[1].title.set_text(f"Hz spectrum from mean of {arr.shape[0]} trials")
    fig.canvas.draw_idle()


# register the update function with each slider
y_0_slider.on_changed(update)
y_1_slider.on_changed(update)
x_0_slider.on_changed(update)
x_1_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
reset_ax = plt.axes([0.06, 0.06, 0.11, 0.04])
window_ax = plt.axes([0.06, 0.1, 0.11, 0.04])
printspectrum_ax = plt.axes([0.06, 0.14, 0.11, 0.04])
button_3 = Button(printspectrum_ax, "Print spectrum", hovercolor="0.975")
button_2 = Button(window_ax, "change mode", hovercolor="0.975")
button_1 = Button(reset_ax, "Reset", hovercolor="0.975")


def reset(event):
    y_0_slider.reset()
    y_1_slider.reset()
    x_0_slider.reset()
    x_1_slider.reset()


def free(event):
    global free_move, change
    change = True
    free_move = not free_move
    if free_move:
        text_var_1.set_visible(False)
        text_var_2.set_visible(True)
    else:
        text_var_1.set_visible(True)
        text_var_2.set_visible(False)


def printSpectrum(event):
    global power, threshold
    print(f"freq:{power[0][np.where(power[1]>threshold)]}")
    print(f"powe:{power[1][np.where(power[1]>threshold)]}")


button_1.on_clicked(reset)
button_2.on_clicked(free)
button_3.on_clicked(printSpectrum)
ax[0].legend()


plt.show()
