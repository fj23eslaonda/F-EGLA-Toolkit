# plot_style.py
import matplotlib.pyplot as plt

def apply_plot_style():
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 16,
        "axes.labelsize": 14,
        "axes.titlesize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "figure.titlesize": 16,
        "axes.edgecolor": "black",
        "axes.linewidth": 1.2,
    })