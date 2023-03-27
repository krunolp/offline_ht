import matplotlib.pyplot as plt
import numpy as np
from figures.plots.get_plots import get_plot_data_5


def main():
    high_est, high_min, high_max, med_est, med_min, med_max, low_est, low_min, low_max, high_alpha, med_alpha, low_alpha = get_plot_data_5()  # (n_lr_choices=9, n_pts-1=11-1=10, n_lr=10)
    true_ns = [10, 20, 30, 50, 100, 200, 300, 400, 500, 750, 1000]
    selected_ns = [10, 20, 50, 100, 500]
    indices = np.array([np.argwhere(np.array(true_ns) == x) for x in selected_ns]).squeeze()

    scale = .8
    fig, ax = plt.subplots(1, 3, figsize=(12 * scale, 4 * scale))
    etas = [0.01, 0.025, 0.04]

    for ax_ in ax:
        ax_.set_xlabel('$n$', fontsize=20)
        ax_.tick_params(axis='both', which='major', labelsize=15)
    ax[0].set_ylabel(r'$\hat{\alpha}^{(n)}$', fontsize=20)


    e1 = (high_max - high_min)[indices]
    e2 = (med_max - med_min)[indices]
    e3 = (low_max - low_min)[indices]
    x = np.arange(len(e1))

    ax[0].errorbar(x, high_est[indices], yerr=e1, fmt='o', capsize=5, ecolor='green', color='green')
    ax[1].errorbar(x, med_est[indices], yerr=e2, fmt='o', capsize=5, ecolor='green', color='green')
    ax[2].errorbar(x, low_est[indices], yerr=e3, fmt='o', capsize=5, ecolor='green', color='green')

    ax[0].axhline(y=high_alpha, linestyle='--', color='b')
    ax[1].axhline(y=med_alpha, linestyle='--', color='b')
    ax[2].axhline(y=low_alpha, linestyle='--', color='b')

    for index, ax_ in enumerate(ax):
        ax_.set_xticks(np.arange(len(selected_ns)), selected_ns)
        ax_.set_title('$\eta =$ ' + str(etas[index]), fontsize=20)
        ax_.set_ylim(.3, 2.3)
        plt.rc('grid', linestyle="dotted", color='black', alpha=0.15)
        ax_.grid()

    ax[2].axhline(y=low_alpha, color='b', linestyle='--', label=r'Online $\hat{\alpha}$')
    fig.tight_layout()
    fig.legend(loc=(0.74, 0.675), prop={'size': 20})
    fig.show()
    fig.savefig(
        '/Users/krunoslavlehmanpavasovic/PycharmProjects/heavy_tails_new/plots_png_output/2_error_bars/plot.png')
    fig.savefig(
        '/Users/krunoslavlehmanpavasovic/PycharmProjects/heavy_tails_new/plots_pdf_output/2_error_bars/plot.pdf')
    print("done")


if __name__ == '__main__':
    # n_iter: 1000
    # d: 100
    # lr n bs all say
    main()
