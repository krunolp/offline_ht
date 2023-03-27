import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib
import pickle
from matplotlib.ticker import NullFormatter


def main():
    # plt.rcParams['figure.dpi'] = 300
    # plt.rcParams['savefig.dpi'] = 300
    # sns.set(rc={"figure.dpi": 300, 'savefig.dpi': 300})
    sns.set_context('notebook')
    sns.set_style("white")
    with open('/figures/plots/1_histograms/hist_data.pickle',
              'rb') as handle:
        final_iterates = pickle.load(handle)

    final_iterates = np.array(final_iterates) / 100000
    index = [0, 2, 3]
    ns = np.array([20, 50, 250, 1000])[index]
    N = len(ns)
    scale = 1.
    fig, axs = plt.subplots(N, figsize=(20 * scale, 6 * scale))
    colors = np.flip(matplotlib.cm.viridis(np.linspace(.2, .6, N)), axis=0)
    for i, iterate in enumerate(np.array(final_iterates)[index]):
        counts, bins = np.histogram(np.array(iterate), bins=100, range=(0 / 100000, 2e5 / 100000), density=False)
        axs[i].hist(bins[:-1], bins, weights=counts, range=(0 / 100000, 2e5 / 100000), color=colors[i], label=ns[i],
                    log=True)
        # axs[i].set_xlim(10000, 200000)
        axs[i].set_xlim(10000 / 100000, 200000 / 100000)
        axs[i].set_ylim(0.2)
        axs[i].tick_params(axis='both', which='major', labelsize=40 * scale)
        if i < 2:
            axs[i].xaxis.set_major_formatter(NullFormatter())
            axs[i].xaxis.set_major_formatter(NullFormatter())

        axs[i].yaxis.set_ticks([0.2, 100])
        axs[i].set_yticklabels([0.2, 100])

    leg = fig.legend(loc=(0.805, 0.435), ncol=1, prop={'size': 36 * scale}, framealpha=1)
    leg.set_title(title="n:", prop={'size': 40 * scale})

    fig.tight_layout()
    fig.show()
    fig.savefig(
        '/Users/krunoslavlehmanpavasovic/PycharmProjects/heavy_tails_new/plots_png_output/1_histograms/hist_plot.png')
    fig.savefig(
        '/Users/krunoslavlehmanpavasovic/PycharmProjects/heavy_tails_new/plots_pdf_output/1_histograms/hist_plot.pdf')
    print("done")


if __name__ == '__main__':
    # eta = 0.025
    # d = 100
    # bs = 1
    # n_iter = 1000

    main()

    # ax1.fill_between(
    #     np.squeeze(np.array(data['x_linspace'])), np.array(data['lower_conf']).squeeze(),
    #     np.array(data['upper_conf']).squeeze(), alpha=0.15,
    #     color='blue')
    # ax1.plot(data['pts_xs'], data['pts_ys'], 'o', linewidth=2, label='$(x_i, y_i)_1$', alpha=0.8,
    #          markeredgecolor='blue', color='white',
    #          markeredgewidth=0.5, markersize=8)
    # ax1.legend(loc='lower right')
    # ax1.set_ylim([0.5,7.8])
