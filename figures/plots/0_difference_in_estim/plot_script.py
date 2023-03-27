import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from figures.plots.get_plots import get_plot_data_3
import cycler
import matplotlib
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter


def main():
    matr, true_alpha = get_plot_data_3()  # (ns,bs,n_etas)
    bs = [1, 2, 3, 4, 5, 10, 20]
    true_ns = [10, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900]
    selected_ns = [10, 20, 50, 100, 200, 500]
    indices = np.array([np.argwhere(np.array(true_ns)==x) for x in selected_ns]).squeeze()
    legend = True
    batch_size = 20


    etas = np.linspace(0.001, 0.01, 10).round(3)

    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    sns.set(rc={"figure.dpi": 300, 'savefig.dpi': 300})
    sns.set_context('notebook')
    sns.set_style("white")
    color = matplotlib.pyplot.cm.viridis(np.linspace(0, 1, len(etas)))
    matplotlib.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)


    scatter_weights = (true_alpha - np.mean(true_alpha)) / np.std(true_alpha)
    scatter_weights += np.abs(scatter_weights.min()) + 0.5
    scatter_weights **= 2.5
    scatter_weights *= 10

    scale = .8
    fig, ax = plt.subplots(1, 3, figsize=(12 * scale, 4 * scale))
    plt.rc('grid', linestyle="dotted", color='black', alpha=0.15)

    batch_sizes = [1, 5, 20]
    ax[0].set_ylabel(r'$|\hat{\alpha}-\hat{\alpha}^{(n)}$|')
    for ax_ in ax:
        ax_.set_xlabel('$n$', fontsize=20*scale)
        ax_.tick_params(axis='both', which='major', labelsize=15)
    ax[0].set_ylabel(r'$|\hat{\alpha}-\hat{\alpha}^{(n)}|$', fontsize=20*scale)

    for k, bs_ in enumerate(batch_sizes):
        i = np.argwhere(np.array(batch_sizes) == bs_).squeeze()
        for j in range(matr.shape[-1]):
            ax[k].scatter(np.arange(matr[indices].shape[0]), np.abs(matr[indices])[:, i, j], marker='_', label=etas[j],
                        alpha=0.75,
                        linewidth=1.5, s=scatter_weights[i, j])
        ax[k].set_xticks(np.arange(len(selected_ns)), selected_ns)

        ax[k].set_ylim(-0.05, 0.5)
        ax[k].set_xlim(-0.35, 5.35)
        ax[k].grid()
        ax[k].set_title('Batch size: ' + str(bs_), size=15)

    ax[1].yaxis.set_major_formatter(NullFormatter())
    ax[2].yaxis.set_major_formatter(NullFormatter())

    if legend:
        legend_elements = [Line2D([0], [0], marker='_', color=clr, label=lr,
                                  markerfacecolor=clr, markersize=5) for clr, lr in zip(color[::2], etas[::2])]
        legend2_label = np.around(np.linspace(np.min(true_alpha), np.max(true_alpha), len(etas[::2])), 3)
        legend2_size = np.linspace(np.min(scatter_weights), np.max(scatter_weights), len(etas[::2]))
        legend_elements2 = [Line2D([0], [0], marker='_', color='b', label=l,
                                   markerfacecolor='b', linewidth=0., ms=s / 20 + 1) for l, s in
                            zip(legend2_label, legend2_size)]
        legend1 = ax[-1].legend(handles=legend_elements, loc=(0.375, 0.4), prop={'size': 9})
        legend1.set_title(title=r'$\eta$', prop={'size': 12})

        ax[-1].add_artist(legend1)
        legend2 = ax[-1].legend(handles=legend_elements2, loc=(0.685, 0.4), prop={'size': 9})
        legend2.set_title(title=r'$\hat{\alpha}$', prop={'size': 12})

    fig.tight_layout()
    fig.show()
    fig.savefig(
        '/Users/krunoslavlehmanpavasovic/PycharmProjects/heavy_tails_new/plots_png_output/0_difference_in_estim/plot.png')
    fig.savefig(
        '/Users/krunoslavlehmanpavasovic/PycharmProjects/heavy_tails_new/plots_pdf_output/0_difference_in_estim/plot.pdf')
    print("done")


if __name__ == '__main__':
    # n_iter: 1000
    # d: 100
    # lr n bs all say
    main()
