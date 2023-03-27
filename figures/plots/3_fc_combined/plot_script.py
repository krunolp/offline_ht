import numpy as np
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import os, json
import seaborn as sns
from matplotlib.ticker import NullFormatter

DATA_PATH_MNIST = '/ht_nns/results_saved/fc_mnist/'
DATA_PATH_CIFAR10 = '/ht_nns/results_saved/fc_cifar10/'


def get_data_mnist(folder, ind=-1):
    path_to_json = DATA_PATH_MNIST + folder + '/weights/'
    json_files = [pos_json for pos_json in os.listdir(path_to_json)]
    paths = sorted(json_files, key=lambda x: int(x.split('_')[1]))[:ind]
    lrs = [float(x.split('_')[3]) for x in paths]
    bs = [int(x.split('_')[5]) for x in paths]
    perc = [float(x.split('_')[7][:-4]) for x in paths]

    all_alphas = []
    for path in tqdm(paths):
        try:
            a = torch.load(path_to_json + path, map_location=torch.device('cpu'))

            x_mcs = []
            for key in a['last_iterate'].keys():
                mean_ = a['run_average'][key].numpy().reshape(-1, 1).mean()
                temp = a['last_iterate'][key].numpy().reshape(-1, 1) - mean_
                x_mcs.append(temp)

            estim_alphas = []
            for Xmc in x_mcs:
                alp_tmp = []
                for mm in [2, 5, 10, 20, 50, 100, 500, 1000]:
                    alp_tmp.append(alpha_estimator_one(mm, Xmc))

                estim_alphas.append(np.nanmedian(np.array(alp_tmp)))

            all_alphas.append(estim_alphas)
        finally:
            pass

    final_alphas = np.array(all_alphas)
    ratio = np.array([float(eta / b) for eta, b in zip(lrs, bs)])

    return final_alphas, ratio, perc


def get_data_cifar10(folder, ind=-1):
    path_to_json = DATA_PATH_CIFAR10 + folder + '/weights/'
    json_files = [pos_json for pos_json in os.listdir(path_to_json)]
    paths = sorted(json_files, key=lambda x: int(x.split('_')[1]))[:ind]
    lrs = [float(x.split('_')[3]) for x in paths]
    bs = [int(x.split('_')[5]) for x in paths]
    perc = [float(x.split('_')[7][:-4]) for x in paths]

    all_alphas = []
    for path in tqdm(paths):
        try:
            a = torch.load(path_to_json + path, map_location=torch.device('cpu'))

            x_mcs = []
            for key in a['last_iterate'].keys():
                mean_ = a['run_average'][key].numpy().reshape(-1, 1).mean()
                temp = a['last_iterate'][key].numpy().reshape(-1, 1) - mean_
                x_mcs.append(temp)

            estim_alphas = []
            for Xmc in x_mcs:
                alp_tmp = []
                for mm in [2, 5, 10, 20, 50, 100, 500, 1000]:
                    alp_tmp.append(alpha_estimator_one(mm, Xmc))

                estim_alphas.append(np.nanmedian(np.array(alp_tmp)))

            all_alphas.append(estim_alphas)
        finally:
            pass

    final_alphas = np.array(all_alphas)
    ratio = np.array([float(eta / b) for eta, b in zip(lrs, bs)])

    return final_alphas, ratio, perc


def alpha_estimator_one(m, X):
    N = len(X)
    n = int(N / m)  # must be an integer

    X = X[0:n * m]

    Y = np.sum(X.reshape(n, m), 1)
    eps = np.spacing(1)

    Y_log_norm = np.log(np.abs(Y) + eps).mean()
    X_log_norm = np.log(np.abs(X) + eps).mean()
    diff = (Y_log_norm - X_log_norm) / math.log(m)
    return 1 / diff


def get_plots(cmap='viridis'):
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    sns.set(rc={"figure.dpi": 300, 'savefig.dpi': 300})
    sns.set_context('notebook')
    sns.set_style("white")
    plt.rcParams['image.cmap'] = cmap

    folder = '2023-03-25_19_19_53.506303'
    final_alphas1, ratio1, perc1 = get_data_mnist(folder)
    folder = '2023-03-25_19_19_41.642338'
    final_alphas2, ratio2, perc2 = get_data_cifar10(folder)

    final_alphas1, ratio1, perc1 = final_alphas1[:-3], ratio1[:-3], perc1[:-3]
    final_alphas2, ratio2, perc2 = final_alphas2[:-1], ratio2[:-1], perc2[:-1]

    scale = .5
    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, figsize=(25 * scale, 5 * scale))
    ext = []
    for i, ax_ in enumerate([ax1, ax2, ax3, ax4, ax5, ax6]):
        plt.rc('grid', linestyle="dotted", color='black', alpha=0.15)
        ax_.grid()
        ax_.set_ylim(1.0, 2.35)
        ax_.set_xlabel(r'$\eta/b$', fontsize=25 * scale)
        ax_.set_title('Layer: ' + str((i % 3 + 1)))
        ext.append([ax_.get_window_extent().x0, ax_.get_window_extent().width])

    ax1.set_ylabel(r'$\hat\alpha$', fontsize=25*scale)

    pointsize = 20
    scatter = ax1.scatter(ratio1, final_alphas1[:, 0], c=perc1, s=pointsize * scale)
    ax2.scatter(ratio1, final_alphas1[:, 1], c=perc1, s=pointsize * scale)
    ax3.scatter(ratio1, final_alphas1[:, 2], c=perc1, s=pointsize * scale)

    # cifar10

    ax4.scatter(ratio2, final_alphas2[:, 0], c=perc2, s=pointsize * scale)
    ax5.scatter(ratio2, final_alphas2[:, 1], c=perc2, s=pointsize * scale)
    ax6.scatter(ratio2, final_alphas2[:, 2], c=perc2, s=pointsize * scale)

    legend = ax6.legend(*scatter.legend_elements(), bbox_to_anchor=(1.1, 1.05), title="Data %:",
                        fontsize=25 * scale, title_fontsize=25 * scale)
    ax6.add_artist(legend)

    ax2.yaxis.set_major_formatter(NullFormatter())
    ax3.yaxis.set_major_formatter(NullFormatter())
    ax4.yaxis.set_major_formatter(NullFormatter())
    ax5.yaxis.set_major_formatter(NullFormatter())
    ax6.yaxis.set_major_formatter(NullFormatter())

    inv = fig.transFigure.inverted()
    width_left = 1.075*(ext[0][0] + (ext[0][0] + ext[0][0] + ext[0][0] - ext[0][0]) / 2.)
    left_center = inv.transform((width_left, 1))
    width_right = 1.05*(ext[3][0]*1.025 + (ext[3][0] + ext[0][0] + ext[0][0] - ext[3][0]) / 2.)
    right_center = inv.transform((width_right, 1))
    plt.figtext(left_center[0], 0.95, "MNIST data", va="center", ha="center", size=30 * scale)
    plt.figtext(right_center[0], 0.95, "CIFAR10 data", va="center", ha="center", size=30 * scale)
    fig.tight_layout(rect=[0, 0., .9, .95])

    plt.show()
    fig.savefig(
        '/Users/krunoslavlehmanpavasovic/PycharmProjects/heavy_tails_new/plots_png_output/3_fc_combined/plot.png')
    fig.savefig(
        '/Users/krunoslavlehmanpavasovic/PycharmProjects/heavy_tails_new/plots_pdf_output/3_fc_combined/plot.pdf')


if __name__ == '__main__':
    get_plots()
