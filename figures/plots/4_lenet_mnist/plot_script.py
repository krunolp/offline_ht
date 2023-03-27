import numpy as np
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import os, json
import seaborn as sns
from matplotlib.ticker import NullFormatter

DATA_PATH = '/ht_nns/results_saved/lenet_mnist/'


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


def get_data(folder, ind=-1):
    path_to_json = DATA_PATH + folder + '/weights/'
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
                if "weight" in key:
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


def get_plots(final_alphas, ratio, perc, dataset, model, cmap='viridis'):
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    sns.set(rc={"figure.dpi": 300, 'savefig.dpi': 300})
    sns.set_context('notebook')
    sns.set_style("white")
    plt.rcParams['image.cmap'] = cmap
    final_alphas, ratio, perc = final_alphas[:-3], ratio[:-3], perc[:-3]

    scale = .5
    if final_alphas.shape[-1] > 8:
        final_alphas = final_alphas[:, :8]

    fig, ax = plt.subplots(1, 5, figsize=(40 * scale, 8 * scale))

    for i in range(5):
        ax[i].scatter(ratio, final_alphas[:, i], c=perc)
        ax[i].tick_params(axis='both', which='major', labelsize=45 * scale)

    for i in range(5):
        ax[i].set_title('Layer: ' + str(i + 1), size=45 * scale)
        ax[i].set_xlabel(r'$\eta/b$', fontsize=45 * scale)
        plt.rc('grid', linestyle="dotted", color='black', alpha=0.15)
        ax[i].grid()
        ax[i].set_ylim(.9, 2.5)
        if i > 0:
            ax[i].yaxis.set_major_formatter(NullFormatter())
    ax[0].set_ylabel(r'$\hat\alpha$', fontsize=50 * scale)
    scatter = ax[-1].scatter(ratio, final_alphas[:, -1], c=perc)
    legend = ax[-1].legend(*scatter.legend_elements(), bbox_to_anchor=(1.05, 1.), title="Data %:",
                           fontsize=45 * scale, title_fontsize=45 * scale)
    ax[-1].add_artist(legend)

    fig.tight_layout(rect=[0, 0., .9, 1.])
    plt.show()

    fig.savefig(
        '/Users/krunoslavlehmanpavasovic/PycharmProjects/heavy_tails_new/plots_png_output/4_lenet_mnist/plot.png')
    fig.savefig(
        '/Users/krunoslavlehmanpavasovic/PycharmProjects/heavy_tails_new/plots_pdf_output/4_lenet_mnist/plot.pdf')


if __name__ == '__main__':
    folder = '2023-03-25_10_37_23.616183'
    # best 2023-03-25_10_37_23.616183
    # good 023-03-25_10_37_17.919439

    f = open(DATA_PATH + folder + '/parameters.log.json')
    data = json.load(f)

    final_alphas, ratio, perc = get_data(folder)
    get_plots(final_alphas, ratio, perc, dataset=data['dataset'], model=data['model'])
