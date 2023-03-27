import numpy as np
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import os, json

DATA_PATH = '/ht_nns/results_temp/'


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
            keys = ['linear1.weight', 'linear2.weight', 'linear3.weight']
            # for key in a['last_iterate'].keys():
            for key in keys:
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


def get_plots(final_alphas, ratio, perc, dataset, model):
    if final_alphas.shape[-1] == 3:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        for ax_ in [ax1, ax2, ax3]:
            plt.rc('grid', linestyle="dotted", color='black', alpha=0.15)
            ax_.grid()

        ax1.scatter(ratio, final_alphas[:, 0], c=perc)
        ax2.scatter(ratio, final_alphas[:, 1], c=perc)

        scatter = ax3.scatter(ratio, final_alphas[:, 2], c=perc)
        legend3 = ax3.legend(*scatter.legend_elements(), loc="upper right", title="Proportion of data:")
        ax3.add_artist(legend3)

        ax1.set(ylabel=r'$\hat\alpha$')
        for i, ax in enumerate([ax1, ax2, ax3]):
            ax.set(xlabel=r'$\eta/b$')
            ax.set_title('Layer: ' + str(i + 1))

        fig.suptitle(dataset + ' on ' + model + ' model')
        plt.show()

    elif final_alphas.shape[-1] >= 9:
        if final_alphas.shape[-1] > 9:
            final_alphas = final_alphas[:, :9]
        fig, ax = plt.subplots(3, 3, figsize=(15, 15))

        for i in range(3):
            ax[0, i].scatter(ratio, final_alphas[:, i], c=perc)
            ax[1, i].scatter(ratio, final_alphas[:, i + 3], c=perc)
            ax[2, i].scatter(ratio, final_alphas[:, i + 6], c=perc)

        scatter = ax[-1, 1].scatter(ratio, final_alphas[:, -1], c=perc)
        legend = ax[-1, 1].legend(*scatter.legend_elements(), loc="upper right", title="Proportion of data:")
        ax[-1, 1].add_artist(legend)

        ax[0, 0].set(ylabel=r'$\hat\alpha$')
        for i in range(3):
            ax[0, i].set(xlabel=r'$\eta/b$')
            ax[0, i].set_title('Layer: ' + str(i + 1))
            ax[1, i].set(xlabel=r'$\eta/b$')
            ax[1, i].set_title('Layer: ' + str(i + 1 + 3))
            ax[2, i].set(xlabel=r'$\eta/b$')
            ax[2, i].set_title('Layer: ' + str(i + 1 + 6))

            ax[i, 0].set_ylim(0, None)
            ax[i, 1].set_ylim(0, None)

        for ax_ in ax:
            for ax__ in ax_:
                plt.rc('grid', linestyle="dotted", color='black', alpha=0.15)
                ax__.grid()

        # fig.suptitle(dataset + ' on ' + model + ' model')
        fig.tight_layout()
        fig.gca().set_ylim(bottom=0)
        plt.show()


if __name__ == '__main__':
    folder = '2023-01-22_22_07_40.211615'

    f = open(DATA_PATH + folder + '/parameters.log.json')
    data = json.load(f)

    final_alphas, ratio, perc = get_data(folder, ind=205)
    get_plots(final_alphas, ratio, perc, dataset=data['dataset'], model=data['model'])
