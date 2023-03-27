import numpy as np
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import os, json
import matplotlib
import seaborn as sns
from matplotlib.ticker import NullFormatter

DATA_PATH = '/figures/plots/3_fc_mnist/'


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
    x_mcs = None

    all_params = []
    for path in tqdm(paths):
        try:
            a = torch.load(path_to_json + path, map_location=torch.device('cpu'))

            x_mcs = []
            for key in a['last_iterate'].keys():
                temp = a['last_iterate'][key].numpy()
                x_mcs.append(temp)
            all_params.append(x_mcs)
        finally:
            pass

    return all_params, bs, lrs, perc


def get_plots(all_params, bs, lrs, perc, dataset, model, cmap='viridis'):
    # plt.rcParams['figure.dpi'] = 300
    # plt.rcParams['savefig.dpi'] = 300
    # sns.set(rc={"figure.dpi": 300, 'savefig.dpi': 300})
    sns.set_context('notebook')
    sns.set_style("white")

    # all_params: (n_runs, layers, )
    bs, lrs, perc = np.array(bs), np.array(lrs), np.array(perc)
    first_l, second_l, third_l = [], [], []
    for runs in all_params:
        first_l.append(runs[0])

    first_l = np.array(first_l).reshape((len(all_params), -1, 1))  # (n_runs, n_params, 1)

    selected_bs = 1
    select_lr = 0.045

    first_l_plt = first_l[(bs == selected_bs) & (lrs == select_lr), :, :]

    ns = [1000, 2000, 5000]
    perc_plt = np.unique(perc)
    assert len(perc_plt) == len(first_l_plt)
    N = len(first_l_plt)

    scale = 1.
    fig, axs = plt.subplots(N-3, figsize=(20 * scale, 6 * scale))
    colors = np.flip(matplotlib.cm.viridis(np.linspace(.2, .6, N-3)), axis=0)
    for i, iterate in enumerate(first_l_plt[[1,2,3]]):
        # counts, bins = np.histogram(np.array(iterate), bins=200, density=False, range=(-2.5, 3))
        counts, bins = np.histogram(np.abs(np.array(iterate)), bins=100, density=False)#, range=(-2.5, 3))
        axs[i].hist(bins[:-1], bins, weights=counts, color=colors[i], label=ns[i], log=True, range=(-2.5, 3))
        axs[i].set_xlim(-0.01, 1.75)
        axs[i].set_ylim(0.2)
        axs[i].tick_params(axis='both', which='major', labelsize=40*scale)
        if i < 2:
            axs[i].xaxis.set_major_formatter(NullFormatter())
            axs[i].xaxis.set_major_formatter(NullFormatter())
    leg = fig.legend(loc=(0.835, 0.45), ncol=1, prop={'size': 36 * scale}, framealpha=1)
    leg.set_title(title="n:", prop={'size': 40 * scale})

    # fig.suptitle('First layer')
    fig.tight_layout()
    fig.show()

    fig.savefig(
        '/Users/krunoslavlehmanpavasovic/PycharmProjects/heavy_tails_new/plots_png_output/5_hist_nns_fc_mnist/plot.png')
    fig.savefig(
        '/Users/krunoslavlehmanpavasovic/PycharmProjects/heavy_tails_new/plots_pdf_output/5_hist_nns_fc_mnist/plot.pdf')




if __name__ == '__main__':
    folder = 'FC_MNIST'
    f = open(DATA_PATH + folder + '/parameters.log.json')
    data = json.load(f)

    all_params, bs, lrs, perc = get_data(folder)
    get_plots(all_params, bs, lrs, perc, dataset=data['dataset'], model=data['model'])
