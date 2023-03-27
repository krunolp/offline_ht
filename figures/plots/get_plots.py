import pandas as pd
import numpy as np
import pickle
from jax import vmap


def get_plot_data(num_batches: int = None) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv('batch_size_' + str(num_batches) + '.csv', sep=',')

    alpha_10000_mean, alpha_10000_min, alpha_10000_max = df['n_pts: 10000 - alpha_estim'], df[
        'n_pts: 10000 - alpha_estim__MIN'], df['n_pts: 10000 - alpha_estim__MAX']
    alpha_5000_mean, alpha_5000_min, alpha_5000_max = df['n_pts: 5000 - alpha_estim'], df[
        'n_pts: 5000 - alpha_estim__MIN'], df['n_pts: 5000 - alpha_estim__MAX']
    alpha_1000_mean, alpha_1000_min, alpha_1000_max = df['n_pts: 1000 - alpha_estim'], df[
        'n_pts: 1000 - alpha_estim__MIN'], df['n_pts: 1000 - alpha_estim__MAX']
    alpha_500_mean, alpha_500_min, alpha_500_max = df['n_pts: 500 - alpha_estim'], df['n_pts: 500 - alpha_estim__MIN'], \
                                                   df['n_pts: 500 - alpha_estim__MAX']
    alpha_100_mean, alpha_100_min, alpha_100_max = df['n_pts: 100 - alpha_estim'], df['n_pts: 100 - alpha_estim__MIN'], \
                                                   df['n_pts: 100 - alpha_estim__MAX']
    alpha_50_mean, alpha_50_min, alpha_50_max = df['n_pts: 50 - alpha_estim'], df['n_pts: 50 - alpha_estim__MIN'], df[
        'n_pts: 50 - alpha_estim__MAX']
    alpha_20_mean, alpha_20_min, alpha_20_max = df['n_pts: 20 - alpha_estim'], df['n_pts: 20 - alpha_estim__MIN'], df[
        'n_pts: 20 - alpha_estim__MAX']
    alpha_30_mean, alpha_30_min, alpha_30_max = df['n_pts: 30 - alpha_estim'], df['n_pts: 30 - alpha_estim__MIN'], df[
        'n_pts: 30 - alpha_estim__MAX']
    alpha_10_mean, alpha_10_min, alpha_10_max = df['n_pts: 10 - alpha_estim'], df['n_pts: 10 - alpha_estim__MIN'], df[
        'n_pts: 10 - alpha_estim__MAX']

    true_alpha = np.array(alpha_10000_mean)
    diff_10 = np.array(alpha_10_mean) - true_alpha
    diff_20 = np.array(alpha_20_mean) - true_alpha
    diff_30 = np.array(alpha_30_mean) - true_alpha
    diff_50 = np.array(alpha_50_mean) - true_alpha
    diff_100 = np.array(alpha_100_mean) - true_alpha
    diff_500 = np.array(alpha_500_mean) - true_alpha
    diff_1000 = np.array(alpha_1000_mean) - true_alpha
    diff_5000 = np.array(alpha_5000_mean) - true_alpha

    matr = np.vstack((diff_10, diff_20, diff_30, diff_50, diff_100, diff_500, diff_1000, diff_5000))
    return matr, true_alpha


def get_plot_data_2() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv('first_new_plot.csv', sep=',')

    alpha_10000_mean, alpha_10000_min, alpha_10000_max = df['n_pts: 10000 - alpha_estim'], df[
        'n_pts: 10000 - alpha_estim__MIN'], df['n_pts: 10000 - alpha_estim__MAX']
    alpha_5000_mean, alpha_5000_min, alpha_5000_max = df['n_pts: 5000 - alpha_estim'], df[
        'n_pts: 5000 - alpha_estim__MIN'], df['n_pts: 5000 - alpha_estim__MAX']
    alpha_1000_mean, alpha_1000_min, alpha_1000_max = df['n_pts: 1000 - alpha_estim'], df[
        'n_pts: 1000 - alpha_estim__MIN'], df['n_pts: 1000 - alpha_estim__MAX']
    alpha_500_mean, alpha_500_min, alpha_500_max = df['n_pts: 500 - alpha_estim'], df[
        'n_pts: 500 - alpha_estim__MIN'], df['n_pts: 500 - alpha_estim__MAX']
    alpha_100_mean, alpha_100_min, alpha_100_max = df['n_pts: 100 - alpha_estim'], df[
        'n_pts: 100 - alpha_estim__MIN'], df['n_pts: 100 - alpha_estim__MAX']
    alpha_50_mean, alpha_50_min, alpha_50_max = df['n_pts: 50 - alpha_estim'], df['n_pts: 50 - alpha_estim__MIN'], \
                                                df['n_pts: 50 - alpha_estim__MAX']
    alpha_20_mean, alpha_20_min, alpha_20_max = df['n_pts: 20 - alpha_estim'], df['n_pts: 20 - alpha_estim__MIN'], \
                                                df['n_pts: 20 - alpha_estim__MAX']
    alpha_30_mean, alpha_30_min, alpha_30_max = df['n_pts: 30 - alpha_estim'], df['n_pts: 30 - alpha_estim__MIN'], \
                                                df['n_pts: 30 - alpha_estim__MAX']
    alpha_10_mean, alpha_10_min, alpha_10_max = df['n_pts: 10 - alpha_estim'], df['n_pts: 10 - alpha_estim__MIN'], \
                                                df['n_pts: 10 - alpha_estim__MAX']

    matr = np.vstack((alpha_10_mean, alpha_20_mean, alpha_30_mean, alpha_50_mean, alpha_100_mean, alpha_500_mean,
                      alpha_1000_mean, alpha_5000_mean, alpha_10000_mean,)).T
    matr_low = np.vstack((alpha_10_min, alpha_20_min, alpha_30_min, alpha_50_min, alpha_100_min, alpha_500_min,
                          alpha_1000_min, alpha_5000_min, alpha_10000_min)).T
    matr_high = np.vstack((alpha_10_max, alpha_20_max, alpha_30_max, alpha_50_max, alpha_100_max, alpha_500_max,
                           alpha_1000_max, alpha_5000_max, alpha_10000_max)).T

    return matr, matr_low, matr_high


def get_plot_data_3() -> tuple[np.ndarray, np.ndarray]:
    log_dir = "/ht_lin_reg/exp_data_saved/n_pts_"
    all_lists = []
    for n_pts in [10, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
        with open(log_dir + str(n_pts), "rb") as fp:  # Pickling
            all_lists.append(pickle.load(fp))

    estims = np.array(all_lists)[:-1, ...]  # (n_pts, n_bs, n_lr)
    true_alpha = np.array(all_lists)[-1, ...]
    matr = vmap(lambda x: x - true_alpha)(estims)

    return matr, true_alpha


def get_plot_data_4() -> tuple[np.ndarray, np.ndarray]:
    log_dir = "/Users/krunoslavlehmanpavasovic/PycharmProjects/heavy_tails_new/difference_in_estim_data/"
    all_lists = []
    for lr_choice in [1, 2, 3, 4, 5, 6, 7, 8]:
        with open(log_dir + str('lr_') + str(float(lr_choice)), "rb") as fp:  # Pickling
            all_lists.append(pickle.load(fp))  # (n_lr_choices=9, n_pts=11, n_lr=10)

    estims = np.array(all_lists)[:, :-1, :]
    true_alpha = np.array(all_lists)[:, -1, :]
    matr = vmap(lambda x: x - true_alpha, 1, 1)(estims)  # (n_lr_choices=9, n_pts-1=11-1, n_lr=10)

    return matr, true_alpha

def get_plot_data_5():
    import pandas as pd
    df = pd.read_csv('/Users/krunoslavlehmanpavasovic/PycharmProjects/heavy_tails_new/experiments_script_error_bars/data.csv', sep=',')

    high_est = np.array(df['eta: 0.012105263157894735 - alpha_estim'][:10])
    high_min = np.array(df['eta: 0.012105263157894735 - alpha_estim__MIN'][:10])
    high_max = np.array(df['eta: 0.012105263157894735 - alpha_estim__MAX'][:10])


    med_est = np.array(df['eta: 0.02894736842105263 - alpha_estim'][:10])
    med_min = np.array(df['eta: 0.02894736842105263 - alpha_estim__MIN'][:10])
    med_max = np.array(df['eta: 0.02894736842105263 - alpha_estim__MAX'][:10])

    low_est = np.array(df['eta: 0.03736842105263158 - alpha_estim'][:10])
    low_min = np.array(df['eta: 0.03736842105263158 - alpha_estim__MIN'][:10])
    low_max = np.array(df['eta: 0.03736842105263158 - alpha_estim__MAX'][:10])

    high_alpha = np.array(df['eta: 0.012105263157894735 - alpha_estim'][-1:])
    med_alpha = np.array(df['eta: 0.02894736842105263 - alpha_estim'][-1:])
    low_alpha = np.array(df['eta: 0.03736842105263158 - alpha_estim'][-1:])

    return high_est, high_min, high_max, med_est, med_min, med_max, low_est, low_min, low_max, high_alpha, med_alpha, low_alpha
