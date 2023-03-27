from functools import partial
from jax import jit
import jax
import jax.numpy as jnp
from tqdm import tqdm
import numpy as np
from tensorflow_probability.substrates import jax as tfp
import itertools

tfd = tfp.distributions


class ToySGD:
    def __init__(self, dim: int) -> None:
        self.sigma_x = 3.
        self.sigma_y = 3.
        self.sigma_a = 1.

        self.true_x = self.sigma_x * np.random.randn(dim)
        self.distr = tfd.Normal(0., 1.)
        self.dim = dim

    @partial(jit, static_argnums=(0, 1))
    def alpha_estimator(self, m, x, eps=1e-16):
        # x is N by d matrix
        assert x.ndim == 2
        N = len(x)
        n = int(N / m)  # must be an integer
        y = jnp.sum(x.reshape(n, m, -1), 1)
        y_log_norm = jnp.log(jnp.linalg.norm(y, axis=1) + eps).mean()
        x_log_norm = jnp.log(jnp.linalg.norm(x, axis=1) + eps).mean()
        diff = (y_log_norm - x_log_norm) / jnp.log(m)
        return 1 / diff

    @partial(jit, static_argnums=(0, 3))
    def f_derivative(self, x, key, b):
        """ The function is 0.5 * (ax-y)**2. """
        a = jax.random.normal(key, (b, self.dim)) * self.sigma_a
        y = jnp.dot(a, self.true_x) + self.sigma_y * jax.random.normal(key, (b,))
        grad = jnp.dot(a.T, jnp.dot(a, x)) - jnp.dot(a.T, y)
        grad = grad / b
        return grad

    @partial(jit, static_argnums=(0, 3, 4))
    def sgd_update(self, i, pack, b, eta):
        x_old, run_avg_, key = pack
        gradient = self.f_derivative(x_old, key, b)
        x_new = x_old - gradient * eta
        run_avg_ = (run_avg_ * i + x_new) / (i + 1)
        key, _ = jax.random.split(key)
        return x_new, run_avg_, key

    @partial(jit, static_argnums=(0, 1, 2, 4))
    def one_sample(self, b, eta, key: jax.random.PRNGKey, n_iter: int = 1e6):
        x_init = tfd.Normal(0., 1.).sample(self.dim, key) * self.sigma_x
        run_avg = jnp.zeros_like(x_init)

        x_post_burn_in, _, key = jax.lax.fori_loop(lower=0, upper=int(n_iter * 0.5), init_val=(x_init, run_avg, key),
                                                   body_fun=lambda i, x: self.sgd_update(i, x, b, eta),
                                                   )

        final_x, run_avg, _ = jax.lax.fori_loop(lower=0, upper=int(n_iter * 0.5),
                                                init_val=(x_post_burn_in, run_avg, key),
                                                body_fun=lambda i, x: self.sgd_update(i, x, b, eta), )
        return final_x, run_avg

    def alpha_estim_iter(self, b, eta, key, x_star, n_iter, m):
        x_final, x_mc = [], []
        for _ in tqdm(range(int(m ** 2))):
            key, subkey = jax.random.split(key)
            final_xs_, run_avgs_ = self.one_sample(b, eta, subkey, n_iter)
            x_final.append(final_xs_)
            x_mc.append(run_avgs_)
        x_final, x_mc = jnp.array(x_final), jnp.array(x_mc)
        Xmc = x_mc - x_star

        alp_tmp = []
        for mm in [2, 5, 10, 20, 50, 100, m]:
            alp_tmp.append(self.alpha_estimator(mm, Xmc))
        return jnp.array(alp_tmp)

    def train(self, key: jax.random.PRNGKey, n_iter: int = 1e6):
        # Numer of repetitions
        m = 40

        # batch size
        bs = [1, 2, 3, 4, 5, 10, 20]
        # step-size
        etas = np.linspace(0.001, 0.2, 10)

        n_iter_main = n_iter
        alpha_mc_sas = []

        for i, (b, eta) in enumerate(itertools.product(bs, etas)):
            print('b =', b, ' eta =', eta)
            nan_flag = True
            counter, alp_tmp = 0, 0
            while nan_flag:
                key, subkey = jax.random.split(key)
                alp_tmp = self.alpha_estim_iter(b, eta, subkey, self.true_x, n_iter, m)
                n_iter = int(n_iter / 2)
                if jnp.isnan(alp_tmp).sum() == 0 or counter > 20:
                    nan_flag = False
                    n_iter = n_iter_main
                counter += 1

            alpha_estim = jnp.nanmedian(alp_tmp)
            alpha_mc_sas.append(alpha_estim)

            print('\t', alp_tmp)
            print("alpha_mc_sas: ", jnp.median(alp_tmp))


if __name__ == '__main__':
    seed = 2
    main_key = jax.random.PRNGKey(seed)
    d = 100
    est = ToySGD(dim=d)
    est.train(main_key, n_iter=int(1e5))
