from functools import partial
from jax import jit
import jax
import jax.numpy as jnp
from tqdm import tqdm
import numpy as np
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


class ToySGD:
    def __init__(self, dim: int, sigma: float = 1.) -> None:
        self.sigma_x = 3.
        self.sigma_y = 3.
        self.sigma_a = sigma

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

    @partial(jit, static_argnums=(0, 1, 2, 5))
    def one_sample(self, b, eta, key: jax.random.PRNGKey, init_seed: jax.random.PRNGKey, n_iter: int = 1e6):
        x_init = tfd.Normal(0., 1.).sample(self.dim, init_seed) * self.sigma_x
        run_avg = jnp.zeros_like(x_init)

        x_post_burn_in, _, key = jax.lax.fori_loop(lower=0, upper=int(n_iter * 0.5), init_val=(x_init, run_avg, key),
                                                   body_fun=lambda i, x: self.sgd_update(i, x, b, eta),
                                                   )

        final_x, run_avg, _ = jax.lax.fori_loop(lower=0, upper=int(n_iter * 0.5),
                                                init_val=(x_post_burn_in, run_avg, key),
                                                body_fun=lambda i, x: self.sgd_update(i, x, b, eta), )
        return final_x, run_avg

    def alpha_estim_iter(self, n, b, eta, key, x_star, n_iter, m):
        x_final, x_mc = [], []
        main_seed = key
        init_seed = key
        for i in tqdm(range(int(m ** 2))):
            if i % n == 0:
                key = main_seed
            init_seed, _ = jax.random.split(init_seed)
            key, subkey = jax.random.split(key)
            final_xs_, run_avgs_ = self.one_sample(b, eta, subkey, init_seed, n_iter)
            x_final.append(final_xs_)
            x_mc.append(run_avgs_)
        x_final, x_mc = jnp.array(x_final), jnp.array(x_mc)
        Xmc = x_mc - x_star

        alp_tmp = []
        for mm in [2, 5, 10, 20, 50, 100, m]:
            alp_tmp.append(self.alpha_estimator(mm, Xmc))
        return jnp.array(alp_tmp)

    def train(self, key: jax.random.PRNGKey, n_iter: int = 1e6, n: int = None, b: int = None):
        # Numer of repetitions
        m = 40

        # step-size
        etas = np.linspace(0.005, 0.025, 10)
        alpha_mc_sas = []

        for i, eta in enumerate(etas):
            key, subkey = jax.random.split(key)
            alp_tmp = self.alpha_estim_iter(n, b, eta, subkey, self.true_x, n_iter, m)

            alpha_estim = jnp.nanmedian(alp_tmp)
            alpha_mc_sas.append(alpha_estim)
            print('n = ', n, 'b =', b, ' eta =', round(eta, 3), " alpha_med= ", alpha_estim)


if __name__ == '__main__':  #
    main_key = jax.random.PRNGKey(2)
    d = 1
    n_pts_ = int(10)
    n_iter_ = int(1e2)
    batch_size = 1

    est = ToySGD(dim=d)
    est.train(main_key, n_iter=n_iter_, n=n_pts_, b=batch_size)
