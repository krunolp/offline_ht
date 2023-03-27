from functools import partial
from jax import jit
import jax
import jax.numpy as jnp
from tqdm import tqdm
import numpy as np
from tensorflow_probability.substrates import jax as tfp
import matplotlib.pyplot as plt

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

    def moments_estim_iter(self, b, eta, key, n_iter, m):
        main_seed, init_seed = key, key

        final_iterates = []
        ns = [1000, 2000, 5000]
        for n in ns:
            final_xs = []
            for i in tqdm(range(int(m ** 2))):
                if i % n == 0:
                    key = main_seed
                init_seed, _ = jax.random.split(init_seed)
                key, subkey = jax.random.split(key)
                final_x, _ = self.one_sample(b, eta, subkey, init_seed, n_iter)
                final_xs.append(final_x)
            final_iterates.append(jnp.array(final_xs))

        for iterate, n in zip(final_iterates, ns):
            counts, bins = np.histogram(jnp.array(iterate), bins=30,
                                        range=(np.array(final_iterates).min(), np.array(final_iterates).max()))
            plt.hist(bins[:-1], bins, weights=counts)
            plt.suptitle("n: " + str(n))
            plt.show()

        mom_tmp = []
        return jnp.array(mom_tmp)

    def train(self, etas: np.array, key: jax.random.PRNGKey, n_iter: int = 1e6, b: int = None):
        # Numer of repetitions
        m = 40

        # step-size
        for i, eta in enumerate(etas):
            key, subkey = jax.random.split(key)
            _ = self.moments_estim_iter(b, eta, subkey, n_iter, m)


if __name__ == '__main__':  #
    main_key = jax.random.PRNGKey(2)
    d = 100
    n_iter_ = int(1000)
    batch_size = 1
    etas = np.linspace(1e-5, 1e-1, 10)

    est = ToySGD(dim=d)
    est.train(etas, main_key, n_iter=n_iter_, b=batch_size)
