"""
Based on Schmidt et al. 2009, "Bayesian non-negative matrix factorization"
fit() method uses their notation.
truncated_normal_sample is directly adapted from their code.
The rest of the implementation was guided by the paper.
Original Matlab version at http://mikkelschmidt.dk/code.html in the file gibbsnmf.m
The author also shared code for Chib's method and ICM algorithm via email
The author indicated that what they refer to as a "rectified" gaussian is actually a truncated gaussian.
"""
import os
import time
import math

import numpy as np
import scipy.stats
from scipy.special import erfc, erfcinv
#from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import NMF
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed

math_log = np.vectorize(math.log)
MACHINE_PREC = np.finfo(float).eps
LOG_ZERO = math.log(MACHINE_PREC)


class BayesianNMF(NMF):  # BaseEstimator, TransformerMixin
    """Bayesian Non-negative Matrix Factorization

    Assumes Gaussian error and exponentially distribution of W and H.

    Parameters
    ----------

    n_components : int or None
        Number of components, if n_components is not set all features
        are kept.

    tol : float, default 1E-6
        Convergence criterion in iterated conditional modes (ICM) algorithm

    mode : string, 'icm' | 'gibbs', default: 'icm'
        Algorithm for fitting the model.
        Valid options:

        -'icm': Iterated conditional modes.
           Approximates the maximum a posteriori (MAP) solution.

        -'gibbs': Gibbs sampling.
           Samples from the posterior distribution and averages.

    max_iter : int, default: 10000
        Maximum iterations of ICM or number of Gibbs samples ('gibbs' mode).

    burnin_fraction : float, default: 0.5
        0 <= burnin_fraction <= 1.
        Fraction of initial Gibbs samples to discard ('gibbs' mode only)

    mean_only : bool, default: True
        Store only mean of parameters, rather than samples.
         To save memory when using mode='gibbs'.

    random_state : int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : bool, default: False
        Whether to print output

    Attributes
    ----------
    components_ : array, [n_components, n_features]
        H matrix.

    bases_ : array, [n_samples, n_components]
        W matrix

    reconstruction_err_ : number
        Frobenius norm of the matrix difference, or beta-divergence, between
        the training data ``X`` and the reconstructed data ``WH`` from
        the fitted model.

    n_iter_ : int
        Actual number of iterations.

    Examples
    --------

    References
    ----------
    Schmidt, Mikkel N., Ole Winther, and Lars Kai Hansen.
    "Bayesian Non-negative Matrix Factorization."
    In ICA, vol. 9, pp. 540-547. 2009.

    Chib, Siddhartha. "Marginal likelihood from the Gibbs output."
    Journal of the American Statistical Association 90,
    no. 432 (1995): 1313–1321
    """

    def __init__(self, n_components=None,
                 mode='icm',
                 tol=1E-6,
                 max_iter=10000,
                 burnin_fraction=0.5,
                 mean_only=True,
                 random_state=None,
                 verbose=False):

        if mode not in ['icm', 'gibbs']:
            raise ValueError("{} is not a supported mode. Try 'icm' or 'gibbs'".format(mode))

        self.n_components = n_components
        self.mode = mode
        self.max_iter = max_iter
        self.tol = tol
        self.burnin_fraction = burnin_fraction
        self.mean_only = mean_only
        if isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        else:
            self.random_state = np.random.RandomState(seed=random_state)
        self.verbose = verbose
        self.bases_prior_ = None
        self.components_prior_ = None
        self.variance_prior_shape_ = None
        self.variance_prior_scale_ = None
        self.burnin_bases_samples_ = None
        self.burnin_components_samples_ = None
        self.burnin_variance_samples_ = None
        self.bases_samples_ = None
        self.components_samples_ = None
        self.variance_samples_ = None
        self.bases_ = None
        self.components_ = None
        self.variance_ = None
        self.reconstruction_err_ = None
        self.last_sampled_bases_ = None
        self.last_sampled_components_ = None
        self.last_sampled_variance_ = None
        self.n_free_params_ = None
        self.input_shape_ = None

    def fit(self, X, y=None,
            bases_cols_to_sample=None,
            components_rows_to_sample=None,
            sample_sigma=True,
            print_every=None,
            bases_init=None,
            components_init=None,
            variance_init=None,
            alpha_prior_scalar=None,
            beta_prior_scalar=None):
        """Learn a Bayesian NMF model for the data X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Data matrix to be decomposed. Must be non-negative.

        y : ignored

        Returns
        -------
        self : object
        """
        X = check_array(X)
        self.input_shape_ = X.shape
        N = self.n_components
        M = self.max_iter
        burnin_index = int(M * self.burnin_fraction)
        n_after_burnin = M - burnin_index
        I, J = X.shape
        if bases_cols_to_sample is None:
            bases_cols_to_sample = [True] * N
        if components_rows_to_sample is None:
            components_rows_to_sample = [True] * N
        alpha_prior_scalar = 1 if alpha_prior_scalar is None else alpha_prior_scalar
        # in Schmidt et al. they have all alpha_i,n = 1
        # and choose the beta prior to match the amplitude of the data
        beta_prior_scalar = N / X.mean() if beta_prior_scalar is None else beta_prior_scalar  # over N so that the product of A and B will be X.mean()?
        # In Schmidt et al., they use the rate param lambda. Then the mean of the exponential is 1 / its param
        # However, np.random.exponential and scipy.stats.expon use the alternative parameterization
        # with scale parameter beta = 1 / lambda

        # alpha_prior_scalar, beta_prior_scalar = 0, 0 # flat prior used in other part of Schmidt et al.
        k = 0  # uninformative prior used in Schmidt et al.
        theta = 0  # uninformative prior used in Schmidt et al.
        self.variance_prior_shape_ = k
        self.variance_prior_scale_ = theta
        alpha = np.ones((I, N)) * alpha_prior_scalar
        self.bases_prior_ = alpha
        beta = np.ones((N, J)) * beta_prior_scalar
        self.components_prior_ = beta
        A = self.random_state.exponential(scale=1 / alpha_prior_scalar,
                                          size=(I, N)) if bases_init is None else bases_init
        B = self.random_state.exponential(scale=1 / beta_prior_scalar,
                                          size=(N, J)) if components_init is None else components_init
        A_mean = np.zeros(A.shape)
        B_mean = np.zeros(B.shape)
        mu2_mean = 0
        As = []
        Bs = []
        mu2s = []
        chi = 0.5 * np.square(X).sum()
        mu2 = np.var(X - np.dot(A, B)) if variance_init is None else variance_init
        m = 0
        diff = self.tol + 1
        obj = mean_squared_error(X, np.dot(A, B))
        t = time.time()
        while m < M and diff > self.tol:
            if print_every is not None:
                if (m + 1) % print_every == 0:
                    new_t = time.time()
                    elapsed = new_t - t
                    t = new_t
                    if m > 0:
                        timestring = ": last {} iters in {:.3} seconds".format(print_every, elapsed)
                    else:
                        timestring = ""
                    print("iter {}{}".format(m + 1, timestring))
            C = np.dot(B, B.T)
            D = np.dot(X, B.T)
            for n in range(N):
                if bases_cols_to_sample[n]:
                    notn = list(range(n)) + list(range(n + 1, N))
                    an = (D[:, n] - np.dot(A[:, notn], C[notn, n]) - alpha[:, n] * mu2) / (C[n, n] + MACHINE_PREC)
                    if self.mode == 'gibbs':
                        rnorm_variance = mu2 / (C[n, n] + MACHINE_PREC)
                        A[:, n] = truncated_normal_sample(an, rnorm_variance, alpha[:, n],
                                                          random_state=self.random_state)
                    else:
                        A[:, n] = an.clip(min=0)
            ac_2d_diff = np.dot(A, C) - (2 * D)
            xi = 0.5 * np.multiply(A, ac_2d_diff).sum()
            if sample_sigma:
                if self.mode == 'gibbs':
                    mu2 = scipy.stats.invgamma.rvs(a=(I * J / 2) + k + 1, scale=chi + theta + xi,
                                                   random_state=self.random_state)
                else:
                    mu2 = (theta + chi + xi) / ((I * J / 2) + k + 1)
            E = np.dot(A.T, A)
            F = np.dot(A.T, X)
            for n in range(N):
                if components_rows_to_sample[n]:
                    notn = list(range(n)) + list(range(n + 1, N))
                    bn = (F[n] - np.dot(E[n, notn], B[notn]) - beta[n] * mu2) / (E[n, n] + MACHINE_PREC)
                    if self.mode == 'gibbs':
                        rnorm_variance = mu2 / (E[n, n] + MACHINE_PREC)
                        B[n] = truncated_normal_sample(bn, rnorm_variance, beta[n], random_state=self.random_state)
                    else:
                        B[n] = bn.clip(min=0)
            if self.mode == 'gibbs':
                if self.mean_only:
                    if m >= burnin_index:
                        A_mean += A / n_after_burnin
                        B_mean += B / n_after_burnin
                        mu2_mean += mu2 / n_after_burnin
                else:
                    As.append(A.copy())
                    Bs.append(B.copy())
                    mu2s.append(mu2)
            else:
                new_obj = mean_squared_error(X, np.dot(A, B))
                diff = obj - new_obj
                obj = new_obj
                if self.verbose:
                    print("MSE: ", obj)
            m += 1
        if self.mode == 'gibbs':
            As, Bs, mu2s = [np.array(arr) for arr in [As, Bs, mu2s]]
            self.A_mean = A_mean
            self.B_mean = B_mean
            self.mu2_mean = mu2_mean
            self.last_sampled_variance_ = float(mu2)
            self.last_sampled_bases_ = A.copy()
            self.last_sampled_components_ = B.copy()
            if self.mean_only:
                A = A_mean
                B = B_mean
                mu2 = mu2_mean
            else:
                self.burnin_bases_samples_ = As[:burnin_index]
                self.burnin_components_samples_ = Bs[:burnin_index]
                self.burnin_variance_samples_ = mu2s[:burnin_index]
                self.bases_samples_ = As[burnin_index:]
                self.components_samples_ = Bs[burnin_index:]
                self.variance_samples_ = mu2s[burnin_index:]
                A = self.bases_samples_.mean(axis=0)
                B = self.components_samples_.mean(axis=0)
                mu2 = self.variance_samples_.mean(axis=0)
        obj = mean_squared_error(X, np.dot(A, B))
        self.bases_ = A
        self.components_ = B
        self.variance_ = mu2
        self.reconstruction_err_ = obj
        self.n_free_params_ = np.count_nonzero(self.bases_) + np.count_nonzero(self.components_) + 1
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """Learn a NMF model for the data X and returns the transformed data.

        This is more efficient than calling fit followed by transform.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data matrix to be decomposed. Must be non-negative.

        y : ignored
        """
        # Todo: look into supporting sparse matrices
        self.fit(X, **fit_params)
        return self.bases_

    def aic(self, X):
        log_likelihood = self.score(X)
        return 2 * self.n_free_params_ - 2 * log_likelihood

    def score(self, X):
        X = check_array(X)
        check_is_fitted(self, ['input_shape_'])
        self.check_input_shape(X)
        X_model = np.dot(self.bases_, self.components_)
        err_std_dev = np.sqrt(self.variance_)
        log_likelihood = scipy.stats.norm.logpdf(X, loc=X_model, scale=err_std_dev).sum()
        return log_likelihood

    def bic(self, X):
        log_likelihood = self.score(X)
        I, J = X.shape
        n_samples = I * J
        return np.log(n_samples) * self.n_free_params_ - 2 * log_likelihood

    def check_input_shape(self, X):
        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape != self.input_shape_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')

    def marginal_likelihood(self, X, max_iter=1000, burnin_fraction=0.5, random_state=None, n_jobs=1):
        X = check_array(X)
        check_is_fitted(self, ['input_shape_'])
        self.check_input_shape(X)
        # Should burn-in not be necessary for this? If starting from the equilibrium distribution...
        # todo: check Schmidt code for this.
        if isinstance(random_state, np.random.RandomState):
            rs = random_state
        else:
            rs = np.random.RandomState(seed=random_state)
        M = max_iter
        A, B, mu2, alpha, beta, k, theta = [self.bases_,
                                            self.components_,
                                            self.variance_,
                                            self.bases_prior_,
                                            self.components_prior_,
                                            self.variance_prior_shape_,
                                            self.variance_prior_scale_]
        N = self.n_components
        X_model = np.dot(A, B)
        log_p_x_g_theta = scipy.stats.norm.logpdf(X, loc=X_model, scale=mu2).sum()
        log_p_a = scipy.stats.expon.logpdf(A,
                                           scale=1 / alpha).sum()  # if expon were param'd with rate lambda, it would not be inverse
        log_p_b = scipy.stats.expon.logpdf(B, scale=1 / beta).sum()
        minval = MACHINE_PREC
        log_p_mu2 = scipy.stats.invgamma.logpdf(mu2, a=max(minval, k), scale=max(minval, theta))
        log_numerator = sum([log_p_x_g_theta, log_p_a, log_p_b, log_p_mu2])

        # Now to do Gibbs sampling for each parameter block (columns of A, rows of B, so 2N runs total)
        # What about the variance parameter? Should it be Gibbs sampled? Maybe sampled along with each block?
        # p(t | X) =
        # p(t1 | X) * p(t2 | t1, X) ... p(tK| tK-1, ... t1, X)
        # p(tk | tk-1...t1, x) =
        # mean over gibbs samples of
        # p(tk | t1, t2, ... tk-1, tk+1(sampled), ...tK(sampled), X)
        chi = 0.5 * np.square(X).sum()
        gibbs_param_inputs = [(param_block_idx, X, A, B, alpha, beta, mu2, chi, k, theta, M, burnin_fraction, rs) for
                              param_block_idx in range(2 * N)]
        if n_jobs == -1:
            n_jobs = os.cpu_count()
        param_prob_means = Parallel(n_jobs=n_jobs)(delayed(gibbs_sample_param_block)(inputs) for inputs in gibbs_param_inputs)
        log_denom = np.array(param_prob_means).sum()
        return log_numerator - log_denom


def gibbs_sample_param_block(inputs):
    # In Python3.x multiprocessing can pickle instance methods,
    # so I could include this in the NMF object
    # so I don't have to pass around so many parameters
    # but for Python2 compatibility, it can't be an instance method
    # I could pass the NMF object, however.
    # However, this may prevent joblib from memmapping the arrays (if bound to NMF object)
    pbidx, X, A, B, alpha, beta, mu2, chi, k, theta, M, burnin_fraction, rs = inputs
    burnin_index = int(M * burnin_fraction)
    n_after_burnin = M - burnin_index
    I, N = A.shape
    N, J = B.shape
    nB = pbidx - N
    An = A.copy()
    Bn = B.copy()
    prob_dim = I if pbidx < N else J
    prob_mean = np.zeros((prob_dim,))  # 0
    for m in range(M):
        C = np.dot(Bn, Bn.T)
        D = np.dot(X, Bn.T)
        An_temp = An.copy()
        for n in range(min(pbidx + 1, N), N):
            notn = list(range(n - 1)) + list(range(n + 1, N))
            an = (D[:, n] - np.dot(An[:, notn], C[notn, n]) - alpha[:, n] * mu2) / (C[n, n] + MACHINE_PREC)
            rnorm_variance_a = mu2 / (C[n, n] + MACHINE_PREC)
            An_temp[:, n] = truncated_normal_sample(an, rnorm_variance_a, alpha[:, n], random_state=rs)
        An = An_temp
        ac_2d_diff = np.dot(An, C) - (2 * D)
        xi = 0.5 * np.multiply(An, ac_2d_diff).sum()
        # Should I be sampling mu2 or just using the fitted value?
        mu2 = scipy.stats.invgamma.rvs(a=(I * J / 2) + k + 1, scale=chi + theta + xi, random_state=rs)
        E = np.dot(An.T, An)
        F = np.dot(An.T, X)
        Bn_temp = Bn.copy()
        for n in range(nB + 1, N):
            notn = list(range(n - 1)) + list(range(n + 1, N))
            bn = (F[n] - np.dot(E[n, notn], Bn[notn]) - beta[n] * mu2) / (E[n, n] + MACHINE_PREC)
            rnorm_variance_b = mu2 / (E[n, n] + MACHINE_PREC)
            Bn_temp[n] = truncated_normal_sample(bn, rnorm_variance_b, beta[n], random_state=rs)
        Bn = Bn.copy()
        if pbidx < N:
            C = np.dot(Bn, Bn.T)
            D = np.dot(X, Bn.T)
            x = A[:, pbidx]
            notn = list(range(pbidx)) + list(range(pbidx + 1, N))
            param_mean = (D[:, pbidx] -
                          np.dot(An[:, notn], C[notn, pbidx]) -
                          alpha[:, pbidx] * mu2) / (C[pbidx, pbidx] + MACHINE_PREC)
            param_variance = mu2 / (C[pbidx, pbidx] + MACHINE_PREC)
            param_scale = alpha[:, pbidx]
        else:
            E = np.dot(An.T, An)
            F = np.dot(An.T, X)
            x = B[nB]
            notn = list(range(nB)) + list(range(nB + 1, N))
            param_mean = (F[nB] - np.dot(E[nB, notn], Bn[notn]) - beta[nB] * mu2) / (
                E[nB, nB] + MACHINE_PREC)
            param_variance = mu2 / (E[nB, nB] + MACHINE_PREC)
            param_scale = beta[nB]
        prob_sample = truncated_normal_pdf(x, param_mean, param_variance, param_scale, log=True)
        if m >= burnin_index:
            prob_mean = log_add(prob_mean, prob_sample - np.log(n_after_burnin))
    return np.sum(prob_mean)


def log_add(X, Y):
    # Taken from Schmidt's chib.m
    maxXY = np.max(np.array([X, Y]), axis=0)
    return maxXY + np.log(np.exp(X - maxXY) + np.exp(Y - maxXY))


def truncated_normal_sample(m, s, l, random_state=None):
    """
    Return random number from distribution with density
    p(x)=K*exp(-(x-m)^2/s-l'x), x>=0.
    m and l are vectors and s is scalar
    Adapted from randr function at http://mikkelschmidt.dk/code/gibbsnmf.html
    which is Copyright 2007 Mikkel N. Schmidt, ms@it.dk, www.mikkelschmidt.dk
    """
    if isinstance(random_state, np.random.RandomState):
        rs = random_state
    else:
        rs = np.random.RandomState(seed=random_state)
    sqrt_2s = np.sqrt(2 * s)
    ls = l * s
    lsm = ls - m
    A = lsm / sqrt_2s
    a = A > 26
    x = np.zeros(m.shape)
    y = rs.random_sample(m.shape)
    x[a] = -np.log(y[a]) / (lsm[a] / s)
    na = np.logical_not(a)
    R = erfc(abs(A[na]))
    x[na] = erfcinv(y[na] * R - (A[na] < 0) * (2 * y[na] + R - 2)) * sqrt_2s + m[na] - ls[na]
    x[np.isnan(x)] = 0
    x[x < 0] = 0
    x[np.isinf(x)] = 0
    return x.real


def truncated_normal_pdf(x, m, s, l, log=False):
    """
    It is a truncated normal according to Mikkel Schmidt.
    So the answer at
    https://stats.stackexchange.com/questions/179618/how-to-sample-from-a-rectified-gaussian-distribution
    is correct.

    Using formula for pdf from Wikipedia truncated normal page

    To visually verify it is the same distribution as that used in the above sampling function:
    plt.hist(truncated_normal_sample(np.zeros((100000,)), 1, np.ones((100000,))), bins=50, alpha=0.25, normed=True)
    xs = np.arange(-1, 3, 0.01)
    plt.plot(xs, truncated_normal_pdf(xs, np.zeros_like(xs), 1, np.ones_like(xs)))
    plt.show()

    Would it be possible to use scipy.stats.truncnorm instead?

    :param x: values at which to evaluate the density
    :param m: means of normal distributions
    :param s: variance (scalar)
    :param l: scale parameters from exponential distribution
    :param log: whether to return log of pdf
    :return:
    """
    pdf = np.array(x)
    pdf[x < 0] = 0
    nn = [x >= 0]
    newm = m - s * l
    std = np.sqrt(s)
    stdn = scipy.stats.norm()
    if log:
        pdf[x < 0] = LOG_ZERO
        normpdf = scipy.stats.norm(loc=newm[nn], scale=std).logpdf(x[nn])
        log_denom = math.log(std) + math_log(stdn.sf(- newm[nn] / std).clip(min=MACHINE_PREC))
        pdf[nn] = np.subtract(normpdf, log_denom)
    else:
        pdf[x < 0] = 0
        normpdf = scipy.stats.norm(loc=newm[nn], scale=std).pdf(x[nn])
        denom = std * stdn.sf(- newm[nn] / std).clip(min=MACHINE_PREC)
        pdf[nn] = np.divide(normpdf, denom)
    return pdf


