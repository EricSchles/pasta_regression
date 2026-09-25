import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import mean_squared_error


class SGDRegressor(BaseEstimator, RegressorMixin):
    """
    Simple scikit-learn-compatible SGD linear regression.

    The estimator minimizes:

        1/(2n) * ||Xw - y||^2

    using minibatch stochastic gradient descent.
    """

    def __init__(
        self,
        learning_rate=0.01,
        batch_size=32,
        max_iter=1000,
        learning_rate_decay=0.0,
        fit_intercept=True,
        random_state=None,
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.learning_rate_decay = learning_rate_decay
        self.fit_intercept = fit_intercept
        self.random_state = random_state

    def _prepare_X(self, X):
        if self.fit_intercept:
            return np.column_stack([np.ones(len(X)), X])
        return X

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        X = self._prepare_X(X)

        rng = np.random.default_rng(self.random_state)

        n_samples, n_features = X.shape

        self.coef_ = np.zeros(n_features)
        self.loss_curve_ = []

        for iteration in range(self.max_iter):

            indices = rng.choice(
                n_samples,
                size=min(self.batch_size, n_samples),
                replace=False,
            )

            X_batch = X[indices]
            y_batch = y[indices]

            residual = X_batch @ self.coef_ - y_batch

            gradient = (
                X_batch.T @ residual
                / len(indices)
            )

            lr = self.learning_rate / (
                1.0 + self.learning_rate_decay * iteration
            )

            self.coef_ -= lr * gradient

            prediction = X @ self.coef_

            self.loss_curve_.append(
                0.5 * np.mean((prediction - y) ** 2)
            )

        if self.fit_intercept:
            self.intercept_ = self.coef_[0]
            self.coef_ = self.coef_[1:]
        else:
            self.intercept_ = 0.0

        return self

    def predict(self, X):
        check_is_fitted(self, ["coef_", "intercept_"])

        X = check_array(X)

        return self.intercept_ + X @ self.coef_

    def score(self, X, y):
        prediction = self.predict(X)

        return 1.0 - (
            np.sum((y - prediction) ** 2)
            / np.sum((y - np.mean(y)) ** 2)
        )


class PASTARegressor(BaseEstimator, RegressorMixin):
    """
    PASTA-style linear regression.

    Implements the epoch-based proximal anchoring mechanism from:

        Fazla et al. (2026)
        "Lower Bounds and Proximally Anchored SGD
         for Non-Convex Minimization Under Unbounded Variance"

    Inner-loop update:

        w_{t+1} =
            beta * anchor
            + (1 - beta) * w_t
            - eta * gradient

    with:

        beta = lambda * eta

    The anchor is reset after every epoch.
    """

    def __init__(
        self,
        learning_rate=0.01,
        regularization=0.01,
        batch_size=32,
        epoch_length=100,
        n_epochs=20,
        learning_rate_decay=0.0,
        fit_intercept=True,
        random_state=None,
    ):
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.batch_size = batch_size
        self.epoch_length = epoch_length
        self.n_epochs = n_epochs
        self.learning_rate_decay = learning_rate_decay
        self.fit_intercept = fit_intercept
        self.random_state = random_state

    def _prepare_X(self, X):
        if self.fit_intercept:
            return np.column_stack([np.ones(len(X)), X])
        return X

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        X = self._prepare_X(X)

        rng = np.random.default_rng(self.random_state)

        n_samples, n_features = X.shape

        # Initial point.
        w = np.zeros(n_features)

        # Epoch-level anchor.
        anchor = w.copy()

        self.loss_curve_ = []
        self.anchor_history_ = []

        iteration = 0

        for epoch in range(self.n_epochs):

            # Freeze the anchor during this epoch.
            anchor = w.copy()

            self.anchor_history_.append(anchor.copy())

            for _ in range(self.epoch_length):

                indices = rng.choice(
                    n_samples,
                    size=min(self.batch_size, n_samples),
                    replace=False,
                )

                X_batch = X[indices]
                y_batch = y[indices]

                residual = X_batch @ w - y_batch

                gradient = (
                    X_batch.T @ residual
                    / len(indices)
                )

                lr = self.learning_rate / (
                    1.0 + self.learning_rate_decay * iteration
                )

                # PASTA coupling:
                #
                # beta = lambda * eta
                #
                beta = self.regularization * lr

                # Keep beta in a valid Halpern range.
                beta = min(beta, 1.0)

                w = (
                    beta * anchor
                    + (1.0 - beta) * w
                    - lr * gradient
                )

                prediction = X @ w

                self.loss_curve_.append(
                    0.5 * np.mean((prediction - y) ** 2)
                )

                iteration += 1

        if self.fit_intercept:
            self.intercept_ = w[0]
            self.coef_ = w[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = w

        return self

    def predict(self, X):
        check_is_fitted(self, ["coef_", "intercept_"])

        X = check_array(X)

        return self.intercept_ + X @ self.coef_

    def score(self, X, y):
        prediction = self.predict(X)

        return 1.0 - (
            np.sum((y - prediction) ** 2)
            / np.sum((y - np.mean(y)) ** 2)
        )


class StateDependentLinearRegression:
    """
    Generates a linear regression problem with BG-0-style noise.

    y = X beta + epsilon

    where

        Var(epsilon | w)
            ~
        base_variance
        +
        variance_growth * ||w - w0||^2

    The resulting stochastic gradient therefore becomes
    increasingly noisy as the optimizer moves away from w0.
    """

    def __init__(
        self,
        n_samples=5000,
        n_features=20,
        noise_std=1.0,
        variance_growth=1.0,
        random_state=None,
    ):
        self.n_samples = n_samples
        self.n_features = n_features
        self.noise_std = noise_std
        self.variance_growth = variance_growth
        self.random_state = random_state

    def generate(self):
        rng = np.random.default_rng(self.random_state)

        X = rng.normal(
            size=(self.n_samples, self.n_features)
        )

        beta = rng.normal(
            size=self.n_features
        )

        # Clean response.
        y_clean = X @ beta

        self.X = X
        self.beta = beta
        self.y_clean = y_clean

        return X, y_clean

    def stochastic_batch(
        self,
        X_batch,
        y_batch,
        w,
        w0,
        rng,
    ):
        """
        Produce a noisy response whose variance depends
        quadratically on distance from the anchor.
        """

        distance_sq = np.sum(
            (w - w0) ** 2
        )

        variance = (
            self.noise_std ** 2
            +
            self.variance_growth * distance_sq
        )

        noise = rng.normal(
            scale=np.sqrt(variance),
            size=len(y_batch),
        )

        return y_batch + noise


def run_benchmark(
    variance_growth=1.0,
    n_repeats=10,
    random_state=42,
):
    """
    Compare ordinary SGD against PASTA.

    Returns a dictionary containing the final MSE and R²
    for each repetition.
    """

    rng = np.random.default_rng(random_state)

    results = {
        "SGD": [],
        "PASTA": [],
    }

    for repeat in range(n_repeats):

        seed = int(rng.integers(0, 1_000_000))

        generator = StateDependentLinearRegression(
            n_samples=5000,
            n_features=20,
            noise_std=1.0,
            variance_growth=variance_growth,
            random_state=seed,
        )

        X, y = generator.generate()

        # Hold out a clean test set.
        n_train = int(0.8 * len(X))

        X_train = X[:n_train]
        y_train = y[:n_train]

        X_test = X[n_train:]
        y_test = y[n_train:]

        # -------------------------------------------------
        # SGD
        # -------------------------------------------------

        sgd = SGDRegressor(
            learning_rate=0.01,
            batch_size=32,
            max_iter=2000,
            learning_rate_decay=1e-4,
            random_state=seed,
        )

        sgd.fit(X_train, y_train)

        sgd_prediction = sgd.predict(X_test)

        results["SGD"].append({
            "mse": mean_squared_error(
                y_test,
                sgd_prediction,
            ),
            "r2": sgd.score(X_test, y_test),
        })

        # -------------------------------------------------
        # PASTA
        # -------------------------------------------------

        pasta = PASTARegressor(
            learning_rate=0.01,
            regularization=0.1,
            batch_size=32,
            epoch_length=100,
            n_epochs=20,
            learning_rate_decay=1e-4,
            random_state=seed,
        )

        pasta.fit(X_train, y_train)

        pasta_prediction = pasta.predict(X_test)

        results["PASTA"].append({
            "mse": mean_squared_error(
                y_test,
                pasta_prediction,
            ),
            "r2": pasta.score(X_test, y_test),
        })

    return results


if __name__ == "__main__":

    results = run_benchmark(
        variance_growth=2.0,
        n_repeats=10,
    )

    for algorithm, runs in results.items():

        mse = np.mean([
            result["mse"]
            for result in runs
        ])

        r2 = np.mean([
            result["r2"]
            for result in runs
        ])

        print(
            f"{algorithm:>8} | "
            f"MSE = {mse:10.4f} | "
            f"R² = {r2:8.4f}"
        )
