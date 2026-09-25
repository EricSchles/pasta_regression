import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import (
    check_X_y,
    check_array,
    check_is_fitted,
)
from sklearn.metrics import accuracy_score


class SGDLinearSVM(BaseEstimator, ClassifierMixin):
    """
    Linear SVM trained with stochastic subgradient descent.

    Objective:

        lambda / 2 * ||w||^2
        +
        mean(max(0, 1 - y * (Xw + b)))

    Labels are internally converted to {-1, +1}.
    """

    def __init__(
        self,
        learning_rate=0.01,
        regularization=0.01,
        batch_size=32,
        max_iter=1000,
        learning_rate_decay=0.0,
        fit_intercept=True,
        random_state=None,
    ):
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.learning_rate_decay = learning_rate_decay
        self.fit_intercept = fit_intercept
        self.random_state = random_state

    def _prepare_X(self, X):
        if self.fit_intercept:
            return np.column_stack(
                [np.ones(len(X)), X]
            )
        return X

    def _encode_labels(self, y):
        self.classes_ = np.unique(y)

        if len(self.classes_) != 2:
            raise ValueError(
                "SGDLinearSVM requires exactly two classes."
            )

        self._negative_class = self.classes_[0]
        self._positive_class = self.classes_[1]

        return np.where(
            y == self._positive_class,
            1.0,
            -1.0,
        )

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        y_encoded = self._encode_labels(y)
        X_aug = self._prepare_X(X)

        rng = np.random.default_rng(
            self.random_state
        )

        n_samples, n_features = X_aug.shape

        w = np.zeros(n_features)

        self.loss_curve_ = []

        for iteration in range(self.max_iter):

            batch_size = min(
                self.batch_size,
                n_samples,
            )

            indices = rng.choice(
                n_samples,
                size=batch_size,
                replace=False,
            )

            X_batch = X_aug[indices]
            y_batch = y_encoded[indices]

            scores = X_batch @ w

            margins = y_batch * scores

            active = margins < 1.0

            # Hinge-loss subgradient.
            gradient = np.zeros_like(w)

            if np.any(active):
                gradient = -(
                    X_batch[active].T
                    @ y_batch[active]
                ) / batch_size

            # L2 regularization.
            if self.fit_intercept:
                gradient[1:] += (
                    self.regularization
                    * w[1:]
                )
            else:
                gradient += (
                    self.regularization * w
                )

            lr = self.learning_rate / (
                1.0
                + self.learning_rate_decay
                * iteration
            )

            w -= lr * gradient

            # Track objective.
            scores_full = X_aug @ w
            margins_full = y_encoded * scores_full

            hinge = np.maximum(
                0.0,
                1.0 - margins_full,
            )

            if self.fit_intercept:
                reg = (
                    0.5
                    * self.regularization
                    * np.sum(w[1:] ** 2)
                )
            else:
                reg = (
                    0.5
                    * self.regularization
                    * np.sum(w ** 2)
                )

            loss = np.mean(hinge) + reg

            self.loss_curve_.append(loss)

        if self.fit_intercept:
            self.intercept_ = w[0]
            self.coef_ = w[1:].copy()
        else:
            self.intercept_ = 0.0
            self.coef_ = w.copy()

        return self

    def decision_function(self, X):
        check_is_fitted(
            self,
            ["coef_", "intercept_"],
        )

        X = check_array(X)

        return X @ self.coef_ + self.intercept_

    def predict(self, X):
        scores = self.decision_function(X)

        return np.where(
            scores >= 0,
            self._positive_class,
            self._negative_class,
        )

    def score(self, X, y):
        return accuracy_score(
            y,
            self.predict(X),
        )


class PASTASVM(BaseEstimator, ClassifierMixin):
    """
    Linear SVM trained with PASTA-style stochastic
    subgradient descent.

    Within each epoch the anchor is fixed.

    Update:

        w_{t+1}
            =
            beta_t * anchor
            +
            (1 - beta_t) * w_t
            -
            eta_t * g_t

    with

        beta_t = lambda_pasta * eta_t.

    The ordinary SVM L2 regularization and the PASTA
    proximal anchoring are separate quantities.
    """

    def __init__(
        self,
        learning_rate=0.01,
        svm_regularization=0.01,
        pasta_regularization=0.1,
        batch_size=32,
        epoch_length=100,
        n_epochs=20,
        learning_rate_decay=0.0,
        fit_intercept=True,
        random_state=None,
    ):
        self.learning_rate = learning_rate
        self.svm_regularization = svm_regularization
        self.pasta_regularization = pasta_regularization
        self.batch_size = batch_size
        self.epoch_length = epoch_length
        self.n_epochs = n_epochs
        self.learning_rate_decay = learning_rate_decay
        self.fit_intercept = fit_intercept
        self.random_state = random_state

    def _prepare_X(self, X):
        if self.fit_intercept:
            return np.column_stack(
                [np.ones(len(X)), X]
            )
        return X

    def _encode_labels(self, y):
        self.classes_ = np.unique(y)

        if len(self.classes_) != 2:
            raise ValueError(
                "PASTASVM requires exactly two classes."
            )

        self._negative_class = self.classes_[0]
        self._positive_class = self.classes_[1]

        return np.where(
            y == self._positive_class,
            1.0,
            -1.0,
        )

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        y_encoded = self._encode_labels(y)
        X_aug = self._prepare_X(X)

        rng = np.random.default_rng(
            self.random_state
        )

        n_samples, n_features = X_aug.shape

        w = np.zeros(n_features)

        self.loss_curve_ = []
        self.anchor_history_ = []

        iteration = 0

        for epoch in range(self.n_epochs):

            # -----------------------------------------
            # PASTA anchor
            # -----------------------------------------

            anchor = w.copy()

            self.anchor_history_.append(
                anchor.copy()
            )

            for _ in range(self.epoch_length):

                batch_size = min(
                    self.batch_size,
                    n_samples,
                )

                indices = rng.choice(
                    n_samples,
                    size=batch_size,
                    replace=False,
                )

                X_batch = X_aug[indices]
                y_batch = y_encoded[indices]

                scores = X_batch @ w

                margins = (
                    y_batch * scores
                )

                active = margins < 1.0

                # -------------------------------------
                # Stochastic hinge subgradient
                # -------------------------------------

                gradient = np.zeros_like(w)

                if np.any(active):
                    gradient = -(
                        X_batch[active].T
                        @ y_batch[active]
                    ) / batch_size

                # -------------------------------------
                # Ordinary SVM regularization
                # -------------------------------------

                if self.fit_intercept:
                    gradient[1:] += (
                        self.svm_regularization
                        * w[1:]
                    )
                else:
                    gradient += (
                        self.svm_regularization
                        * w
                    )

                # -------------------------------------
                # Step size
                # -------------------------------------

                lr = self.learning_rate / (
                    1.0
                    + self.learning_rate_decay
                    * iteration
                )

                # -------------------------------------
                # PASTA coupling
                #
                # beta = lambda * eta
                # -------------------------------------

                beta = (
                    self.pasta_regularization
                    * lr
                )

                beta = min(beta, 1.0)

                # -------------------------------------
                # PASTA update
                # -------------------------------------

                w = (
                    beta * anchor
                    + (1.0 - beta) * w
                    - lr * gradient
                )

                # -------------------------------------
                # Objective
                # -------------------------------------

                scores_full = X_aug @ w

                margins_full = (
                    y_encoded
                    * scores_full
                )

                hinge = np.maximum(
                    0.0,
                    1.0 - margins_full,
                )

                if self.fit_intercept:
                    reg = (
                        0.5
                        * self.svm_regularization
                        * np.sum(w[1:] ** 2)
                    )
                else:
                    reg = (
                        0.5
                        * self.svm_regularization
                        * np.sum(w ** 2)
                    )

                loss = (
                    np.mean(hinge)
                    + reg
                )

                self.loss_curve_.append(loss)

                iteration += 1

        if self.fit_intercept:
            self.intercept_ = w[0]
            self.coef_ = w[1:].copy()
        else:
            self.intercept_ = 0.0
            self.coef_ = w.copy()

        return self

    def decision_function(self, X):
        check_is_fitted(
            self,
            ["coef_", "intercept_"],
        )

        X = check_array(X)

        return X @ self.coef_ + self.intercept_

    def predict(self, X):
        scores = self.decision_function(X)

        return np.where(
            scores >= 0,
            self._positive_class,
            self._negative_class,
        )

    def score(self, X, y):
        return accuracy_score(
            y,
            self.predict(X),
        )

if __name__ == '__main__':
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(
        n_samples=5000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        class_sep=1.0,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    sgd = SGDLinearSVM(
        learning_rate=0.01,
        regularization=0.01,
        batch_size=32,
        max_iter=5000,
        random_state=42,
    )

    pasta = PASTASVM(
        learning_rate=0.01,
        svm_regularization=0.01,
        pasta_regularization=0.1,
        batch_size=32,
        epoch_length=100,
        n_epochs=50,
        random_state=42,
    )

    sgd.fit(X_train, y_train)
    pasta.fit(X_train, y_train)

    print("SGD accuracy:", sgd.score(X_test, y_test))
    print("PASTA accuracy:", pasta.score(X_test, y_test))
