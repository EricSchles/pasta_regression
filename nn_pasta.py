import numpy as np
import torch
import torch.nn as nn

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import r2_score, accuracy_score
from torch.optim import Optimizer


class PASTA(Optimizer):
    """
    PASTA-style optimizer.

    Parameters
    ----------
    params : iterable
        Model parameters.

    lr : float
        Learning rate.

    regularization : float
        PASTA anchoring strength.

    epoch_length : int
        Number of optimizer steps before refreshing the anchor.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        regularization=0.01,
        epoch_length=100,
    ):
        if lr <= 0:
            raise ValueError("lr must be positive")

        if regularization < 0:
            raise ValueError(
                "regularization must be non-negative"
            )

        if epoch_length <= 0:
            raise ValueError(
                "epoch_length must be positive"
            )

        defaults = dict(
            lr=lr,
            regularization=regularization,
            epoch_length=epoch_length,
        )

        super().__init__(params, defaults)

        self._step_count = 0
        self._anchors = {}

    @torch.no_grad()
    def step(self, closure=None):

        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1

        for group in self.param_groups:

            lr = group["lr"]
            lam = group["regularization"]
            epoch_length = group["epoch_length"]

            # Refresh the anchor.
            if (
                self._step_count == 1
                or
                (self._step_count - 1)
                % epoch_length == 0
            ):
                for p in group["params"]:
                    if p.requires_grad:
                        self._anchors[p] = (
                            p.detach().clone()
                        )

            beta = min(lam * lr, 1.0)

            for p in group["params"]:

                if p.grad is None:
                    continue

                anchor = self._anchors[p]

                p.mul_(1.0 - beta)

                p.add_(
                    anchor,
                    alpha=beta,
                )

                p.add_(
                    p.grad,
                    alpha=-lr,
                )

        return loss


class MLP(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        n_hidden_layers=2,
        output_dim=1,
    ):
        super().__init__()

        layers = []

        in_features = input_dim

        for _ in range(n_hidden_layers):

            layers.append(
                nn.Linear(
                    in_features,
                    hidden_dim,
                )
            )

            layers.append(nn.ReLU())

            in_features = hidden_dim

        layers.append(
            nn.Linear(
                in_features,
                output_dim,
            )
        )

        self.network = nn.Sequential(*layers)

    def forward(self, X):
        return self.network(X)


class PASTARegressor(
    BaseEstimator,
    RegressorMixin,
):
    """
    Scikit-learn style MLP regression using PASTA.

    Example
    -------
    model = PASTARegressor(
        hidden_dim=128,
        n_epochs=100,
        batch_size=64,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    """

    def __init__(
        self,
        hidden_dim=128,
        n_hidden_layers=2,
        learning_rate=1e-3,
        pasta_regularization=0.01,
        epoch_length=100,
        batch_size=64,
        n_epochs=100,
        random_state=None,
        device="cpu",
    ):
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        self.learning_rate = learning_rate
        self.pasta_regularization = pasta_regularization
        self.epoch_length = epoch_length
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.device = device

    def _set_seed(self):
        if self.random_state is not None:
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)

    def fit(self, X, y):

        X, y = check_X_y(
            X,
            y,
            y_numeric=True,
        )

        self._set_seed()

        self.n_features_in_ = X.shape[1]

        X_tensor = torch.tensor(
            X,
            dtype=torch.float32,
            device=self.device,
        )

        y_tensor = torch.tensor(
            y.reshape(-1, 1),
            dtype=torch.float32,
            device=self.device,
        )

        self.model_ = MLP(
            input_dim=self.n_features_in_,
            hidden_dim=self.hidden_dim,
            n_hidden_layers=self.n_hidden_layers,
            output_dim=1,
        ).to(self.device)

        self.optimizer_ = PASTA(
            self.model_.parameters(),
            lr=self.learning_rate,
            regularization=self.pasta_regularization,
            epoch_length=self.epoch_length,
        )

        criterion = nn.MSELoss()

        rng = np.random.default_rng(
            self.random_state
        )

        n_samples = len(X)

        self.loss_curve_ = []

        for epoch in range(self.n_epochs):

            permutation = rng.permutation(
                n_samples
            )

            epoch_losses = []

            for start in range(
                0,
                n_samples,
                self.batch_size,
            ):

                indices = permutation[
                    start:start + self.batch_size
                ]

                xb = X_tensor[indices]
                yb = y_tensor[indices]

                self.optimizer_.zero_grad()

                prediction = self.model_(xb)

                loss = criterion(
                    prediction,
                    yb,
                )

                loss.backward()

                self.optimizer_.step()

                epoch_losses.append(
                    loss.item()
                )

            self.loss_curve_.append(
                np.mean(epoch_losses)
            )

        return self

    def predict(self, X):

        check_is_fitted(
            self,
            ["model_", "n_features_in_"],
        )

        X = check_array(X)

        X_tensor = torch.tensor(
            X,
            dtype=torch.float32,
            device=self.device,
        )

        self.model_.eval()

        with torch.no_grad():

            prediction = self.model_(
                X_tensor
            ).cpu().numpy()

        return prediction.ravel()

    def score(self, X, y):

        return r2_score(
            y,
            self.predict(X),
        )

if __name__ == '__main__':
    from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

X, y = make_regression(
    n_samples=5000,
        n_features=20,
        n_informative=15,
        noise=10,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = PASTARegressor(
        hidden_dim=128,
        n_hidden_layers=2,
        learning_rate=1e-3,
        pasta_regularization=0.1,
        epoch_length=100,
        batch_size=64,
        n_epochs=100,
        random_state=42,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("R²:", model.score(X_test, y_test))
