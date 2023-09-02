from matrix_factorization.kernel_matrix_factorization import KernelMF, _sgd
import pandas as pd
import numpy as np
from typing import Optional

class RetrainableKernelMF(KernelMF):
    '''Splits KernelMF fit function into separate parts'''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prep_data(self, X: pd.DataFrame, y: pd.Series, bound=False) -> np.ndarray:
        ans = self._preprocess_data(X=X, y=y, type="fit")
        # after preprocessing, ans["rating"] = y
        self.global_mean = ans["rating"].mean()
        if bound:
            self.min_rating = ans["rating"].min()
            self.max_rating = ans["rating"].max()
        return ans.to_numpy()
    
    def initialize(self) -> None:
        # Initialize vector bias parameters
        self.user_biases = np.zeros(self.n_users)
        self.item_biases = np.zeros(self.n_items)

        # Initialize latent factor parameters of matrices P and Q
        self.user_features = np.random.normal(
            self.init_mean, self.init_sd, (self.n_users, self.n_factors)
        )
        self.item_features = np.random.normal(
            self.init_mean, self.init_sd, (self.n_items, self.n_factors)
        )

    def train(self, prepped_data: np.ndarray, n_epochs: Optional[int]=None) -> None:
        if n_epochs is None:
            n_epochs = self.n_epochs
        # Perform stochastic gradient descent
        (
            self.user_features,
            self.item_features,
            self.user_biases,
            self.item_biases,
            self.train_rmse,
        ) = _sgd(
            X=prepped_data,
            global_mean=self.global_mean,
            user_biases=self.user_biases,
            item_biases=self.item_biases,
            user_features=self.user_features,
            item_features=self.item_features,
            n_epochs=n_epochs,
            kernel=self.kernel,
            gamma=self.gamma,
            lr=self.lr,
            reg=self.reg,
            min_rating=self.min_rating,
            max_rating=self.max_rating,
            verbose=self.verbose,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series, bound: bool=False):
        '''Just clarifying how this matches the original fit function'''
        X = self.prep_data(X, y, bound=bound)
        self.initialize()
        self.train(X)
        return self