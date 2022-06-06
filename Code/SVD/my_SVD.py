import numpy as np
import pandas as pd
import time

from my_fast_methods import _shuffle, _initialization, _run_epoch, _run_epoch_mixed, _compute_val_metrics, _compute_val_metrics_mixed
from utils import _timer



class my_SVD:
    def __init__(self, lr=.005, reg=.02, n_epochs=20, n_factors=100,
                 early_stopping=False, shuffle=False, min_delta=.001,
                 min_rating=1, max_rating=5, mode="rating"):

        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.n_factors = n_factors
        self.early_stopping = early_stopping
        self.shuffle = shuffle
        self.min_delta = min_delta
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.mode = mode
        # 2 modes: "rating" → rating only, "mixed" → rating + nlp

    @_timer(text='\nTraining took ')
    def fit(self, X, X_val=None):
        """Learns model weights from input data.

        Parameters
        ----------
        X : pandas.DataFrame
            Training set, must have 'u_id' for user ids, 'i_id' for item ids,
            and 'rating' column names.
        X_val : pandas.DataFrame, default=None
            Validation set with the same column structure as X.

        Returns
        -------
        self : SVD object
            The current fitted object.
        """

        X = self._preprocess_data(X, mode=self.mode)

        if X_val is not None:
            X_val = self._preprocess_data(X_val, train=False, verbose=False, mode=self.mode)
        self._init_metrics()

        
        # self.userMean = 
        self._run_sgd(X, X_val, mode=self.mode)

        return self
    
    def _preprocess_data(self, X : pd.DataFrame, train=True, verbose=True, mode="rating"):
        """Maps user and item ids to their indexes.

        Parameters
        ----------
        X : pandas.DataFrame
            Dataset, must have 'u_id' for user ids, 'i_id' for item ids, and
            'rating' column names.
        train : boolean
            Whether or not X is the training set or the validation set.

        Returns
        -------
        X : numpy.array
            Mapped dataset.
        """
        print('Preprocessing data...\n')
        
        X = X.copy()
        if mode == "rating":
            X = X.iloc[:, 0:3]

        if train:  # Mappings have to be created
            user_ids = X.iloc[:, 0].unique().tolist()
            item_ids = X.iloc[:, 1].unique().tolist()

            n_users = len(user_ids)
            n_items = len(item_ids)

            user_idx = range(n_users)
            item_idx = range(n_items)

            self.user_mapping_ = dict(zip(user_ids, user_idx))
            self.item_mapping_ = dict(zip(item_ids, item_idx))

            self.global_rating_mean_ = np.mean(X.iloc[:, 2])

            if mode == "mixed":
                self.global_nlp_mean_ = np.mean(X[X.iloc[:, 3] > 0].iloc[:, 3])
                # X[X.iloc[:, 3] == -1].iloc[:, 3] = self.global_nlp_mean_
                X.iloc[list(X.iloc[:, 3] == -1), 3] = self.global_nlp_mean_

        X.iloc[:, 0] = X.iloc[:, 0].map(self.user_mapping_)
        X.iloc[:, 1] = X.iloc[:, 1].map(self.item_mapping_)

        # Tag validation set unknown users/items with -1 (enables
        # `fast_methods._compute_val_metrics` detecting them)
        X.fillna(-1, inplace=True)

        X.iloc[:, 0] = X.iloc[:, 0].astype(np.int32)
        X.iloc[:, 1] = X.iloc[:, 1].astype(np.int32)

        if train:
            user_idx = X.iloc[:, 0].unique().tolist()
            item_idx = X.iloc[:, 1].unique().tolist()

            user_rating_means = X.groupby(X.columns[0], as_index=False).mean().iloc[:, 2].to_list()
            item_rating_means = X.groupby(X.columns[1], as_index=False).mean().iloc[:, 2].to_list()

            self.user_rating_means_mapping_ = dict(zip(user_idx, user_rating_means))
            self.item_rating_means_mapping_ = dict(zip(item_idx, item_rating_means))

            if mode == "mixed":
                user_nlp_means = X.groupby(X.columns[0], as_index=False).mean().iloc[:, 3].to_list()
                item_nlp_means = X.groupby(X.columns[1], as_index=False).mean().iloc[:, 3].to_list()

                self.user_nlp_means_mapping_ = dict(zip(user_idx, user_nlp_means))
                self.item_nlp_means_mapping_ = dict(zip(item_idx, item_nlp_means))
        
        X["user_rating_mean"] = X.iloc[:, 0].map(self.user_rating_means_mapping_)
        X["item_rating_mean"] = X.iloc[:, 1].map(self.item_rating_means_mapping_)

        # Tag validation set unknown users/items mean with global mean
        X.fillna(self.global_rating_mean_, inplace=True)

        if mode == "mixed":
            X["user_nlp_mean"] = X.iloc[:, 0].map(self.user_nlp_means_mapping_)
            X["item_nlp_mean"] = X.iloc[:, 1].map(self.item_nlp_means_mapping_)
            X.fillna(self.global_nlp_mean_, inplace=True)
        
        if mode == "mixed":
            return X.iloc[:, [0, 1, 2, 4, 5, 6, 7]].values
        else:
            return X.iloc[:, :5].values

    def _init_metrics(self):
        metrics = np.zeros((self.n_epochs, 6), dtype=float)
        self.metrics_ = pd.DataFrame(metrics, columns=['Train Loss', 'Train RMSE', 'Train MAE', 'Valid Loss', 'Valid RMSE', 'Valid MAE'])

    # todo
    def _run_sgd(self, X, X_val, mode="rating"):
        """Runs SGD algorithm, learning model weights.

        Parameters
        ----------
        X : numpy.array
            Training set, first column must be user indexes, second one item
            indexes, and third one ratings.
        X_val : numpy.array or None
            Validation set with the same structure as X.
        """
        n_users = len(np.unique(X[:, 0]))
        n_items = len(np.unique(X[:, 1]))

        if mode == "rating":
            bu_k1, _, bu_c, bi_k1, _, bi_c, pu, qi = _initialization(n_users, n_items, self.n_factors)
        else:
            bu_k1, bu_k2, bu_c, bi_k1, bi_k2, bi_c, pu, qi = _initialization(n_users, n_items, self.n_factors)

        # Run SGD
        for epoch_ix in range(self.n_epochs):
            start = self._on_epoch_begin(epoch_ix)

            if self.shuffle:
                X = _shuffle(X)
            
            if mode == 'rating':
                bu_k1, bu_c, bi_k1, bi_c, pu, qi = _run_epoch(
                    X,
                    bu_k1, bu_c, bi_k1, bi_c, pu, qi,
                    self.n_factors,
                    self.global_rating_mean_,
                    self.lr,
                    self.reg
                )
                self.metrics_.loc[epoch_ix, ['Train Loss', 'Train RMSE', 'Train MAE']] = _compute_val_metrics(
                    X,
                    bu_k1, bu_c, bi_k1, bi_c, pu, qi,
                    self.n_factors,
                    self.global_rating_mean_
                )
            elif mode == "mixed":
                bu_k1, bu_k2, bu_c, bi_k1, bi_k2, bi_c, pu, qi = _run_epoch_mixed(
                    X,
                    bu_k1, bu_k2, bu_c, bi_k1, bi_k2, bi_c, pu, qi,
                    self.n_factors,
                    self.global_rating_mean_,
                    self.global_nlp_mean_,
                    self.lr,
                    self.reg
                )
                self.metrics_.loc[epoch_ix, ['Train Loss', 'Train RMSE', 'Train MAE']] = _compute_val_metrics_mixed(
                    X,
                    bu_k1, bu_k2, bu_c, bi_k1, bi_k2, bi_c, pu, qi,
                    self.n_factors,
                    self.global_rating_mean_,
                    self.global_nlp_mean_
                )
            
            if X_val is not None:
                if mode == "rating":
                    self.metrics_.loc[epoch_ix, ['Valid Loss', 'Valid RMSE', 'Valid MAE']] = _compute_val_metrics(
                        X_val,
                        bu_k1, bu_c, bi_k1, bi_c, pu, qi,
                        self.n_factors,
                        self.global_rating_mean_
                    )
                else:
                    self.metrics_.loc[epoch_ix, ['Valid Loss', 'Valid RMSE', 'Valid MAE']] = _compute_val_metrics_mixed(
                        X_val,
                        bu_k1, bu_k2, bu_c, bi_k1, bi_k2, bi_c, pu, qi,
                        self.n_factors,
                        self.global_rating_mean_,
                        self.global_nlp_mean_
                    )
                self._on_epoch_end(
                    start,
                    train_loss=self.metrics_.loc[epoch_ix, 'Train Loss'],
                    train_rmse=self.metrics_.loc[epoch_ix, 'Train RMSE'],
                    train_mae=self.metrics_.loc[epoch_ix, 'Train MAE'],
                    val_loss=self.metrics_.loc[epoch_ix, 'Valid Loss'],
                    val_rmse=self.metrics_.loc[epoch_ix, 'Valid RMSE'],
                    val_mae=self.metrics_.loc[epoch_ix, 'Valid MAE']
                )
            else:
                self._on_epoch_end(
                    start,
                    train_loss=self.metrics_.loc[epoch_ix, 'Train Loss'],
                    train_rmse=self.metrics_.loc[epoch_ix, 'Train RMSE'],
                    train_mae=self.metrics_.loc[epoch_ix, 'Train MAE']
                )
            
            # Early stopping
            if self.early_stopping:
                if X_val is not None:
                    val_rmse = self.metrics_['Valid RMSE'].tolist()
                    if self._early_stopping(epoch_ix, self.min_delta, val_rmse=val_rmse):
                        break
                else:
                    train_rmse = self.metrics_['Train RMSE'].tolist()
                    if self._early_stopping(epoch_ix, self.min_delta, train_rmse=train_rmse):
                        break

        self.bu_k1_ = bu_k1
        self.bu_c_ = bu_c
        self.bi_k1_ = bi_k1
        self.bi_c_ = bi_c
        self.pu_ = pu
        self.qi_ = qi
        if mode == "mixed":
            self.bu_k2_ = bu_k2
            self.bi_k2_ = bi_k2
    
    def predict(self, X, clip=True):
        """Returns estimated ratings of several given user/item pairs.

        Parameters
        ----------
        X : pandas.DataFrame
            Storing all user/item pairs we want to predict the ratings. Must
            contains columns labeled 'u_id' and 'i_id'.
        clip : bool, default=True
            Whether to clip the predictions or not.

        Returns
        -------
        predictions : list
            Predictions belonging to the input user/item pairs.
        """
        return [
            self.predict_pair(u_id, i_id, clip)
            for u_id, i_id in zip(X.iloc[:, 0], X.iloc[:, 1])
        ]
    
    def predict_pair(self, u_id, i_id, clip=True, mode="rating"):
        """Returns the model rating prediction for a given user/item pair.

        Parameters
        ----------
        u_id : int
            A user id.
        i_id : int
            An item id.
        clip : bool, default=True
            Whether to clip the prediction or not.

        Returns
        -------
        pred : float
            The estimated rating for the given user/item pair.
        """
        user_known, item_known = False, False
        pred = self.global_rating_mean_

        if mode == "rating":
            if u_id in self.user_mapping_:
                user_known = True
                u_ix = self.user_mapping_[u_id]
                u_rating_mean = self.user_rating_means_mapping_[u_ix]
                pred += self.bu_k1_[u_ix] * (u_rating_mean - self.global_rating_mean_) + self.bu_c_[u_ix]

            if i_id in self.item_mapping_:
                item_known = True
                i_ix = self.item_mapping_[i_id]
                i_rating_mean = self.item_rating_means_mapping_[i_ix]
                pred += self.bi_k1_[i_ix] * (i_rating_mean - self.global_rating_mean_) + self.bi_c_[i_ix]
        else:
            if u_id in self.user_mapping_:
                user_known = True
                u_ix = self.user_mapping_[u_id]
                u_rating_mean = self.user_rating_means_mapping_[u_ix]
                u_nlp_mean = self.user_nlp_means_mapping_[u_ix]
                pred += self.bu_k1_[u_ix] * (u_rating_mean - self.global_rating_mean_) + self.bu_k2_[u_ix] * (u_nlp_mean - self.global_nlp_mean_) + self.bu_c_[u_ix]

            if i_id in self.item_mapping_:
                item_known = True
                i_ix = self.item_mapping_[i_id]
                i_rating_mean = self.item_rating_means_mapping_[i_ix]
                i_nlp_mean = self.item_nlp_means_mapping_[i_ix]
                pred += self.bi_k1_[i_ix] * (i_rating_mean - self.global_rating_mean_) + self.bi_k2_[i_ix] * (i_nlp_mean - self.global_nlp_mean_) + self.bi_c_[i_ix]

        if user_known and item_known:
            pred += np.dot(self.pu_[u_ix], self.qi_[i_ix])

        if clip:
            pred = self.max_rating if pred > self.max_rating else pred
            pred = self.min_rating if pred < self.min_rating else pred

        return pred

    def _early_stopping(self, epoch_idx, min_delta, train_rmse=None, val_rmse=None):
        """Returns True if validation rmse is not improving.

        Last rmse (plus `min_delta`) is compared with the second to last.

        Parameters
        ----------
        val_rmse : list
            Validation RMSEs.
        min_delta : float
            Minimun delta to argue for an improvement.

        Returns
        -------
        early_stopping : bool
            Whether to stop training or not.
        """
        if epoch_idx > 0:
            if val_rmse is not None:
                if val_rmse[epoch_idx] + min_delta > val_rmse[epoch_idx - 1]:
                    self.metrics_ = self.metrics_.loc[:(epoch_idx + 1), :]
                    return True
            else:
                if train_rmse[epoch_idx] + min_delta > train_rmse[epoch_idx - 1]:
                    self.metrics_ = self.metrics_.loc[:(epoch_idx + 1), :]
                    return True
        return False
    
    def _on_epoch_begin(self, epoch_ix):
        """Displays epoch starting log and returns its starting time.

        Parameters
        ----------
        epoch_ix : int
            Epoch index.

        Returns
        -------
        start : float
            Starting time of the current epoch.
        """
        start = time.time()
        end = '  | ' if epoch_ix < 9 else ' | '
        print('Epoch {}/{}'.format(epoch_ix + 1, self.n_epochs), end=end)

        return start
    
    def _on_epoch_end(self, start, train_loss=None, train_rmse=None, train_mae=None, val_loss=None, val_rmse=None, val_mae=None):
        """Displays epoch ending log.

        If self.verbose, computes and displays validation metrics (loss, rmse,
        and mae).

        Parameters
        ----------
        start : float
            Starting time of the current epoch.
        val_loss : float, default=None
            Validation loss.
        val_rmse : float, default=None
            Validation rmse.
        val_mae : float, default=None
            Validation mae.
        """
        end = time.time()

        if train_loss is not None:
            print(f'train_loss: {train_loss:.3f}', end=' - ')
            print(f'train_rmse: {train_rmse:.3f}', end=' - ')
            print(f'train_mae: {train_mae:.3f}', end=' - ')

        if val_loss is not None:
            print(f'val_loss: {val_loss:.3f}', end=' - ')
            print(f'val_rmse: {val_rmse:.3f}', end=' - ')
            print(f'val_mae: {val_mae:.3f}', end=' - ')
        

        print(f'took {end - start:.1f} sec')