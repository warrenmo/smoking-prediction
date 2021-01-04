import numpy as np
import pandas as pd
from multiprocessing import Pool

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from sklearn.manifold import TSNE

from os import listdir
from collections import Counter

from config import Y_TRAIN_PATH, TSFM_TRAIN_DIR, MASK_DIR


__all__ = [
    'read_csv_custom', 'read_csv_mp', 'plot_class_freq',
    'oversample', 'undersample', 'undersample_split', 
    'merge_classes', 'tsne_plot', 'corrcoef', 'two_mat_corrcoef',
    'corr_filter_basic', 'corr_filter', 'mask_path', 'save_mask',
    'CorrCoefChunk',
    ]


def _asarray(X):
    return X if type(X) is np.ndarray else np.asarray(X)


def read_csv_custom(path, *, asarray, **kwargs):
    X = pd.read_csv(path,
                    header=None,
                    engine='c',
                    delimiter=',',
                    na_filter=False,
                    low_memory=False,
                    dtype=np.float32,
                    **kwargs
                   )
    return np.asarray(X) if asarray else X


def _read_csv_custom_df(path):
    return read_csv_custom(path, asarray=False)


def read_csv_mp(split_subdir, *, asarray, axis=1, processes=None):
    path_list = sorted(split_subdir/csv for csv in listdir(split_subdir))
    with Pool(processes=processes) as pool:
        df_list = pool.map(_read_csv_custom_df, path_list)
    X = pd.concat(df_list, axis=axis, ignore_index=True)
    return np.asarray(X) if asarray else X


def plot_class_freq(y, ticklabels=('Non-smoker', 'Smoker', 'Former Smoker')):
    if type(y) is pd.core.series.Series:
        y = pd.DataFrame(y)

    counts = Counter(y[0])
    assert len(counts) == len(ticklabels), \
           "the length of ticklabels kwarg must match number of unique values in y"

    total = sum(counts.values())

    sns.set(style='whitegrid')
    ax = sns.countplot(x=0, data=y)
    plt.xlabel('')
    ax.set_xticklabels(ticklabels)
    plt.ylabel('Frequency')
    plt.title('Class Frequencies')
    for i in range(len(ticklabels)):
        plt.text(x=i, y=2,
                 s=f"{counts[i]} ({counts[i]/total:.1%})", 
                 color='white', horizontalalignment='center')


def _join_data_targets(X, y=None):
    if y is not None:
        data = pd.concat([X, y], axis=1)
    else:
        data = X
    return data


def oversample(X, y=None, shuffle=False, *, random_state):
    # TODO: convert to numpy array
    data = _join_data_targets(X, y=y)
    max_group_num = data.iloc[:,-1].value_counts().max()
    data_list = [data]
    for _, group in data.groupby(data.iloc[:,-1]):
        if len(group) == max_group_num:
            continue
        over = group.sample(max_group_num-len(group),
                            replace=True,
                            random_state=random_state)
        data_list.append(over)
    over = pd.concat(data_list)
    if shuffle:
        over = over.sample(frac=1, random_state=random_state)
    return over.iloc[:,:-1], over.iloc[:,-1]


def undersample(X, y=None, *, random_state):
    # TODO: convert to numpy array
    data = _join_data_targets(X, y=y)
    min_group_num = data.iloc[:,-1].value_counts().min()
    data_list = []
    for _, group in data.groupby(data.iloc[:,-1]):
        if len(group) == min_group_num:
            data_list.append(group)
        else:
            sample = group.sample(n=min_group_num,
                                  random_state=random_state)
            data_list.append(sample)
    under = pd.concat(data_list)
    return under.iloc[:,:-1], under.iloc[:,-1]


def _combine_min_maj(min_groups, majority_splits, *, num_splits):
    """
    NOTE: this function "wastes" memory by copying the minority
    group(s) k times where k=num_splits, but allows for more
    easily understood output
    """
    splits = []
    for split_id in range(num_splits):
        under = pd.concat([split[split_id] for split in majority_splits] + min_groups)
        X_under, y_under = under.iloc[:,:-1], under.iloc[:,-1]
        splits.append((X_under, y_under))
    return splits


def undersample_split(X, y=None, *, num_splits, random_state):
    # TODO: convert to numpy array
    data = _join_data_targets(X, y=y)
    min_group_num = data.iloc[:,-1].value_counts().min()
    min_groups = []
    majority_splits = []
    for _, group in data.groupby(data.iloc[:,-1]):
        if len(group) == min_group_num:
            min_groups.append(group)
        else:
            under = np.array_split(group.sample(frac=1,
                                                random_state=random_state),
                                   num_splits)
            majority_splits.append(under)

    splits = _combine_min_maj(min_groups, majority_splits, 
                              num_splits=num_splits)
    return splits


def merge_classes(y, classes, col=None):
    assert type(y) is pd.core.frame.DataFrame or col is not None, \
           "col kwarg cannot be None if y is a DataFrame"

    min_class = min(classes)
    replace_key = {c:min_class for c in classes if c != min_class}
    if col is None:
        ans = y.replace(replace_key)
    else:
        ans = pd.DataFrame(y[col].replace(replace_key))
    return ans


def tsne_plot(transform_name, y_train_path=Y_TRAIN_PATH, n_components=2, perplexities=[5, 30, 50]):
    # https://scikit-learn.org/stable/auto_examples/manifold/plot_t_sne_perplexity.html
    tsfm_path = lambda x: TSFM_TRAIN_DIR/f'{transform_name}.csv'

    X = np.loadtxt(tsfm_path(transform_name), delimiter=',')
    (fig, subplots) = plt.subplots(1, len(perplexities), figsize=(15, 4))

    y = np.loadtxt(y_train_path, delimiter=',')

    red = y == 0
    blue = y == 1
    green = y == 2

    for i, perplexity in enumerate(perplexities):
            tsne = TSNE(n_components=n_components, init='random',
                        random_state=0, perplexity=perplexity)
            X_tsne = tsne.fit_transform(X)
                   
            ax = subplots[i]
            ax.set_title(f"Perplexity={perplexity} ({transform_name})")
            ax.scatter(X_tsne[red,0], X_tsne[red,1], c="r")
            ax.scatter(X_tsne[blue,0], X_tsne[blue,1], c="b")
            ax.scatter(X_tsne[green,0], X_tsne[green,1], c="g")
            ax.xaxis.set_major_formatter(NullFormatter())
            ax.yaxis.set_major_formatter(NullFormatter())
            ax.axis('tight')


def corrcoef(m, return_moments=False):
    A = _asarray(m)
    ave = np.average(A, axis=0)[None,:]
    A -= ave
    c = np.dot(A.T, A)        
    stdev = np.sqrt(np.diag(c))
    corr = c / stdev[:,None] / stdev[None,:]
    if return_moments:
        return corr, ave, stdev
    return corr


def two_mat_corrcoef(m1, m2):
    A = _asarray(m1)
    B = _asarray(m2)

    A -= np.average(A, axis=0)
    B -= np.average(B, axis=0)

    c = np.dot(B.T, A)
 
    ssq_A = np.sum(np.square(A), axis=0)
    ssq_B = np.sum(np.square(B), axis=0)
    norm = np.sqrt(np.outer(ssq_B, ssq_A))    
    return c / norm


def corr_filter_basic(X, *, threshold):
    corr = np.abs(corrcoef(X))
    mask = np.all(np.triu(np.abs(corr), k=1) < threshold, axis=0)
    return mask

def corr_filter(X, *, threshold):
    corr = np.abs(corrcoef(X))
    avg_corr = np.average(corr, axis=0)
    upper = np.triu(corr, k=1)
    indices = np.argwhere(upper > threshold)
    maybe_drop = np.argmax(np.vectorize(avg_corr.get)(indices), axis=1) 



def mask_path(directory, prefix, threshold, random_state=None):
    assert ('over' not in prefix and 'under' not in prefix) or random_state is not None, \
           "random_state kwarg cannot be None with oversampled or undersampled data"
    parent = MASK_DIR / directory
    fname = f'{prefix}_'
    if 'over' in prefix or 'under' in prefix:
        fname += f'{random_state}_'
    fname += f'{int(threshold*1000):03d}.csv'
    return parent / fname


def save_mask(mask, *args, **kwargs):
    path = mask_path(*args, **kwargs)
    np.savetxt(path, mask, delimiter=',', fmt="%d")


def _ceil_int_div(dividend, divisor):
    return ((dividend - 1) // divisor) + 1


class CorrCoefChunk:
    def __init__(self, X, num_splits=None, chunk_size=None):
        assert num_splits is not chunk_size, \
               "EXACTLY one of num_splits or chunk_size kwarg must be specified"

        self.X = _asarray(X)

        if chunk_size is not None:
            self.num_splits = _ceil_int_div(X.shape[1], chunk_size)
        else:
            self.num_splits = num_splits
        if num_splits is not None:
            self.chunk_size = _ceil_int_div(X.shape[1], num_splits)
        else:
            self.chunk_size = chunk_size

        self.corrs = []
        self.two_mat_corrs = {}
        self.moments = []

    def _calc_single(self, X):
        corr, ave, stdev = corrcoef(X, return_moments=True)
        self.corrs.append(corr)
        self.moments.append((ave, stdev))

    def _corrs_moments(self):
        self._calc_single(X[:,:self.chunk_size])
        
        for i in range(0, self.X.shape[1], self.chunk_size):
            Xi = self.X[:,i:i+self.chunk_size]
            for j in range(i+self.chunk_size, self.X.shape[1], self.chunk_size):
                Xj = self.X[:,j:j+self.chunk_size]
                self._calc_single(Xj)
                corr = two_mat_corrcoef(Xi, Xj)
                self.two_mat_corrs[(i,j)] = corr

    def get_mask(self, threshold):
        pass

        

    #@staticmethod
    #def _precompute(self):
    #    lookup = []
    #    for i in range(0, self.X.shape[1], self.chunk_size):
    #        diff = self.X.iloc[:,i:i+self.chunk_size] \
    #               - np.average(self.X.iloc[:,i:i+self.chunk_size], axis=0)
    #        ssq = np.sum(np.square(diff), axis=0)
    #    lookup.append((diff, ssq))

    #def cross(self, i, j):
    #    diff_A, ssq_A = self.lookup[i]
    #    diff_B, ssq_B = self.lookup[j]
    #    return np.dot(diff_B.T, diff_A) / np.sqrt(np.dot(ssq_B.T, ssq_A))
