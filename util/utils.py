#from pylab import *
import random
import pandas as pd
from deeprig.inits import *

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(adj, train_arr, test_arr, labels, AM):
    n_gene = AM.shape[0]

    logits_train = sp.csr_matrix((labels[train_arr, 2], (labels[train_arr, 1], labels[train_arr, 1])), shape = (n_gene, n_gene)).toarray()
    logits_train = logits_train.reshape([-1, 1])
    
    logits_test = sp.csr_matrix((labels[test_arr, 2], (labels[test_arr, 0], labels[test_arr, 1])), shape = (n_gene, n_gene)).toarray()
    logits_test = logits_test.reshape([-1, 1])

    adj = preprocess_adj(adj)
    labels = abs(labels)

    train_mask = np.array(logits_train[:, 0], dtype=np.bool).reshape([-1, 1])
    test_mask = np.array(logits_test[:, 0], dtype=np.bool).reshape([-1, 1])


    return adj, n_gene, logits_train, logits_test, train_mask, test_mask, labels


def generate_mask(labels, ratio, N, n_gene):
    num = 0
    A = sp.csr_matrix((labels[:, 2], (labels[:, 0], labels[:, 1])), shape=(n_gene, n_gene)).toarray()
    mask = np.zeros((n_gene, n_gene))
    label_neg = np.zeros((ratio * N, 2))
    while (num < ratio * N):
        a = random.randint(0, n_gene - 1)
        b = random.randint(0, n_gene - 1)
        if A[a, b] != 1 and mask[a, b] != 1:
            mask[a, b] = 1
            label_neg[num, 0] = a
            label_neg[num, 1] = b
            num += 1
    mask = np.reshape(mask, [-1, 1])
    return mask, label_neg


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj.toarray()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj) + np.eye(adj.shape[0])
    return adj_normalized


def construct_feed_dict(adj, features, labels, labels_mask, negative_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['adjacency_matrix']: adj})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['negative_mask']: negative_mask})
    return feed_dict


def div_list(ls, n):
    ls_len = len(ls)
    j = ls_len // n
    ls_return = []
    for i in range(0, (n - 1) * j, j):
        ls_return.append(ls[i:i + j])
    ls_return.append(ls[(n - 1) * j:])
    return ls_return


def ROC(outs, labels, train_mask, test_mask, label_neg, gene_names, result_path):
    # scores = []
    results = pd.DataFrame(columns = ['Gene1', 'Gene2', 'Label', 'EdgeWeight'])
    
    train_mask = train_mask[:, 0].reshape(outs.shape)
    test_mask = test_mask[:, 0].reshape(outs.shape)
    
    for i in range(outs.shape[0]):
        for j in range(outs.shape[1]):
            if train_mask[i, j] == 1:
                continue
            else:
                if test_mask[i, j] == 1:
                    label_true = 1
                else:
                    label_true = 0
                new_df = pd.DataFrame({'Gene1': gene_names[i],
                                       'Gene2': gene_names[j],
                                       'Label': label_true,
                                       'EdgeWeight': outs[i, j]}, index=[1])
                results = results.append(new_df, ignore_index=True)

    results = results.sort_values(by = ['EdgeWeight'], axis = 0, ascending = False)
    results.to_csv(result_path, header=True, index=False)
