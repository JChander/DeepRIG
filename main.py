import tensorflow.compat.v1 as tf
import numpy as np
import pandas as pd
from train import *
from util.utils import div_list
import time

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model', 'DeepRIG', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_string('input_path', './Datasets/500_ChIP-seq_mESC/', 'Input data path')
flags.DEFINE_string('output_path', './output/', 'Output data path')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('cv', 3, 'Folds for cross validation.')
flags.DEFINE_integer('epochs', 500, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 200, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.7, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 1e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_integer('ratio', 1, 'Ratio of negetive samples to positive samples.')
flags.DEFINE_integer('dim', 300, 'The size of latent factor vector.')

def computCorr(data, t = 0.0):

    genes = data.columns
    corr = data.corr(method = "spearman")

    adj = np.array(corr.values)
    return adj

def prepareData(FLAGS, data_path, label_path, reverse_flags = 0):
    # Reading data from disk
    label_file = pd.read_csv(label_path, header=0, sep = ',')
    data = pd.read_csv(data_path, header=0, index_col = 0).T                 ###transpose for six datasets of BEELINE
    print("Read data completed! Normalize data now!")
    data = data.transform(lambda x: np.log(x + 1))
    print("Data normalized and logged!")

    TF = set(label_file['Gene1'])
    # Adjacency matrix transformation
    labels = []
    if reverse_flags == 0:
        var_names = list(data.columns)
        num_genes = len(var_names)
        AM = np.zeros([num_genes, num_genes])
        for row_index, row in label_file.iterrows():
            AM[var_names.index(row[0]), var_names.index(row[1])] = 1
            label_triplet = []
            label_triplet.append(var_names.index(row[0]))
            label_triplet.append(var_names.index(row[1]))
            label_triplet.append(1)
            labels.append(label_triplet)

    labels = np.array(labels)
    print("Start to compute correlations between genes!")
    adj = computCorr(data)
    node_feat = data.T.values
    return labels, adj, AM, var_names, TF, node_feat


# Preparing data for training
input_path = FLAGS.input_path
output_path = FLAGS.output_path
dataset = input_path.split('/')[-2]
data_file = input_path + dataset + '-ExpressionData.csv'
label_file = input_path + dataset + '-network.csv'

reverse_flags = 0   ###whether label file exists reverse regulations, 0 for DeepSem data, 1 for CNNC data
labels, adj, AM, gene_names, TF = prepareData(FLAGS, data_file, label_file, reverse_flags)
reorder = np.arange(labels.shape[0])
np.random.shuffle(reorder)

T = 1  # Number of training rounds
cv_num = FLAGS.cv  # k-flod Cross-validation (CV)
start = time.time()
for t in range(T):
    order = div_list(reorder.tolist(), cv_num)
    pred_results = []
    for i in range(cv_num):
        print("T:", '%01d' % (t))
        print("cross_validation:", '%01d' % (i))
        result_path_cv = output_path + '/pred_result_' + dataset + '_CV' + str(cv_num) + '_' + str(i) + '.csv'
        test_arr = order[i]
        arr = list(set(reorder).difference(set(test_arr)))
        np.random.shuffle(arr)
        train_arr = arr
        pred_matrix = train(FLAGS, adj, node_feat, train_arr, test_arr, labels, AM, gene_names, TF, result_path_cv)
        pred_results.append(pred_matrix)

    output = pred_results[0]
    for i in range(1, cv_num):
    	output = pd.concat([output, pred_results[i]])
    output['EdgeWeight'] = abs(output['EdgeWeight'])

    result_path = output_path + '/Inferred_result_' + dataset + '.csv'
    output.to_csv(result_path, header=True, index=False)
        
print("Predict complete!")
print("RunTimes is:", "{:.5f}".format(time.time() - start))


