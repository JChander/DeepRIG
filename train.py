from __future__ import division
from __future__ import print_function
import numpy as np
import time
import tensorflow.compat.v1 as tf
from util.utils import *
from deeprig.models import DeepRIG


def train(FLAGS, adj, features, train_arr, test_arr, labels, AM, gene_names, TF, result_path):
    # Load data
    adj, size_gene, logits_train, logits_test, train_mask, test_mask, labels= load_data(
        adj, train_arr, test_arr, labels, AM)
    
    # Some preprocessing
    if FLAGS.model == 'DeepRIG':
        model_func = DeepRIG
    else:
        raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

    # Define placeholders
    tf.compat.v1.disable_eager_execution()
    placeholders = {
        'adjacency_matrix': tf.placeholder(tf.int32, shape=adj.shape),
        'features': tf.placeholder(tf.float32, shape= features.shape),
        'labels': tf.placeholder(tf.float32, shape=(None, logits_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'negative_mask': tf.placeholder(tf.int32)
    }
    
    input_dim = features.shape[1]
    # Create model
    model = model_func(placeholders, input_dim, size_gene, FLAGS.dim)

    # Initialize session
    sess = tf.Session()

    # Define model evaluation function
    def evaluate(adj, features, labels, mask, negative_mask, placeholders):
        t_test = time.time()
        feed_dict_val = construct_feed_dict(adj, features, labels, mask, negative_mask, placeholders)
        outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
        return outs_val[0], 1 - outs_val[1], (time.time() - t_test)

    # Init variables
    sess.run(tf.global_variables_initializer())

    cost_val = []

    # Train model
    for epoch in range(FLAGS.epochs):
        t = time.time()
        # Construct feed dictionary
        negative_mask, label_neg = generate_mask(labels, FLAGS.ratio, len(train_arr), size_gene)

        feed_dict = construct_feed_dict(adj, features, logits_train, train_mask, negative_mask, placeholders)

        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

        # Print results
        print("Epoch:", '%04d' % (epoch + 1),
              "train_loss=", "{:.5f}".format(outs[1]),
              "train_acc=", "{:.5f}".format(1 - outs[2]),
              "time=", "{:.5f}".format(time.time() - t))

    print("Optimization Finished!")

    # # Save trained model
    # model.save(sess, './saved_models/')
    
    # Testing
    test_negative_mask, test_label_neg = generate_mask(labels, FLAGS.ratio, len(test_arr), size_gene)
    test_cost, test_acc, test_duration = evaluate(adj, features, logits_test, test_mask, test_negative_mask, placeholders)
    print("Test set results:", "cost=", "{:.5f}".format(test_cost),
          "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

    #Save results
    feed_dict_val = construct_feed_dict(adj, features, logits_test, test_mask, test_negative_mask, placeholders)
    outs = sess.run(model.outputs, feed_dict=feed_dict_val)
    outs = np.array(outs)[:, 0]
    outs = outs.reshape((size_gene, size_gene))

    # save the predicted matrix
    logits_train = logits_train.reshape(outs.shape)
    TF_mask = np.zeros(outs.shape)
    for i, item in enumerate(gene_names):
        for j in range(len(gene_names)):
            if i == j or (logits_train[i, j] == 1):
                continue
            if item in TF:
                TF_mask[i, j] = 1
    geneNames = np.array(gene_names)
    idx_rec, idx_send = np.where(TF_mask)
    results = pd.DataFrame(
        {'Gene1': geneNames[idx_rec], 'Gene2': geneNames[idx_send], 'EdgeWeight': (outs[idx_rec, idx_send])})
    results = results.sort_values(by = ['EdgeWeight'], axis = 0, ascending = False)
    return results
