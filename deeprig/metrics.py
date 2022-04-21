import tensorflow.compat.v1 as tf

def masked_accuracy(preds, labels, mask, negative_mask):
    """Accuracy with masking."""
    preds = tf.cast(preds, tf.float32)
    labels = tf.cast(labels, tf.float32)
    error = tf.square(preds - labels)
    mask += negative_mask
    mask = tf.cast(mask, dtype=tf.float32)
    error *= mask
    return tf.sqrt(tf.reduce_mean(error))


def euclidean_loss(preds, labels):
    euclidean_loss = tf.sqrt(tf.reduce_sum(tf.square(preds - labels), 0))
    return euclidean_loss
