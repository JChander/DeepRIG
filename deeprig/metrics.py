import tensorflow.compat.v1 as tf

def masked_accuracy_mse(preds, labels, mask, negative_mask):
    """Accuracy with masking."""
    preds = tf.cast(preds, tf.float32)
    labels = tf.cast(labels, tf.float32)
    error = tf.square(preds - labels)
    mask += negative_mask
    mask = tf.cast(mask, dtype=tf.float32)
    error *= mask
    return tf.sqrt(tf.reduce_mean(error))

def masked_loss(preds, labels, mask, negative_mask):
    """Accuracy with masking."""
    preds = tf.cast(preds, tf.float32)
    labels = tf.cast(labels, tf.float32)
    mask += negative_mask
    mask = tf.cast(mask, dtype=tf.float32)
    preds *= mask
    labels *= mask
    bce = tf.keras.losses.BinaryCrossentropy(from_logits = True)
    loss = bce(labels, preds)
    return loss

def masked_accuracy(preds, labels, mask, negative_mask):
    """Accuracy with masking."""
    preds = tf.cast(preds, tf.float32)
    labels = tf.cast(labels, tf.float32)
    mask += negative_mask
    mask = tf.cast(mask, dtype=tf.float32)
    preds *= mask
    labels *= mask
    correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds), 0.5), tf.int32),
                                           tf.cast(labels, tf.int32))
    return tf.cast(correct_prediction, tf.float32)


def euclidean_loss(preds, labels):
    euclidean_loss = tf.sqrt(tf.reduce_sum(tf.square(preds - labels), 0))
    return euclidean_loss
