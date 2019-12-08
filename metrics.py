import tensorflow as tf
import constants as c

def bins_mean_squared_error(y_true, y_pred):
    yt = tf.math.scalar_mul(10 ** (-c.BIN_POWER), tf.dtypes.cast(tf.math.argmax(y_true, axis=-1), dtype = tf.float32))
    yp = tf.math.scalar_mul(10 ** (-c.BIN_POWER), tf.dtypes.cast(tf.math.argmax(y_pred, axis=-1), dtype = tf.float32))
    return tf.keras.losses.MeanSquaredError()(yt, yp)