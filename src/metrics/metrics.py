import tensorflow as tf
from keras import backend as K

def auc(y_true, y_pred, curve='PR'):
    score, up_opt = tf.compat.v1.metrics.auc(y_true, y_pred, curve=curve, summation_method="careful_interpolation")
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([up_opt]):
        score = tf.identity(score)
    return score

def auprc(y_true, y_pred):
    return auc(y_true, y_pred, curve='PR')

def auroc(y_true, y_pred):
    return auc(y_true, y_pred, curve='ROC')
