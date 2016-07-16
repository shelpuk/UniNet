import numpy as np

class errorFunction(object):
    def __init__(self, technology, activationFunction = None):
        self.technology = technology
        self.activationFunction = activationFunction

class squreError(errorFunction):
    def getValue(self):
        def getValueNumpy(h, y):
            return 0.5 / len(h) * sum(sum((h - y) ** 2))
        def getValueTF(h, y):
            try:
                import tensorflow as tf
                return tf.reduce_mean(tf.square(h - y))
            except ImportError:
                raise ImportError(
                    'TensorFlow is not installed on your computer. Please use other technology for building your network or install tensorflow.')
        if self.technology=='numpy': return getValueNumpy
        if self.technology == 'tensorflow': return getValueTF

    def getGradient(self, h, y):
        return h - y


class logLoss(errorFunction):
    def getValue(self):
        def getValueTF(h, y):
            try:
                import tensorflow as tf
                return tf.reduce_mean(y * -tf.log(h) + (1 - y) * -tf.log(1 - h))
            except ImportError:
                raise ImportError(
                    'TensorFlow is not installed on your computer. Please use other technology for building your network or install tensorflow.')
        def getSoftmaxCrossEntropyWithLogitsTF(z, y):
            try:
                import tensorflow as tf
                return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(z, y))
            except ImportError:
                raise ImportError(
                    'TensorFlow is not installed on your computer. Please use other technology for building your network or install tensorflow.')
        if self.technology == 'tensorflow' and self.activationFunction == 'softmax': return getSoftmaxCrossEntropyWithLogitsTF
        if self.technology == 'tensorflow': return getValueTF


    def getGradient(self, h, y):
        return h - y




class weightedSquare(object):
    def __init__(self):
        self.weights = np.array([8.49543281, 8.14708193, 1., 8.47547075, 12.08617941])
    def getValue(self, h, y):
        coefficient = np.sum(self.weights*y, axis=1)
        return 0.5 / len(h) * sum(sum((h - y) ** 2*np.array([coefficient]).T))

    def getGradient(self, h, y):
        coefficient = np.sum(self.weights*y, axis=1)
        return (h - y)*np.array([coefficient]).T
