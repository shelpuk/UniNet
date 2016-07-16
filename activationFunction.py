import numpy as np

class activationFunction(object):
    def __init__(self, technology='numpy'):
        self.technology = technology

class linear(activationFunction):
    def getActivation(self):
        def getActivationTF(prevLayerActivation, weights, biases):
            try:
                import tensorflow as tf
                return tf.matmul(prevLayerActivation, tf.transpose(weights)) + biases
            except ImportError:
                raise ImportError(
                    'TensorFlow is not installed on your computer. Please use other technology for building your network or install tensorflow.')
        if self.technology == 'tensorflow': return getActivationTF
        else:
            raise ImportError('Technology '+self.technology+' currently is not supported. Please check spelling or switch to another technology.')

    def getGradient(self):
        return None

class convolve2d(activationFunction):
    def getActivation(self, strides, padding='SAME'):
        def getActivationTF(prevLayerActivation, weights, biases, strides=strides, padding=padding, gpu=True):
            try:
                import tensorflow as tf
                return tf.nn.conv2d(prevLayerActivation, weights, strides=strides, padding=padding, use_cudnn_on_gpu=gpu) + biases
            except ImportError:
                raise ImportError(
                    'TensorFlow is not installed on your computer. Please use other technology for building your network or install tensorflow.')
        if self.technology == 'tensorflow': return getActivationTF
        else:
            raise ImportError('Technology '+self.technology+' currently is not supported. Please check spelling or switch to another technology.')

    def getGradient(self):
        return None

class logistic(activationFunction):
    def getActivation(self):
        def getActivationNumpy(z):
            return np.true_divide(1., (1 + np.exp(-z)))
        def getActivationTF(z):
            try:
                import tensorflow as tf
                return tf.nn.sigmoid(z)
            except ImportError:
                raise ImportError(
                    'TensorFlow is not installed on your computer. Please use other technology for building your network or install tensorflow.')

        if self.technology=='numpy': return getActivationNumpy
        if self.technology == 'tensorflow': return getActivationTF
        else:
            raise ImportError('Technology '+self.technology+' currently is not supported. Please check spelling or switch to another technology.')

    def getGradient(self):
        def getGradientNumpy(dEda, activation):
            return np.array(activation * (1 - activation) * dEda)
        if self.technology == 'numpy': return getGradientNumpy
        if self.technology == 'tensorflow': return None
        else:
            raise ImportError('There is no gradient function for '+self.technology+'. You can add it by yourself to activationFunction.py or change another technology.')


class softmax(activationFunction):
    def getActivation(self):
        def getActivationNumpy(z):
            exp_z = np.exp(z)
            sum_exp_z = np.sum(exp_z, axis=1)
            return np.true_divide(exp_z, np.array([sum_exp_z]).T)
        def getActivationTF(z):
            try:
                import tensorflow as tf
                return tf.nn.softmax(z)
            except ImportError:
                raise ImportError(
                    'TensorFlow is not installed on your computer. Please use other technology for building your network or install tensorflow.')

        if self.technology=='numpy': return getActivationNumpy
        if self.technology == 'tensorflow': return getActivationTF
        else:
            raise ImportError('Technology '+self.technology+' currently is not supported. Please check spelling or switch to another technology.')

    def getGradient(self):
        def getGradientNumpy(dEda, activation):
            return np.array(activation * (1 - activation) * dEda)
        if self.technology == 'numpy': return getGradientNumpy
        if self.technology == 'tensorflow': return None
        else:
            raise ImportError('There is no gradient function for '+self.technology+'. You can add it by yourself to activationFunction.py or change another technology.')


class relu(activationFunction):
    def getActivation(self):
        def getActivationNumpy(z):
            return (z>0)*z + (z<0)*0
        def getActivationTF(z):
            try:
                import tensorflow as tf
                return tf.nn.relu(z)
            except ImportError:
                raise ImportError(
                    'TensorFlow is not installed on your computer. Please use other technology for building your network or install tensorflow.')

        if self.technology=='numpy': return getActivationNumpy
        if self.technology == 'tensorflow': return getActivationTF
        else:
            raise ImportError('Technology '+self.technology+' currently is not supported. Please check spelling or switch to another technology.')

    def getGradient(self):
        def getGradientNumpy(dEda, activation):
            return dEda*(activation>0)+0*(activation<0)
        if self.technology == 'numpy': return getGradientNumpy
        if self.technology == 'tensorflow': return None
        else:
            raise ImportError('There is no gradient function for '+self.technology+'. You can add it by yourself to activationFunction.py or change another technology.')


class lrelu(activationFunction):
    def getActivation(self):
        def getActivationNumpy(z):
            return (z>0)*z + (z<0)*0.01*z
        def getActivationTF(z):
            try:
                import tensorflow as tf
                return tf.add(tf.mul(z>0,z), tf.mul(z<0,0.01*z))
            except ImportError:
                raise ImportError(
                    'TensorFlow is not installed on your computer. Please use other technology for building your network or install tensorflow.')

        if self.technology=='numpy': return getActivationNumpy
        if self.technology == 'tensorflow': return getActivationTF
        else:
            raise ImportError('Technology '+self.technology+' currently is not supported. Please check spelling or switch to another technology.')

    def getGradient(self):
        def getGradientNumpy(dEda, activation):
            return dEda*(activation>0)+0.01*dEda*(activation<0)
        if self.technology == 'numpy': return getGradientNumpy
        if self.technology == 'tensorflow': return None
        else:
            raise ImportError('There is no gradient function for '+self.technology+'. You can add it by yourself to activationFunction.py or change another technology.')


class elu(activationFunction):
    def getActivation(self):
        def getActivationTF(z):
            try:
                import tensorflow as tf
                return tf.nn.elu(z)
            except ImportError:
                raise ImportError(
                    'TensorFlow is not installed on your computer. Please use other technology for building your network or install tensorflow.')
        if self.technology == 'tensorflow': return getActivationTF
        else:
            raise ImportError('Technology '+self.technology+' currently is not supported. Please check spelling or switch to another technology.')

    def getGradient(self):
        return None