import numpy as np
import activationFunction as af
from scipy.stats import truncnorm

class layer(object):
    def __init__(self, neurons, dropProbability = None, \
                 weightDecay = None, dropout = False, isInput = False, isOutput = False, \
                 learningRate = 0.01, mmsLimit = 1e-10, rmsProp = None):
        self.neurons = neurons
        self.network = None
        self.weights = None
        self.isInput = isInput
        self.isOutput = isOutput
        self.activation = None
        self.dropProbability = dropProbability
        self.weightDecay = weightDecay
        self.dropout = dropout
        self.learningRate = learningRate

        #Technical constants
        self.epsilon=0.12
        self.mmsLimit = mmsLimit
        self.rmsProp = rmsProp

        #technical flags and parameters
        self.saveGradient = False
        self.gradient = None

        self.weights = None
        self.biases = None
        self.z = None

        self.saveZ = False


    def __selfValidate__(self):
        if self.network == None:
            raise Exception('The layer is not associated with the network')

    def __initializeLearningRate__(self, learningRate):
        self.learningRate = learningRate * np.ones(self.weights.shape)

    def __initializeWeights__(self, epsilon=0.12):
        if self.isInput == True: pass
        #self.weights = truncnorm.rvs(-0.1, 0.1, size=(self.neurons, self.prevLayer.neurons))
        thetas = np.random.uniform(0, 1, (self.neurons, self.prevLayer.neurons))
        self.weights = thetas * 2 * epsilon - epsilon
        self.biases = np.random.uniform(0, 1, (1, self.neurons))

    def __initializeMMS__(self):
        self.MovingMeanSquared = np.ones(self.weights.shape)

    def connect(self, nextLayer):
        self.nextLayer = nextLayer
        nextLayer.acceptConnection(self)

    def acceptConnection(self, prevLayer):
        self.prevLayer = prevLayer
        self.__initializeWeights__(self.epsilon)
        self.__initializeMMS__()
        self.__initializeLearningRate__(self.learningRate)

    def __forwardPropagationActivate__(self, prevLayerActivation):
        prevLayerActivation = np.insert(prevLayerActivation, 0, 1, axis=1)
        self.activation = self.__getActivation__(prevLayerActivation, self.weights)
        if self.dropout:
            self.activation = self.activation * np.random.binomial(1, self.dropProbability, self.activation.shape)

    def __predictionActivate__(self, prevLayerActivation):
        prevLayerActivation = np.insert(prevLayerActivation, 0, 1, axis=1)
        weights = self.weights
        if self.dropout:
            weights = weights * self.dropProbability
        self.activation = self.__getActivation__(prevLayerActivation, weights)

    #def forwardPropagate(self, prevLayerActivation):
    #    if self.isInput:
    #        self.activation=prevLayerActivation
    #    else:
    #        self.__forwardPropagationActivate__(prevLayerActivation)

        #for nextLayer in self.nextLayers:
        #    nextLayer.forwardPropagate(self.activation)
    #    if not self.isOutput:
    #        self.nextLayer.forwardPropagate(self.activation)

    def predict(self, prevLayerActivation):
        if self.isInput:
            self.activation=prevLayerActivation
        else:
            z = np.dot(prevLayerActivation, self.weights.T)
            self.activation = self.activationFunction(z)

        if not self.isOutput:
            self.nextLayer.predict(self.activation)


    def getPrevLayersActivations(self):
        activations = None
        #for prevLayer in self.prevLayers:
        #    if activations is None: activations = prevLayer.activation
        #    else: np.concatenate((activations, prevLayer.activation), axis = 1)
        if activations is None: activations = self.prevLayer.activation
        else: np.concatenate((activations, self.prevLayer.activation), axis = 1)

        activations = np.insert(activations, 0, 1, axis=1)
        return activations

    def getGradient(self, dEda):
        dEdz = self.getdEdz(dEda, self.activation)
        dEdW = 1. / len(dEda) * np.dot(dEdz.transpose(), self.getPrevLayersActivations())
        dEdaPrevLayer = np.dot(dEdz, self.weights)
        dEdaPrevLayer = np.delete(dEdaPrevLayer, 0, axis=1)

        if self.weightDecay is not None:
            dEdW += self.weightDecay * self.weights ** 2
        return [dEdW, dEdaPrevLayer]

    def backPropagate(self, dEda):
        if not self.isInput:
            [gradient, dEdaPrevLayers] = self.getGradient(dEda)
            if self.saveGradient: self.gradient = gradient
            if self.rmsProp is not None:
                self.updateMMS(gradient)
                stepLearningRate = np.true_divide(self.learningRate, np.sqrt(self.MovingMeanSquared))
            else:
                stepLearningRate = self.learningRate

            self.weights = self.weights - stepLearningRate*gradient
            #for layer in self.prevLayers:
            #    layer.backPropagate(dEdaPrevLayers[0:layer.neurons-1])
            #    dEdaPrevLayers = np.delete(dEdaPrevLayers, range(layer.neurons), axis = 0)
            self.prevLayer.backPropagate(dEdaPrevLayers)

    def updateMMS(self, newGradient):
        self.MovingMeanSquared = self.rmsProp * self.MovingMeanSquared + (1 - self.rmsProp) * (newGradient ** 2)
        self.MovingMeanSquared = self.MovingMeanSquared * (self.MovingMeanSquared > self.mmsLimit) + \
                                    self.mmsLimit * (self.MovingMeanSquared < self.mmsLimit)


class input(layer):
    def __init__(self, neurons=0, isInput=True, isOutput=False, input_size=[0,0,0]):
        self.neurons = neurons
        if input_size != [0,0,0]:
            self.output_size = input_size
            self.depth = input_size[2]
        self.network = None
        self.weights = None
        self.isInput = isInput
        self.isOutput = isOutput
        self.activation = None


class logistic(layer):
    def initializeLayerSpecificFunctions(self, technology):
        linearFunction = af.linear(technology=technology)
        self.linearFunction = linearFunction.getActivation()

        activationFunction = af.logistic(technology=technology)
        self.activationFunction = activationFunction.getActivation()

        self.getdEdz = activationFunction.getGradient()

class softmax(layer):
    def initializeLayerSpecificFunctions(self, technology):
        self.saveZ = True

        linearFunction = af.linear(technology=technology)
        self.linearFunction = linearFunction.getActivation()

        activationFunction = af.softmax(technology=technology)
        self.activationFunction = activationFunction.getActivation()
        self.getdEdz = activationFunction.getGradient()

class relu(layer):
    def initializeLayerSpecificFunctions(self, technology):

        linearFunction = af.linear(technology=technology)
        self.linearFunction = linearFunction.getActivation()

        activationFunction = af.relu(technology=technology)
        self.activationFunction = activationFunction.getActivation()
        self.getdEdz = activationFunction.getGradient()

class lrelu(layer):
    def initializeLayerSpecificFunctions(self, technology):

        linearFunction = af.linear(technology=technology)
        self.linearFunction = linearFunction.getActivation()

        activationFunction = af.relu(technology=technology)
        self.activationFunction = activationFunction.getActivation()
        self.getdEdz = activationFunction.getGradient()

class elu(layer):
    def initializeLayerSpecificFunctions(self, technology):

        linearFunction = af.linear(technology=technology)
        self.linearFunction = linearFunction.getActivation()

        activationFunction = af.relu(technology=technology)
        self.activationFunction = activationFunction.getActivation()
        self.getdEdz = activationFunction.getGradient()

class convolutional(layer):
    def __init__(self, patchSize, strides, depth, padding='same', activationFunction = af.relu, dropProbability=None, \
                 weightDecay=None, dropout=False, isInput=False, isOutput=False, \
                 learningRate=0.01, mmsLimit=1e-10, rmsProp=None):
        self.patchSize = patchSize
        self.depth = depth
        self.strides = strides
        self.network = None
        self.weights = None
        self.isInput = isInput
        self.isOutput = isOutput
        self.activation = None
        self.dropProbability = dropProbability
        self.weightDecay = weightDecay
        self.dropout = dropout
        self.learningRate = learningRate
        self.padding = padding.upper()
        self.af = activationFunction

        # Technical constants
        self.epsilon = 0.12
        self.mmsLimit = mmsLimit
        self.rmsProp = rmsProp

        # technical flags and parameters
        self.saveGradient = False
        self.gradient = None

        self.weights = None
        self.biases = None
        self.z = None

        self.saveZ = False

    def __initializeWeights__(self, epsilon=0.12):
        if self.isInput == True: pass
        self.weights = truncnorm.rvs(-0.1, 0.1, size=(self.patchSize, self.patchSize, self.prevLayer.depth, self.depth))
        #thetas = np.random.uniform(0, 1, (self.patchSize, self.patchSize, self.prevLayer.depth, self.depth))
        #self.weights = thetas * 2 * epsilon - epsilon
        self.biases = np.random.uniform(0, 1, (1, self.depth))

        if self.padding == 'SAME':
            self.output_size = self.prevLayer.output_size
            self.output_size[2] = self.depth

        self.neurons = self.output_size[0]*self.output_size[1]*self.output_size[2]


    def initializeLayerSpecificFunctions(self, technology):
        linearFunction = af.convolve2d(technology=technology)
        self.linearFunction = linearFunction.getActivation(strides=self.strides, padding=self.padding)

        activationFunction = self.af(technology=technology)
        self.activationFunction = activationFunction.getActivation()

        self.getdEdz = activationFunction.getGradient()

