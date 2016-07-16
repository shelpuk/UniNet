import errorFunction as ef
import cPickle
import numpy as np

class neuralnetwork(object):
    def __init__(self, layers=[], errorFunction=ef.squreError(technology='numpy')):
        self.layers=layers
        self.errorFunction = errorFunction
        self.inputLayer = None
        self.outputLayer = None
        self.weightDecay = False

        if layers != []: self.initializeConnectivity()

    def initializeConnectivity(self):
        for i in range(len(self.layers)-1):
            self.layers[i].connect(self.layers[i+1])
            self.layers[i].isInput = False
            self.layers[i].isOutput = False

        for layer in self.layers:
            layer.network = self
            layer.initializeLayerSpecificFunctions(technology='numpy')

        self.layers[0].isInput=True
        self.inputLayer = self.layers[0]
        self.layers[-1].isOutput=True
        self.outputLayer = self.layers[-1]


    def addLayer(self, layer):
        if layer.isInput and self.inputLayer != None: raise Exception('The network already has an input layer')
        if layer.isOutput and self.outputLayer != None: raise Exception('The network already has an output layer')

        self.layers.append(layer)
        layer.network = self

        if layer.isInput: self.inputLayer = layer
        if layer.isOutput: self.outputLayer = layer

    def forwardPropagate(self, data):
        self.inputLayer.forwardPropagate(data)
        return self.outputLayer.activation

    def getSumSquareWeights(self):
        sumSquareWeights = 0
        for layer in self.layers:
            if layer.weightDecay is None: continue
            sumSquareWeights += layer.weightDecay / 2. * sum(sum(layer.weights ** 2))
        return sumSquareWeights

    def getErrorValue(self, data, labels):
        prediction = self.forwardPropagate(data)
        return self.errorFunction.getValue(prediction, labels)

    def getErrorGradient(self, data, labels):
        prediction = self.forwardPropagate(data)
        return self.errorFunction.getGradient(prediction, labels)

    def backPropagate(self, dEda):
        self.outputLayer.backPropagate(dEda)

    def gradientCheck(self, dataSet, verbose = 0, step=0.00001):
        data = dataSet.examples
        labels = dataSet.labels
        for layer in self.layers: layer.saveGradient = True
        self.forwardPropagate(data)
        dEda = self.errorFunction.getGradient(self.outputLayer.activation, labels)
        self.backPropagate(dEda)
        maxDifference = float('-inf')
        for layer in self.layers:
            if layer.isInput: continue
            if layer.gradient is None: raise Exception('Unable to do numerical gradient check: analytical gradients are empty')
            for i in range(len(layer.weights)):
                for j in range(len(layer.weights[i])):
                    oldWeight = layer.weights[i][j]
                    layer.weights[i][j] += step
                    stepForwardError = self.getErrorValue(data=data, labels=labels)
                    layer.weights[i][j] -= 2*step
                    stepBackwardError = self.getErrorValue(data=data, labels=labels)
                    numericalGradient = (stepForwardError-stepBackwardError)/(2.*step)
                    difference = abs(layer.gradient[i][j]-numericalGradient)
                    if verbose > 0:
                        print 'Gradient = ' + str(layer.gradient[i][j]) + ' | numerical = ' + str(numericalGradient)
                    if difference > maxDifference:
                        maxDifference = difference
                    layer.weights[i][j] = oldWeight
        print 'Maximum difference: '+str(maxDifference)
        for layer in self.layers: layer.saveGradient = False

    def predict(self, dataSet):
        self.inputLayer.predict(dataSet.examples)
        return self.outputLayer.activation

    def predictWithLabel(self, dataSet):
        predicted = self.predict(dataSet=dataSet)
        maxValue = np.max(predicted, axis=1).reshape(-1, 1)
        return (predicted == maxValue) * 1.

    def getAccuracy(self, dataSet):
        numExamples = dataSet.getNumExamples()
        predictedLabels = self.predictWithLabel(dataSet=dataSet)
        errorRate = np.true_divide(np.sum((predictedLabels != dataSet.labels) * 1), (self.outputLayer.neurons * numExamples))
        return 1. - errorRate


    def train(self, trainingSet, cvSet = None, cvCheckPeriod = None, numEpochs=100, minibatchSize=None, verbose = 0):
        trainingExamples = trainingSet.examples
        trainingLabels = trainingSet.labels
        for epoch in range(numEpochs):
            if cvCheckPeriod is not None:
                if epoch % cvCheckPeriod == 0:
                    cvError = self.getErrorValue(data=cvSet.examples, labels=cvSet.labels)
                    print 'Cross validation check error = ', cvError
            if minibatchSize is not None:
                data = trainingSet.getMiniBatch(size=minibatchSize)
                trainingExamples = data['examples']
                trainingLabels = data['labels']
            self.forwardPropagate(trainingExamples)
            error = self.getErrorValue(data=trainingExamples, labels=trainingLabels)
            dEda = self.errorFunction.getGradient(self.outputLayer.activation, trainingLabels)

            if verbose >= 4: print 'Iter = ' + str(epoch) + ', error = '+str(error)
            self.backPropagate(dEda)

    def saveNetwork(self, file):
        cPickle.dump(self, open(file, 'wb'))

    def loadNetwork(self, file):
        self = cPickle.load(open(file, 'rb'))

