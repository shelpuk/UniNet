import tensorflow as tf
import numpy as np
import errorFunction as ef
import matplotlib.pyplot as plt

class neuralnetwork(object):
    def __init__(self, layers=[], errorFunction=ef.squreError):
        self.layers=layers
        if self.layers != []: self.__initializeConnectivity__()
        errorFunctionObject = errorFunction(technology='tensorflow', activationFunction=self.outputLayer.__class__.__name__)
        self.errorFunction = errorFunctionObject.getValue()

    def __initializeConnectivity__(self):
        for i in range(len(self.layers) - 1):
            self.layers[i].connect(self.layers[i + 1])
            self.layers[i].isInput = False
            self.layers[i].isOutput = False

        self.layers[0].isInput = True
        self.inputLayer = self.layers[0]
        self.layers[-1].isOutput = True
        self.outputLayer = self.layers[-1]

        for layer in self.layers:
            layer.network = self
            if layer.isInput == True: continue
            if layer.__class__.__name__ == 'convolutional': self.convolutional = True
            layer.initializeLayerSpecificFunctions(technology='tensorflow')


    def __initializeTrainGraph__(self, datasetShape, labelShape, cvCheckPeriod=None, rmsProp=None, mmsLimit=1*10**-10, learningRate=0.01, weightDecay=None):
        self.activateGraph = tf.Graph()

        with self.activateGraph.as_default():
            self.tfTrainSet = tf.placeholder(tf.float32, shape=datasetShape)
            self.tfTrainLabels = tf.placeholder(tf.float32, shape=labelShape)
            if cvCheckPeriod != None:
                tfCVSetShape = datasetShape
                tfCVLabelShape = labelShape
                tfCVLabelShape[0] = None
                self.tfCVSet = tf.placeholder(tf.float32, shape=tfCVSetShape)
                self.tfCVLabels = tf.placeholder(tf.float32, shape=tfCVLabelShape)

            for layer in self.layers:
                if layer.isInput==True: continue
                layer.weights = tf.Variable(tf.convert_to_tensor(layer.weights, dtype=tf.float32))
                layer.biases = tf.Variable(tf.convert_to_tensor(layer.biases, dtype=tf.float32))

            def activate(data):
                for layer in self.layers:
                    if layer.isInput==True:
                        layer.activation = data
                        continue

                    #print layer.prevLayer.activation.get_shape()
                    #print layer.weights.get_shape()

                    z = layer.linearFunction(layer.prevLayer.activation, layer.weights, layer.biases)
                    if layer.saveZ: layer.z = z
                    layer.activation = layer.activationFunction(z)
                    if (layer.__class__.__name__ == 'convolutional' or layer.__class__.__name__ == 'pooling') and layer.nextLayer.__class__.__name__ != 'convolutional' and layer.nextLayer.__class__.__name__ != 'pooling':
                        shape = layer.activation.get_shape().as_list()
                        #print layer.activation
                        layer.activation = tf.reshape(layer.activation, [shape[0], shape[1] * shape[2] * shape[3]])

            activate(self.tfTrainSet)

            self.activation = self.outputLayer.activation

            if self.outputLayer.__class__.__name__ == 'softmax': self.loss = self.errorFunction(self.outputLayer.z, self.tfTrainLabels)
            else: self.loss = self.errorFunction(self.activation, self.tfTrainLabels)

            if rmsProp != None: self.optimizer = tf.train.RMSPropOptimizer(learning_rate=learningRate, decay=rmsProp, momentum=0.0, epsilon=mmsLimit)
            else: self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningRate)

            self.optimizer = self.optimizer.minimize(self.loss)

    def getAccuracy(self, predictions, labels):
        return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
                / predictions.shape[0])

    def __restoreLayerVariablesToNumpy__(self, session):
        for layer in self.layers:
            if layer.isInput == True: continue
            layer.weights = session.run(layer.weights)
            layer.biases = session.run(layer.biases)

    def train(self, trainingSet, numEpochs,
              minibatchSize=None,
              cvSet = None,
              rmsProp=None,
              mmsLimit=1e-10,
              learningRate=0.01,
              weightDecay=None,
              errorCheckPeriod=None,
              cvAccuracyCheck = False,
              visualization = False):
        if visualization == True:
            plt.ion()
            plt.show()
            plt.axis([0, numEpochs, 0.0, 1.0])
            ax = plt.gca()
            ax.set_autoscale_on(False)
            iterLog = []
            minibatchTrainErrorLog = []
            fullTrainErrorLog = []
            cvErrorLog = []
            trainAccuracy = []
            cvAccuracy = []

        if self.convolutional:
            trainingSet.rearrangeToCubic()
            if cvSet != None: cvSet.rearrangeToCubic()

        batchExamplesShape = list(trainingSet.examples.shape)
        batchLabelsShape = list(trainingSet.labels.shape)
        if minibatchSize != None:
            batchExamplesShape[0] = minibatchSize
            batchLabelsShape[0] = minibatchSize
        self.__initializeTrainGraph__(datasetShape=batchExamplesShape, labelShape=batchLabelsShape)
        with tf.Session(graph=self.activateGraph) as session:
            tf.initialize_all_variables().run()
            for iter in range(numEpochs):
                if minibatchSize is not None:
                    data = trainingSet.getMiniBatch(size=minibatchSize)
                    trainingExamples = data['examples']
                    trainingLabels = data['labels']
                else:
                    trainingExamples = trainingSet.examples
                    trainingLabels = trainingSet.labels

                feed_dict = {self.tfTrainSet: trainingExamples, self.tfTrainLabels: trainingLabels}
                _, l = session.run([self.optimizer, self.loss], feed_dict=feed_dict)
                if errorCheckPeriod != None:
                    if iter % errorCheckPeriod == 0:
                        print 'Iter: ', iter, ', error = ', l

                # Everything below is cross-validation check and visualization
                if errorCheckPeriod!= None and iter % errorCheckPeriod == 0 and cvSet != None:
                    cv_feed_dict = {self.tfTrainSet: cvSet.examples, self.tfTrainLabels: cvSet.labels}
                    train_feed_dict = {self.tfTrainSet: trainingSet.examples, self.tfTrainLabels: trainingSet.labels}
                    if cvAccuracyCheck:
                        cvLoss, cvPrediction = session.run([self.loss, self.activation], feed_dict=cv_feed_dict)
                        trainLoss, trainPrediction = session.run([self.loss, self.activation], feed_dict=train_feed_dict)
                    else:
                        cvLoss = session.run(self.loss, feed_dict=cv_feed_dict)
                        trainLoss = session.run(self.loss, feed_dict=train_feed_dict)
                    print 'Iter: ', iter
                    print 'Cross-valodation check: train error = ', trainLoss, ', CV error = ', cvLoss
                    if cvAccuracyCheck: print 'Train accuracy = ', self.getAccuracy(trainPrediction, trainingSet.labels), '%, CV accuracy = ', self.getAccuracy(cvPrediction, cvSet.labels), '%'

                    if visualization == True:
                        minibatchTrainErrorLog.append(l)
                        fullTrainErrorLog.append(trainLoss)

                        if cvAccuracyCheck:
                            trainAccuracy.append(trainAccuracy)
                            cvAccuracy.append(cvAccuracy)

                        cvErrorLog.append(cvLoss)
                        iterLog.append(iter)
                        plt.clf()
                        plt.axis([0, numEpochs, 0.0, 1.0])
                        ax = plt.gca()
                        ax.set_autoscale_on(False)
                        plt.plot(iterLog, minibatchTrainErrorLog)
                        plt.plot(iterLog, fullTrainErrorLog)
                        plt.plot(iterLog, cvErrorLog)
                        plt.legend(['Train error (minibatch)', 'Train error', 'CV error'], loc=2, prop={'size': 10})
                        plt.xlabel('Iterations')
                        plt.ylabel('Error')
                        plt.grid(True)
                        plt.draw()
                        plt.pause(0.00001)


            self.__restoreLayerVariablesToNumpy__(session)

    def predict(self, dataset):
        self.__initializeTrainGraph__(datasetShape=dataset.examples.shape, labelShape=dataset.labels.shape)
        with tf.Session(graph=self.activateGraph) as session:
            tf.initialize_all_variables().run()
            feed_dict = {self.tfTrainSet: dataset.examples, self.tfTrainLabels: dataset.labels}
            activation = session.run(self.activation, feed_dict=feed_dict)
            self.__restoreLayerVariablesToNumpy__(session)

        return activation











