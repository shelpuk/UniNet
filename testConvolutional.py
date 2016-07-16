import uninet as ssb
import neuralnetwork_tensorflow as nntf
import cPickle
import activationFunction as af
import numpy as np

X = [[[1,0,0,0],
      [0,1,0,0],
      [0,0,1,0],
      [0,0,0,1]],
     [[1, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 0]
      ],
     [[1, 0, 1, 0],
      [0, 1, 1, 0],
      [0, 0, 1, 0],
      [0, 0, 1, 1]
      ]]

y = [[1,0],[1,0],[0,1]]

trainigSet = ssb.dataset(examples=np.array(X), labels=y)

trainigSet.rearrangeToCubic()

l1 = ssb.input(input_size=[4,4,1])
l2 = ssb.convolutional(patchSize=3, strides=[1,1,1,1], depth=16, activationFunction=af.relu)
l3 = ssb.softmax(neurons=2)

net = nntf.neuralnetwork(layers=[l1,l2,l3], errorFunction=ssb.logLoss)

net.train(trainingSet=trainigSet, numEpochs=50, minibatchSize=2, learningRate=0.1, errorCheckPeriod=10)

print net.predict(trainigSet)
