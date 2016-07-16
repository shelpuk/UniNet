import uninet as ssb
import time
import numpy as np
import errorFunction as ef

X = [[1,0.5,0.2], [0.2, 0.8, 0.9], [0.1, 0.2, 1], [0.2, 0.1, 0.9]]
y = [[1,0],[0,1],[1,0],[1,0]]

trainigSet = ssb.dataset(examples=X, labels=y)

l1 = ssb.linear(neurons=3, learningRate=0.1)
l2 = ssb.linear(neurons=12, learningRate=0.1)
l3 = ssb.linear(neurons=2, learningRate=0.1)

net = ssb.neuralnetwork(layers=[l1, l2, l3])

#print net.predict(trainigSet)
#start_time = time.time()

net.train(trainingSet=trainigSet, numEpochs=5000, minibatchSize=None, verbose=5)
#print time.time() - start_time, "seconds"

net.gradientCheck(dataSet=trainigSet, verbose=5, step=0.0000001)

#print l3.activation

#print net.predict(trainigSet)
