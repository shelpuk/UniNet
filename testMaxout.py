import uninet as ssb
import time
import errorFunction as ef

X = [[1,0.5,0.2], [0.2, 0.8, 0.9], [0.1, 0.2, 1], [0.1, 0.1, 1]]
y = [[1,0],[0,1],[1,0],[1,0]]

trainigSet = ssb.dataset(examples=X, labels=y)

l1 = ssb.logistic(neurons=3, learningRate=0.1)
l2 = ssb.maxout(units=6, neuronsPerPool=2, learningRate=0.001, rmsProp=0.9)
l3 = ssb.maxout(units=2, neuronsPerPool=2, learningRate=0.001, rmsProp=0.9)
net = ssb.neuralnetwork(layers=[l1, l2, l3])

net.train(trainingSet=trainigSet, numEpochs=10000, minibatchSize=None, verbose=5)

print net.predict(trainigSet)

#net.gradientCheck(dataSet=trainigSet, verbose=5)

