import uninet as ssb
import cPickle

trainigSet = cPickle.load(open("data/mnist/MNISTTrainingSet", 'rb'))
testSet = cPickle.load(open("data/mnist/MNISTTestSet", 'rb'))

l1 = ssb.logistic(neurons=784, learningRate=0.1)
l2 = ssb.maxout(units=500, neuronsPerPool=2 , learningRate=0.0001, rmsProp=0.9)
l3 = ssb.maxout(units=100, neuronsPerPool=2, learningRate=0.0001, rmsProp=0.9)
l4 = ssb.softmax(neurons=10, learningRate=0.0001, rmsProp=0.9)

net = ssb.neuralnetwork(layers=[l1, l2, l3, l4])

print net.getAccuracy(trainigSet)

net.train(trainingSet=trainigSet, numEpochs=200, minibatchSize=100, verbose=5)

print "Training set accuracy: ",net.getAccuracy(trainigSet)
print "Test set accuracy: ",net.getAccuracy(testSet)