import neuralnetwork_tensorflow as nntf
import uninet as ssb
import activationFunction as af


X = [[1,0.5,0.2], [0.2, 0.8, 0.9], [0.9, 0.2, 0.3], [0.3, 0.1, 0.1]]
y = [[1,0],[0,1],[1,0],[1,0]]

X_cv = [[0.8,0.3,0.2], [0.1, 0.7, 0.8], [1.0, 0.3, 0.2]]
y_cv = [[1,0],[0,1],[1,0],[1,0]]

trainigSet = ssb.dataset(examples=X, labels=y)
cvSet = ssb.dataset(examples=X, labels=y)

l1 = ssb.input(neurons=3)
l2 = ssb.relu(neurons=20)
l3 = ssb.softmax(neurons=2)

l4 = ssb.convolutional(patchSize=3, depth=10, numChannels=1, activationFunction=af.relu)

net = nntf.neuralnetwork(layers=[l1,l2,l3], errorFunction=ssb.logLoss)

net.train(trainingSet=trainigSet, cvSet=cvSet, numEpochs=1, minibatchSize=None, learningRate=0.001, rmsProp=0.9, cvCheckPeriod=50, cvAccuracyCheck=True)

#print net.predict(trainigSet)
