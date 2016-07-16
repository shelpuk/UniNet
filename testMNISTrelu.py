import uninet as ssb
import neuralnetwork_tensorflow as nntf
import cPickle


trainigSet = cPickle.load(open("/media/tassadar/Work/Google Drive/My/NeuralNet/data/mnist/MNISTTrainingSet", 'rb'))
cvSet = cPickle.load(open("/media/tassadar/Work/Google Drive/My/NeuralNet/data/mnist/MNISTTestSet", 'rb'))

l1 = ssb.input(neurons=784)
l2 = ssb.elu(neurons=1000)
l3 = ssb.elu(neurons=500)
l4 = ssb.softmax(neurons=10)

net = nntf.neuralnetwork(layers=[l1,l2,l3,l4], errorFunction=ssb.logLoss)

net.train(trainingSet=trainigSet, cvSet=cvSet, numEpochs=20000, minibatchSize=100, learningRate=0.1, rmsProp=0.9, cvCheckPeriod=100, cvAccuracyCheck=True, visualization=True)