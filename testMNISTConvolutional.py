import uninet as ssb
import neuralnetwork_tensorflow as nntf
import cPickle
import activationFunction as af
import dataset

old_trainingSet = cPickle.load(open("/media/tassadar/Work/Google Drive/My/NeuralNet/data/mnist/MNISTTrainingSet_square", 'rb'))
#cvSet = cPickle.load(open("/media/tassadar/Work/Google Drive/My/NeuralNet/data/mnist/MNISTTestSet_square", 'rb'))

trainingSet = dataset.dataset(examples=old_trainingSet.examples, labels=old_trainingSet.labels)
trainingSet.rearrangeToCubic()

l1 = ssb.input(input_size=[28,28,1])
l2 = ssb.convolutional(patchSize=3, strides=[1,1,1,1], depth=16, activationFunction=af.relu)
l3 = ssb.softmax(neurons=10)

net = nntf.neuralnetwork(layers=[l1,l2,l3], errorFunction=ssb.logLoss)

net.train(trainingSet=trainingSet, numEpochs=10000, minibatchSize=100, learningRate=0.1, errorCheckPeriod=100)