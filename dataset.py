import random
import numpy as np

class dataset(object):
    def __init__(self, examples=None, labels = None):
        self.examples = np.array(examples)
        self.labels = np.array(labels)
        self.numberExamples = None
        self.cubic = False

        self.__initializeExamples__()

    def __initializeExamples__(self):
        if self.examples is None: pass
        self.examples = np.array(self.examples)
        self.numberExamples = len(self.examples)
        if self.labels is not None:
            if len(self.labels) != len(self.examples): raise Exception ('There are different amount of labels and examples in the dataset')
            self.labels = np.array(self.labels)

    def getMiniBatch(self, size):
        exampleIds = random.sample(range(self.numberExamples), size)
        return {'id': exampleIds, 'examples': self.examples[exampleIds], 'labels': self.labels[exampleIds]}

    def getData(self):
        return {'examples': self.examples, 'labels': self.labels}

    def getNumExamples(self):
        return len(self.examples)

    def normalizeExamples(self):
        X_normalized = []
        for i in range(len(self.examples)):
            minX = min(self.examples[i])
            maxX = max(self.examples[i])
            X_normalized.append(np.true_divide((np.array(self.examples[i]) - minX), (np.array(maxX) - minX)))

        self.examples = X_normalized

    def rearrangeToCubic(self):
        if self.cubic: return 0
        if len(self.examples[0].shape) < 3: num_channels = 1
        else: num_channels = self.examples[0].shape[3]
        self.examples = np.array(self.examples).reshape(
            (-1, self.examples[0].shape[0], self.examples[0].shape[1], num_channels)).astype(np.float32)
        self.cubic = True





