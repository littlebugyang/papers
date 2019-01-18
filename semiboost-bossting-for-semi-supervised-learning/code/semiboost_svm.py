from sklearn import svm

class SemiBoostSvm:
    def __init__(self):
        self.classifiers = []
        self.weights = []
        pass

    def predict(self, X):
        sum = 0
        for i in range(len(self.classifiers)):
            sum += self.weights[i] * self.classifiers[i].predict(X.reshape(1,-1))
        if sum >= 0:
            return 1
        else:
            return -1
        pass

    def boost(self, classifier, weight):
        self.classifiers.append(classifier)
        self.weights.append(weight)
        pass


