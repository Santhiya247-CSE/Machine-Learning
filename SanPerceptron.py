import numpy as np;
class Perceptron(object):
    def __init__(self, input_size, lr=1, epochs=2):
        self.W = np.zeros(input_size+1)
        self.epochs = epochs
        self.lr = lr
    def activation_fn(self, x):
        return 1 if x >= 0 else 0
    def predict(self, x):
        z = self.W.T.dot(x)
        a = self.activation_fn(z)
        return a
    def fit(self, X, d):
        for _ in range(self.epochs):
            for i in range(d.shape[0]):
                x = np.insert(X[i], 0, 1)
                y = self.predict(x)
                e = d[i] - y
                self.W = self.W + self.lr * e * x
if __name__ == '__main__':
    Xi = np.array([[0, 0],[0, 1],[1, 0],[1, 1]])
    targets = np.array([0, 1, 1, 1])
    perceptron = Perceptron(input_size=2)
    perceptron.fit(Xi, targets)
print("*****************Perceptron Network**************")
print("Inputs\tTargets\n Xi\t t")
for i in range(4):
    print(Xi[i],"\t",targets[i],"\n")
print("Final Weights & Bias",perceptron.W)
print("On Epoch",perceptron.epochs)
