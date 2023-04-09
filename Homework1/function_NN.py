import numpy as np

# 定义全连接网络
class LinearLayer:
    def __init__(self, input_D, output_D):
        self._W = np.random.normal(0, 0.1, (input_D, output_D))
        self._b = np.random.normal(0, 0.1, (1, output_D))
        self._grad_W = np.zeros((input_D, output_D))
        self._grad_b = np.zeros((1, output_D))

    def forward(self, X):
        return np.matmul(X, self._W) + self._b

    def backward(self, X, grad):
        self._grad_W = np.matmul(X.T, grad)
        self._grad_b = np.matmul(grad.T, np.ones(X.shape[0]))
        return np.matmul(grad, self._W.T)

    def update_L2(self, learn_rate, lamda):
        self._W = self._W - (self._grad_W + self._W * lamda) * learn_rate
        self._b = self._b - self._grad_b * learn_rate

class Relu:
    def __init__(self):
        pass

    def forward(self, X):
        return np.where(X < 0, 0, X)

    def backward(self, X, grad):
        return np.where(X > 0, X, 0) * grad

class Softmax:
    def __init__(self):
        pass

    def forward(self, X):
        exp_X = np.exp(X)
        return (exp_X.T/np.sum(exp_X, axis=1)).T

    def backward(self, X, grad):
        exp_X = np.exp(X)
        temp_X = (exp_X.T/np.sum(exp_X, axis=1)).T
        return (temp_X - temp_X ** 2) * grad

def MSE(y, y_):
    return np.sum((y-y_)**2)/y.size

# 定义网络模型
linear1 = LinearLayer(784, 10)
relu1 = Relu()
linear2 = LinearLayer(10, 10)
softmax2 = Softmax()