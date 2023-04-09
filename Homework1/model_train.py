import numpy as np
from function_NN import * 
from function_MNIST import * 

# 测试验证集路径
x_train_path = '/Users/qiaoqiaoqiaoqiaoqiao/Desktop/复旦/研一下/神经网络和深度学习 张力/Homework1/Mnist_data/train-images-idx3-ubyte.gz'
y_train_path = '/Users/qiaoqiaoqiaoqiaoqiao/Desktop/复旦/研一下/神经网络和深度学习 张力/Homework1/Mnist_data/train-labels-idx1-ubyte.gz'
x_test_path = '/Users/qiaoqiaoqiaoqiaoqiao/Desktop/复旦/研一下/神经网络和深度学习 张力/Homework1/Mnist_data/t10k-images-idx3-ubyte.gz'
y_test_path = '/Users/qiaoqiaoqiaoqiaoqiao/Desktop/复旦/研一下/神经网络和深度学习 张力/Homework1/Mnist_data/t10k-labels-idx1-ubyte.gz'
(x_train, y_train), (x_test, y_test) = load_mnist(x_train_path, y_train_path, x_test_path, y_test_path)

# 超参数设定
batch_size = 1000; epoch = 100; lamda = 0.00001; learn_rate = 0.1

# 神经训练网络
loss_train = []
loss_test = []
ACC_test = []
for i in range(epoch):
    state = np.random.get_state()
    np.random.shuffle(x_train)
    np.random.set_state(state)
    np.random.shuffle(y_train)
    pos = 0
    loss = []
    loss_2 = []
    acc = []
    while pos < x_train.shape[0]:
        x_train_batch = x_train[pos:min(pos+batch_size, x_train.shape[0])]
        y_train_batch = y_train[pos:min(pos+batch_size, y_train.shape[0])]
        o0 = x_train_batch
        a1 = linear1.forward(o0)
        o1 = relu1.forward(a1)
        a2 = linear2.forward(o1)
        o2 = softmax2.forward(a2)
        y = o2
    # 获得网络当前输出，计算损失loss
        loss_temp = MSE(y, y_train_batch) + lamda/2*(np.sum(linear1._W**2)+np.sum(linear2._W**2))
        loss.append(loss_temp)
    # 反向传播，获取梯度
        grad = (y - y_train_batch)/y.size*2
        grad = softmax2.backward(a2, grad)
        grad = linear2.backward(o1, grad)
        grad = relu1.backward(a1, grad)
        grad = linear1.backward(o0, grad)
    # 更新网络中线性层的参数
        linear1.update_L2(learn_rate, lamda)
        linear2.update_L2(learn_rate, lamda)
        pos = pos + batch_size
    # 在测试集中验证分类准确率：
        o0 = x_test
        a1 = linear1.forward(o0)
        o1 = relu1.forward(a1)
        a2 = linear2.forward(o1)
        o2 = softmax2.forward(a2)
        y = np.argmax(o2, axis=1)
        y_ = np.argmax(y_test, axis=1)
        loss_temp = MSE(o2, y_test) + lamda / 2 * (np.sum(linear1._W ** 2) + np.sum(linear2._W ** 2))
        count = np.sum((y-y_)==0)
        acc_temp = count/y.size
        acc.append(acc_temp)
        loss_2.append(loss_temp)
    print('Loss for epoch', i+1, ':', np.mean(loss))  # train loss
    loss_train.append(np.mean(loss))
    print('Test Loss for epoch', i+1, ':', np.mean(loss_2))  # test loss
    loss_test.append(np.mean(loss_2))
    print('ACC for epoch', i+1, ':', np.mean(acc))  # acc
    ACC_test.append(np.mean(acc))

# 保存模型
np.save("dense1_W", linear1._W)
np.save("dense1_b", linear1._b)
np.save("dense2_W", linear2._W)
np.save("dense2_b", linear2._b)

# 导入模型
W1 = np.load("dense1_W.npy")
b1 = np.load("dense1_b.npy")
dense1 = LinearLayer(W1.shape[0], W1.shape[1])
dense1._W = W1
dense1._b = b1
W2 = np.load("dense2_W.npy")
b2 = np.load("dense2_b.npy")
dense2 = LinearLayer(W2.shape[0], W2.shape[1])
dense2._W = W2
dense2._b = b2