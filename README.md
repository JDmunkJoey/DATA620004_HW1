# DATA620004
## Homework1
### MNIST_data
MNIST数据集（训练集和验证集）

### function_MNIST.py
导入MNIST数据

### function_NN.py
激活函数、反向传播，

### model_train.py
loss以及梯度的计算、学习率下降策略、L2正则化、优化器SGD、保存模型
参数查找：学习率，隐藏层大小，正则化强度

### model_visual.py
测试：导入模型，用经过参数查找后的模型进行测试，输出分类精度
 
### mlp
模型参数训练结果（npy格式）

### visual
可视化训练和测试的loss曲线，测试的accuracy曲线，以及可视化每层的网络参数。

### 运行逻辑
直接运行文件model_visual.py，可在model_train中修改所涉参数
