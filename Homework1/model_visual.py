from model_train import *
import matplotlib.pyplot as plt

# 可视化
## train loss
plt.plot(range(1, epoch+1), loss_train, 'b')
plt.xlabel('epoch')
plt.ylabel('Loss in Train')
plt.savefig('Train_loss.png')
plt.close()
## test loss
plt.plot(range(1, epoch+1), loss_test, 'r')
plt.xlabel('epoch')
plt.ylabel('Loss in Test')
plt.savefig('Test_loss.png')
plt.close()
## test acc
plt.plot(range(1, epoch+1), ACC_test, 'g')
plt.xlabel('epoch')
plt.ylabel('Accuracy in Test')
plt.savefig('Test_acc.png')
plt.close()
## W
for i in range(10):
    plt.imshow(linear1._W[:,i].reshape(28,28))
    name = 'W' + str(i) + '.png'
    plt.savefig(name)
    plt.close()