import gzip
import numpy as np
from struct import unpack

# 读取 MNIST
def __read_image(path):
    with gzip.open(path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        img = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)
    return img

def __read_label(path):
    with gzip.open(path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        lab = np.frombuffer(f.read(), dtype=np.uint8)
        # print(lab[1])
    return lab

def __normalize_image(image):
    img = image.astype(np.float32) / 255.0
    return img

def __one_hot_label(label):
    lab = np.zeros((label.size, 10))
    for i, row in enumerate(lab):
        row[label[i]] = 1
    return lab

def load_mnist(x_train_path, y_train_path, x_test_path, y_test_path, normalize=True, one_hot=True):
    image = {
        'train': __read_image(x_train_path),
        'test': __read_image(x_test_path)
    }
    label = {
        'train': __read_label(y_train_path),
        'test': __read_label(y_test_path)
    }
    if normalize:
        for key in ('train', 'test'):
            image[key] = __normalize_image(image[key])
    if one_hot:
        for key in ('train', 'test'):
            label[key] = __one_hot_label(label[key])
    return (image['train'], label['train']), (image['test'], label['test'])
