import numpy as np
import struct
import gzip

class MNISTDataset:
    def __init__(self, images_path, labels_path):
        self.images = self.load_images(images_path)
        self.labels = self.load_labels(labels_path)
        self.num_samples = self.images.shape[0]

    def load_images(self, path):
        with gzip.open(path, 'rb') as f:
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)
            images = images.astype(np.float32) / 255.0  # 归一化到 [0, 1]
        return images

    def load_labels(self, path):
        with gzip.open(path, 'rb') as f:
            magic, num = struct.unpack(">II", f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

class DataLoader:
    def __init__(self, dataset, batch_size=64, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(self.dataset.num_samples)
        self.current = 0
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        self.current = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self):
        if self.current >= self.dataset.num_samples:
            raise StopIteration
        batch_indices = self.indices[self.current:self.current + self.batch_size]
        batch_images = self.dataset.images[batch_indices]
        batch_labels = self.dataset.labels[batch_indices]
        self.current += self.batch_size
        return batch_images, batch_labels

def load_mnist(batch_size=64):
    # 请确保以下路径正确指向你的 MNIST 数据集文件
    train_images_path = './data/mnist/train-images-idx3-ubyte.gz'
    train_labels_path = './data/mnist/train-labels-idx1-ubyte.gz'
    test_images_path = './data/mnist/t10k-images-idx3-ubyte.gz'
    test_labels_path = './data/mnist/t10k-labels-idx1-ubyte.gz'

    train_dataset = MNISTDataset(train_images_path, train_labels_path)
    test_dataset = MNISTDataset(test_images_path, test_labels_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
