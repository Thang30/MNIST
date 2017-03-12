import numpy as np


def load_data(dataset):
    array_X = []
    array_Y = []
    for i in range(10):
        if dataset == 'train':
            instance_X = np.loadtxt('dataset/train%d.txt' % i)
        elif dataset == 'test':
            instance_X = np.loadtxt('dataset/test%d.txt' % i)
        else:
            raise ValueError("have to be 'train' or 'test' dataset!")
        instance_Y = np.empty([instance_X.shape[0], 1])
        instance_Y.fill(i)
        array_X.append(instance_X)
        array_Y.append(instance_Y)
    array_X = np.concatenate(array_X, axis=0)
    array_Y = np.concatenate(array_Y, axis=0)
    return array_X, array_Y
