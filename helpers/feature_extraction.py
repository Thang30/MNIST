import numpy as np


def feature_extraction(array):
    new_samples = []
    for i in range(array.shape[0]):
        new_features = []
        for j in range(0, array.shape[1], 4):
            new_value = array[i, range(j, j + 4)].mean()
            new_features.append(new_value)
        new_features = np.array(new_features)
        new_samples.append(new_features)
    array = np.array(new_samples)
    return array
