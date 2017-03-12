from IPython.display import Image, display
import numpy as np
import scipy.misc
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd


def create_image(array):
    if array.shape == (784,):
        img_arr = np.reshape(array, (28, 28))
    else:
        raise ValueError('the array have to be size (784,)!')
    scipy.misc.imsave('figures/original character.jpg', img_arr)
    image_name = 'figures/original character.jpg'
    return image_name


def display_image(image_name):
    display(Image(filename=image_name, width=100, height=100))


def create_image_extracted(array):
    if array.shape == (196,):
        img_arr = np.reshape(array, (14, 14))
    else:
        raise ValueError('the array have to be size (196,)!')
    scipy.misc.imsave('figures/extracted character.jpg', img_arr)
    image_name = 'figures/extracted character.jpg'
    return image_name


def plot_confusion_matrix(conf_matrix, labels):
    df_conf_matrix = pd.DataFrame(conf_matrix, index=[i for i in labels], columns=[i for i in labels])
    plt.figure(figsize=(8, 6))
    sn.heatmap(df_conf_matrix, annot=True)
    plt.title('Confusion Matrix')
