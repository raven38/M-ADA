import numpy as np
import os
import pickle
from PIL import Image
import tensorflow.keras.datasets as datasets

def resize_images(image_arrays, size=[32, 32]):
    image_arrays = (image_arrays * 255).astype('uint8')

    resized_image_arrays = np.zeros([image_arrays.shape[0]] + size)
    for i, image_array in enumerate(image_arrays):
        image = Image.fromarray(image_array)
        resized_image = image.resize(size=size, resample=Image.ANTIALIAS)

        resized_image_arrays[i] = np.asarray(resized_image)

    return np.expand_dims(resized_image_arrays, 3)

def download_and_process_mnist():
    if not os.path.exists('./data/mnist'):
        os.makedirs('./data/mnist')

    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

    train = {'X': resize_images(x_train.reshape(-1, 28, 28)),
             'y': y_train}

    test = {'X': resize_images(x_test.reshape(-1, 28, 28)),
            'y': y_test}

    with open('./data/mnist/train.pkl', 'wb') as f:
        pickle.dump(train, f, protocol=-1)

    with open('./data/mnist/test.pkl', 'wb') as f:
        pickle.dump(test, f, protocol=-1)


if __name__ == "__main__":
    download_and_process_mnist()
