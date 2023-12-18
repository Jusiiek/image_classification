import random
import os
import itertools

import pandas as pd
import numpy as np
import math
import sys
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist, cifar10
from matplotlib import pyplot as plt

# problem https://www.tensorflow.org/datasets/catalog/fashion_mnist

HERE = os.getcwd()
DATA_DIR = os.path.join(HERE, 'data')
RESULTS_DIR = os.path.join(HERE, 'results')
LOGS_DIR = os.path.join(HERE, 'logs')
MODELS_DIR = os.path.join(HERE, 'models')


class NeuralNetwork:
    def __init__(
            self,
            n_of_neural_layers,
            n_of_neural_per_layer,
            dropout_rate=0.2,
            output_units=1,
            learning_rate=0.2,
            epochs=50,
            batch_size=64,
            shape=None,
            activation='relu',
            model_type='dense'
    ):
        self.n_of_neural_layers = n_of_neural_layers
        self.n_of_neural_per_layer = n_of_neural_per_layer
        self.dropout_rate = dropout_rate
        self.output_units = output_units
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.activation = activation
        self.shape = shape
        self.tensorboard_callback = {}
        self.model_type = model_type
        self.models = {
            'dense': self.__create_dense_model,
            'conv': self.__create_conv_model
        }
        self.model_path = os.path.join(MODELS_DIR, self.model_type)

        self.__create_models_dir()
        self.__create_logs_dir()
        self.__create_model()

    def __create_models_dir(self):
        """
        Creates logs directory
        """
        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR)
            print('Created logs directory: {}'.format(MODELS_DIR))

    def __create_logs_dir(self):
        """
        Creates logs directory
        """
        if not os.path.exists(LOGS_DIR):
            os.makedirs(LOGS_DIR)
            print('Created logs directory: {}'.format(LOGS_DIR))

    def __create_model(self):
        try:
            self.model = tf.keras.models.load_model(self.model_path)
        except:
            self.model = self.models[self.model_type]()

    def __create_dense_model(self):
        """
        Creates model with dense layers
        """
        print("CREATING DENSE MODEL")
        model = models.Sequential()

        model.add(layers.Dense(self.n_of_neural_per_layer, activation=self.activation, input_shape=self.shape))
        model.add(layers.Dropout(self.dropout_rate))

        for _ in range(self.n_of_neural_layers):
            model.add(layers.Dense(self.n_of_neural_per_layer, activation=self.activation))
            model.add(layers.Dropout(self.dropout_rate))

        model.add(layers.Dense(self.output_units, activation='sigmoid'))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        model.summary()
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOGS_DIR, histogram_freq=0)
        return model

    def __create_conv_model(self):
        """
        Creates model with dense Conv2D
        """
        print("CREATING CONV MODEL")
        model = models.Sequential()
        model.add(layers.Conv2D(
            self.n_of_neural_per_layer, (3, 3), 1, activation=self.activation, input_shape=self.shape, padding='same'
        ))
        model.add(layers.MaxPool2D(2, 2))
        for _ in range(self.n_of_neural_layers):
            model.add(layers.Conv2D(
                self.n_of_neural_per_layer, (3, 3), 1, input_shape=self.shape, activation=self.activation, padding='same'
            ))
            model.add(layers.MaxPool2D(2, 2))

        # the number of filters creates a channel
        model.add(layers.Flatten())

        model.add(layers.Dense(self.shape[0], activation=self.activation))
        model.add(layers.Dense(self.output_units, activation='softmax'))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        # check how model will transform data
        model.summary()
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOGS_DIR, histogram_freq=0)
        return model

    def evaluate_model(
            self,
            img_train,
            train_label,
            img_val,
            val_label,
            img_test,
            test_label,
            save_config,
            get_evaluate_model,
            show_diagrams=True
    ):
        history = self.model.fit(
            img_train,
            train_label,
            validation_data=(img_val, val_label),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[self.tensorboard_callback]
        )
        self.model.save(self.model_path)
        self.model.evaluate(img_test, test_label)
        self.model.save(self.model_path)

        if save_config:
            self.__save_config(history.history['accuracy'][-1])
        if get_evaluate_model:
            loss, _ = self.model.evaluate(img_train, train_label)
            return loss
        if show_diagrams:
            fig = plt.figure()
            plt.plot(history.history['loss'], color='teal', label='loss')
            plt.plot(history.history['accuracy'], color='blue', label='accuracy')
            fig.suptitle('Loss', fontsize=20)
            plt.legend(loc='upper left')
            plt.show()

    def __save_config(self, val_accuracy):

        config = {
            "n_of_neural_layers": self.n_of_neural_layers,
            "n_of_neural_per_layer": self.n_of_neural_per_layer,
            "dropout_rate": self.dropout_rate,
            "output_units": self.output_units,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "val_accuracy": val_accuracy
        }

        path = os.path.join(RESULTS_DIR, 'result.xlsx')
        if os.path.exists(path):
            df = pd.read_excel(path)
            df = df.append(pd.DataFrame([config]), ignore_index=True)
        else:
            os.makedirs(RESULTS_DIR, exist_ok=True)
            df = pd.DataFrame([config])

        df.to_excel(path, index=False)

    def test_image(self, path):
        try:
            img = tf.keras.utils.load_img(path)
        except:
            print("Couldn't load image")
            sys.exit(1)

        img = tf.image.resize(img, [28, 28])
        img = tf.image.rgb_to_grayscale(img)
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = img.copy() / 255.0
        img = img.reshape((28, 28))
        img = np.expand_dims(img, axis=0)
        prediction = self.model.predict(img)
        print("prediction", prediction)
        max_pred = np.argmax(prediction)
        print("max_pred", max_pred)


class Data:
    def __init__(self):
        self.X_train = self.y_train = self.X_val = self.y_val = self.X_test = self.y_test = None
        self.shape = []
        self.__tensor_dataset()

    def split_data_train_validation_test(self, size, training_size=0.8, validation_size=0.1):
        """
        Function that splits a set of images into a validation, training, and testing set.

        Parameters:
            size (int): Number of images.
            training_size (float): Size of training set.
            validation_size (float): Size of validation set.
        Returns:
            int: Size of training, validation and test set.
        """
        training = math.floor(training_size * size)
        validation = math.floor(validation_size * size)
        test = size - training - validation

        return training, validation, test

    def __get_data(self):
        return fashion_mnist.load_data()

    def __data_excel(self):
        path = os.path.join(DATA_DIR, 'fashion-mnist_train.csv')
        df = pd.read_csv(path).to_numpy()

        X = df[:, 1:] / 255.0
        y = df[:, 0]

        training_size, validation_size, test_size = self.split_data_train_validation_test(df.shape[0])

        train_images, train_labels = X[:training_size], y[:training_size]
        val_images, val_labels = (X[training_size:validation_size + training_size],
                                  y[training_size:validation_size + training_size])
        test_images, test_labels = X[training_size + validation_size:], y[training_size + validation_size:]

        self.X_train, self.y_train = train_images, train_labels
        self.X_val, self.y_val = val_images, val_labels
        self.X_test, self.y_test = test_images, test_labels

    def __tensor_dataset(self):
        (train_images, train_labels), (test_images, test_labels) = self.__get_data()

        train_images = np.reshape(
            train_images, (train_images.shape[0], train_images.shape[1], train_images.shape[2], 1)
        )
        test_images = np.reshape(
            test_images, (test_images.shape[0], test_images.shape[1], test_images.shape[2], 1)
        )

        self.shape = (train_images.shape[1], train_images.shape[2], 1)

        train_data = list(zip(train_images, train_labels))
        test_data = list(zip(test_images, test_labels))

        # random.shuffle(train_data)
        # random.shuffle(test_data)
        #
        # train_images, train_labels = zip(*train_data)
        # test_images, test_labels = zip(*test_data)

        train_images = train_images / 255.0
        test_images = test_images / 255.0
        #
        # train_labels = np.array(train_labels)
        # test_labels = np.array(test_labels)

        train_size = int(len(train_images) * .8)
        test_size = int(len(test_images) * .1)

        self.X_train, self.y_train = train_images[:train_size], train_labels[:train_size]
        self.X_val, self.y_val = train_images[:test_size], train_labels[:test_size]
        self.X_test, self.y_test = test_images[:test_size], test_labels[:test_size]

    def run(self, image_path=''):
        nn = NeuralNetwork(
            3,
            32,
            epochs=100,
            learning_rate=0.001,
            shape=self.shape,
            batch_size=128,
            output_units=10,
            model_type='conv'
        )
        nn.evaluate_model(
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val,
            self.X_test,
            self.y_test,
            True,
            False
        )

        if image_path:
            nn.test_image(image_path)

    def grid_search(self, random_ten):

        best_params = None
        best_score = float('inf')

        param_grid = {
            'n_of_neural_layers': [1, 2, 3],
            'n_of_neural_per_layer': [32, 64, 128],
            'dropout_rate': [0.2, 0.3, 0.4],
            'activation': ['relu', 'elu']
        }
        param_combinations = list(itertools.product(*param_grid.values()))

        if random_ten:
            param_combinations = random.sample(param_combinations, 10)

        for params in param_combinations:
            print(f"Parameters: {params}")

            nn = NeuralNetwork(
                n_of_neural_layers=params[0],
                n_of_neural_per_layer=params[1],
                dropout_rate=params[2],
                activation=params[3]
            )

            score = nn.evaluate_model(
                self.X_test,
                self.y_test,
                self.X_val,
                self.y_val,
                self.X_test,
                self.y_test,
                False,
                True
            )

            if score < best_score:
                best_score = score
                best_params = dict(zip(param_grid.keys(), params))

        print(f"Best score: {best_score}")
        print(f"\nBest params: {best_params}")


class FashionMnist(Data):

    def __get_data(self):
        return fashion_mnist.load_data()


class CIFAR10(Data):

    def __get_data(self):
        return cifar10.load_data()


def main():
    f = FashionMnist()
    f.run()

    # c = CIFAR10()
    # c.run()


if __name__ == '__main__':
    main()
