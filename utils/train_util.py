import os
import numpy as np
from random import shuffle
import support.config as c
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
import matplotlib.pyplot as plt
from keras.callbacks import History, Callback


class accCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc') > 0.998):
            print("\nReached 99.8% accuracy so stopping training!")
            self.model.stop_training = True


def process_string(string):
    temp_list = []
    for char in string:
        temp_list.append([int(char)])
    return temp_list


def create_x():
    string_alias = '{0:0' + str(c.str_len) + 'b}'
    train_input = [string_alias.format(i) for i in range(2**c.str_len)]
    shuffle(train_input)
    #train_input = [map(int, i) for i in train_input]
    ti = []
    for i in train_input:
        ti.append(np.array(process_string(i)))
    train_input = ti
    return np.array(train_input)


def create_y(x):
    train_output = []
    for i in x:
        count = 0
        for j in i:
            if j[0] == 1:
                count += 1
        temp_list = ([0]*c.str_classes)
        temp_list[count] = 1
        train_output.append(np.array(temp_list))
    return np.array(train_output)


def create_data():
    x = create_x()
    y = create_y(x)
    return x, y


def plot_loss(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


def model_training():
    x, y = create_data()
    model = Sequential()
    model.add(LSTM(c.lstm_len, input_shape=c.input_shape))
    model.add(Dense(c.output_shape, activation='softmax'))
    model.compile(optimizer=c.optimizer,
                  loss=c.loss,
                  metrics=c.metrics)

    acc_callbacks = accCallback()
    history = model.fit(x, y, callbacks=[acc_callbacks], epochs=c.epochs,
                        batch_size=c.batch_size, validation_split=c.validation_split)
    model.save(c.model_path)
    if c.plot_loss:
        plot_loss(history)


if __name__ == "__main__":
    model_training()
