import numpy as np

np.random.seed(42)
import tensorflow as tf

tf.set_random_seed(42)

import argparse
import pandas as pd
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, Callback
from keras import backend as K

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, required=True, help="number of epochs")
parser.add_argument("--dir", required=True, help="where to put output files")
a = parser.parse_args()


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get("loss"))


class NN:
    def __init__(self, epochs):
        self.epochs = epochs

    @staticmethod
    def rmsle(y_true, y_pred):
        a = K.log(y_pred + 1)
        b = K.log(y_true + 1)
        return K.mean(K.square(a - b), axis=-1) ** (1 / 2)

    def load_data(self):
        train = pd.read_csv("data/train.csv", parse_dates=["timestamp"])
        test = pd.read_csv("data/test.csv", parse_dates=["timestamp"])
        return train, test

    def clean_feature(self, data):
        # transform non-numerical variables
        for c in data.columns:
            if data[c].dtype == "object":
                lbl = preprocessing.LabelEncoder()
                lbl.fit(list(data[c].values))
                data[c] = lbl.transform(list(data[c].values))

        # replace missing values with mean values
        for c in data.columns:
            data[c].fillna(data[c].mean(), inplace=True)
        return data

    def transform_data(self):
        train, test = self.load_data()
        y_train = train["price_doc"]
        x_train = train.drop(["timestamp", "price_doc", "id"], axis=1)
        x_train = self.clean_feature(x_train)

        x_test = test.drop(["timestamp", "id"], axis=1)
        x_test = self.clean_feature(x_test)

        return y_train, x_train, x_test

    def define_model(self):
        _, x_train, _ = self.transform_data()
        model = Sequential()
        model.add(Dense(1024, input_dim=x_train.shape[1]))
        model.add(Activation("sigmoid"))
        model.add(Dense(512))
        model.add(Activation("sigmoid"))
        model.add(Dense(256))
        model.add(Activation("sigmoid"))
        model.add(Dense(128))
        model.add(Activation("sigmoid"))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation("linear"))
        model.compile(optimizer="rmsprop", loss=self.rmsle)
        return model

    def main(self):
        _, test = self.load_data()
        y_train, x_train, x_test = self.transform_data()
        early_stop = EarlyStopping(monitor="val_loss", patience=20)
        model = self.define_model()
        history = LossHistory()
        model.fit(x_train.values, y_train.values, epochs=self.epochs,
                  validation_split=0.01, callbacks=[early_stop, history],
                  verbose=1)
        model.save(a.dir + "model.h5")
        pred = np.reshape(model.predict(x_test.values), -1)
        output = pd.DataFrame({"id": test.id, "price_doc": pred})
        output.to_csv(a.dir + "submissions_nn.csv", index=False)


if __name__ == "__main__":
    # python neural_network_model --dir /output/ --epochs 10
    run = NN(epochs=a.epochs)
    run.main()
