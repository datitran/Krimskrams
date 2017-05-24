import numpy as np

np.random.seed(42)
import tensorflow as tf

tf.set_random_seed(42)

import argparse
import pandas as pd
from sklearn import preprocessing
from keras.models import Model
from keras.layers import Dense, Input, Dropout, average
from keras.optimizers import Adam
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
        macro = pd.read_csv("data/macro.csv", parse_dates=["timestamp"])
        return train, test, macro

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
        train, test, macro = self.load_data()

        y_train = train["price_doc"]
        x_train = train.drop(["timestamp", "price_doc", "id"], axis=1)
        x_train = self.clean_feature(x_train)

        x_test = test.drop(["timestamp", "id"], axis=1)
        x_test = self.clean_feature(x_test)

        # join macro with train data
        train_macro = pd.merge(train, macro, how="left", on="timestamp")
        test_macro = pd.merge(test, macro, how="left", on="timestamp")

        test_macro = test_macro[np.append(macro.columns.values, "id")].copy()
        test_macro.dropna(axis=1, how="all", inplace=True)
        train_macro = train_macro[np.append(test_macro.columns.values, "price_doc")]

        assert test_macro.shape[1] == 61
        assert train_macro.shape[1] == 62

        x_train_macro = train_macro.drop(["timestamp", "price_doc"], axis=1)
        x_train_macro = self.clean_feature(x_train_macro)

        x_test_macro = test_macro.drop(["timestamp"], axis=1)
        x_test_macro = self.clean_feature(x_test_macro)

        x_train_macro.set_index("id", inplace=True)
        x_test_macro.set_index("id", inplace=True)

        return y_train, x_train, x_test, x_train_macro, x_test_macro

    def define_model(self):
        _, x_train, _, x_train_macro, _, = self.transform_data()
        a_input = Input(shape=(x_train.shape[1],))
        a_1 = Dense(1024, activation="sigmoid")(a_input)
        a_2 = Dense(512, activation="sigmoid")(a_1)
        a_3 = Dense(256, activation="sigmoid")(a_2)
        a_4 = Dense(128, activation="sigmoid")(a_3)
        a_5 = Dropout(0.5)(a_4)
        a_6 = Dense(64, activation="sigmoid")(a_5)
        a_7 = Dense(1, activation="linear")(a_6)

        b_input = Input(shape=(x_train_macro.shape[1],))
        b_1 = Dense(256, activation="sigmoid")(b_input)
        b_2 = Dense(192, activation="sigmoid")(b_1)
        b_3 = Dense(128, activation="sigmoid")(b_2)
        b_4 = Dense(64, activation="sigmoid")(b_3)
        b_5 = Dropout(0.5)(b_4)
        b_6 = Dense(32, activation="sigmoid")(b_5)
        b_7 = Dense(1, activation="linear")(b_6)

        merge_output = average([a_7, b_7])

        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        model = Model(inputs=[a_input, b_input], outputs=merge_output)
        model.compile(optimizer=adam, loss=self.rmsle)

        return model

    def main(self):
        _, test, _ = self.load_data()
        y_train, x_train, x_test, x_train_macro, x_test_macro = self.transform_data()
        early_stop = EarlyStopping(monitor="val_loss", patience=20)
        model = self.define_model()
        history = LossHistory()
        model.fit([x_train.values, x_train_macro.values], y_train.values, epochs=self.epochs,
                  validation_split=0.01, callbacks=[early_stop, history],
                  verbose=1)
        model.save(a.dir + "model_macro.h5")
        pred = np.reshape(model.predict([x_test.values, x_test_macro.values]), -1)
        output = pd.DataFrame({"id": test.id, "price_doc": pred})
        output.to_csv(a.dir + "submissions_nn_macro.csv", index=False)


if __name__ == "__main__":
    # python neural_network_model --dir /output/ --epochs 10
    run = NN(epochs=a.epochs)
    run.main()
