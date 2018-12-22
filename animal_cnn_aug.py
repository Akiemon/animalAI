from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
import numpy as np
import keras

classes = ["monkey", "boar", "crow"]
num_classes = len(classes)
image_size = 50



# 関数を定義しよう
def main():
    # データを読み込む
    X_train, X_test, y_train, y_test = np.load("./animal_aug.npy")
    # 正規化（データを０〜１に変換する作業）
    ### astypeとは？
    X_train = X_train.astype("float")/256
    X_test = X_test.astype("float")/256
    ###ここ何してるん？？
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    
    model = model_train(X_train, y_train)
    model_eval(model, X_test, y_test)

# modelの作成 #Aidemyの見直ししたら多分よくわかると思う！！
def model_train(X, y):
    model = Sequential()
    model.add(
        Conv2D(32, (3, 3),
        padding = "same",
        input_shape = X.shape[1:]) # X_trainの値は４つ入ってて、今回は1.2.3の要素が欲しいので取り出してる
        )
    model.add(Activation("relu"))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding = "same"))
    model.add(Activation("relu"))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(3))
    model.add(Activation("softmax"))


    # ここは何をやってるか、忘れた・・・。損失関数をいれてるのはわかる。optってなにしてんの？
    opt = keras.optimizers.rmsprop(lr = 0.0001, decay = 1e-6)
    model.compile(
        loss = "categorical_crossentropy",
        optimizer = opt,
        metrics=["accuracy"]
        )

    model.fit(X, y, batch_size=64, epochs=100)

    # modelの保存
    model.save("./animal_cnn_aug.h5")

    return model

def model_eval(model, X, y):
    scores = model.evaluate(X, y, verbose = 1)
    print("test loss: ", scores[0])
    print("tesr Accuracy: ", scores[1])

# 何しているのかマジで謎
if __name__ == "__main__":
    main()
