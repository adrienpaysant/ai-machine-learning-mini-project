import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn import model_selection
from sklearn import preprocessing

def example():
    df = pd.read_csv("data/gray_75_balanced.csv", header=None, names=["sound", "acc1", "acc2"])

    print("\n== Dataframe description ==")
    print(df.describe())  # general description of the dataframe
    print("\n", df.head())  # print the first 5 lines
    print("\n", df.tail(3))  # print the last 3 lines

    # plot accel. data
    df[['acc1','acc2']].plot()
    plt.show()

    # plot sound data
    df['sound'].plot()

    plt.show()

    # merge mulple dataset
    print("\n== Dataframe concatenation ==")

    df2 = pd.read_csv("data/gray_75_unbalanced.csv", header=None, names=["sound", "acc1", "acc2"])
    print("\nDataframe description")
    print("\n", df2.describe())  # general description of the dataframe
    print("\n", df.head())  # print the first 5 lines

    result = pd.concat([df, df], axis=1) # concatenate 2 df (colum-wise)
    print("\nConcatenation results")
    print("\n", result.describe())
    print("\n", result.head())


    # convert dataframe to nparray 
    print("\n== Conversion to numpy ==")
    X_train = df.to_numpy()
    print("Shape: ", X_train.shape)
    print(X_train[0])
    print(X_train[1])

def loadData():
    labels=["balanced","unbalanced","electric_fault"]
    speedLabels=["75","80","85","90","95","100"]
    X = []
    y = []

    for i in range(len(labels)):
        for speedLabel in speedLabels:
            try:
                df = pd.read_csv(f"data/gray_{speedLabel}_{labels[i]}.csv", header=None, names=["sound", "acc1", "acc2"])
                window_size = 50
                for k in range(0, df.shape[0]-window_size*2, window_size):
                    ac1 = df[k:k+window_size]['sound']
                    ac2 = df[k:k+window_size]['acc1']
                    ac3 = df[k:k+window_size]['acc2']
                    X.append([min(ac1), min(ac2), min(ac3),
                              max(ac1), max(ac2), max(ac3),
                              ac1.mean(),ac2.mean(),ac3.mean(),
                              ac1.std(),ac2.std(),ac3.std()
                              ])
                    y.append(i)
            except Exception as e:
                print(e)

    X = np.array(X)
    y = np.array(y)

    return X, y

if __name__=="__main__":
    X,y = loadData()
    X_train,  X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=True, shuffle=True)

    #https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler.fit
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    model = Sequential()    #https://keras.io/api/models/sequential/

    print(model.summary)

    model.add(Dense(60, activation='relu',input_shape=(12,)))
    model.add(Dense(8, activation='relu', input_shape=(12,)))
    model.add(Dense(3, activation='softmax', input_shape=(12,)))

    print(model.summary()) #https://machinelearningmastery.com/visualize-deep-learning-neural-network-model-keras/#:~:text=%27sigmoid%27))-,Summarize%20Model,output%20shape%20of%20each%20layer.
    
    #https://keras.io/api/models/model_training_apis/
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    #print shape of X 
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)


    # fit the model -> https://keras.io/api/models/model_training_apis/
    history = model.fit(
        X_train,
        to_categorical(y_train),
        epochs=50,
        batch_size=64, #32 default
        validation_data=(X_test, to_categorical(y_test))
    )

    #Show graph 
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Graph')
    plt.ylim([0.5, 1])
    plt.show()

    # Evaluate the model.
    model.evaluate(X_test,to_categorical(y_test), verbose=2)