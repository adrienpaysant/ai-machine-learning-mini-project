import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
import numpy as np

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
    i=0
    dataFrame = []
    for label in labels:
        for speedLabel in speedLabels:
            dataFrame[i] = pd.read_csv(f"data/gray_{speedLabel}_{label}.csv", header=None, names=["sound", "acc1", "acc2"])
            i+=1
    return dataFrame

if __name__=="__main__":
    labels=["balanced","unbalanced","electric_fault"]
    speedLabels=["75","80","85","90","95","100"]
    dataFrame = loadData()
    window_size = 50
    X=[]
    y=[]
    for label in labels:
        for speedLabel in speedLabels:
            df = pd.read_csv(f"data/gray_{speedLabel}_{label}.csv", header=None, names=["sound", "acc1", "acc2"])
            for k in range(0, df.shape[0]-window_size*2, window_size):
                sound = df[k:k+window_size]['sound']
                acc1 = df[k:k+window_size]['acc1']
                acc2 = df[k:k+window_size]['acc2']
                X.append([min(sound),min(acc1),min(acc2),max(sound),max(acc1),max(acc2),sound.mean(),acc1.mean(),acc2.mean(),sound.std(),acc1.std(),acc2.std()])
                y.append(labels.index(label))
    X = np.array(X)
    y = np.arrray(y)
    X_train,  X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=True, shuffle=True)