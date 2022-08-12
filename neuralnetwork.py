import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

def load_coffee_data():
    rng = np.random.default_rng(2)
    X = rng.random(400).reshape(-1,2)
    X[:,1] = X[:,1] * 4 + 11.5          # 12-15 min is best
    X[:,0] = X[:,0] * (285-150) + 150  # 350-500 F (175-260 C) is best
    Y = np.zeros(len(X))
    
    i=0
    for t,d in X:
        y = -3/(260-175)*t + 21
        if (t > 175 and t < 260 and d > 12 and d < 15 and d<=y ):
            Y[i] = 1
        else:
            Y[i] = 0
        i += 1

    return (X, Y.reshape(-1,1))

train_x, train_y = load_coffee_data()

norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(train_x)

try:
    model = tf.keras.models.load_model("savedNN")
except:

    Xn = norm_l(train_x)

    x_train = np.tile(Xn,(1000,1))
    y_train= np.tile(train_y,(1000,1))

    tf.random.set_seed(1234)  # applied to achieve consistent results

    model = Sequential(
        [
            tf.keras.Input(shape=(2,)),
            Dense(3, activation='sigmoid', name = 'layer1'),
            Dense(1, activation='sigmoid', name = 'layer2')
        ]
    )

    model.compile(
        loss = tf.keras.losses.BinaryCrossentropy(),
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
    )

    model.fit(
        x_train, y_train,            
        epochs=10,
    )

    model.save("savedNN")

def neural_network_predict(x_test):

    X_testn = norm_l(x_test)
    predictions = model.predict(X_testn)

    yhat = np.zeros_like(predictions)
    for i in range(len(predictions)):
        if predictions[i] >= 0.5:
            yhat[i] = 1
        else:
            yhat[i] = 0
    return yhat

# x_test = np.array([[200, 13.9]])
# neural_network_predict(x_test)
