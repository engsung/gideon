import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# problems with loading data with tf, issue resolved with this
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

# getting real handwritten digits
fashion = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion.load_data()

# scale down the data to make it easier to compute
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1) # we dont scale down the y values bc they're 0-9

# create basic neural network
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)  # the model will see the data three times

loss, accuracy = model.evaluate(x_test, y_test)
print(accuracy)
print(loss)

model.save('digits.model')

for x in range(1, 3):
    img = cv.imread(f'{x}.png')[:,:,0]
    img = np.array([img])
    prediction = model.predict(img)  # predict our img
    print(f'The result is probably: {np.argmax(prediction)}')  # give us the index of the highest value
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()
