from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

mnist = tf.keras.datasets.mnist        # Load and prepare the MNIST dataset

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def create_model():
    # Build the structure of the model
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10)
    ])
    # this is so that for each example the model returns a vector of "logits" for each class.
    predictions = model(x_train[:1]).numpy()
    predictions

    tf.nn.softmax(predictions).numpy()    # This converts all the logits to probabilities

    # the loss function takes a vector of logits and a True index and returns a scalar loss for each example.
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss_fn(y_train[:1], predictions).numpy()

    # this compiles the model using the adam optimizer and the loss function specified
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])

    return model


model = create_model()

model.fit(x_train, y_train, epochs=13)   # adjusts the model parameters to minimize the loss
model.evaluate(x_test,  y_test, verbose=2) # checks the models performance

model.save('my_model.h5')   # saves the model



