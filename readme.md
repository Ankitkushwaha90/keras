- Here is a collection of Keras code examples that span basic concepts to advanced architectures, designed to help you learn how to use the Keras API effectively.

### 1. Basic Neural Network with Keras (Sequential API)
```python
import tensorflow as tf
from tensorflow.keras import layers

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize the data
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define a simple feedforward neural network
model = tf.keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5)

# Evaluate the model
model.evaluate(x_test, y_test)
```
2. Convolutional Neural Network (CNN) with Keras
```python
import tensorflow as tf
from tensorflow.keras import layers

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize the data
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define the CNN model
model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Evaluate the model
model.evaluate(x_test, y_test)
```
### 3. Recurrent Neural Network (RNN) for Text Classification (IMDb)
```python
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb

# Load the IMDb dataset
max_features = 10000
maxlen = 100

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# Preprocess the data by padding sequences
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

# Define the RNN model
model = tf.keras.Sequential([
    layers.Embedding(max_features, 128),
    layers.SimpleRNN(128),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Evaluate the model
model.evaluate(x_test, y_test)
```
### 4. LSTM (Long Short-Term Memory) Network for Text Classification
```python
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb

# Load the IMDb dataset
max_features = 10000
maxlen = 100

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# Preprocess the data by padding sequences
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

# Define the LSTM model
model = tf.keras.Sequential([
    layers.Embedding(max_features, 128),
    layers.LSTM(128),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Evaluate the model
model.evaluate(x_test, y_test)
```
### 5. Transfer Learning with Pre-trained Models (VGG16)
```python
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models

# Load the pre-trained VGG16 model without the top layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model
base_model.trainable = False

# Add new layers for classification
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1024, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Example data loading (use real datasets)
# x_train, y_train = ...
# x_test, y_test = ...

# Train the model
# model.fit(x_train, y_train, epochs=5)

# Evaluate the model
# model.evaluate(x_test, y_test)
```
### 6. Autoencoder for Data Compression
```python
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Generate dummy data (use real-world data in practice)
data = np.random.random((1000, 32))

# Define the autoencoder
input_data = layers.Input(shape=(32,))
encoded = layers.Dense(16, activation='relu')(input_data)
decoded = layers.Dense(32, activation='sigmoid')(encoded)

autoencoder = models.Model(input_data, decoded)

# Compile the autoencoder
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the autoencoder
autoencoder.fit(data, data, epochs=50, batch_size=256)

# Use the encoder part
encoder = models.Model(input_data, encoded)
encoded_data = encoder.predict(data)
print("Encoded data shape:", encoded_data.shape)
```
### 7. Functional API for Complex Models
```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Functional API for a complex neural network with shared layers

input_1 = layers.Input(shape=(32,))
input_2 = layers.Input(shape=(32,))

shared_layer = layers.Dense(64, activation='relu')

x1 = shared_layer(input_1)
x2 = shared_layer(input_2)

merged = layers.concatenate([x1, x2])
output = layers.Dense(1, activation='sigmoid')(merged)

model = models.Model(inputs=[input_1, input_2], outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy')

# Example data (use real datasets)
import numpy as np
data_1 = np.random.random((1000, 32))
data_2 = np.random.random((1000, 32))
labels = np.random.randint(2, size=(1000, 1))

# Train the model
model.fit([data_1, data_2], labels, epochs=10, batch_size=32)
```
### 8. Custom Loss Function with Keras
```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Custom loss function
def custom_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Define a simple model
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(32,)),
    layers.Dense(1)
])

# Compile the model with the custom loss function
model.compile(optimizer='adam', loss=custom_loss)

# Example data
import numpy as np
data = np.random.random((1000, 32))
labels = np.random.random((1000, 1))

# Train the model
model.fit(data, labels, epochs=10, batch_size=32)
```
### 9. Saving and Loading Models
```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Define a simple model
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(32,)),
    layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model (use real data)
# model.fit(...)

# Save the model
model.save('my_model.h5')

# Load the model
new_model = models.load_model('my_model.h5')
```
### 10. Callbacks (Early Stopping, Model Checkpoints)
```python
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Define a simple model
model = tf.keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(32,)),
    layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Early stopping and model checkpoint callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)

# Train the model (use real data)
# model.fit(x_train, y_train, validation_data=(x_val, y_val),
#           epochs=50, callbacks=[early_stopping, model_checkpoint])
```
These examples cover a wide range of applications and techniques in Keras, from basic feedforward networks to more complex architectures like CNNs, RNNs, LSTMs, and autoencoders, as well as custom loss functions, transfer learning, and model saving/loading.
