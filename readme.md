6th: 
```
import tensorflow as 􀆞
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
# Load MNIST dataset
(x_train, _), (x_test, _) = 􀆞.keras.datasets.mnist.load_data()
# Normalize and reshape
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
# -------------------------------
# Build Convolu􀆟onal Autoencoder
# -------------------------------
input_img = layers.Input(shape=(28, 28, 1))
# Encoder
x = layers.Conv2D(32, (3, 3), ac􀆟va􀆟on='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(16, (3, 3), ac􀆟va􀆟on='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
# Decoder
x = layers.Conv2D(16, (3, 3), ac􀆟va􀆟on='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(32, (3, 3), ac􀆟va􀆟on='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(1, (3, 3), ac􀆟va􀆟on='sigmoid', padding='same')(x)
# Model
autoencoder = models.Model(input_img, decoded)
# Compile
autoencoder.compile(op􀆟mizer='adam', loss='binary_crossentropy')
# Summary
autoencoder.summary()
# -------------------------------
# Train the Model
# -------------------------------
history = autoencoder.fit(
x_train, x_train,
epochs=10,
batch_size=128,
shuffle=True,
valida􀆟on_data=(x_test, x_test)
)
# -------------------------------
# Evaluate & Visualize Results
# -------------------------------
decoded_imgs = autoencoder.predict(x_test)
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
# Original
ax = plt.subplot(2, n, i + 1)
plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
plt.􀆟tle("Original")
plt.axis('off')
# Reconstructed
ax = plt.subplot(2, n, i + 1 + n)
plt.imshow(decoded_imgs[i].reshape(28, 28), cmap='gray')
plt.􀆟tle("Reconstructed")
plt.axis('off')
plt.show()
# -------------------------------
# Plot Loss Curve
# -------------------------------
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Valida􀆟on Loss')
plt.legend()
plt.􀆟tle("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()
```
