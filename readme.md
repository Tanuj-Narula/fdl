2nd :
```
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

# Generate sample classification data
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ANN Model
model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')   # Binary classification
])

# Compile model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)
```

3rd: cnn for mri
```
import tensorflow as tf
from tensorflow.keras import layers, models
train_path = "/kaggle/input/brain-tumor-mri-dataset/Training"
val_path = "/kaggle/input/brain-tumor-mri-dataset/Tes􀆟ng"
# Load MRI dataset (folder structure required)
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
train_path, image_size=(128, 128), batch_size=16
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
val_path, image_size=(128, 128), batch_size=16
)
class_names = train_ds.class_names
print("Classes:", class_names)
# Normalize images
train_ds = train_ds.map(lambda x, y: (x/255.0, y))
val_ds = val_ds.map(lambda x, y: (x/255.0, y))
# Build a small CNN
model = models.Sequen􀆟al([
layers.Conv2D(32, (3,3), ac􀆟va􀆟on='relu', input_shape=(128,128,3)),
layers.MaxPooling2D(),
layers.Conv2D(64, (3,3), ac􀆟va􀆟on='relu'),
layers.MaxPooling2D(),
layers.Fla􀆩en(),
layers.Dense(64, ac􀆟va􀆟on='relu'),
layers.Dense(len(class_names), ac􀆟va􀆟on='so􀅌max')
])
model.compile(op􀆟mizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])
# Train
model.fit(train_ds, epochs=10, valida􀆟on_data=val_ds)
# Evaluate
model.evaluate(val_ds)
```
4th: autoencoder for dimensionality reduction
```
import numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.reshape((len(x_train), 784)).astype("float32") / 255
x_test = x_test.reshape((len(x_test), 784)).astype("float32") / 255
input_dim = 784
latent_dim = 16
inputs = Input(shape=(input_dim,))
encoded = Dense(128, activation="relu")(inputs)
encoded = Dense(64, activation="relu")(encoded)
latent = Dense(latent_dim, activation="linear")(encoded)
decoded = Dense(64, activation="relu")(latent)
decoded = Dense(128, activation="relu")(decoded)
outputs = Dense(input_dim, activation="sigmoid")(decoded)
autoencoder = Model(inputs, outputs)
autoencoder.compile(optimizer="adam", loss="mse")
encoder = Model(inputs, latent)
autoencoder.fit(
x_train, x_train,
epochs=20,
batch_size=256,
shuffle=True,
validation_data=(x_test, x_test),
verbose=2
)
compressed_features = encoder.predict(x_test)
print("Original dimension:", x_test.shape[1])
print("Reduced dimension:", compressed_features.shape[1])
print("Compressed feature shape:", compressed_features.shape)
```
5th: autoencoder for image
```
import numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.reshape(-1, 784).astype("float32") / 255
x_test = x_test.reshape(-1, 784).astype("float32") / 255
input_dim = 784
latent_dim = 32
input_img = Input(shape=(input_dim,))
encoded = Dense(128, activation="relu")(input_img)
latent = Dense(latent_dim, activation="relu")(encoded)
decoded = Dense(128, activation="relu")(latent)
output = Dense(input_dim, activation="sigmoid")(decoded)
autoencoder = Model(input_img, output)
autoencoder.compile(optimizer="adam", loss="mse")
encoder = Model(input_img, latent)
autoencoder.fit(
x_train, x_train,
epochs=10,
batch_size=256,
shuffle=True,
validation_data=(x_test, x_test)
)
compressed = encoder.predict(x_test)
print("Original shape:", x_test.shape)
print("Compressed shape:", compressed.shape)
decoded_imgs = autoencoder.predict(x_test)
n = 5
plt.figure(figsize=(10, 4))
for i in range(n):
ax = plt.subplot(2, n, i + 1)
plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
plt.title("Original")
plt.axis("off")
ax = plt.subplot(2, n, i + 1 + n)
plt.imshow(decoded_imgs[i].reshape(28, 28), cmap="gray")
plt.title("Reconstructed")
plt.axis("off")
plt.show()
```

6th: improving autoencoders performance 
```
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
# Load MNIST dataset
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
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
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
# Decoder
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
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
  plt.title("Original")
  plt.axis('off')
  # Reconstructed
  ax = plt.subplot(2, n, i + 1 + n)
  plt.imshow(decoded_imgs[i].reshape(28, 28), cmap='gray')
  plt.title("Reconstructed")
  plt.axis('off')
  plt.show()
# -------------------------------
# Plot Loss Curve
# -------------------------------
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Valida􀆟on Loss')
plt.legend()
plt.title("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()
```
7th: stock
```
# 1. Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# ============================================================
# 2. Load Dataset
# ============================================================
# Make sure your CSV has at least a 'Close' column
data = pd.read_csv("stock_prices.csv")

# Keep only 'Close' price
data = data[['Close']]

# Convert to numpy array
dataset = data.values

# ============================================================
# 3. Normalize Data
# ============================================================
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# ============================================================
# 4. Create Time-Series Data
# ============================================================
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60
X, y = create_dataset(scaled_data, time_step)

# ============================================================
# 5. Reshape Input for LSTM
# ============================================================
X = X.reshape(X.shape[0], X.shape[1], 1)

# ============================================================
# 6. Train-Test Split
# ============================================================
train_size = int(len(X) * 0.8)

X_train = X[:train_size]
X_test = X[train_size:]

y_train = y[:train_size]
y_test = y[train_size:]

# ============================================================
# 7. Build LSTM Model
# ============================================================
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# ============================================================
# 8. Train Model
# ============================================================
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# ============================================================
# 9. Make Predictions
# ============================================================
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform predictions
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Inverse transform actual values
y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# ============================================================
# 10. Plot Results
# ============================================================
plt.figure(figsize=(12, 6))

plt.plot(y_test_actual, label="Actual Price")
plt.plot(test_predict, label="Predicted Price")

plt.title("Stock Price Prediction using LSTM")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()
plt.show()
# ============================================================
# 11. Predict Future Value (Next Day)
# ============================================================
last_60_days = scaled_data[-60:]
last_60_days = last_60_days.reshape(1, 60, 1)
next_day_prediction = model.predict(last_60_days)
next_day_prediction = scaler.inverse_transform(next_day_prediction)
print("Next Day Predicted Price:", next_day_prediction[0][0])
```
8th: weather
```
# 1. Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# ============================================================
# 2. Load Dataset
# ============================================================
# Dataset should contain columns like:
# Date, Temperature, Humidity, Pressure, WindSpeed

data = pd.read_csv("weather_data.csv")

# Select relevant features
features = ['Temperature', 'Humidity', 'Pressure', 'WindSpeed']
data = data[features]

# Convert to numpy
dataset = data.values

# ============================================================
# 3. Normalize Data
# ============================================================
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# ============================================================
# 4. Create Time-Series Data
# ============================================================
def create_dataset(data, time_step=30):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step)])
        y.append(data[i + time_step][0])  # Predict Temperature
    return np.array(X), np.array(y)

time_step = 30
X, y = create_dataset(scaled_data, time_step)

# ============================================================
# 5. Train-Test Split
# ============================================================
train_size = int(len(X) * 0.8)

X_train = X[:train_size]
X_test = X[train_size:]

y_train = y[:train_size]
y_test = y[train_size:]

# ============================================================
# 6. Build LSTM Model
# ============================================================
model = Sequential()

model.add(LSTM(units=64, return_sequences=True, input_shape=(time_step, len(features))))
model.add(Dropout(0.2))

model.add(LSTM(units=64))
model.add(Dropout(0.2))

model.add(Dense(1))  # Predict Temperature

model.compile(optimizer='adam', loss='mean_squared_error')

# ============================================================
# 7. Train Model
# ============================================================
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

# ============================================================
# 8. Predictions
# ============================================================
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# ============================================================
# 9. Inverse Scaling
# ============================================================
# Create dummy arrays to inverse transform only temperature
def inverse_transform(predictions):
    dummy = np.zeros((len(predictions), len(features)))
    dummy[:, 0] = predictions[:, 0]
    return scaler.inverse_transform(dummy)[:, 0]

train_predict = inverse_transform(train_predict)
test_predict = inverse_transform(test_predict)

y_train_actual = inverse_transform(y_train.reshape(-1, 1))
y_test_actual = inverse_transform(y_test.reshape(-1, 1))

# ============================================================
# 10. Plot Results
# ============================================================
plt.figure(figsize=(12,6))

plt.plot(y_test_actual, label="Actual Temperature")
plt.plot(test_predict, label="Predicted Temperature")

plt.title("Weather Prediction using LSTM")
plt.xlabel("Time")
plt.ylabel("Temperature")

plt.legend()
plt.show()

# ============================================================
# 11. Predict Future Weather (Next Day)
# ============================================================
last_sequence = scaled_data[-time_step:]
last_sequence = last_sequence.reshape(1, time_step, len(features))

future_prediction = model.predict(last_sequence)
future_temp = inverse_transform(future_prediction)

print("Next Day Predicted Temperature:", future_temp[0])
```
