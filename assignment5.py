from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import MeanSquaredError
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, KFold
from tensorflow.keras import regularizers
import warnings
warnings.filterwarnings("ignore")

# 1. Use the training set, validation set, and test set from Assignment 3
olivetti_faces = fetch_olivetti_faces()

X = olivetti_faces.data  #  flattened 1D array format
y = olivetti_faces.target  # The target labels (person identifier)

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=51)
for train_idx, temp_idx in sss.split(X, y):
    X_train, X_temp = X[train_idx], X[temp_idx]
    y_train, y_temp = y[train_idx], y[temp_idx]


sss_val_test = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=51)
for val_idx, test_idx in sss_val_test.split(X_temp, y_temp):
    X_val, X_test = X_temp[val_idx], X_temp[test_idx]
    y_val, y_test = y_temp[val_idx], y_temp[test_idx]

# 2. Use PCA preserving 99% of the variance to reduce the dataset’s dimensionality as in Assignment 4 (Gaussian Mixture Models) and use it to train the autoencoder

pca = PCA(n_components=0.99, random_state=51)
X_train_pca = pca.fit_transform(X_train)

# Step 2.2: Transform validation and test sets using the trained PCA model
X_val_pca = pca.transform(X_val)
X_test_pca = pca.transform(X_test)

n_components = pca.n_components_
print(f'Number of PCA components: {n_components}')
X_train_pca.shape

# 3. Define an autoencoder
# a) Use k-fold cross validation to fine tune the model’s learning rate and hyperparameter of the regularizer.  Due to the long training requirements, for the number of hidden units, 
# try two or three different values for each hidden layer

# Parameters for tuning
learning_rates = [0.0001, 0.001, 0.01]
regularizers_list = [0.0001, 0.001, 0.01]
k = 5  # Number of folds for cross-validation
best_params = {}
best_val_loss = np.inf

for lr in learning_rates:
    for reg in regularizers_list:
        print(f"Testing with learning rate {lr} and regularizer {reg}")
        kfold = KFold(n_splits=k, shuffle=True, random_state=51)
        fold_losses = []

        for train_index, val_index in kfold.split(X_train_pca):
            # Split the training data into k-fold training and validation sets
            X_ktrain, X_kval = X_train_pca[train_index], X_train_pca[val_index]

            input_size = X_train_pca.shape[1] 

            # Input layer
            input_img = Input(shape=(input_size,))

            # Use the L2 regularizer with the current regularizer strength
            regularizer = regularizers.l2(reg)  # Apply the correct method from Keras regularizers

            # Encoder with regularizer applied
            encoded = Dense(256, activation='relu', kernel_regularizer=regularizer)(input_img)
            encoded = Dense(128, activation='relu', kernel_regularizer=regularizer)(encoded)  # Hidden layer 1
            encoded = Dense(64, activation='relu', kernel_regularizer=regularizer)(encoded)  # Central layer (bottleneck)

            # Decoder with regularizer applied
            decoded = Dense(128, activation='relu', kernel_regularizer=regularizer)(encoded)  # Hidden layer 2
            decoded = Dense(256, activation='relu', kernel_regularizer=regularizer)(decoded)
            output = Dense(input_size, activation='sigmoid')(decoded)

            # Build and compile the autoencoder model
            autoencoder = Model(input_img, output)
            autoencoder.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')

            # Train the model on the k-fold split
            history = autoencoder.fit(X_ktrain, X_ktrain, epochs=10, batch_size=16, validation_data=(X_kval, X_kval), verbose=0)
            fold_losses.append(history.history['val_loss'][-1])

        avg_val_loss = np.mean(fold_losses)
        print(f"Average validation loss for learning rate {lr} and regularizer {reg}: {avg_val_loss}")

        # Track the best hyperparameters
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_params = {'learning_rate': lr, 'regularizer': reg}

print(f"Best parameters: {best_params}, with validation loss: {best_val_loss}")

# 4. Run the best model with the test set and display the original image and the reconstructed image

noise_factor = 0.4
x_train_noisy = X_train_pca + noise_factor *np.random.normal(size=X_train_pca.shape)
x_test_noisy = X_test_pca + noise_factor *np.random.normal(size=X_test_pca.shape)

x_train_noisy=np.clip(x_train_noisy,0,1)
x_test_noisy=np.clip(x_test_noisy,0,1)

# Train the final model with the best hyperparameters on the entire training and validation set
final_model = Model(input_img, output)
final_model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']), loss='mean_squared_error')
final_model.fit(np.concatenate((X_train_pca, X_val_pca)), np.concatenate((X_train_pca, X_val_pca)), epochs=50, batch_size=32)

# Reconstruct images from the test set
reconstructed_imgs = final_model.predict(X_test_pca)

n = 10
plt.figure(figsize=(20, 6))

for i in range(n):
    # Original Image
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(pca.inverse_transform(X_test_pca[i]).reshape(64, 64), cmap="gray")
    plt.title("Original")
    plt.axis("off")

    # Noisy Image
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(pca.inverse_transform(x_test_noisy[i]).reshape(64, 64), cmap="gray")
    plt.title("Noisy")
    plt.axis("off")

    # Reconstructed Image
    ax = plt.subplot(3, n, i + 1 + 2 * n)
    plt.imshow(pca.inverse_transform(reconstructed_imgs[i]).reshape(64, 64), cmap="gray")
    plt.title("Reconstructed")
    plt.axis("off")

plt.show()


