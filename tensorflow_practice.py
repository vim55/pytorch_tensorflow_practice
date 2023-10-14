import tensorflow as tf
import numpy as np
from tqdm import tqdm

# Generate some example data
data = np.random.rand(100, 1)  # Input data
labels = 2 * data + 1 + np.random.randn(100, 1) * 0.1  # Linear relationship with some noise

# Split the data into training and testing sets
split_ratio = 0.8
split_index = int(data.shape[0] * split_ratio)
x_train, x_test = data[:split_index], data[split_index:]
y_train, y_test = labels[:split_index], labels[split_index:]

# # Create a simple neural network model
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(1, input_shape=(1,), activation='linear')
# ])


# Create a more complex neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, input_shape=(1,), activation='relu'),  # 1st hidden layer
    tf.keras.layers.Dense(32, activation='relu'),  # 2nd hidden layer
    tf.keras.layers.Dense(32, activation='relu'),  # 3rd hidden layer
    tf.keras.layers.Dense(1, activation='linear')  # Output layer
])

# Compile the model
model.compile(optimizer='sgd', loss='mean_squared_error')

# Number of epochs and batch size
num_epochs = 50
batch_size = 16

# Training loop with tqdm
for epoch in tqdm(range(num_epochs), desc="Training"):
    # Shuffle the training data
    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    x_train_shuffled = x_train[indices]
    y_train_shuffled = y_train[indices]

    for i in range(0, len(x_train_shuffled), batch_size):
        x_batch = x_train_shuffled[i:i + batch_size]
        y_batch = y_train_shuffled[i:i + batch_size]

        # Train on the batch
        model.train_on_batch(x_batch, y_batch)

    # Evaluate the model on training data
    train_loss = model.evaluate(x_train, y_train, verbose=0)
    train_acc = 1 - train_loss  # Accuracy is not commonly used for regression tasks, but you can calculate it this way

    # Optionally, print the loss and accuracy at the end of each epoch
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}')

# Evaluate the model on the test data
test_loss = model.evaluate(x_test, y_test)
test_acc = 1 - test_loss  # Calculate accuracy in the same way

print("Test loss:", test_loss)
print("Test accuracy:", test_acc)

# save trained model
model.save("trained_model.h5")
# Load the model from the SavedModel format
# loaded_model = tf.keras.models.load_model("trained_model.h5")

# Use the model to make predictions on new data
new_data = np.array([[0.7], [0.8]])
predictions = model.predict(new_data)
print("Predictions for new data:")
print(predictions)