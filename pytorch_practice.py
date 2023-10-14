import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm


# Generate some example data
# y = 2x + 1
data = np.random.rand(100, 1)  # Input data
labels = 2 * data + 1 + np.random.randn(100, 1) * 0.1  # Linear relationship with some noise

# Split the data into training and testing sets
split_ratio = 0.8
split_index = int(data.shape[0] * split_ratio)
x_train, x_test = data[:split_index], data[split_index:]
y_train, y_test = labels[:split_index], labels[split_index:]

# trying out
# Convert NumPy arrays to PyTorch tensors
x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train)
x_test = torch.FloatTensor(x_test)
y_test = torch.FloatTensor(y_test)

# Define a custom neural network model in PyTorch
# class SimpleModel(nn.Module):
#     def __init__(self):
#         super(SimpleModel, self).__init()
#         self.fc = nn.Linear(1, 1)  # Input size: 1, Output size: 1

#     def forward(self, x):
#         return self.fc(x)

# model = SimpleModel()

# Define the model directly without a custom class
# Define a more complex neural network model
model = nn.Sequential(
    nn.Linear(1, 64),  # Input size: 1, Output size: 64 (1st hidden layer)
    nn.ReLU(),         # Activation function
    nn.Linear(64, 32),  # Input size: 64, Output size: 32 (2nd hidden layer)
    nn.ReLU(),         # Activation function
    nn.Linear(32, 1)    # Input size: 32, Output size: 1 (output layer)
)


# Define loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent optimizer


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

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

    # Calculate loss and optionally print it
    train_loss = criterion(model(x_train), y_train).item()
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}')

# Evaluate the model on the test data
test_loss = criterion(model(x_test), y_test).item()
print("Test loss:", test_loss)

# save trained model
torch.save(model.state_dict(), "trained_model.pth")
# Assuming 'model' is the same architecture as when it was saved
# model.load_state_dict(torch.load("trained_model.pth"))
# model.eval()  # Set the model to evaluation mode


# Use the model to make predictions on new data
new_data = torch.FloatTensor([[0.7], [0.8]])
predictions = model(new_data)

print("Predictions for new data:")
print(predictions)