import numpy as np

def rnn_forward(x, Wxh, Whh, Why, bh, by, h0):
    h = h0  # Initialize hidden state
    hs = []  # To store all hidden states
    ys = []  # To store all outputs
    
    for t in range(len(x)):
        xt = np.array([[x[t]]])  # Input at time t (make it a column vector)
        h = np.tanh(np.dot(Whh, h) + np.dot(Wxh, xt) + bh)  # Hidden state update
        y = np.dot(Why, h) + by  # Output calculation
        hs.append(h)  # Save hidden state
        ys.append(y)  # Save output
    
    return ys, hs

# Example usage:

# Input sequence
x = [1, 2, 3]

# Hyperparameters
input_size = 1  # Since x is a sequence of numbers
hidden_size = 4  # You can choose any size for hidden state
output_size = 1  # Output is a single number at each time step

# Random initialization of weights and biases
np.random.seed(0)  # For reproducibility
Wxh = np.random.randn(hidden_size, input_size) * 0.01
Whh = np.random.randn(hidden_size, hidden_size) * 0.01
Why = np.random.randn(output_size, hidden_size) * 0.01
bh = np.zeros((hidden_size, 1))
by = np.zeros((output_size, 1))
h0 = np.zeros((hidden_size, 1))

# Run the RNN forward function
ys, hs = rnn_forward(x, Wxh, Whh, Why, bh, by, h0)

# Print outputs at each time step
print("Outputs at each time step:")
for t, y in enumerate(ys):
    print(f"Time step {t+1}: y = {y.flatten()}")
