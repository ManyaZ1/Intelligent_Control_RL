# %% [markdown]
# # Intelligent Control (ECE-DK807)
# 
# ## Electrical and Computer Engineering Department, University of Patras Greece
# 
# **Instructor:** Konstantinos Chazilygeroudis (costashatz@upatras.gr)
# 
# ## Lab 3
# 
# ### Gradient Descent
# 
# **Adam Optimizer**
# 
# 1. Initialize $\boldsymbol{m}_0 = \boldsymbol{v}_0 = \boldsymbol{0}, \beta_1,\beta_2\in[0,1), \epsilon>0$
# 2. $\boldsymbol{g}_{k+1} = \nabla_{\boldsymbol{x}}f(\boldsymbol{x}_k)$
# 3. $\boldsymbol{m}_{k+1} = \beta_1\boldsymbol{m}_k + (1 - \beta_1)\boldsymbol{g}_{k+1}$
# 4. $\boldsymbol{v}_{k+1} = \beta_2\boldsymbol{v}_k + (1 - \beta_2)\boldsymbol{g}^2_{k+1}$
# 5. $\boldsymbol{x}_{k+1} = \boldsymbol{x}_k - \alpha\frac{\frac{\boldsymbol{m}_{k+1}}{1 - \beta_1^{k+1}}}{\sqrt{\frac{\boldsymbol{v}_{k+1}}{1 - \beta_2^{k+1}}}+\epsilon}$
# 6. $k=k+1$ and back to step 2 until we converge
# 
# Let's implement it!

# %%
# Let's first import modules
import numpy as np # Linear Algebra
import matplotlib.pyplot as plt # Plotting
import copy

# %%
# First we initialize values
adamM = 0.
adamV = 0.
adamB1n = 0.9
adamB2n = 0.999

def reset_adam():
    global adamM, adamV, adamB1n, adamB2n
    adamM = 0.
    adamV = 0.
    adamB1n = 0.9
    adamB2n = 0.999

# Adam step
# x is the current iterate, x_k
# df is the derivative/gradient of the function we want to optimize
# a is the learning rate
# b1 is the β_1 parameter
# b2 is the β_2 parameter
# eps is the ε parameter
def adam_step(x, df, a = 0.001, b1 = 0.9, b2 = 0.999, eps = 1e-8):
    global adamM, adamV, adamB1n, adamB2n
    ### TO-DO: Implement the steps (1-5) of the Adam algorithm. You need to return x_{k+1}
    ### adamM, adamV global variables are used to store the current m_k and v_k values (updated at each iteration)
    ### adamB1n, adamB2n global variables are used to store β_1^{k+1} and β_2^{k+1} (to avoid having to know the number of iterations)
    ### ANSWER: Insert code here
    gk1=df(x)
    adamM=adamB1n*adamM+(1-adamB1n)*gk1
    adamV=adamB2n*adamV+(1-adamB2n)*gk1*gk1  
    adamB1n *= b1   #beta1(t) = beta1^t
    adamB2n *= b2
    x=x-a*((adamM /(1-adamB1n))/(np.sqrt(adamV/(1-adamB2n))+eps))

    return x
    ### END of ANSWER

# %%
# First reset Adam
reset_adam()

def simple_f(x):
    return np.square(x)

def simple_df(x):
    return 2. * x

x = 10.
lr = 1e-3
x = adam_step(x, simple_df, lr)
assert(np.isclose(x, 9.999, rtol=1e-4))
x = adam_step(x, simple_df, lr)
assert(np.isclose(x, 9.998, rtol=1e-4))
x = adam_step(x, simple_df, lr)
assert(np.isclose(x, 9.997, rtol=1e-4))
x = adam_step(x, simple_df, lr)
assert(np.isclose(x, 9.996, rtol=1e-4))


reset_adam()

# %%
# Let's optimize the Rastrigin function
def f(x):
    ### TO-DO: Implement the Rastrigin function for any dimension as defined in https://www.sfu.ca/~ssurjano/rastr.html
    ### we assume that the input is of the form (dimensions, batch_size)
    ### f(x) should return a (batch_size,) numpy array
    ### ANSWER: Insert code here
    x = np.asarray(x)  # Ensure input is a NumPy array
    d, batch_size = x.shape  # Get dimensions and batch size
    summation = np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=0)  # Sum across dimensions
    y = 10 * d + summation  # Compute function value
    return y  # Return function values and batch size
    ### END of ANSWER

def df(x):
    ### TO-DO: Write the derivative of the Rastrigin function
    ### we assume that the input is of the form (dimensions, batch_size)
    ### df should return a (dimensions, batch_size) numpy array
    ### ANSWER: Insert code here
    x = np.asarray(x)  # Ensure input is a NumPy array
    grad = sum(2 * x + 20 * np.pi * np.sin(2 * np.pi * x) ) # Compute derivative
    return grad
    ### END of ANSWER

# %%
x = np.array([[5., 4., 3., 2., 1., 0., -1., -2., -3., -4., -5.]])
assert(np.isclose(f(x), np.array([25., 16., 9., 4., 1., 0., 1., 4., 9., 16., 25.])).all())


# %%
x = np.array([[5., 4., 3., 2., 1., 0., -1., -2., -3., -4., -5.]])
assert(np.isclose(df(x), np.array([[10., 8., 6., 4., 2., 0., -2., -4., -6., -8., -10.]])).all())


# %%
# Let's plot the function
x = np.linspace(-5.12, 5.12, 1000)

plt.close() # close previous
fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(x, f(x.reshape((1, -1))));

# %%
# Let's optimize
# Initial point!
x_init = np.array([[1.5]])

ax.plot(x_init, f(x_init), 'rx')

# Optimization!
x_new = np.copy(x_init)

fig # show figure again with updated point(s)

# %%
# Let's do a few steps of Adam
lr = 0.01 # learning rate
for i in range(50):
    x_new = adam_step(x_new, df, lr)
    ax.plot(x_new, f(x_new), 'rx')

print(x_new)

fig # show figure again with updated point(s)

# %% [markdown]
# ### Neural Networks
# 
# Now let's learn a nice neural network to approximate our Rastrigin function!
# 
# First let's create a dataset. For this we sample $K$ points randomly in the input space of our function. The Rastrigin function is defined in $[-5.12, 5.12]$. So, let's do this:

# %%
# Number of points
K = 2000

np.random.seed(2)

### TO-DO: Sample random inputs in [-5.12, 5.12]. We need to sample a dim x K matrix. Store the result in a variable named 'Xn'
### ANSWER: Insert code here
Xn=np.random.uniform(-5.12, 5.12, (1, K))
### END of ANSWER

### TO-DO: Evaluate the function in the sampled points. Store the result in a output_dim x K matrix, and a variable named 'Yn'
### ANSWER: Insert code here
Yn=f(Xn).reshape(1,K)
### END of ANSWER

# %%
assert(Xn.shape == (1, 2000))
assert(Yn.shape == (1, 2000))

assert(np.isclose(Xn[0, 0], -0.655, rtol=1e-3))
assert(np.isclose(Yn[0, 0], 16.029, rtol=1e-3))


# %%
# Let's plot our dataset!
plt.close() # close previous
fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(Xn, Yn, 'b.');

# %%
# Now let's import pytorch
import torch # For neural networks and automatic differentiation
torch.set_num_threads(1)

# Convert numpy arrays to tensors
# We transpose the arrays because torch assumes (batch_size, dimension)
X = torch.Tensor(Xn.T)
Y = torch.Tensor(Yn.T)

# %% [markdown]
# Now we need to make our neural network! Let's create a feedforward neural network with 2 hidden layers (32 and 64 units respectively), let's use `tanh` as the activation functions for the hidden layers, and no activation for the output layer!

# %%
# MLP Model
model = torch.nn.Sequential(
    ### TO-DO: Fill up the model with layers and activation functions as described above!
    ### ANSWER: Insert code here
    torch.nn.Linear(1, 32),  # Input: 1 -> Hidden layer 1 (32 units)
    torch.nn.Tanh(),
    torch.nn.Linear(32, 64),  # Hidden layer 1 -> Hidden layer 2 (64 units)
    torch.nn.Tanh(),
    torch.nn.Linear(64, 1)
    ### END of ANSWER
)

# %%
params = list(model.named_parameters())
assert(params[0][1].size() == torch.Size([32, 1]))
assert(params[4][1].size() == torch.Size([1, 64]))


# %% [markdown]
# Now we need to make a learning loop:
# 
# 1. Compute model prediction $y_{pred}$ for the input $X$
# 2. Compute loss given the actual labels $Y$
# 3. Zero the gradients of the optimizer
# 4. Automatically compute gradients (`backward` pass)
# 5. Perform one step of the optimizer
# 
# Let's implement those with the Adam optimizer!

# %%
# Let's first create a function that computes the predictions of the model
def prediction(model, x):
    ### TO-DO: Return the prediction of the model when given as input X
    ### ANSWER: Insert code here
    return model.forward(x)
    ### END of ANSWER

# %%


# %%
# Now let's create a function that computes the mean squared error loss
def loss(y_pred, y_true):
    ### TO-DO: Compute and return the mean squared error. Remember that torch assumes (batch_size, dimension) for batched operations.
    ### ANSWER: Insert code here
    mse = torch.mean((y_pred - y_true) ** 2)
    return mse.unsqueeze(0)  # Ensures output shape is (1,)
    ### END of ANSWER

# %%
Y_perturbed = Y + 2.
assert(torch.isclose(loss(Y_perturbed, Y), torch.ones(Y.shape[1]) * 4.).all())
Y_perturbed = Y - 2.
assert(np.isclose(loss(Y_perturbed, Y), torch.ones(Y.shape[1]) * 4.).all())

assert(loss(Y_perturbed, Y).size() == torch.Size((1,)))

# %%
# Now let's create a function that performs steps 3, 4 and 5 of the above procedure. Assume that the loss value is given as input, and the optimizer is given as input
def update_optimizer(optimizer, loss_value):
    ### ANSWER: Insert code here
    optimizer.zero_grad()  # Step 3: Zero gradients
    loss_value.backward()  # Step 4: Compute gradients
    optimizer.step() 
    ### END of ANSWER

# %%
tmp_model = copy.deepcopy(model)

# Initialize weights of the model
def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        m.weight.data.fill_(1.)
        m.bias.data.fill_(0.01)

tmp_model.apply(init_weights)

lr = 1e-3
optim = torch.optim.Adam(tmp_model.parameters(), lr=lr) # Adam optimizer

myloss = loss(prediction(tmp_model, X), Y)
assert(update_optimizer(optim, myloss) == None)
myloss = loss(prediction(tmp_model, X), Y)
assert(update_optimizer(optim, myloss) == None)

params = list(tmp_model.parameters())
assert(torch.isclose(params[0].grad, torch.ones_like(params[0].grad) * 0.9356, rtol=1e-4).all())
assert(torch.isclose(params[1].grad, torch.ones_like(params[1].grad) * 0.5833, rtol=1e-4).all())
assert(torch.isclose(params[2].grad, torch.ones_like(params[2].grad) * 0.0147, rtol=1e-4, atol=1e-4).all())
assert(torch.isclose(params[3].grad, torch.ones_like(params[3].grad) * 0.009, rtol=1e-4, atol=1e-4).all())
assert(torch.isclose(params[4].grad, torch.ones_like(params[4].grad) * 128.829, rtol=1e-4).all())
assert(torch.isclose(params[5].grad, torch.ones_like(params[5].grad) * (-44.53), rtol=1e-4).all())

# %%
# Let's create the optimizer
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # Adam optimizer

for t in range(20000):
    y_pred = prediction(model, X)

    myloss = loss(y_pred, Y)
    if t == 0 or (t + 1) % 200 == 0:
        print(t+1, myloss.item() / K)

    update_optimizer(optimizer, myloss)

# %%
# Let's plot the function
x = np.linspace(-5.12, 5.12, 1000)

plt.close() # close previous
fig = plt.figure()
ax = fig.add_subplot(111)

# ax.plot(X, Y, '.')
ax.plot(x, f(x.reshape((1, -1))), label='Rastrigin')

ax.plot(x, [prediction(model, torch.Tensor(x[i].reshape((1, -1)))).detach().numpy()[0, 0] for i in range(len(x))], label='NN')

ax.legend();

# %%
# We can also create our model using the nn.Module class of pytorch
# The model should be identical to the previous one!
import torch.nn as nn
import torch.nn.functional as F
class MyModel(torch.nn.Module):
    # Constructor
    def __init__(self):
        super().__init__()
        # Define the layers
        self.fc1 = torch.nn.Linear(1, 32)  # Input -> 32 units
        self.fc2 = torch.nn.Linear(32, 64)  # 32 -> 64 units
        self.fc3 = torch.nn.Linear(64, 1)  # 64 -> Output (1 unit)

    # Define forward pass
    def forward(self, x):
        x = F.tanh(self.fc1(x))  # First hidden layer with tanh activation
        x = F.tanh(self.fc2(x))  # Second hidden layer with tanh activation
        x = self.fc3(x)  # Output layer (no activation)
        return x


# %%
tmp_model = MyModel()

# Initialize weights of the model
def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        m.weight.data.fill_(1.)
        m.bias.data.fill_(0.01)

tmp_model.apply(init_weights)

lr = 1e-3
optim = torch.optim.Adam(tmp_model.parameters(), lr=lr) # Adam optimizer

myloss = loss(prediction(tmp_model, X), Y)
assert(update_optimizer(optim, myloss) == None)
myloss = loss(prediction(tmp_model, X), Y)
assert(update_optimizer(optim, myloss) == None)

params = list(tmp_model.parameters())
assert(torch.isclose(params[0].grad, torch.ones_like(params[0].grad) * 0.9356, rtol=1e-4).all())
assert(torch.isclose(params[1].grad, torch.ones_like(params[1].grad) * 0.5833, rtol=1e-4).all())
assert(torch.isclose(params[2].grad, torch.ones_like(params[2].grad) * 0.0147, rtol=1e-4, atol=1e-4).all())
assert(torch.isclose(params[3].grad, torch.ones_like(params[3].grad) * 0.009, rtol=1e-4, atol=1e-4).all())
assert(torch.isclose(params[4].grad, torch.ones_like(params[4].grad) * 128.829, rtol=1e-4).all())
assert(torch.isclose(params[5].grad, torch.ones_like(params[5].grad) * (-44.53), rtol=1e-4).all())

# %%
# Create the new model
model = MyModel()

# Let's create the optimizer
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # Adam optimizer

for t in range(20000):
    y_pred = prediction(model, X)

    myloss = loss(y_pred, Y)
    if t == 0 or (t + 1) % 200 == 0:
        print(t+1, myloss.item() / K)

    update_optimizer(optimizer, myloss)

# %%
# Let's plot the function
x = np.linspace(-5.12, 5.12, 1000)

plt.close() # close previous
fig = plt.figure()
ax = fig.add_subplot(111)

# ax.plot(X, Y, '.')
ax.plot(x, f(x.reshape((1, -1))), label='Rastrigin')

ax.plot(x, [prediction(model, torch.Tensor(x[i].reshape((1, -1)))).detach().numpy()[0, 0] for i in range(len(x))], label='NN')

ax.legend();
plt.show()
# %%
# We can also use pytorch's built-in functions for learning with batches
from torch.utils.data import Dataset, DataLoader

# We create a custom dataset class
class NumpyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.Tensor(X) # Convert NumPy to Tensor
        self.y = torch.Tensor(y) # Convert NumPy to Tensor

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# %%
# Let's create the actual dataset and dataloader (aka the worker that makes the batches)
dataset = NumpyDataset(Xn.T, Yn.T)

# We want a batch size of 32, and we randomly shuffle the samples
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# %%
# Create the new model
model = MyModel()

# Let's create the optimizer
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # Adam optimizer

# Training Loop
num_epochs = 200
for epoch in range(num_epochs):
    epoch_loss = 0.0

    for batch_X, batch_y in dataloader:
        outputs = prediction(model, batch_X)

        myloss = loss(outputs, batch_y)

        update_optimizer(optimizer, myloss)

        epoch_loss += myloss.item()

    # if epoch == 0 or (epoch + 1) % 20 == 0:
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader):.4f}")


# %%
# Let's plot the function
x = np.linspace(-5.12, 5.12, 1000)

plt.close() # close previous
fig = plt.figure()
ax = fig.add_subplot(111)

# ax.plot(X, Y, '.')
ax.plot(x, f(x.reshape((1, -1))), label='Rastrigin')

ax.plot(x, [prediction(model, torch.Tensor(x[i].reshape((1, -1)))).detach().numpy()[0, 0] for i in range(len(x))], label='NN')

ax.legend();

# %%



