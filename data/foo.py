import torch
from torch.optim import Adam

# Step 1: Define the function you want to minimize
def quadratic_function(x):
    return x**2 

# Step 2: Create a variable (tensor) for the parameter you want to optimize
x = torch.tensor([1.0], requires_grad=True)  # Initialize x with some initial value

# Step 3: Choose the Adam optimizer
optimizer = Adam([x], lr=0.1)  # Specify the learning rate

# Step 4: Perform the optimization loop
num_epochs = 100  # Number of optimization iterations
for epoch in range(num_epochs):
    # Calculate the function value and gradient
    y = quadratic_function(x)
    
    # Clear the gradients from the previous iteration
    optimizer.zero_grad()
    
    # Compute the gradient of the function with respect to x
    y.backward()
    
    # Update x using the optimizer
    optimizer.step()
    
    # Print the progress (optional)
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], x: {x.item():.4f}, f(x): {y.item():.4f}')

# After optimization, x should be close to the minimum point of the function
print(f'Optimized x: {x.item():.4f}')