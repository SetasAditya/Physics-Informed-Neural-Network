import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

import torch.nn.functional as F

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden_layer1 = nn.Linear(2142, 16)
        self.hidden_layer2 = nn.Linear(16, 16)
        self.hidden_layer3 = nn.Linear(16, 16)
        self.output_layer = nn.Linear(16, 1071)

    def forward(self, x, y):
        # Concatenate x and y along the first dimension
        input = torch.cat([x, y], dim=0)

        layer1_out = Mish()(self.hidden_layer1(input))
        layer2_out = Mish()(self.hidden_layer2(layer1_out))
        layer3_out = Mish()(self.hidden_layer3(layer2_out))
        output = self.output_layer(layer3_out)
        return output

# Define the dimensions
x_length = 1.0
y_length = 0.4
x_points = 51
y_points = 21

# Calculate the spacing between points
x_spacing = x_length / (x_points - 1)
y_spacing = y_length / (y_points - 1)

# Generate the grid
x_points = np.linspace(0, x_length, x_points)
y_points = np.linspace(0, y_length, y_points)

# Create a meshgrid
x_mesh, y_mesh = np.meshgrid(x_points, y_points)

# Flatten the meshgrid to get the point cloud
point_cloud = np.column_stack((x_mesh.ravel(), y_mesh.ravel()))

# Identify boundary points
left_wall_points = point_cloud[point_cloud[:, 0] == 0]
right_wall_points = point_cloud[point_cloud[:, 0] == x_length]
top_wall_points = point_cloud[(point_cloud[:, 1] == 0)]
bottom_wall_points = point_cloud[(point_cloud[:, 1] == y_length)]

print(left_wall_points[3])

# Visualize the mesh
plt.scatter(point_cloud[:, 0], point_cloud[:, 1], color='b', label='Mesh Points', s=5)
plt.title('Mesh Points')
plt.xlabel('x')
plt.ylabel('y')

# Visualize left wall points
plt.scatter(left_wall_points[:, 0], left_wall_points[:, 1], color='r', label='Left Inlet', s=10)

# Visualize right wall points
plt.scatter(right_wall_points[:, 0], right_wall_points[:, 1], color='g', label='Right Outlet', s=10)

# Visualize top and bottom wall points
plt.scatter(top_wall_points[:, 0], top_wall_points[:, 1], color='purple', label='Top Wall', s=10)

# Visualize top and bottom wall points
plt.scatter(bottom_wall_points[:, 0], bottom_wall_points[:, 1], color='purple', label='Bottom Wall', s=10)

# Set the aspect ratio to equal for better visualization
plt.gca().set_aspect('equal', adjustable='box')

# Add legend
plt.legend()

# Show the plot
plt.show()

def vel_x(points, count):
  vel = np.zeros_like(points[:,0])
  for i in range(len(points)):
    vel[i] = (0.1 + 0.05 * count) * (points[i][1]) * (0.4 - (points[i][1]))
  return vel

net = Net()

# Apply the initialization to your network
# net.apply(init_weights)

net = net.to(device)

mse_cost_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.02)
# Use SGD with momentum
# optimizer = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.9)

def f(x, y, net):
    # Pass x and y through the neural network
    u = net(x, y)

    # Reshape u to a 2D NumPy array
    u_2d_array = u.reshape(len(x_points), len(y_points)).detach().numpy()

    # delta_x and delta_y are the spacing between points in x and y directions
    delta_x = 0.02
    delta_y = 0.02

    # Compute the second derivatives using finite differences along x and y
    # u_xx = (u_2d_array[:, 1:-1] - 2 * u_2d_array[1:-1, 1:-1] + u_2d_array[2:-2, 1:-1]) / (delta_x**2)
    # u_yy = (u_2d_array[2:, :] - 2 * u_2d_array[1:-1, :] + u_2d_array[:-2, :]) / (delta_y**2)

    # Get the shape of u_2d_array
    rows, cols = u_2d_array.shape

    # Initialize an array for u_xx with the same shape as u_2d_array
    u_xx = np.zeros_like(u_2d_array)
    u_yy = np.zeros_like(u_2d_array)

    # Iterate through the indices excluding the first and last points along the x-axis
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            # Compute the central difference for u_xx
            u_xx[i, j] = (u_2d_array[i-1, j] - 2 * u_2d_array[i, j] + u_2d_array[i+1, j]) / (delta_x**2)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            # Compute the central difference for u_xx
            u_yy[i, j] = (u_2d_array[i, j-1] - 2 * u_2d_array[i, j] + u_2d_array[i, j+1]) / (delta_y**2)

    # Combine the second derivatives to get the PDE of the same shape as u_2d_array
    pde = u_xx + u_yy

    # Exclude the boundary points from the PDE
    interior_pde = pde[1:-1, 1:-1]

    # Flatten the interior points
    flattened_pde = interior_pde.flatten()

    return flattened_pde

import torch
from torch import nn, optim, autograd
from torch.autograd import Variable
import matplotlib.pyplot as plt

# Assuming you have defined your network, optimizer, and other components

# Define the learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=750, gamma=0.95)

iterations = 4551

# Create a tensor of zeros for later use
pt_all_zeros = torch.zeros((1071,), requires_grad=True).to(device)

# Lists to store training loss values
data_losses = []
eqn_losses = []
total_losses = []

# Lists to store values for plotting
y_coordinates_list = []
abs_grad_values_list = []
residue_lw = []
average_residue_list = []  # New list to store average residue values

# Training loop
for epoch in range(iterations):
    optimizer.zero_grad()
    data_loss = 0.0
    total_mae_lw = 0.0
    total_mae_rw = 0.0
    total_mae_tb = 0.0
    total_mae_b = 0.0
    eqn_loss = 0.0
    total_grad_phi_out_left_wall_x = 0.0
    total_residue_lw = 0.0  # Accumulator for total residue
    total_residue_rw = 0.0
    count = 0

    for times in range(80):
        pt_x_batch = Variable(torch.from_numpy(point_cloud[:, 0]).float(), requires_grad=True).to(device)
        pt_y_batch = Variable(torch.from_numpy(point_cloud[:, 1]).float(), requires_grad=True).to(device)

        # Pass pt_x_batch and pt_y_batch to f, not net
        f_out = f(pt_x_batch, pt_y_batch, net)  # Output of f(x, y, net) which gives R = phi_xx + phi_yy
        f_output = Variable(torch.from_numpy(f_out).float(), requires_grad=True).to(device)

        # Now, both tensors should have the same shape, and we can calculate the L1 loss
        mae_f = torch.nn.L1Loss()(f_output, torch.zeros_like(f_output))

        phi_out = net(pt_x_batch, pt_y_batch)

        # Convert left_wall_points batch to tensors
        pt_x_left_wall = Variable(torch.from_numpy(left_wall_points[:, 0]).float(), requires_grad=True).to(device)
        pt_y_left_wall = Variable(torch.from_numpy(left_wall_points[:, 1]).float(), requires_grad=True).to(device)


        # Extract the x-coordinates from the flattened point cloud
        x_coordinates = point_cloud[:, 0]

        # Find indices where x is minimum (on the left wall)
        left_wall_indices = np.where(x_coordinates == x_coordinates.min())[0]

        # Extract corresponding u values on the left wall
        phi_out_left_wall = phi_out[left_wall_indices]


        # Find pt_u_bc with phi values to get the boundary condition
        vel_l = vel_x(left_wall_points, count)
        pt_u_batch = Variable(torch.from_numpy(vel_l).float(), requires_grad=True).to(device)

        # Calculate the gradient of phi_out_left_wall with respect to left_wall_points
        grad_phi_out_left_wall_x = autograd.grad(phi_out, pt_x_batch, torch.ones_like(phi_out), create_graph=True)[0]

        # Calculate the absolute values of grad_phi_out_left_wall_x for the last sample
        # abs_grad_values_last_sample = torch.abs(grad_phi_out_left_wall_x[180:205]).detach().cpu().numpy()

        # Calculate the residue for the left wall
        residue_lw = torch.abs(pt_u_batch - grad_phi_out_left_wall_x[left_wall_indices])

        # Accumulate the individual residues
        total_residue_lw += residue_lw.detach().cpu().numpy()

        # Check if gradient computation is successful
        if grad_phi_out_left_wall_x is not None:
            # Use grad_phi_out_left_wall_x in some computation
            # some_computation = torch.sum(grad_phi_out_left_wall_x[180:205])
            mae_lw = torch.nn.L1Loss()(grad_phi_out_left_wall_x[left_wall_indices], pt_u_batch)
        else:
            # Handle the case when gradient computation fails
            mae_lw = torch.tensor(0.0).to(device)  # or any other suitable default value

        # Find pt_u_bc with phi values to get the boundary condition
        vel_r = vel_x(right_wall_points, count)
        pt_ur_batch = Variable(torch.from_numpy(vel_r).float(), requires_grad=True).to(device)

        # Calculate the gradient of phi_out_left_wall with respect to left_wall_points
        grad_phi_out_right_wall_x = autograd.grad(phi_out, pt_x_batch, torch.ones_like(phi_out), create_graph=True)[0]

        # Calculate the absolute values of grad_phi_out_left_wall_x for the last sample
        # abs_grad_values_last_sample = torch.abs(grad_phi_out_left_wall_x[180:205]).detach().cpu().numpy()


        # Find indices where x is minimum (on the left wall)
        right_wall_indices = np.where(x_coordinates == x_coordinates.max())[0]


        # Calculate the residue for the left wall
        residue_rw = torch.abs(pt_ur_batch - grad_phi_out_right_wall_x[right_wall_indices])

        # Accumulate the individual residues
        total_residue_rw += residue_rw.detach().cpu().numpy()

        # Check if gradient computation is successful
        if grad_phi_out_right_wall_x is not None:
            # Use grad_phi_out_left_wall_x in some computation
            # some_computation = torch.sum(grad_phi_out_left_wall_x[180:205])
            mae_rw = torch.nn.L1Loss()(grad_phi_out_right_wall_x[right_wall_indices], pt_ur_batch)

        # Convert top_wall_points batch to tensors
        pt_y_top_wall = Variable(torch.from_numpy(top_wall_points[:, 1]).float(), requires_grad=True).to(device)

        # Calculate the gradient of phi_out_tb_wall with respect to top_wall_points
        grad_phi_out_top_wall_points_y = autograd.grad(phi_out, pt_y_batch, torch.ones_like(phi_out), create_graph=True)[0]


        # Extract the y-coordinates from the flattened point cloud
        y_coordinates = point_cloud[:, 1]

        # Find indices where y is maximum (on the top wall)
        top_wall_indices = np.where(y_coordinates == y_coordinates.max())[0]
        bottom_wall_indices = np.where(y_coordinates == y_coordinates.min())[0]


        mae_tb = torch.nn.L1Loss()(grad_phi_out_top_wall_points_y[top_wall_indices], torch.zeros_like(grad_phi_out_top_wall_points_y[top_wall_indices]))

        # Convert bottom_wall_points batch to tensors
        pt_y_bottom_wall = Variable(torch.from_numpy(bottom_wall_points[:, 1]).float(), requires_grad=True).to(device)

        # Calculate the gradient of phi_out_bottom_wall with respect to bottom_wall_points
        grad_phi_out_bottom_wall_points_y = autograd.grad(phi_out, pt_y_batch, torch.ones_like(phi_out), create_graph=True)[0]

        mae_b = torch.nn.L1Loss()(grad_phi_out_bottom_wall_points_y[bottom_wall_indices], torch.zeros_like(grad_phi_out_bottom_wall_points_y[bottom_wall_indices]))

        # Accumulate the individual losses
        data_loss += mae_lw + mae_tb + mae_b + mae_rw
        eqn_loss += mae_f

        total_mae_rw += mae_rw
        total_mae_lw += mae_lw
        total_mae_tb += mae_tb
        total_mae_b += mae_b

        count += 1

        # Append values to the lists for plotting
        # y_coordinates_list.append(left_wall_points[:, 1])
        # abs_grad_values_list.append(abs_grad_values_last_sample)

    # Calculate the average over all mini-batches
    # average_grad_phi_out_left_wall_x = total_grad_phi_out_left_wall_x / len(points_matrix)

    # Calculate the average residue over all mini-batches
    average_residue_lw = total_residue_lw / len(point_cloud)
    # average_residue_list.append(average_residue_lw)
    average_residue_rw = total_residue_rw / len(point_cloud)

    # Divide the losses by the number of batches
    data_loss /= len(point_cloud)
    eqn_loss /= len(point_cloud)
    total_mae_lw /= len(point_cloud)
    total_mae_rw /= len(point_cloud)
    total_mae_tb /= len(point_cloud)
    total_mae_b /= len(point_cloud)

    # Calculate the total loss
    total_loss = data_loss + eqn_loss

    # Perform backward pass
    total_loss.backward()
    optimizer.step()

    # Update the learning rate
    scheduler.step()

    # Store training losses for later plotting
    data_losses.append(data_loss.item())
    eqn_losses.append(eqn_loss.item())
    total_losses.append(total_loss.item())


    # Print training loss every 50 epochs
    if epoch % 50 == 0:
        with torch.autograd.no_grad():
            print("")
            print("=============================================================")
            # print("Epoch:", epoch, "BC Loss:", round(data_loss.item(), 5))
            print("Epoch:", epoch, "BC MAE:", round(data_loss.item(), 5), " | For Left Inlet:", round(total_mae_lw.item(), 5), " | For Right Outlet:", round(total_mae_rw.item(), 5), " | For Top Wall:", round(total_mae_tb.item(), 5), " | For Bottom Wall:", round(total_mae_b.item(), 5))
            print("")
            # print("-------------------------------------------------------------")
            # print("Average Grad Values for Left Inlet:")
            # print(average_grad_phi_out_left_wall_x)
            # print("-------------------------------------------------------------")
            # print("MAE(left Inlet): ", mae_lw.detach().cpu().numpy())
            # print("-------------------------------------------------------------")
            # print("-------------------------------------------------------------")
            print("LW Averaged Absolute Residue: ", average_residue_lw)
            # print("-------------------------------------------------------------")
            print("")
            print("PDE MAE:", round(eqn_loss.item(), 5), "Total MAE:", round(total_loss.item(), 5), "Learning Rate:", round(optimizer.param_groups[0]['lr'], 5))
            print("=============================================================")
            print("")

            y_coordinates = left_wall_points[:, 1]
            bar_width = 0.2
            x_positions = np.linspace(0, len(y_coordinates) - 1, len(y_coordinates))

# Plot the losses
fig = plt.figure(figsize=(8, 5))
ax = plt.axes()

ax.plot(range(0, iterations), data_losses, label='BC Loss')
ax.plot(range(0, iterations), eqn_losses, label='PDE Loss')
ax.plot(range(0, iterations), total_losses, label='Total Loss')

ax.set_xlabel('Epochs', fontsize=16)
ax.set_ylabel('Loss', fontsize=16)

plt.legend()
plt.show()

# Set the model to evaluation mode
net.eval()

