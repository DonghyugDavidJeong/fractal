import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X = np.random.default_rng().standard_normal(2)
Y = np.random.default_rng().standard_normal(2)
Z = np.random.default_rng().standard_normal(2)
x = torch.Tensor(X)
y = torch.Tensor(Y)
z = torch.Tensor(Z)
x = x.to(device)
y = y.to(device)
z = z.to(device)

# vertices = [x, y, z]

start_vertex = np.random.randint(3)
match start_vertex:
    case 0:
        start_position = (y + z)/2
    case 1:
        start_position = (x + z)/2
    case 2:
        start_position = (x + y)/2

start_position = start_position.to(device)

plt.plot(x[0], x[1], 'ro')
plt.plot(y[0], y[1], 'go')
plt.plot(z[0], z[1], 'bo')

for i in range(20000):
    match start_vertex:
        case 0:
            start_position = (start_position + x)/2
        case 1:
            start_position = (start_position + y)/2
        case 2:
            start_position = (start_position + z)/2
    start_vertex = np.random.randint(3)
    plt.plot(start_position[0], start_position[1], 'co')
plt.tight_layout(pad=0)
plt.show()