#########################################

# Выполнил Шпак В. С. гр. 09-251
# Нейронные сети, тема 2, задание 4. 

#########################################


import torch
import numpy as np
import matplotlib.pyplot as plt

x = torch.tensor([[5., 10.],
                  [1.,2.]], requires_grad=True)
var_history = []
fn_history = []

optimizer = torch.optim.SGD([x], lr=0.001)

def function_parabola(variable):
    return torch.prod(torch.log(torch.log(variable + 7)))

def function_skewed_np(variable):
    gramma = np.array([[1, -1], [1, 1]]) @ np.array([[1.0, 0.0], [0.0, 4.0]])
    res = 10 * (variable.transpose(1, 0) @ (gramma @ variable)).sum()
    return res


def function_skewed(variable):
    gramma = torch.tensor([[1., -1.], [1., 1.]]) @ torch.tensor([[1.0, 0.0], [0.0, 4.0]])
    res = 10 * (variable.unsqueeze(0) @ (gramma @ variable.unsqueeze(1))).sum()
    return res

def make_gradient_step(function, variable):
    function_result = function(variable)
    function_result.backward()
    optimizer.step()
    optimizer.zero_grad()
    
step_hist = []

for i in range(500):
    var_history.append(x.data.numpy().copy())
    fn_history.append(function_skewed(x).data.cpu().numpy().copy())
    step_hist = make_gradient_step(function_skewed, x)

# plt.scatter(np.array(var_history)[:,0], np.array(var_history)[:,1], s=10, c='r');
# plt.show

def show_contours(objective,
                  x_lims=[-10.0, 10.0], 
                  y_lims=[-10.0, 10.0],
                  x_ticks=100,
                  y_ticks=100):
    x_step = (x_lims[1] - x_lims[0]) / x_ticks
    y_step = (y_lims[1] - y_lims[0]) / y_ticks
    X, Y = np.mgrid[x_lims[0]:x_lims[1]:x_step, y_lims[0]:y_lims[1]:y_step]
    res = []
    for x_index in range(X.shape[0]):
        res.append([])
        for y_index in range(X.shape[1]):
            x_val = X[x_index, y_index]
            y_val = Y[x_index, y_index]
            res[-1].append(objective(np.array([[x_val, y_val]]).T))
    res = np.array(res)
    plt.figure(figsize=(7,7))
    plt.contour(X, Y, res, 100)
    plt.xlabel('x_1')
    plt.ylabel('x_2')


# plt.figure(figsize=(7,7))
# plt.plot(fn_history);
# plt.xlabel('step')
# plt.ylabel('function value');
# plt.show()

plt.scatter(np.array(var_history)[:,0], np.array(var_history)[:,1], s=10, c='r');
plt.show()



show_contours(function_skewed_np)
plt.scatter(np.array(var_history), np.array(var_history), s=10, c='r');
plt.show()