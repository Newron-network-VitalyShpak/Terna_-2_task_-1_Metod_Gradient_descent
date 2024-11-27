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

def make_gradient_step(function, variable):
    function_result = function(variable)
    function_result.backward()
    optimizer.step()
    optimizer.zero_grad()
    
for i in range(50000):
    var_history.append(x.data.numpy().copy())
    fn_history.append(function_parabola(x).data.cpu().numpy().copy())
    make_gradient_step(function_parabola, x)

# plt.scatter(np.array(var_history)[:,0], np.array(var_history)[:,1], s=10, c='r');
# plt.show

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

# отдельная функция, в которой задаются x, y, z



# plt.figure(figsize=(7,7))
# plt.plot(fn_history);
# plt.xlabel('step')
# plt.ylabel('function value');
# plt.show()

plt.scatter(np.array(var_history)[:,0], np.array(var_history)[:,1], s=10, c='r');
plt.show()