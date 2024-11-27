###############################################################################

# Выполнил Шпак В. С. гр. 09-251
# Нейронные сети, тема 2, задание 3. 
# Задние 2 + SGD

###############################################################################

import torch

x = torch.tensor([[5., 10.],
                  [1.,2.]], requires_grad=True)
var_history = []
fn_history = []

optimizer = torch.optim.SGD([x], lr = 0.001)

def function_parabola(variable):
    return torch.prod(torch.log(torch.log(variable + 7)))

def make_gradient_step(function, variable):
    function_result = function(variable)
    function_result.backward()
    optimizer.step()
    optimizer.zero_grad()
    
for i in range(500):
    var_history.append(x.data.numpy().copy())
    fn_history.append(function_parabola(x).data.cpu().numpy().copy())
    make_gradient_step(function_parabola, x)

print(fn_history[-1])
print(var_history[-1])

