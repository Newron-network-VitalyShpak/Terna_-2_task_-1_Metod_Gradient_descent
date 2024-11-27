###############################################################################

# Выполнил Шпак В. С. гр. 09-251
# Нейронные сети, тема 2, задание 2. 
# Реализуйте градиентный спуск для той же функции. Пусть начальным приближением
# будет w^(t=0) = [[5,10],[1,2]], шаг градиентного спуска alpha = 0.001.
# Чему будет равен w^(t=500)

###############################################################################

import torch

# Создаем тензор X и включаем режим отслеживания градиента
x = torch.tensor(
    [[5.,  10.],
     [1., 2.]], requires_grad=True)

# Проверяем доступен ли GPU и при возможности переносим на GPU и иначе остаемся на CPU
device = torch.device('cuda:0' 
                      if torch.cuda.is_available() 
                      else 'cpu')
x = x.to(device)

for iteration in range(500):
    function = torch.prod(torch.log(torch.log(x + 7)))
    
    print(f'Iteration {iteration}: Value = {function.item()}')

    function.backward()
    
    with torch.no_grad():
        x -= 0.001 * x.grad
    x.grad.zero_()

print(f'\nFinal value: {function.item()}')
print(f'Gradient: {x}')


# Реализация 2

# import torch

x = torch.tensor([[5., 10.],
                  [1., 2.]], requires_grad=True)
var_history = []
fn_history = []

def function_(variable):
    return torch.prod(torch.log(torch.log(variable + 7)))

def make_gradient_step(function, variable):
    function_result = function(variable)
    function_result.backward()
    variable.data -= 0.001 * variable.grad
    variable.grad.zero_()

for i in range(500):
    var_history.append(x.data.cpu().numpy().copy())
    fn_history.append(function_(x).data.cpu().numpy().copy())
    make_gradient_step(function_, x)


print(fn_history[-1])
print(var_history[-1])

