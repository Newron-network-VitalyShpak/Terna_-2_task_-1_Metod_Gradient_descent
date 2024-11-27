#########################################

# Выполнил Шпак В. С. гр. 09-251
# Нейронные сети, тема 2, задание 1. 
# Реализуйте расчет градиента для функции

#########################################

import torch

x = torch.tensor(
    [[5.,  10.],
     [1., 2.]], requires_grad=True)

device = torch.device('cuda:0' 
                      if torch.cuda.is_available() 
                      else 'cpu')
x = x.to(device)

function = torch.prod(torch.log(torch.log(x + 7)))

print(function, '<- Значение функции')

function.backward()

print(x.grad, '<- Градиент')