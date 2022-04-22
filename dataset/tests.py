import torch
hashmap={}
hashmap[1]="1"

a=torch.tensor(
              [
                  [1, 5, 5, 2],
                  [9, -6, 2, 8],
                  [-3, 7, -9, 1]
              ])
b=torch.argmax(a,dim=0)

a = torch.tensor([
    [
        [1, 5, 5, 2],
        [9, -6, 2, 8],
        [-3, 7, -9, 1]
    ],

    [
        [-1, 7, -5, 2],
        [9, 6, 2, 8],
        [3, 7, 9, 1]
    ]])
b = torch.argmax(a, dim=1)
print(a.shape)
print(b)
