print("This is working!")

def hello():
    print("Inside the function")
hello()

import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())