import torch 

class jensen_shannon :
    def __init__(self):
        self.name = 'js'
    
    def f(self,t):
        return torch.log(torch.ones(t.shape)*2) -torch.log(1+torch.exp(-t))
    
    def f_star(self,t):
        return -torch.log(2 - torch.exp(t))

class Kullback_Leibler:
    def __init__(self):
        self.name = 'KL'
    
    def f(self,t):
        return t
    
    def f_star(self,t):
        return torch.exp(t-1)

class Pearson:
    def __init__(self):
        self.name = 'Pearson'
    
    def f(self,t):
        return t
    
    def f_star(self,t):
        return 0.25*t**2 + t
    
