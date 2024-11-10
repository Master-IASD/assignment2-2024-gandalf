import torch 

class f_divergence:
    """
    Definine the class that every f-divergence will inherit from.

    Parameters
    ------------------------------------------------------------------
        name : str, name of the f-divergence, no incidence
        fdiv : python function, "f" in the f-divergence
        fenchel_conjugate : python function, Fenchel conjugate of fdiv
        threshold : threshold used for accuracy computation, should be f'(1) based on paper
    """
    def __init__(self,name,fdiv,fenchel_conjugate,threshold):
        self.name = name
        self.threshold = threshold
        self.fdiv = fdiv
        self.fenchel = fenchel_conjugate
    
    def activation(self,t):
        """Compute the activation function used on the latest layer of the dsicriminator"""
        return self.fdiv(t)
    
    def f_star(self,t):
        return self.fenchel(t)

class jensen_shannon(f_divergence):
    def __init__(self):
        super().__init__(name = 'JS',
                        fdiv = lambda t : torch.log(torch.ones(t.shape)*2) -torch.log(1+torch.exp(-t)),
                        fenchel_conjugate = lambda t : -torch.log(2 - torch.exp(t)),
                        threshold = 0)

class Kullback_Leibler(f_divergence):
    def __init__(self):
        super().__init__(name = 'KL',
                        fdiv = lambda t : t,
                        fenchel_conjugate = lambda t : torch.exp(t-1),
                        threshold = 1)

class reverse_KL(f_divergence):
    def __init__(self):
        super().__init__(name = 'rKL',
                        fdiv = lambda t : -torch.exp(t),
                        fenchel_conjugate = lambda t : -1-torch.log(-t),
                        threshold = -1)
        
class Pearson_chi2(f_divergence):
    def __init__(self):
        super().__init__(name = 'Pearson chi2',
                        fdiv = lambda t : t,
                        fenchel_conjugate = lambda t : 0.25 * t**2 + t,
                        threshold = 0)
        
class Squared_Hellinger(f_divergence):
    def __init__(self):
        super().__init__(name = 'Squared Hellinger',
                        fdiv = lambda t : 1-torch.exp(t),
                        fenchel_conjugate = lambda t : t / (1-t),
                        threshold = 0)  