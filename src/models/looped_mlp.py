import torch
import torch.nn as nn
import torch.nn.functional as F

class LoopedMLP(nn.Module):
    """
    A simple weight-tied Looped MLP.
    Dynamics: h_{t+1} = (1-alpha)h_t + alpha * tanh(W h_t + W_x x + b)
    """
    def __init__(self, input_dim, hidden_dim, output_dim, alpha=0.5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.alpha = alpha

        # Parameters
        self.Wx = nn.Linear(input_dim, hidden_dim, bias=True)
        self.Wh = nn.Linear(hidden_dim, hidden_dim, bias=True) # Weight tied across time
        self.Head = nn.Linear(hidden_dim, output_dim, bias=True)
        
        # Initialize weights for stability (often orthogonal or small)
        nn.init.orthogonal_(self.Wh.weight)
        # Ensure spectral radius < 1 roughly by scaling
        with torch.no_grad():
            self.Wh.weight.mul_(0.9)

    def forward_step(self, h, x):
        # h: [batch, hidden]
        # x: [batch, input]
        pre_act = self.Wx(x) + self.Wh(h)
        h_new = torch.tanh(pre_act)
        return (1 - self.alpha) * h + self.alpha * h_new

    def forward(self, x, steps=30):
        batch_size = x.size(0)
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        
        for _ in range(steps):
            h = self.forward_step(h, x)
            
        return self.Head(h)
    
    def energy(self, h, x):
        """
        Scalar energy function E(h; x).
        Assumes symmetric weights for valid energy definition: Wh = Wh.T
        E = 0.5 * ||h||^2 - Sum(LogCosh(Wx x + Wh h))
        """
        # Note: We rely on the autograd to compute dE/dTheta.
        # This energy function implies the fixed point h = tanh(Wx x + Wh h).
        
        # Self-interaction term: 0.5 * ||h||^2
        term1 = 0.5 * torch.sum(h ** 2)
        
        # Interaction potential: Integral of tanh is LogCosh
        pre_act = self.Wx(x) + self.Wh(h)
        # Stable LogCosh implementation
        # log cosh(x) = log( (e^x + e^-x)/2 )
        #             = x + softplus(-2x) - log2
        abs_pre = torch.abs(pre_act)
        log_cosh = abs_pre + torch.nn.functional.softplus(-2 * abs_pre) - 0.693147
        term2 = torch.sum(log_cosh)
        
        return term1 - term2
