import torch
import torch.nn as nn

class ToroidalMLP(nn.Module):
    """
    Toroidal Equilibrium Propagation (TEP) MLP.
    Includes a recirculation buffer to stabilize dynamics using historical states.
    
    Dynamics:
    h_{t+1} = (1-alpha)h_t + alpha * tanh(Wx x + Wh h_t + BufferTerm)
    BufferTerm = Sum(w_k * h_{t-k})
    """
    def __init__(self, input_dim, hidden_dim, output_dim, alpha=0.5, buffer_size=5, decay=0.9):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.alpha = alpha
        self.decay = decay
        
        self.Wx = nn.Linear(input_dim, hidden_dim, bias=True)
        self.Wh = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.Head = nn.Linear(hidden_dim, output_dim, bias=True)
        
        self.buffer_gamma = nn.Parameter(torch.tensor(0.1))

        nn.init.orthogonal_(self.Wh.weight)
        with torch.no_grad():
            self.Wh.weight.mul_(0.9)

    def forward_step(self, h, x, buffer_state):
        # buffer_state: EMA of past history
        if buffer_state is None:
            buffer_input = torch.zeros_like(h)
            # Init next state
            new_buffer = h.detach() * self.decay 
        else:
            buffer_input = buffer_state
            # Update: new_buffer = old_buffer * decay + current_h * decay
            new_buffer = buffer_state * self.decay + h.detach() * self.decay
            
        pre_act = self.Wx(x) + self.Wh(h) + self.buffer_gamma * buffer_input
        h_new = torch.tanh(pre_act)
        
        return (1 - self.alpha) * h + self.alpha * h_new, new_buffer

    def forward(self, x, steps=30):
        batch_size = x.size(0)
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        buffer_state = None
        
        for _ in range(steps):
            h, buffer_state = self.forward_step(h, x, buffer_state)
            
        return self.Head(h)

    def energy(self, h, x, buffer_state=None):
        term1 = 0.5 * torch.sum(h ** 2)
        
        if buffer_state is None:
             buffer_input = torch.zeros_like(h)
        else:
             buffer_input = buffer_state

        pre_act = self.Wx(x) + self.Wh(h) + self.buffer_gamma * buffer_input
        
        abs_pre = torch.abs(pre_act)
        log_cosh = abs_pre + torch.nn.functional.softplus(-2 * abs_pre) - 0.693147
        term2 = torch.sum(log_cosh)
        
        return term1 - term2
