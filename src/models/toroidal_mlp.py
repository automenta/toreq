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
        self.buffer_size = buffer_size
        
        # Buffer weights: exponential decay
        raw_alphas = [decay ** (k+1) for k in range(buffer_size)]
        self.register_buffer('buffer_alphas', torch.tensor(raw_alphas))

        self.Wx = nn.Linear(input_dim, hidden_dim, bias=True)
        self.Wh = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.Head = nn.Linear(hidden_dim, output_dim, bias=True)
        
        # Recirculation mixing weight
        self.buffer_gamma = nn.Parameter(torch.tensor(0.1))

        nn.init.orthogonal_(self.Wh.weight)
        with torch.no_grad():
            self.Wh.weight.mul_(0.9)

    def forward_step(self, h, x, buffer_list):
        # buffer_list: [h_{t-1}, h_{t-2}, ...]
        
        buffer_input = torch.zeros_like(h)
        if buffer_list and self.buffer_size > 0:
            current_buffer_len = len(buffer_list)
            # Weights: [History, 1, 1] for broadcasting
            weights = self.buffer_alphas[:current_buffer_len].view(-1, 1, 1)
            stack = torch.stack(buffer_list) # [History, Batch, Dim]
            buffer_input = (stack * weights).sum(dim=0)
            
        pre_act = self.Wx(x) + self.Wh(h) + self.buffer_gamma * buffer_input
        h_new = torch.tanh(pre_act)
        return (1 - self.alpha) * h + self.alpha * h_new

    def forward(self, x, steps=30):
        batch_size = x.size(0)
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        buffer = []
        
        for _ in range(steps):
            h_next = self.forward_step(h, x, buffer)
            
            # Update buffer
            buffer.insert(0, h.detach()) # Store history
            if len(buffer) > self.buffer_size:
                buffer.pop()
            
            h = h_next
            
        return self.Head(h)

    def energy(self, h, x, buffer_list=None):
        """
        Approximate Energy function. 
        E = 0.5 * ||h||^2 - Sum(LogCosh(Wx x + Wh h + Gamma * Buffer))
        """
        term1 = 0.5 * torch.sum(h ** 2)
        
        # Calculate buffer term exactly as in forward_step
        buffer_input = torch.zeros_like(h)
        if buffer_list and self.buffer_size > 0:
            current_buffer_len = len(buffer_list)
            weights = self.buffer_alphas[:current_buffer_len].view(-1, 1, 1)
            stack = torch.stack(buffer_list) 
            buffer_input = (stack * weights).sum(dim=0)

        pre_act = self.Wx(x) + self.Wh(h) + self.buffer_gamma * buffer_input
        
        abs_pre = torch.abs(pre_act)
        log_cosh = abs_pre + torch.nn.functional.softplus(-2 * abs_pre) - 0.693147
        term2 = torch.sum(log_cosh)
        
        return term1 - term2
