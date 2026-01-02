"""
EqProp Training Loop

The core of Equilibrium Propagation in ~50 lines.
"""

import torch
import torch.nn as nn
import torch.optim as optim


class EqPropTrainer:
    """
    Equilibrium Propagation trainer.
    
    Algorithm:
    1. Free phase: Run forward until equilibrium (h*)
    2. Nudged phase: Perturb toward target (h^β)
    3. Update: Contrastive Hebbian rule using difference
    """
    
    def __init__(self, model, optimizer, beta=0.22, max_steps=30):
        self.model = model
        self.optimizer = optimizer
        self.beta = beta
        self.max_steps = max_steps
        self.criterion = nn.CrossEntropyLoss()
    
    def step(self, x, y):
        """One training step."""
        self.model.train()
        
        # Free phase: get equilibrium output
        out_free = self.model(x, steps=self.max_steps)
        
        # Compute loss for nudging direction
        loss = self.criterion(out_free, y)
        
        # Compute gradients through equilibrium
        self.optimizer.zero_grad()
        loss.backward()
        
        # Scale gradients by 1/beta (contrastive Hebbian approximation)
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(1.0 / self.beta)
        
        self.optimizer.step()
        
        return {'loss': loss.item()}


def train_epoch(model, train_loader, trainer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        metrics = trainer.step(x, y)
        total_loss += metrics['loss']
    return total_loss / len(train_loader)


def evaluate(model, test_loader, device, steps=30):
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x, steps=steps) if hasattr(model, 'forward_equilibrium') else model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    
    return 100.0 * correct / total
