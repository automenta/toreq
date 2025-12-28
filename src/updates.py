"""Update strategies for Equilibrium Propagation training.

This module implements different strategies for computing parameter updates
from equilibrium states. All strategies follow the UpdateStrategy interface.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from abc import ABC, abstractmethod
from typing import Dict


class UpdateStrategy(ABC):
    """Base class for EqProp update strategies."""
    
    def __init__(self, beta: float):
        """Initialize update strategy.
        
        Args:
            beta: Nudge strength parameter
        """
        self.beta = beta
    
    @abstractmethod
    def compute_model_update(self, model: nn.Module, h_free: Tensor, 
                            h_nudged: Tensor, x: Tensor) -> Tensor:
        """Compute loss for model parameter update.
        
        Args:
            model: The equilibrium model
            h_free: Free phase equilibrium state
            h_nudged: Nudged phase equilibrium state  
            x: Input tensor
            
        Returns:
            Loss tensor for backpropagation
        """
        pass
    
    @abstractmethod
    def compute_head_update(self, output_head: nn.Module, h_free: Tensor, 
                           y: Tensor) -> Tensor:
        """Compute loss for output head parameter update.
        
        Args:
            output_head: Classification/output head
            h_free: Free phase equilibrium state
            y: Target labels
            
        Returns:
            Loss tensor for backpropagation
        """
        pass


class MSEProxyUpdate(UpdateStrategy):
    """MSE proxy loss update mechanism.
    
    Minimizes the distance between the model's one-step output from the free
    equilibrium and the nudged equilibrium state. The loss is scaled by 1/β
    to approximate the correct gradient magnitude.
    
    Theory: delta ~ 1/β * (h_free - h_nudged) approximates the adjoint state.
    """
    
    def compute_model_update(self, model: nn.Module, h_free: Tensor,
                            h_nudged: Tensor, x: Tensor) -> Tensor:
        """Compute MSE proxy loss between model output and nudged equilibrium."""
        h_free_detached = h_free.detach()
        h_out = model(h_free_detached, x)
        
        # Scale by 1/beta for correct gradient magnitude
        loss = (1.0 / self.beta) * F.mse_loss(h_out, h_nudged.detach())
        return loss
    
    def compute_head_update(self, output_head: nn.Module, h_free: Tensor,
                           y: Tensor) -> Tensor:
        """Standard cross-entropy loss at free equilibrium."""
        y_pred = output_head(h_free.detach().mean(dim=0))
        return F.cross_entropy(y_pred, y)


class VectorFieldUpdate(UpdateStrategy):
    """Vector field backpropagation update mechanism.
    
    Backpropagates the vector v = (h_nudged - h_free) / β through the model.
    This is theoretically cleaner as it directly computes the gradient without
    requiring a proxy loss.
    
    Theory: Gradient ≈ 1/β * (h_nudged - h_free) * df/dθ via the vector field.
    """
    
    def compute_model_update(self, model: nn.Module, h_free: Tensor,
                            h_nudged: Tensor, x: Tensor) -> None:
        """Backpropagate vector field through model (no explicit loss)."""
        delta = h_nudged - h_free.detach()
        v = delta / self.beta
        
        # Re-compute output at free equilibrium with gradients enabled
        h_at_free = h_free.detach()
        with torch.enable_grad():
            out_free = model(h_at_free, x)
        
        # Backward the vector v through model parameters
        out_free.backward(gradient=v)
        
        # Return None since gradients are accumulated directly
        return None
    
    def compute_head_update(self, output_head: nn.Module, h_free: Tensor,
                           y: Tensor) -> Tensor:
        """Standard cross-entropy loss at free equilibrium."""
        with torch.enable_grad():
            y_pred = output_head(h_free.mean(dim=0))
            return F.cross_entropy(y_pred, y)
