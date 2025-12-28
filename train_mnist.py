import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from src.models import LoopedTransformerBlock
from src.solver import EquilibriumSolver
from src.trainer import EqPropTrainer
import time

def train_mnist():
    print("Starting MNIST Training with TorEqProp...")

    # Config
    batch_size = 64
    d_model = 128
    n_heads = 4
    d_ff = 512
    epochs = 1 # Just 1 epoch for quick check, or more
    lr = 1e-3
    beta = 0.5 # Larger beta for training? README suggests [0.01, 0.5]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    try:
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    except Exception as e:
        print(f"Failed to download MNIST: {e}")
        return

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model
    # Use Symmetric + Linear Attention for EqProp compatibility
    model = LoopedTransformerBlock(
        d_model, n_heads, d_ff,
        use_linear_attn=True,
        symmetric=True
    ).to(device)

    output_head = nn.Linear(d_model, 10).to(device)

    solver = EquilibriumSolver(max_iters=30, tol=1e-4, damping=0.9) # Fewer iters for speed
    trainer = EqPropTrainer(model, solver, output_head, beta=beta, lr=lr)

    # Input Projection (e.g. 7x7 patches of 4x4 pixels = 16 dim -> 128 dim)
    # MNIST 28x28.
    # Patch size 4x4.
    # Grid 7x7. Seq len 49.
    # Input dim 16.
    input_proj = nn.Linear(16, d_model).to(device)
    trainer.optimizer.add_param_group({'params': input_proj.parameters()})

    print("Training...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_acc = 0
        start_time = time.time()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # Prepare input: [Batch, 1, 28, 28] -> [Seq, Batch, d_model]
            # Patches
            B = data.size(0)
            patches = data.unfold(2, 4, 4).unfold(3, 4, 4) # [B, 1, 7, 7, 4, 4]
            patches = patches.contiguous().view(B, 49, 16)
            patches = patches.permute(1, 0, 2) # [Seq, Batch, 16]

            x = input_proj(patches) # [Seq, Batch, d_model]

            metrics = trainer.train_step(x, target)

            total_loss += metrics['loss']
            total_acc += metrics['accuracy']

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {metrics['loss']:.4f} Acc: {metrics['accuracy']:.4f} "
                      f"Iters: {metrics['iters_free']}/{metrics['iters_nudged']}")

        avg_loss = total_loss / len(train_loader)
        avg_acc = total_acc / len(train_loader)
        print(f"Epoch {epoch} Done. Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.4f}, Time: {time.time()-start_time:.1f}s")

        # Test
        model.eval()
        test_acc = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                B = data.size(0)
                patches = data.unfold(2, 4, 4).unfold(3, 4, 4).contiguous().view(B, 49, 16).permute(1, 0, 2)
                x = input_proj(patches)

                # Solve free phase
                h0 = torch.zeros_like(x)
                h_free, _ = solver.solve(model, h0, x)
                y_pred = output_head(h_free.mean(dim=0))
                test_acc += (y_pred.argmax(-1) == target).float().sum().item()

        test_acc /= len(test_dataset)
        print(f"Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    train_mnist()
