import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from src.models import LoopedTransformerBlock
import time

def train_mnist_bp():
    print("Starting MNIST Training with Backpropagation (Baseline)...")

    # Config
    batch_size = 64
    d_model = 128
    n_heads = 4
    d_ff = 512
    epochs = 1
    lr = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model: Same configuration as EqProp
    model = LoopedTransformerBlock(
        d_model, n_heads, d_ff,
        use_linear_attn=True,
        symmetric=True
    ).to(device)

    output_head = nn.Linear(d_model, 10).to(device)
    input_proj = nn.Linear(16, d_model).to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(output_head.parameters()) + list(input_proj.parameters()),
        lr=lr
    )

    print("Training...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_acc = 0
        start_time = time.time()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # Prepare input
            B = data.size(0)
            patches = data.unfold(2, 4, 4).unfold(3, 4, 4).contiguous().view(B, 49, 16).permute(1, 0, 2)
            x = input_proj(patches) # [Seq, Batch, d_model]

            # Forward to equilibrium (unrolled)
            # Use same solver logic manually
            h = torch.zeros_like(x)
            for _ in range(30): # 30 iters like EqProp
                h = (1 - 0.9) * h + 0.9 * model(h, x)

            # Prediction
            y_pred = output_head(h.mean(dim=0))
            loss = F.cross_entropy(y_pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            acc = (y_pred.argmax(-1) == target).float().mean().item()
            total_acc += acc

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f} Acc: {acc:.4f}")

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

                h = torch.zeros_like(x)
                for _ in range(30):
                    h = (1 - 0.9) * h + 0.9 * model(h, x)

                y_pred = output_head(h.mean(dim=0))
                test_acc += (y_pred.argmax(-1) == target).float().sum().item()

        test_acc /= len(test_dataset)
        print(f"Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    train_mnist_bp()
