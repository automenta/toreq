import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.models import LoopedTransformerBlock
from src.solver import EquilibriumSolver
import torch.nn.functional as F
import time

# Global PyTorch optimizations
torch.backends.cudnn.benchmark = True  # Optimize conv/attention for fixed input sizes
torch.set_float32_matmul_precision('high')  # Use TensorCores if available

def train_bp():
    # Configuration
    config = {
        "d_model": 128,
        "n_heads": 4,
        "d_ff": 512,
        "batch_size": 128,
        "max_iters": 50,
        "damping": 0.9,
        "lr": 1e-3,
        "epochs": 5,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

    print(f"Training Baseline (BP) on {config['device']}")

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(784))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    # Model
    embedding = nn.Linear(784, config["d_model"]).to(config["device"])
    model = LoopedTransformerBlock(config["d_model"], config["n_heads"], config["d_ff"]).to(config["device"])
    output_head = nn.Linear(config["d_model"], 10).to(config["device"])

    solver = EquilibriumSolver(
        max_iters=config["max_iters"],
        tol=1e-5,
        damping=config["damping"]
    )

    optimizer = optim.Adam(
        list(embedding.parameters()) + list(model.parameters()) + list(output_head.parameters()),
        lr=config["lr"]
    )

    for epoch in range(config["epochs"]):
        model.train()
        embedding.train()
        output_head.train()

        total_loss = 0
        total_acc = 0
        start_time = time.time()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(config["device"]), target.to(config["device"])

            optimizer.zero_grad()

            x_emb = embedding(data).unsqueeze(0)
            h0 = torch.zeros_like(x_emb)

            # Forward (BPTT through solver)
            h_fixed, iters = solver.solve(model, h0, x_emb)

            y_pred = output_head(h_fixed.mean(dim=0))
            loss = F.cross_entropy(y_pred, target)

            loss.backward()
            optimizer.step()

            acc = (y_pred.argmax(-1) == target).float().mean().item()
            total_loss += loss.item()
            total_acc += acc

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} Acc: {acc:.4f} Iters: {iters}")

        avg_loss = total_loss / len(train_loader)
        avg_acc = total_acc / len(train_loader)
        duration = time.time() - start_time

        print(f"Epoch {epoch} Completed in {duration:.2f}s. Avg Loss: {avg_loss:.4f} Avg Acc: {avg_acc:.4f}")

        # Validation
        model.eval()
        embedding.eval()
        output_head.eval()
        test_acc = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(config["device"]), target.to(config["device"])
                x_emb = embedding(data).unsqueeze(0)
                h0 = torch.zeros_like(x_emb)
                h_fixed, _ = solver.solve(model, h0, x_emb)
                y_pred = output_head(h_fixed.mean(dim=0))
                test_acc += (y_pred.argmax(-1) == target).float().mean().item()

        test_acc /= len(test_loader)
        print(f"Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    train_bp()
