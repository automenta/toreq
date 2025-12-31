import optuna
import torch
import torch.optim as optim
import time
from src.models import LoopedMLP, ToroidalMLP, BackpropMLP
from src.training import EqPropTrainer, get_mnist_loaders

def objective(trial, model_type="LoopedMLP", time_budget=None, device="cpu"):
    """
    Optuna objective function.
    """
    # Hyperparameters
    alpha = trial.suggest_float("alpha", 0.1, 0.9)
    beta = trial.suggest_float("beta", 0.01, 0.5) if model_type != "BackpropMLP" else 0.0
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    
    # Model
    input_dim = 784
    hidden_dim = 256
    output_dim = 10
    
    if model_type == "LoopedMLP":
        model = LoopedMLP(input_dim, hidden_dim, output_dim, alpha=alpha).to(device)
    elif model_type == "ToroidalMLP":
        model = ToroidalMLP(input_dim, hidden_dim, output_dim, alpha=alpha).to(device)
    elif model_type == "BackpropMLP":
        model = BackpropMLP(input_dim, hidden_dim, output_dim).to(device)
        
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    # Data
    # Use small subset for speed in hyperopt
    train_loader, test_loader = get_mnist_loaders(batch_size=64, train_size=1000, test_size=500)
    
    # Training Loop
    trainer = EqPropTrainer(model, optimizer, beta=beta, alpha=alpha)
    
    start_time = time.time()
    for epoch in range(1): # Just 1 epoch for quick signal or limited time?
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            if model_type == "BackpropMLP":
                optimizer.zero_grad()
                out = model(x)
                loss = torch.nn.functional.cross_entropy(out, y)
                loss.backward()
                optimizer.step()
            else:
                trainer.step(x, y)
                
            # Check time budget
            if time_budget and (time.time() - start_time) > time_budget:
                break
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x) # Uses default steps for inference
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
                
        accuracy = correct / total
        trial.report(accuracy, epoch)
        
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return accuracy

def run_study(study_name, model_type="LoopedMLP", n_trials=10, time_budget=60):
    study = optuna.create_study(direction="maximize", study_name=study_name, storage=f"sqlite:///{study_name}.db", load_if_exists=True)
    
    def func(trial):
        return objective(trial, model_type, time_budget, device="cuda" if torch.cuda.is_available() else "cpu")
        
    # Optimize with timeout
    study.optimize(func, n_trials=n_trials, timeout=time_budget*5) 
    
    print(f"Best trial for {model_type}:")
    print(f"  Value: {study.best_value}")
    print(f"  Params: {study.best_params}")
    return study.best_value
