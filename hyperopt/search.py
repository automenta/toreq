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
    symmetric = trial.suggest_categorical("symmetric", [True, False]) if model_type == "LoopedMLP" else False
    buffer_decay = trial.suggest_float("buffer_decay", 0.5, 0.99) if model_type == "ToroidalMLP" else 0.9
    
    # Dynamics parameters
    if model_type != "BackpropMLP":
        max_steps = trial.suggest_categorical("max_steps", [20, 50, 100])
        epsilon = trial.suggest_categorical("epsilon", [1e-3, 1e-4, 1e-5])
    else:
        max_steps = 50
        epsilon = 1e-4
    
    # Model
    input_dim = 784
    hidden_dim = 256
    output_dim = 10
    
    if model_type == "LoopedMLP":
        model = LoopedMLP(input_dim, hidden_dim, output_dim, alpha=alpha, symmetric=symmetric).to(device)
    elif model_type == "ToroidalMLP":
        model = ToroidalMLP(input_dim, hidden_dim, output_dim, alpha=alpha, decay=buffer_decay).to(device)
    elif model_type == "BackpropMLP":
        model = BackpropMLP(input_dim, hidden_dim, output_dim).to(device)
        
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    # Data
    # Use small subset for speed in hyperopt
    train_loader, test_loader = get_mnist_loaders(batch_size=64, train_size=1000, test_size=500)
    
    # Training Loop
    trainer = EqPropTrainer(model, optimizer, beta=beta, alpha=alpha, epsilon=epsilon, max_steps=max_steps)
    
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
        # Pruning not supported for multi-objective in this version
        # trial.report(accuracy, epoch)
        # if trial.should_prune():
        #     raise optuna.exceptions.TrialPruned()
            
    param_count = sum(p.numel() for p in model.parameters())

    return accuracy, (time.time() - start_time), param_count

def run_study(study_name, model_type="LoopedMLP", n_trials=10, time_budget=60):
    # Multi-objective: Maximize Acc, Minimize Time, Minimize Params
    study = optuna.create_study(directions=["maximize", "minimize", "minimize"], 
                                study_name=study_name, 
                                storage=f"sqlite:///{study_name}.db", 
                                load_if_exists=True)
    
    def func(trial):
        return objective(trial, model_type, time_budget, device="cuda" if torch.cuda.is_available() else "cpu")
        
    study.optimize(func, n_trials=n_trials, timeout=time_budget*5) 
    
    print(f"Pareto Front for {model_type}:")
    for t in study.best_trials:
        acc, dur, params = t.values
        print(f"  Trial {t.number}: Acc={acc:.4f}, Time={dur:.2f}s, Params={params}, Config={t.params}")
        
    best_acc = max([t.values[0] for t in study.best_trials])
    return best_acc
