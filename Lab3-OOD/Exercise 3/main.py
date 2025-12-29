import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from model import CNN
from utils import get_loaders

def load_model(device):
    model = CNN(in_channels=3, out_channels=64, n_class=10).to(device)
    # Load the best model from Exercise 1.1
    # Assuming the path relative to Exercise 3
    checkpoint_path = '../Exercise 1.1 Build a simple OOD detection pipeline/result/CNN-1/model.pth.tar'
    print(f"=> Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

def odin_preprocessing(model, inputs, epsilon, T):
    # Enable gradient calculation for inputs
    inputs = Variable(inputs, requires_grad=True)
    
    # Forward pass
    outputs = model(inputs)
    
    # Calculate softmax with temperature scaling
    outputs = outputs / T
    
    # Calculate max log probability (predicted class)
    # We want to increase the softmax score of the predicted class
    # So we take the gradient of the log-softmax of the predicted class w.r.t input
    
    # Log softmax
    log_outputs = F.log_softmax(outputs, dim=1)
    
    # Get the max log probability for each input
    max_log_probs, _ = torch.max(log_outputs, dim=1)
    
    # Compute gradients
    model.zero_grad()
    # We want to maximize the score, so we compute gradient of the sum of max_log_probs
    max_log_probs.sum().backward()
    
    # Perturb the input
    # Formula: x' = x - epsilon * sign(-gradient) = x + epsilon * sign(gradient)
    # Wait, the paper says: x_star = x - epsilon * sign(-gradient)
    # The objective is to INCREASE the softmax score.
    # Gradient ascent on the log-softmax score.
    # x_new = x + epsilon * sign(gradient)
    
    gradient = inputs.grad.data
    perturbed_inputs = inputs - epsilon * torch.sign(-gradient)
    
    return perturbed_inputs

def eval_odin(model, test_loader, fake_loader, epsilon, T, device):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    
    id_scores = []
    ood_scores = []
    
    # Evaluate on ID data (CIFAR10)
    for data, _ in test_loader:
        data = data.to(device)
        
        if epsilon > 0:
            data = odin_preprocessing(model, data, epsilon, T)
            
        with torch.no_grad():
            outputs = model(data)
            outputs = outputs / T
            softmax_outputs = F.softmax(outputs, dim=1)
            max_scores, _ = torch.max(softmax_outputs, dim=1)
            
            # Using max softmax score as the confidence
            # ID samples should have higher max scores than OOD
            id_scores.extend(max_scores.cpu().numpy())

    # Evaluate on OOD data (FakeData)
    for data, _ in fake_loader:
        data = data.to(device)
        
        if epsilon > 0:
            data = odin_preprocessing(model, data, epsilon, T)
            
        with torch.no_grad():
            outputs = model(data)
            outputs = outputs / T
            softmax_outputs = F.softmax(outputs, dim=1)
            max_scores, _ = torch.max(softmax_outputs, dim=1)
            
            ood_scores.extend(max_scores.cpu().numpy())

    # ID scores should be higher (1) and OOD scores lower (0) for AUROC calculation?
    # Usually AUROC is calculated for detecting the POSITIVE class. 
    # Can define In-Distribution as Positive (1) and OOD as Negative (0)
    # Or OOD detection task: OOD is Positive (1), ID is Negative (0).
    # Let's say we are detecting OOD. So OOD is 1.
    # OOD samples should have LOWER max softmax scores.
    # So we use (1 - max_score) or negative max_score as the anomaly score.
    
    id_scores = np.array(id_scores)
    ood_scores = np.array(ood_scores)
    
    # Anomaly scores: Higher means more anomalous (OOD)
    # ODIN normally uses Max Softmax Probability. 
    # MSP is higher for ID, lower for OOD.
    # Score for OOD detection = - MSP
    
    scores = np.concatenate([-id_scores, -ood_scores])
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))]) # 0 for ID, 1 for OOD
    
    # AUROC
    auroc = roc_auc_score(labels, scores)
    
    # TNR at TPR 95%
    # We want to detect OOD (Class 1) correctly 95% of the time (TPR=0.95). 
    # What is the TNR (Rate of correctly identifying ID as ID)?
    # TPR = TP / (TP + FN). TP are OOD detected as OOD.
    # TNR = TN / (TN + FP). TN are ID detected as ID.
    
    fpr, tpr, thresholds = roc_curve(labels, scores)
    
    # Search for threshold where TPR is close to 0.95
    # Warning: tpr is increasing. We find the index where tpr >= 0.95
    idx = np.argmax(tpr >= 0.95)
    
    # FPR at this threshold
    fpr_at_95 = fpr[idx]
    
    # TNR = 1 - FPR
    tnr_at_95 = 1 - fpr_at_95
    
    return auroc, tnr_at_95

if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"=> Using device: {device}")
    
    model = load_model(device)
    
    # Parameters for grid search
    temperatures = [1, 10, 100, 1000]
    epsilons = [0, 0.0005, 0.001, 0.0014, 0.002, 0.0024, 0.005, 0.01, 0.02, 0.05]
    
    # Get loaders
    test_loader, fake_loader = get_loaders(batch_size=128, num_workers=4, training=False)
    
    results = []
    
    best_auroc = 0
    best_params = {}
    
    print(f"{'Tempera':<10} | {'Epsilon':<10} | {'AUROC':<10} | {'TNR@95':<10}")
    print("-" * 50)
    
    for T in temperatures:
        for eps in epsilons:
            auroc, tnr = eval_odin(model, test_loader, fake_loader, eps, T, device)
            print(f"{T:<10} | {eps:<10} | {auroc:.4f}     | {tnr:.4f}")
            
            results.append({
                'Temperature': T,
                'Epsilon': eps,
                'AUROC': auroc,
                'TNR': tnr
            })
            
            if auroc > best_auroc:
                best_auroc = auroc
                best_params = {'T': T, 'epsilon': eps, 'tnr': tnr}

    print("\n=> Grid Search Complete")
    print(f"Best Parameters: T={best_params['T']}, Epsilon={best_params['epsilon']}")
    print(f"Best AUROC: {best_auroc:.4f}")
    print(f"TNR@95: {best_params['tnr']:.4f}")
    
    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv('results.csv', index=False)
    print("=> Results saved to results.csv")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    for T in temperatures:
        subset = df[df['Temperature'] == T]
        plt.plot(subset['Epsilon'], subset['AUROC'], marker='o', label=f'T={T}')
        
    plt.xscale('log')
    plt.xlabel('Epsilon (Log Scale)')
    plt.ylabel('AUROC')
    plt.title('ODIN Performance: AUROC vs Epsilon for different Temperatures')
    plt.legend()
    plt.grid(True, which="both", ls="-")
    plt.savefig('odin_results.png')
    print("=> Plot saved to odin_results.png")
