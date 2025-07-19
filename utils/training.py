import torch
import copy
from tqdm import tqdm
import warnings
import sys
warnings.filterwarnings("ignore")

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, max_epochs=10, patience=3, run_id=None):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    epochs_no_improve = 0
    val_accuracies = []

    for epoch in range(max_epochs):
        model.train()
        total, correct = 0, 0
        for frames, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", dynamic_ncols=True, leave=False, file=sys.stdout):
            frames, labels = frames.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total

        val_acc = evaluate(model, val_loader, device)
        scheduler.step(1 - val_acc)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            if run_id is not None:
                torch.save(best_model_wts, f"checkpoints/model_run{run_id}.pth")
            else:
                torch.save(best_model_wts, f"checkpoints/best_model.pth")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    return best_model_wts, val_accuracies

def evaluate(model, val_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for frames, labels in val_loader:
            frames, labels = frames.to(device), labels.to(device)
            outputs = model(frames)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total
