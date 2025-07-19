import torch
import time
from tqdm import tqdm
from itertools import product
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from datasets.ucf101_dataset import UCF101Dataset
from models.conv_lstm import CNNLSTM
from utils.training import train_model
from utils.logger import log_results_to_csv
import sys

def main():
    # Data and transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.ToTensor()
    ])

    train_csv = "D:\\ActionRecognition\\splits\\train.csv"
    val_csv = "D:\\ActionRecognition\\splits\\val.csv"
    root_dir = "D:\\TheNewTwoStream\\data"

    train_dataset = UCF101Dataset(root_dir, train_csv, num_frames=16, transform=transform)
    val_dataset = UCF101Dataset(root_dir, val_csv, num_frames=16, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Training config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = CrossEntropyLoss()
    num_classes = 10

    # Hyperparameter grid
    hidden_sizes = [256, 512, 1024]
    lstm_layers_list = [1, 2]
    learning_rates = [1e-4, 1e-3]
    dropouts = [0.3, 0.5]
    bidirectionals = [False, True]

    all_combos = list(product(hidden_sizes, lstm_layers_list, learning_rates, dropouts, bidirectionals))
    results = []
    best_config = None
    outer_bar = tqdm(all_combos, desc="Hyperparameter Tuning", dynamic_ncols=True, leave=True)
    best_acc = 0.0

    for run_id, (hs, layers, lr, dropout, bidir) in enumerate(outer_bar, start=1):
        outer_bar.set_postfix(run=run_id)
        tqdm.write(f"Run {run_id}: hs={hs}, layers={layers}, lr={lr}, dropout={dropout}, bidir={bidir}")
        model = CNNLSTM(hidden_size=hs, lstm_layers=layers, num_classes=num_classes,
                        bidirectional=bidir, dropout=dropout).to(device)
        optimizer = Adam(model.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

        start_time = time.time()
        _, val_accuracies = train_model(
            model, train_loader, val_loader, criterion, optimizer,
            scheduler, device, max_epochs=10, patience=2, run_id=run_id
        )
        elapsed_time = time.time() - start_time

        best_val_acc = max(val_accuracies)
        if best_val_acc > best_acc:
            best_acc = best_val_acc
            best_config = {
                "run": run_id,
                "hidden_size": hs,
                "lstm_layers": layers,
                "learning_rate": lr,
                "dropout": dropout,
                "bidirectional": bidir,
                "val_acc": round(best_val_acc * 100, 2),
                "time": round(elapsed_time, 2)
            }

        results.append({
            "run": run_id,
            "hidden_size": hs,
            "lstm_layers": layers,
            "learning_rate": lr,
            "dropout": dropout,
            "bidirectional": bidir,
            "val_acc": round(best_val_acc * 100, 2),
            "time": round(elapsed_time, 2)
        })

    # Save results
    log_results_to_csv(results)

    # Print best config
    print("\nTuning complete. Results saved to 'results/conv_lstm_results.csv'")
    print("Best Run:")
    for k, v in best_config.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
