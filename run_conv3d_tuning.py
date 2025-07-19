import torch
import random
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from datasets.conv3d_dataset import UCF101Dataset
from models.conv3d import C3D_QuoVadis
from utils.logger import log_results_to_csv
from utils.training import train_model

def main():
    # Transform: Resize and convert to tensor
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.ToTensor()
    ])

    # Dataset paths
    train_csv = "D:\\ActionRecognition\\splits\\train.csv"
    val_csv = "D:\\ActionRecognition\\splits\\val.csv"
    root_dir = "D:\\TheNewTwoStream\\data"


    #  Static dataset (recreated only if batch size changes)
    train_dataset = UCF101Dataset(root_dir, train_csv, num_frames=16, transform=transform)
    val_dataset = UCF101Dataset(root_dir, val_csv, num_frames=16, transform=transform)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = CrossEntropyLoss()
    num_classes = 10  # Adjust based on your current UCF101 subset

    # Important Hyperparameters to tune
    param_space = {
        "learning_rate": [1e-4, 1e-3],
        "dropout": [0.3, 0.5],
        "weight_decay": [0, 1e-4],
        "batch_size": [8, 16]
    }

    results = []
    n_trials = 20

    print(f"\n Starting random search over {n_trials} trials...\n")
    for run_id in tqdm(range(1, n_trials + 1), desc="Hyperparameter Tuning", unit="trial"):
        # Sample config
        config = {
            "learning_rate": random.choice(param_space["learning_rate"]),
            "dropout": random.choice(param_space["dropout"]),
            "weight_decay": random.choice(param_space["weight_decay"]),
            "batch_size": random.choice(param_space["batch_size"])
        }

        print(f"\n Run {run_id}: {config}")

        # Initialize C3D model
        model = C3D_QuoVadis(num_classes=num_classes).to(device)

        # Optimizer & Scheduler
        optimizer = Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

        # Loaders with tuned batch size
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=2, pin_memory=True)

        # Training
        _, val_accuracies = train_model(
            model, train_loader, val_loader, criterion,
            optimizer, scheduler, device,
            max_epochs=12, patience=2, run_id=run_id
        )

        best_val_acc = max(val_accuracies)
        results.append({
            "run": run_id,
            **config,
            "val_acc": round(best_val_acc * 100, 2)
        })

    # Save results
    log_results_to_csv(results, filename="results/conv3d_results.csv")
    print("\n Random search complete. Results saved to `results/conv3d_results.csv`.")


if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()
    main()
