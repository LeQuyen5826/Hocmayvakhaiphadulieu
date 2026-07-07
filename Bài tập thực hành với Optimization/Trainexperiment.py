
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

def get_dataloaders(batch_size=128, val_ratio=0.1, augment=False, data_root="./data"):
    """Tai CIFAR-10, tach train/val, tra ve DataLoader."""
    if augment:
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    full_train = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=train_tf)
    test_set = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=test_tf)

    n_val = int(len(full_train) * val_ratio)
    n_train = len(full_train) - n_val
    train_set, val_set = random_split(
        full_train, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2)
    return train_loader, val_loader, test_loader


class BaselineCNN(nn.Module):
    """B1 - CNN co ban: KHONG BatchNorm, KHONG Dropout."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                   # 32->16

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                   # 16->8

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                   # 8->4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class ImprovedCNN(nn.Module):
    """B2 - CNN cai tien: CO BatchNorm + Dropout (dropout_rate thay doi duoc)."""
    def __init__(self, num_classes=10, dropout_rate=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate * 0.5),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate * 0.75),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def train_model(model, optimizer, train_loader, val_loader, epochs=30,
                 patience=5, min_delta=0.001, tag="model"):
    """
    Train mot mo hinh, ghi lai:
      - train_loss, val_loss, val_acc theo tung epoch
      - thoi gian train tong cong
      - epoch hoi tu: epoch dau tien ma sau do val_acc khong cai thien
        > min_delta trong `patience` epoch lien tiep
    """
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    best_acc = 0.0
    epochs_no_improve = 0
    convergence_epoch = None
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, n_samples = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            n_samples += x.size(0)
        train_loss = running_loss / n_samples

        model.eval()
        correct, total, val_loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                loss = criterion(out, y)
                val_loss_sum += loss.item() * x.size(0)
                correct += (out.argmax(dim=1) == y).sum().item()
                total += y.size(0)
        val_acc = correct / total
        val_loss = val_loss_sum / total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"[{tag}] Epoch {epoch:02d}/{epochs} "
              f"- train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_acc + min_delta:
            best_acc = val_acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if convergence_epoch is None and epochs_no_improve >= patience:
            convergence_epoch = epoch - patience
            print(f"[{tag}] --> Mo hinh hoi tu tai epoch {convergence_epoch} "
                  f"(khong cai thien them sau {patience} epoch)")

    total_time = time.time() - start_time
    if convergence_epoch is None:
        convergence_epoch = epochs  

    return {
        "tag": tag,
        "history": history,
        "best_val_acc": best_acc,
        "final_val_acc": history["val_acc"][-1],
        "train_time_sec": total_time,
        "convergence_epoch": convergence_epoch,
    }


def run_all_experiments(epochs=30, batch_size=128):
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size)

    results = []

    model = BaselineCNN()
    opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.0)
    results.append(train_model(model, opt, train_loader, val_loader,
                                epochs=epochs, tag="B1_Baseline_SGD"))

    configs = [
        # (nhan, model, ham tao optimizer)
        ("B2_BN_Drop0.3_SGDMomentum", ImprovedCNN(dropout_rate=0.3),
         lambda p: optim.SGD(p, lr=0.01, momentum=0.9)),
        ("B2_BN_Drop0.5_SGDMomentum", ImprovedCNN(dropout_rate=0.5),
         lambda p: optim.SGD(p, lr=0.01, momentum=0.9)),
        ("B2_BN_Drop0.3_Adam", ImprovedCNN(dropout_rate=0.3),
         lambda p: optim.Adam(p, lr=0.001)),
        ("B2_BN_Drop0.5_Adam", ImprovedCNN(dropout_rate=0.5),
         lambda p: optim.Adam(p, lr=0.001)),
    ]

    for tag, m, opt_fn in configs:
        optimizer = opt_fn(m.parameters())
        results.append(train_model(m, optimizer, train_loader, val_loader,
                                    epochs=epochs, tag=tag))

    return results, test_loader


def summarize(results, out_csv="results_summary.csv"):
    rows = [{
        "Model": r["tag"],
        "Best Val Acc": round(r["best_val_acc"], 4),
        "Final Val Acc": round(r["final_val_acc"], 4),
        "Train Time (s)": round(r["train_time_sec"], 1),
        "Convergence Epoch": r["convergence_epoch"],
    } for r in results]
    df = pd.DataFrame(rows)
    print("\n===== BANG TONG HOP KET QUA =====")
    print(df.to_string(index=False))
    df.to_csv(out_csv, index=False)
    print(f"\nDa luu bang ket qua vao: {out_csv}")
    return df


def plot_curves(results, out_png="training_curves.png"):
    plt.figure(figsize=(13, 5))

    plt.subplot(1, 2, 1)
    for r in results:
        plt.plot(r["history"]["train_loss"], label=r["tag"])
    plt.xlabel("Epoch"); plt.ylabel("Training Loss")
    plt.title("Training Loss theo Epoch")
    plt.legend(fontsize=7)

    plt.subplot(1, 2, 2)
    for r in results:
        plt.plot(r["history"]["val_acc"], label=r["tag"])
    plt.xlabel("Epoch"); plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy theo Epoch")
    plt.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Da luu bieu do vao: {out_png}")
    plt.show()



if __name__ == "__main__":
    print(f"Su dung thiet bi: {DEVICE}")
    all_results, test_loader = run_all_experiments(epochs=30, batch_size=128)
    df_summary = summarize(all_results)
    plot_curves(all_results)