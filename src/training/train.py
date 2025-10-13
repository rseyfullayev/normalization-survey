import argparse, csv, os, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.datasets.cifar100 import loadData
from src.models.resnet18 import Resnet18




def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)

def write_csv_header(path, fieldnames):
    new = not Path(path).exists()
    f = open(path, "a", newline="")
    w = csv.DictWriter(f, fieldnames=fieldnames)
    if new:
        w.writeheader()
    return f, w

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    tot_loss, tot_correct, tot = 0.0, 0, 0
    for x, y in tqdm(loader, desc="train", ncols=80, leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item() * y.size(0)
        tot_correct += (out.argmax(1) == y).sum().item()
        tot += y.size(0)
    return tot_loss / tot, 100.0 * tot_correct / tot

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    tot_loss, tot_correct, tot = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        tot_loss += loss.item() * y.size(0)
        tot_correct += (out.argmax(1) == y).sum().item()
        tot += y.size(0)
    return tot_loss / tot, 100.0 * tot_correct / tot



def run_one_norm(norm_type, args, device, epoch_writer=None, save_dir="checkpoints"):

    train_loader, val_loader, test_loader = loadData(
        batch=args.batch, valid=args.valid, workers=args.workers,
        seed=args.seed, norm_type=norm_type
    )


    model = Resnet18(classes=100, pretrained=False).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)


    if args.cosine:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None

    best = {"val_acc": 0.0, "epoch": 0}
    ckpt_path = Path(save_dir) / f"resnet18_{norm_type}_best.pth"
    ensure_dir(ckpt_path.parent)


    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()


        if va_acc > best["val_acc"]:
            best = {"val_acc": va_acc, "epoch": ep}
            torch.save(model.state_dict(), ckpt_path)


        row = {
            "norm": norm_type, "epoch": ep,
            "train_loss": f"{tr_loss:.6f}", "train_acc": f"{tr_acc:.4f}",
            "val_loss": f"{va_loss:.6f}", "val_acc": f"{va_acc:.4f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.6g}",
            "batch": args.batch, "seed": args.seed,
        }
        if epoch_writer is not None:
            epoch_writer.writerow(row)

        print(f"[{norm_type}] epoch {ep:02d} | "
              f"train {tr_acc:5.2f}% (loss {tr_loss:.4f}) | "
              f"val {va_acc:5.2f}% (best {best['val_acc']:5.2f}% @ {best['epoch']})")


    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    te_loss, te_acc = evaluate(model, test_loader, criterion, device)

    return best["val_acc"], te_acc, ckpt_path.as_posix()


def parse_args():
    p = argparse.ArgumentParser("Train across input normalizations with per-epoch CSV logging + test eval")
    p.add_argument("--norms", nargs="+", default=["none", "l2", "linf"],
                   help="choose from {none,l1,l2,linf} (e.g., --norms none l1 l2)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--valid", type=int, default=5000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cosine", action="store_true", help="use cosine LR schedule")
    p.add_argument("--epoch_csv", type=str, default="logs/norm_epochs.csv",
                   help="CSV with one row per epoch & normalization")
    p.add_argument("--summary_csv", type=str, default="logs/norm_summary.csv",
                   help="CSV with best val & test per normalization")
    p.add_argument("--save_dir", type=str, default="checkpoints",
                   help="where to save best checkpoints")
    return p.parse_args()

def main():
    args = parse_args()
    for n in args.norms:
        if n not in {"none","l1","l2","linf"}:
            raise ValueError(f"Bad norm '{n}'. Choose from none,l1,l2,linf.")

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, " norms:", args.norms)


    ensure_dir(Path(args.epoch_csv).parent)
    ensure_dir(Path(args.summary_csv).parent)

    epoch_fields = ["norm","epoch","train_loss","train_acc","val_loss","val_acc","lr","batch","seed"]
    epoch_f, epoch_writer = write_csv_header(args.epoch_csv, epoch_fields)

    summary_fields = ["norm","best_val_top1","test_top1","best_ckpt"]
    summary_f, summary_writer = write_csv_header(args.summary_csv, summary_fields)

    t0 = time.time()
    try:
        for norm in args.norms:
            print(f"\ntraining with {norm.upper()} normalization")
            best_val, test_acc, ckpt = run_one_norm(
                norm_type=norm, args=args, device=device,
                epoch_writer=epoch_writer, save_dir=args.save_dir
            )
            summary_writer.writerow({
                "norm": norm,
                "best_val_top1": f"{best_val:.4f}",
                "test_top1": f"{test_acc:.4f}",
                "best_ckpt": ckpt
            })
            print(f"=> [{norm}] best val: {best_val:.2f}% | test: {test_acc:.2f}% | ckpt: {ckpt}")
    finally:
        epoch_f.close()
        summary_f.close()

    mins = (time.time()-t0)/60
    print(f"\nWrote epoch log → {args.epoch_csv}")
    print(f"Wrote summary   → {args.summary_csv}")
    print(f"Total time: {mins:.1f} min")

if __name__ == "__main__":
    main()
