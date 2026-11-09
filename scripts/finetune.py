"""OPTIONAL: fine-tune a 4-output head on NIH ChestX-ray14.

NOTE: written against the TorchXRayVision API but NOT run in the build
environment (the NIH dataset is ~45 GB). Run it on Kaggle/Colab with a GPU and
treat it as a starting point; expect to adjust paths and batch size.

Usage:
  python scripts/finetune.py --imgpath /path/to/images --csv /path/to/Data_Entry_2017.csv

Strategy: keep the pretrained DenseNet121 features, train a fresh 1024->4 linear
head with BCEWithLogitsLoss (stage 1), then optionally unfreeze everything at a
low learning rate (stage 2). Split is by PATIENT to avoid leakage.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchxrayvision as xrv
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402


class Net(nn.Module):
    def __init__(self, backbone, n_out: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(1024, n_out)

    def forward(self, x):
        return self.head(self.backbone.features2(x))  # logits


def make_loaders(args):
    tf = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)])
    ds = xrv.datasets.NIH_Dataset(imgpath=args.imgpath, csvpath=args.csv,
                                  views=["PA", "AP"], transform=tf)
    xrv.datasets.relabel_dataset(config.TARGETS, ds)        # labels -> our 4 columns
    groups = ds.csv["Patient ID"].values
    tr_idx, te_idx = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(np.zeros(len(ds)), groups=groups))
    sub = tr_idx[: args.limit] if args.limit else tr_idx
    mk = lambda idx, shuf: DataLoader(Subset(ds, idx), batch_size=args.bs, shuffle=shuf, num_workers=2)
    return mk(sub, True), mk(te_idx[: args.limit] if args.limit else te_idx, False), ds


def evaluate(net, loader, dev):
    net.eval(); ys, ps = [], []
    with torch.no_grad():
        for b in loader:
            ps.append(torch.sigmoid(net(b["img"].to(dev))).cpu()); ys.append(b["lab"])
    y, p = torch.cat(ys).numpy(), torch.cat(ps).numpy()
    return {t: roc_auc_score(y[:, i], p[:, i]) for i, t in enumerate(config.TARGETS) if len(set(y[:, i])) > 1}


def run_epochs(net, loader, val, opt, epochs, dev, pos_weight):
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(dev))
    for ep in range(epochs):
        net.train()
        for b in loader:
            x, y = b["img"].to(dev), b["lab"].to(dev).float()
            y = torch.nan_to_num(y)
            opt.zero_grad(); loss = loss_fn(net(x), y); loss.backward(); opt.step()
        print(f"epoch {ep + 1}: val AUROC", {k: round(v, 3) for k, v in evaluate(net, val, dev).items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgpath", required=True); ap.add_argument("--csv", required=True)
    ap.add_argument("--bs", type=int, default=32); ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="use only N samples (quick test)")
    ap.add_argument("--unfreeze", action="store_true", help="stage 2: fine-tune backbone too")
    ap.add_argument("--out", default=str(config.ROOT / "models" / "finetuned.pt"))
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, va, ds = make_loaders(args)
    net = Net(xrv.models.DenseNet(weights=config.WEIGHTS), len(config.TARGETS)).to(dev)
    pos = np.nansum(ds.labels, 0); pos_weight = torch.tensor((len(ds) - pos) / np.maximum(pos, 1), dtype=torch.float32)

    for p in net.backbone.parameters(): p.requires_grad = False
    print("baseline (random head) AUROC:", evaluate(net, va, dev))
    run_epochs(net, tr, va, torch.optim.Adam(net.head.parameters(), 1e-3), args.epochs, dev, pos_weight)
    if args.unfreeze:
        for p in net.backbone.parameters(): p.requires_grad = True
        run_epochs(net, tr, va, torch.optim.Adam(net.parameters(), 1e-4), args.epochs, dev, pos_weight)
    torch.save({"state": net.state_dict(), "targets": config.TARGETS}, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
