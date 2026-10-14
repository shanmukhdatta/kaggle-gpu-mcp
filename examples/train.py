"""Kaggle GPU stage: PyTorch binary classification on output of prep.py.

Use only as a smoke test. Real ML projects require leakage-safe splits,
project-specific validation, and suitable ethical/legal data sourcing.
"""
import csv
import glob
import json
import random
from pathlib import Path

import torch

SEED = 47
random.seed(SEED)
torch.manual_seed(SEED)
if not torch.cuda.is_available():
    raise RuntimeError("The GPU smoke test requires CUDA, but Kaggle did not supply it")
device = torch.device("cuda")
probe = torch.ones((128, 128), device=device)
probe = probe @ probe
torch.cuda.synchronize()
assert probe.is_cuda

paths = sorted(glob.glob("/kaggle/input/**/merged.csv", recursive=True))
if not paths:
    raise FileNotFoundError("Attach the prep Kaggle kernel output as a kernel source")
with open(paths[0], newline="", encoding="utf-8") as source:
    rows = list(csv.DictReader(source))
if not rows or not {"f1", "f2", "label"}.issubset(rows[0]):
    raise ValueError("Demo trainer expects f1, f2, label columns")
x = torch.tensor([[float(r["f1"]), float(r["f2"])] for r in rows], dtype=torch.float32)
y = torch.tensor([[float(r["label"])] for r in rows], dtype=torch.float32)
shuffle = torch.randperm(len(rows))
split = int(len(rows) * .8)
if split <= 0 or split >= len(rows):
    raise ValueError("Need at least 3 data rows")
train, valid = shuffle[:split], shuffle[split:]
x_train, y_train = x[train].to(device), y[train].to(device)
x_valid, y_valid = x[valid].to(device), y[valid].to(device)

model = torch.nn.Sequential(
    torch.nn.Linear(2, 16), torch.nn.ReLU(),
    torch.nn.Linear(16, 1),
).to(device)
loss_fn = torch.nn.BCEWithLogitsLoss()
optim = torch.optim.Adam(model.parameters(), lr=0.03)
for epoch in range(15):
    model.train()
    optim.zero_grad()
    loss = loss_fn(model(x_train), y_train)
    loss.backward()
    optim.step()
    print(f"epoch={epoch+1} loss={loss.item():.5f}")
model.eval()
with torch.no_grad():
    logits = model(x_valid)
    val_loss = loss_fn(logits, y_valid).item()
    accuracy = ((torch.sigmoid(logits) > .5) == y_valid.bool()).float().mean().item()

out = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
out.mkdir(exist_ok=True)
# Save only the state dictionary, not an arbitrary pickled model object.
torch.save(model.cpu().state_dict(), out / "model.pt")
# Prove the saved checkpoint can be safely reloaded and used for inference.
restored = torch.nn.Sequential(
    torch.nn.Linear(2, 16), torch.nn.ReLU(),
    torch.nn.Linear(16, 1),
)
restored.load_state_dict(torch.load(out / "model.pt", map_location="cpu", weights_only=True))
restored.eval()
with torch.no_grad():
    assert restored(torch.tensor([[0.5, -0.2]])).shape == (1, 1)
(out / "metrics.json").write_text(json.dumps({
    "status": "success", "epochs_completed": 15,
    "validation_accuracy": accuracy, "validation_loss": val_loss,
    "train_rows": len(train), "validation_rows": len(valid),
}, indent=2), encoding="utf-8")
(out / "run_manifest.json").write_text(json.dumps({
    "device": "cuda", "gpu_verified": True,
    "gpu_name": torch.cuda.get_device_name(0),
    "torch_version": torch.__version__,
    "data_rows": len(rows), "seed": SEED,
    "checkpoint_reload_verified": True,
}, indent=2), encoding="utf-8")
print(f"TRAINING COMPLETED on {torch.cuda.get_device_name(0)}; accuracy={accuracy:.4f}")
