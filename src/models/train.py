import torch
import torch.nn.functional as F
from tqdm import tqdm


def train_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    i = 0

    for x, y in tqdm(loader, desc="Train", unit="batch"):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = x.size(0)
        total += loss.item() * batch_size
        i += batch_size

    return total / i


@torch.no_grad()
def evaluate(model, loader, device):
    """Compute loss + top-1 accuracy on validation set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    i = 0

    for x, y in tqdm(loader, desc="Val", unit="batch"):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        i += batch_size

    return total_loss / i, correct / i


@torch.no_grad()
def predict(model, loader, device, idx_map, task: str, writer_unknown_threshold: float):
    """Predict for test.csv and format rows for Kaggle submission."""
    model.eval()
    out = []

    for x, image_ids in tqdm(loader, desc="Predict", unit="batch"):
        x = x.to(device, non_blocking=True)
        logits = model(x)

        if task == "writer":
            probs = F.softmax(logits, dim=1)
            pred_confs, pred_indices = probs.max(dim=1)
            for img_id, pred_conf, pred_idx in zip(image_ids, pred_confs.cpu().numpy(), pred_indices.cpu().numpy()):
                if float(pred_conf) < writer_unknown_threshold:
                    out.append((img_id, "-1"))
                else:
                    out.append((img_id, idx_map[int(pred_idx)]))
        elif task == "pen":
            pred_indices = logits.argmax(dim=1).cpu().numpy()
            for img_id, pi in zip(image_ids, pred_indices):
                out.append((img_id, int(idx_map[int(pi)])))
        else:
            raise ValueError("task must be 'writer' or 'pen'")

    return out
