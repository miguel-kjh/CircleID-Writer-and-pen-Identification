import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics import F1Score, Precision, Recall


class CircleIDModule(pl.LightningModule):
    def __init__(self, net: nn.Module, lr: float, task: str,
                 idx_map: dict, writer_unknown_threshold: float = 0.9):
        super().__init__()
        self.net = net
        # Store idx_map with string keys so YAML serialisation is round-trip safe
        str_idx_map = {str(k): v for k, v in idx_map.items()}
        self.save_hyperparameters({"lr": lr, "task": task,
                                   "idx_map": str_idx_map,
                                   "writer_unknown_threshold": writer_unknown_threshold},
                                  ignore=["net"])

        num_classes = len(idx_map)
        metric_kwargs = dict(task="multiclass", num_classes=num_classes, average="macro")
        self.train_f1        = F1Score(**metric_kwargs)
        self.train_precision = Precision(**metric_kwargs)
        self.train_recall    = Recall(**metric_kwargs)
        self.val_f1          = F1Score(**metric_kwargs)
        self.val_precision   = Precision(**metric_kwargs)
        self.val_recall      = Recall(**metric_kwargs)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(1)
        self.train_f1.update(preds, y)
        self.train_precision.update(preds, y)
        self.train_recall.update(preds, y)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.log("train/f1",        self.train_f1.compute())
        self.log("train/precision", self.train_precision.compute())
        self.log("train/recall",    self.train_recall.compute())
        self.train_f1.reset()
        self.train_precision.reset()
        self.train_recall.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(1)
        acc = (preds == y).float().mean()
        self.val_f1.update(preds, y)
        self.val_precision.update(preds, y)
        self.val_recall.update(preds, y)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc",  acc,  on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        self.log("val/f1",        self.val_f1.compute(), prog_bar=True)
        self.log("val/precision", self.val_precision.compute())
        self.log("val/recall",    self.val_recall.compute())
        self.val_f1.reset()
        self.val_precision.reset()
        self.val_recall.reset()

    def predict_step(self, batch, batch_idx):
        x, image_ids = batch
        logits = self(x)
        idx_map   = self.hparams.idx_map        # str keys
        task      = self.hparams.task
        threshold = self.hparams.writer_unknown_threshold
        results = []
        if task == "writer":
            probs = F.softmax(logits, dim=1)
            confs, indices = probs.max(dim=1)
            for img_id, conf, idx in zip(image_ids, confs.cpu(), indices.cpu()):
                label = "-1" if float(conf) < threshold else idx_map[str(int(idx))]
                results.append((img_id, label))
        else:
            for img_id, idx in zip(image_ids, logits.argmax(1).cpu()):
                results.append((img_id, int(idx_map[str(int(idx))])))
        return results

    def configure_optimizers(self):
        return torch.optim.AdamW(self.net.parameters(), lr=self.hparams.lr)

    @classmethod
    def from_checkpoint(cls, ckpt_path: str, net_builder) -> "CircleIDModule":
        """Load module from Lightning checkpoint, rebuilding the net."""
        ckpt = torch.load(ckpt_path, map_location="cpu")
        hparams = ckpt["hyper_parameters"]
        num_classes = len(hparams["idx_map"])
        net = net_builder(num_classes)
        return cls.load_from_checkpoint(ckpt_path, net=net)
