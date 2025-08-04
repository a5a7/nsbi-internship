from sklearn import metrics
import os
import numpy as np
import torch
import torch.nn as nn
import time

def print_metrics(y_preds, y, thresholds=[0.3, 0.5], background_class=0):
    # Compute multiclass AUC
    auc_ovo = metrics.roc_auc_score(
        y,
        y_preds if y_preds.shape[-1] > 2 else y_preds[:, -1],
        multi_class="ovo",
    )
    print(f"AUC: {auc_ovo:.4f}\n")

    accuracy = metrics.accuracy_score(
        y,
        np.argmax(y_preds, axis=1)
    )

    print(f"ACC: {accuracy:.4f}\n")
    
    num_classes = y_preds.shape[1]

    for signal_class in range(num_classes):
        if signal_class == background_class:
            continue

        # Create binary labels: 1 for signal_class, 0 for background_class, ignore others
        mask = (y == signal_class) | (y == background_class)
        y_bin = (y[mask] == signal_class).astype(int)
        scores_bin = y_preds[mask, signal_class] / (
            y_preds[mask, signal_class] + y_preds[mask, background_class]
        )

        # Compute ROC
        fpr, tpr, _ = metrics.roc_curve(y_bin, scores_bin)

        print(f"Signal class {signal_class} vs Background class {background_class}:")

        for threshold in thresholds:
            bineff = np.argmax(tpr > threshold)
            print(
                "Class {} effS at {} 1.0/effB = {}".format(
                    signal_class, tpr[bineff], 1.0 / fpr[bineff]
                )
            )

            
class Trainer:
    def __init__(
        self,
        train_dataset,
        val_dataset,
        model,
        lr,
        optimizer,
        loss_fn=nn.CrossEntropyLoss,
        device='cuda'
    ):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.model = model.to(device)
        self.device = device
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        self.loss_fn = loss_fn()
        self.best_model_wts = None

    def _run_epoch(self, dataloader, training=True):
        self.model.train() if training else self.model.eval()
        losses = []
        for batch in dataloader:
            X = batch["X"].to(self.device, dtype=torch.float)
            y = batch["y"].to(self.device)

            if training:
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(training):
                preds = self.model(X)
                loss = self.loss_fn(preds, y)
                if training:
                    loss.backward()
                    self.optimizer.step()

            losses.append(loss.item())
        return np.mean(losses)

    def train(self, num_epochs, patience=10):
        best_loss = np.inf
        epochs_no_improve = 0
        t0 = time.time()

        for epoch in range(1, num_epochs + 1):
            train_loss = self._run_epoch(self.train_dataset, training=True)
            val_loss = self._run_epoch(self.val_dataset, training=False)

            print(f"Epoch {epoch}: train loss={train_loss:.4f}, validation loss={val_loss:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                epochs_no_improve = 0
                self.best_model_wts = {
                    k: v.clone() for k, v in self.model.state_dict().items()
                }
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"No improvement for {patience} epochs. Early stopping at epoch {epoch}.")
                    break

        if self.best_model_wts is not None:
            self.model.load_state_dict(self.best_model_wts)
        print(f"Training complete. Total time: {time.time() - t0:.1f}s.")

    def evaluate(self, test_loader, batch_size=256):
        self.model.eval()
        preds = []
        labels = []
        with torch.no_grad():
            for batch in test_loader:
                X = batch["X"].to(self.device, dtype=torch.float)
                labels.append(batch["y"])
                preds.append(self.model(X).softmax(-1))
        return torch.cat(preds).cpu().numpy(), torch.cat(labels).cpu().numpy()


def restore_checkpoint(
    model,
    checkpoint_dir,
    checkpoint_name,
    device=0,
):
    device = "cuda:{}".format(device) if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(
        os.path.join(checkpoint_dir, checkpoint_name),
        map_location=device,
    )


    base_model = model.module if hasattr(model, "module") else model
    base_model.to(device)
        
    if base_model.body is not None and "body" in checkpoint:
        body_state = checkpoint["body"]
        model_state = base_model.body.state_dict()
        filtered_state = {}
        for k, v in body_state.items():        
            if k in model_state and model_state[k].shape == v.shape:
                filtered_state[k] = v
            else:
                print(
                    f"Skipping {k}: shape mismatch (checkpoint: {v.shape}, model: {model_state[k].shape if k in model_state else 'missing'})"
                )

        base_model.body.load_state_dict(filtered_state, strict=False)

    if base_model.classifier is not None and "classifier_head" in checkpoint:
        classifier_state = checkpoint["classifier_head"]
        model_state = base_model.classifier.state_dict()
        filtered_state = {}
        for k, v in classifier_state.items():
            if "out." in k:
                print(f"Skipping {k}: explicitly excluded from loading")
                continue
        
            if k in model_state and model_state[k].shape == v.shape:
                filtered_state[k] = v
            else:
                print(
                        f"Skipping {k}: shape mismatch (checkpoint: {v.shape}, model: {model_state[k].shape if k in model_state else 'missing'})"
                )
        base_model.classifier.load_state_dict(filtered_state, strict=False)


    
