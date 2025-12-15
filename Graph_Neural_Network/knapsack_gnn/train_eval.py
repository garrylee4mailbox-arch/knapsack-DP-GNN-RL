import torch


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_nodes = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)  # [total_nodes_in_batch]
        y = batch.y.view(-1)   # [total_nodes_in_batch]
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.numel()
        total_nodes += y.numel()

    return total_loss / max(total_nodes, 1)


@torch.no_grad()
def evaluate_node_accuracy(model, loader, device, threshold: float = 0.5):
    model.eval()
    total_correct = 0
    total_nodes = 0

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()
        y = batch.y.view(-1)

        total_correct += (preds == y).sum().item()
        total_nodes += y.numel()

    return total_correct / max(total_nodes, 1)
