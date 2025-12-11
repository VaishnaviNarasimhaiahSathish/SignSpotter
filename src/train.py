import torch
import torch.nn as nn

# ---------------------------------------------------
# Training loop with metric tracking
# ---------------------------------------------------

def train_model(model, train_loader, val_loader, optimizer, device, num_epochs=10):
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss, running_corrects, total = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(1)
            running_corrects += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = running_corrects / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        # ---- Validation ----
        model.eval()
        val_loss, val_corrects, val_total = 0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * labels.size(0)
                val_corrects += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_corrects / val_total

        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_signspotter_model.pth")

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Acc: {epoch_acc:.4f} | Val Acc: {val_acc:.4f}")

    return train_losses, val_losses, train_accs, val_accs
