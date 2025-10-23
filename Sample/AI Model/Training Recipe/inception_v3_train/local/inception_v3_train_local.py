import os
import argparse
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import Dataset
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import cv2
import io
from PIL import Image
import itertools


class LocalClassificationDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def load_local_dataset(data_path, test_size=0.2):
    """Load dataset from local directory structure"""
    image_paths = []
    labels = []
    class_names = []

    # Assume each class is in a separate subfolder
    for class_idx, class_name in enumerate(sorted(os.listdir(data_path))):
        class_path = os.path.join(data_path, class_name)
        if os.path.isdir(class_path):
            class_names.append(class_name)
            for filename in os.listdir(class_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(class_path, filename))
                    labels.append(class_idx)

    # Split into train and test
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, labels, test_size=test_size, stratify=labels, random_state=42
    )

    return (train_paths, train_labels), (test_paths, test_labels), class_names


def get_transform(input_size=299, mean=0.5, std=0.5):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize((mean, mean, mean), (std, std, std))
    ])


def create_optimizer(optimizer_name, model_parameters, lr):
    if optimizer_name.lower() == "adam":
        return optim.Adam(model_parameters, lr=lr)
    elif optimizer_name.lower() == "sgd":
        return optim.SGD(model_parameters, lr=lr)
    elif optimizer_name.lower() == "adagrad":
        return optim.Adagrad(model_parameters, lr=lr)
    else:
        raise ValueError("Invalid optimizer name")


def plot_loss_curve(epochs, loss_list, title, color='r'):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_list) + 1), loss_list, color, label=title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)

    # Save plot
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100)
    plt.close()

    buffer.seek(0)
    img_arr = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(img_arr, 1)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def plot_confusion_matrix(true_labels, pred_labels, class_names):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Add text annotations
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Save plot
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    plt.close()

    buffer.seek(0)
    img_arr = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(img_arr, 1)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def train(input_path, output_path='./output', epochs=20, batch_size=16, lr=1e-3, test_size=0.2):
    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    print("Loading dataset...")
    (train_paths, train_labels), (test_paths, test_labels), class_names = load_local_dataset(
        input_path, test_size
    )

    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")
    print(f"Train samples: {len(train_paths)}, Test samples: {len(test_paths)}")

    # Create data transforms
    transform = get_transform()

    # Create datasets
    train_dataset = LocalClassificationDataset(train_paths, train_labels, transform)
    test_dataset = LocalClassificationDataset(test_paths, test_labels, transform)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=0
    )

    # Create model
    print("Creating Inception V3 model...")
    model = torchvision.models.inception_v3(num_classes=num_classes, aux_logits=False, init_weights=True)
    model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer("adam", model.parameters(), lr)

    # Training loop
    print("Starting training...")
    train_losses = []
    test_losses = []
    best_accuracy = 0.0

    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        epoch_train_loss = 0.0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        epoch_test_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                epoch_test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_test_loss = epoch_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        accuracy = 100. * correct / total

        print(".4f")

        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'test_loss': avg_test_loss,
                'accuracy': accuracy,
                'class_names': class_names
            }, os.path.join(output_path, 'best_model.pth'))
            print(f"Best model saved with accuracy: {accuracy:.2f}%")

        # Save model every 5 epochs
        if epoch % 5 == 0 or epoch == epochs:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'test_loss': avg_test_loss,
                'accuracy': accuracy,
                'class_names': class_names
            }, os.path.join(output_path, f'model_epoch_{epoch}.pth'))

    # Save final results
    print("Generating result plots...")

    # Loss curves
    train_loss_img = plot_loss_curve(epochs, train_losses, "Training Loss", 'r')
    test_loss_img = plot_loss_curve(epochs, test_losses, "Test Loss", 'b')

    cv2.imwrite(os.path.join(output_path, 'train_loss.png'), cv2.cvtColor(train_loss_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_path, 'test_loss.png'), cv2.cvtColor(test_loss_img, cv2.COLOR_RGB2BGR))

    # Confusion matrix
    conf_matrix_img = plot_confusion_matrix(all_labels, all_preds, class_names)
    cv2.imwrite(os.path.join(output_path, 'confusion_matrix.png'), cv2.cvtColor(conf_matrix_img, cv2.COLOR_RGB2BGR))

    # Save training summary
    with open(os.path.join(output_path, 'training_summary.txt'), 'w') as f:
        f.write(f"Training Summary\n")
        f.write(f"================\n\n")
        f.write(f"Dataset: {input_path}\n")
        f.write(f"Classes: {', '.join(class_names)}\n")
        f.write(f"Number of classes: {num_classes}\n")
        f.write(f"Train samples: {len(train_paths)}\n")
        f.write(f"Test samples: {len(test_paths)}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Learning rate: {lr}\n")
        f.write(f"Best accuracy: {best_accuracy:.2f}%\n")
        f.write(f"Final train loss: {train_losses[-1]:.4f}\n")
        f.write(f"Final test loss: {test_losses[-1]:.4f}\n")

    print(f"Training completed! Results saved to {output_path}")
    print(f"Best accuracy achieved: {best_accuracy:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Inception V3 locally')
    parser.add_argument('--input_path', type=str, required=True,
                       help='Path to dataset directory (each class should be in separate subfolder)')
    parser.add_argument('--output_path', type=str, default='./output',
                       help='Path to save trained model and results')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test size ratio')

    args = parser.parse_args()
    train(args.input_path, args.output_path, args.epochs, args.batch_size, args.lr, args.test_size)
