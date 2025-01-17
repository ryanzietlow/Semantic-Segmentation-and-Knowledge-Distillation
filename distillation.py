import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
from torchvision import models
from PIL import Image
import numpy as np
from tqdm import tqdm
import time
from model import CompactUNetV2
from sklearn.metrics import jaccard_score


# Data preparation function (remains the same)
def get_dataloaders(batch_size=64):
    class VOCSegmentationWithMask(datasets.VOCSegmentation):
        def __getitem__(self, index):
            img, target = super().__getitem__(index)
            img = transforms.Resize((256, 256))(img)
            img = transforms.ToTensor()(img)
            target = transforms.Resize((256, 256), interpolation=Image.NEAREST)(target)
            target = torch.tensor(np.array(target), dtype=torch.long)
            target = target.clamp(0, 20)  # Ensure valid class labels
            return img, target

    root_dir = './data'
    voc_dir = os.path.join(root_dir, 'VOCdevkit')

    train_dataset = VOCSegmentationWithMask(root=root_dir, year='2012', image_set='train',
                                            download=not os.path.exists(voc_dir))
    val_dataset = VOCSegmentationWithMask(root=root_dir, year='2012', image_set='val',
                                          download=not os.path.exists(voc_dir))

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
    }

    return dataloaders


# Distillation loss functions
class ResponseBasedDistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, temperature=4.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, targets):
        # Ensure teacher and student logits have the same spatial dimensions
        teacher_logits = F.interpolate(teacher_logits, size=student_logits.shape[2:], mode='bilinear',
                                       align_corners=False)

        # Apply temperature scaling
        teacher_logits = teacher_logits / self.temperature
        student_logits = student_logits / self.temperature

        # Compute KL divergence loss
        kd_loss = self.kl_div(F.log_softmax(student_logits, dim=1),
                              F.softmax(teacher_logits, dim=1)) * (self.temperature ** 2)

        # Compute cross-entropy loss
        ce_loss = self.ce_loss(student_logits, targets)

        # Combine losses
        return self.alpha * kd_loss + (1 - self.alpha) * ce_loss


# Get teacher and student models
def get_teacher_model():
    model = models.resnet50(weights='IMAGENET1K_V1')
    # Modify ResNet50 to create a segmentation output
    model = nn.Sequential(*list(model.children())[:-2])  # Remove fc and avgpool
    model.add_module('segmentation_head', nn.Conv2d(2048, 21, kernel_size=1))
    return model


def get_student_model():
    model = CompactUNetV2(in_channels=3, out_channels=21)
    # Ensure that the weights file exists and is loaded properly, else skip it
    try:
        model.load_state_dict(torch.load('unet_model.pth', weights_only=True))
    except FileNotFoundError:
        print("Pre-trained weights file not found, skipping weight loading.")
    return model


# Evaluation function
def evaluate_model(model, val_loader, device):
    model.eval()
    total_correct = 0
    total_pixels = 0
    miou_scores = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            preds = outputs.argmax(dim=1)

            total_correct += (preds == labels).sum().item()
            total_pixels += labels.numel()

            # Compute mIoU for each batch
            miou_batch = calculate_miou(preds.cpu().numpy(), labels.cpu().numpy())
            miou_scores.append(miou_batch)

    accuracy = total_correct / total_pixels
    mean_miou = torch.tensor(miou_scores).mean().item()

    return accuracy, mean_miou


# Calculate mIoU
def calculate_miou(preds, labels, num_classes=21):
    miou = []
    for i in range(num_classes):
        intersection = ((preds == i) & (labels == i)).sum()
        union = ((preds == i) | (labels == i)).sum()
        iou = intersection / (union + 1e-6)
        miou.append(iou)
    return np.mean(miou)


# Main function
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    teacher_model = get_teacher_model().to(device)
    student_model = get_student_model().to(device)

    # Freeze teacher model parameters
    for param in teacher_model.parameters():
        param.requires_grad = False

    distillation_loss_fn = ResponseBasedDistillationLoss(temperature=4.0, alpha=0.6)
    optimizer = optim.Adam(student_model.parameters(), lr=1e-4)

    dataloaders = get_dataloaders(batch_size=64)
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']

    num_epochs = 25
    best_model_wts = student_model.state_dict()
    best_loss = float('inf')
    best_accuracy = 0.0

    for epoch in tqdm(range(num_epochs), desc='Epochs', position=0):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')

        # Training phase
        student_model.train()
        teacher_model.eval()
        running_loss = 0.0
        running_corrects = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Get teacher logits
            with torch.no_grad():
                teacher_logits = teacher_model(inputs)
                # Ensure teacher_logits is the logits tensor
                if isinstance(teacher_logits, tuple):
                    teacher_logits = teacher_logits[0]

            # Get student logits
            student_logits, _ = student_model(inputs)

            # Compute the distillation loss
            loss = distillation_loss_fn(student_logits, teacher_logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = student_logits.argmax(dim=1)
            running_corrects += (preds == labels).sum().item()
            total += labels.size(0)

        # Compute training metrics
        train_loss = running_loss / len(train_loader)
        train_accuracy = running_corrects / total
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')

        # Validation phase
        student_model.eval()
        val_accuracy, miou = evaluate_model(student_model, val_loader, device)
        print(f'Val Accuracy: {val_accuracy:.4f}, mIoU: {miou:.4f}')

        # Update best model
        if miou > best_accuracy:
            best_accuracy = miou
            best_model_wts = student_model.state_dict()
            torch.save(best_model_wts, 'best_student_model.pth')

    print('Best validation mIoU: {:.4f}'.format(best_accuracy))


if __name__ == '__main__':
    main()