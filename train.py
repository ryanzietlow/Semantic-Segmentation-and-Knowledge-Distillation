import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
from tqdm import tqdm
import argparse
import time
import matplotlib.pyplot as plt
import os
from datetime import timedelta

# Import your custom model and the ResNet50 teacher
from model import CompactSegmentationModel
from resnet50 import load_resnet50_model, CustomTransform, compute_miou


class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        num_classes = logits.shape[1]

        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        # Calculate Dice score for each class
        dice_scores = []
        for i in range(num_classes):
            intersection = (probs[:, i] * targets_one_hot[:, i]).sum()
            union = probs[:, i].sum() + targets_one_hot[:, i].sum()
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice)

        return 1 - torch.mean(torch.stack(dice_scores))


class CombinedLoss:
    def __init__(self, ce_weight=0.5, dice_weight=0.5):
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.dice_loss = DiceLoss()

    def __call__(self, outputs, targets):
        ce_loss = F.cross_entropy(outputs, targets, ignore_index=255)
        dice_loss = self.dice_loss(outputs, targets)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


def get_teacher_features(teacher, images):
    x = teacher.backbone.conv1(images)
    x = teacher.backbone.bn1(x)
    x = teacher.backbone.relu(x)
    x = teacher.backbone.maxpool(x)

    x = teacher.backbone.layer1(x)
    x = teacher.backbone.layer2(x)
    features = teacher.backbone.layer3(x)
    return features


class ModelTrainer:
    def __init__(self, model, device, mode='vanilla', alpha=0.5, beta=0.5):
        self.model = model.to(device)
        self.device = device
        self.mode = mode
        self.alpha = alpha
        self.beta = beta
        self.num_classes = 21
        self.combined_loss = CombinedLoss()

        if mode in ['knowledge_distillation', 'feature_distillation']:
            self.teacher = load_resnet50_model().to(device)
            self.teacher.eval()
            for param in self.teacher.parameters():
                param.requires_grad = False

    def compute_loss(self, student_output, targets, teacher_output=None, student_features=None, teacher_features=None):
        targets = torch.clamp(targets, min=0, max=self.num_classes - 1)

        if self.mode == 'feature_distillation':
            if student_features.shape != teacher_features.shape:
                channel_adapter = nn.Conv2d(student_features.shape[1], teacher_features.shape[1], kernel_size=1).to(
                    student_features.device)
                student_features = channel_adapter(student_features)
                student_features = F.interpolate(student_features, size=teacher_features.shape[2:], mode='bilinear',
                                                 align_corners=False)

            student_features = F.normalize(student_features.view(student_features.size(0), -1), dim=1)
            teacher_features = F.normalize(teacher_features.view(teacher_features.size(0), -1), dim=1)

            feature_loss = 1 - F.cosine_similarity(student_features, teacher_features).mean()
            gt_loss = self.combined_loss(student_output, targets)

            return (self.alpha * feature_loss) + (self.beta * gt_loss)

        elif self.mode == 'knowledge_distillation':
            T = 2.0
            soft_targets = F.log_softmax(student_output / T, dim=1)
            teacher_probs = F.softmax(teacher_output / T, dim=1)
            distillation_loss = -(teacher_probs * soft_targets).sum(dim=1).mean()

            gt_loss = self.combined_loss(student_output, targets)

            return (self.alpha * distillation_loss * (T ** 2)) + (self.beta * gt_loss)

        else:
            return self.combined_loss(student_output, targets)


def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training Mode: {args.mode}")

    model = CompactSegmentationModel(num_classes=21)
    trainer = ModelTrainer(model, device, mode=args.mode)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    transform = CustomTransform()

    # Check if dataset exists before downloading
    def check_and_download_dataset(root, year, image_set):
        dataset_path = os.path.join(root, f'VOCdevkit/VOC{year}')

        # Check if the dataset directory exists
        if not os.path.exists(dataset_path):
            print(f"Dataset not found. Downloading {image_set} dataset...")
            return VOCSegmentation(root=root, year=year, image_set=image_set, download=True, transforms=transform)
        else:
            print(f"{image_set.capitalize()} dataset already exists. Skipping download.")
            return VOCSegmentation(root=root, year=year, image_set=image_set, download=False, transforms=transform)

    # Download/use existing datasets
    train_dataset = check_and_download_dataset(root='./data', year='2012', image_set='train')
    val_dataset = check_and_download_dataset(root='./data', year='2012', image_set='val')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    best_miou = 0.0
    train_losses = []
    val_losses = []

    starttime = time.time()

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0

        epoch_start_time = time.time()
        with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.num_epochs}') as pbar:
            for images, targets in pbar:
                images, targets = images.to(device), targets.to(device)

                student_output, teacher_output = None, None
                student_features, teacher_features = None, None

                if args.mode == 'feature_distillation':
                    student_output, student_features = model(images, return_features=True)
                    with torch.no_grad():
                        teacher_output = trainer.teacher(images)['out']
                        teacher_features = get_teacher_features(trainer.teacher, images)
                elif args.mode == 'knowledge_distillation':
                    student_output = model(images)
                    with torch.no_grad():
                        teacher_output = trainer.teacher(images)['out']
                else:  # vanilla mode
                    student_output = model(images)

                loss = trainer.compute_loss(student_output, targets, teacher_output, student_features, teacher_features)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                pbar.set_postfix({'loss': total_loss / (pbar.n + 1)})

        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        print(f"Epoch {epoch + 1} took {epoch_time:.2f} seconds.")

        train_losses.append(total_loss / len(train_loader))
        val_losses.append(total_loss / len(val_loader))

        # Validation phase
        model.eval()
        val_miou = 0.0
        num_batches = 0

        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc='Validation'):
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                batch_miou = compute_miou(predicted.cpu().numpy(), targets.cpu().numpy())
                val_miou += batch_miou
                num_batches += 1

        val_miou /= num_batches
        print(f'Epoch {epoch + 1} - Validation mIoU: {val_miou:.4f}')

        # Save best model
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_miou': best_miou,
                'mode': args.mode
            }, f'best_model_{args.mode}.pth')
            print(f'New best model saved with mIoU: {best_miou:.4f}')

    endtime = time.time()

    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Loss vs Epochs ({args.mode} mode)')
    plt.legend()
    plt.savefig(f'loss_plot_{args.mode}.png')
    plt.show()

    # Calculate total time and total images
    total_time = endtime - starttime
    total_train_images = len(train_dataset)
    total_val_images = len(val_dataset)
    total_images = total_train_images + total_val_images

    # Print summary statistics
    print(f"Total time taken: {timedelta(seconds=total_time)}")
    print(f"Best mIoU: {best_miou:.5f}")

    # Calculate time per image
    time_per_image = total_time / (total_images * args.num_epochs)
    print(f"Time per image: {time_per_image:.5f} seconds")

    # Print number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train segmentation model with different modes')
    parser.add_argument('--mode', type=str, choices=['vanilla', 'knowledge_distillation', 'feature_distillation'],
                        default='vanilla', help='Training mode')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')

    args = parser.parse_args()
    train_model(args)