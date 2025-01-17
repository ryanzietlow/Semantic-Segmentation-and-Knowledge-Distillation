import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import ToPILImage
import torchvision.transforms.functional as TVF
import matplotlib.pyplot as plt
import numpy as np
import time
import argparse
import os
from tqdm import tqdm

# Import from train script dependencies
from train import CustomTransform, compute_miou
from model import CompactSegmentationModel
from resnet50 import load_resnet50_model


def load_model(mode, num_classes=21):
    """
    Load the appropriate model based on the mode
    """
    if mode == 'resnet50':
        model = load_resnet50_model()
    else:
        model = CompactSegmentationModel(num_classes=num_classes)

        # Load the best model for the specified mode
        checkpoint_path = f'best_model_{mode}.pth'
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {checkpoint_path}")
        else:
            print(f"Warning: No checkpoint found for mode {mode}")

    return model


def compute_image_miou(prediction, ground_truth):
    """
    Compute mIoU for a single image
    """
    num_classes = prediction.max() + 1
    miou_per_class = []

    for cls in range(num_classes):
        pred_mask = (prediction == cls)
        true_mask = (ground_truth == cls)

        intersection = np.logical_and(pred_mask, true_mask)
        union = np.logical_or(pred_mask, true_mask)

        if union.sum() == 0:
            miou_per_class.append(1.0)  # Perfect score if no pixels of this class
        else:
            miou_per_class.append(intersection.sum() / union.sum())

    return np.mean(miou_per_class)


def visualize_segmentation_results(images, ground_truth, predictions, mode, num_best_worst=2):
    """
    Visualize segmentation results, selecting images with best and worst mIoU
    """
    # Define color map for VOC dataset
    color_map = np.array([
        [0, 0, 0],  # Background
        [128, 0, 0],  # Aeroplane
        [0, 128, 0],  # Bicycle
        [128, 128, 0],  # Bird
        [0, 0, 128],  # Boat
        [128, 0, 128],  # Bottle
        [0, 128, 128],  # Bus
        [128, 128, 128],  # Car
        [64, 0, 0],  # Cat
        [192, 0, 0],  # Chair
        [64, 128, 0],  # Cow
        [192, 128, 0],  # Dining table
        [64, 0, 128],  # Dog
        [192, 0, 128],  # Horse
        [64, 128, 128],  # Motorbike
        [192, 128, 128],  # Person
        [0, 64, 0],  # Potted plant
        [128, 64, 0],  # Sheep
        [0, 192, 0],  # Sofa
        [128, 192, 0],  # Train
        [0, 64, 128]  # TV/Monitor
    ])

    # Compute mIoU for each image
    image_mious = []
    for i in range(len(images)):
        pred_mask = predictions[i].numpy()
        gt_mask = ground_truth[i].numpy()
        image_mious.append(compute_image_miou(pred_mask, gt_mask))

    # Sort indices by mIoU
    sorted_indices = sorted(range(len(image_mious)), key=lambda k: image_mious[k])

    # Select worst and best images
    selected_indices = sorted_indices[:num_best_worst] + sorted_indices[-num_best_worst:]

    plt.figure(figsize=(20, 5 * len(selected_indices)))

    for i, idx in enumerate(selected_indices):
        # Determine if it's a best or worst image
        performance = "Best" if idx >= len(image_mious) - num_best_worst else "Worst"

        # Denormalize and convert image
        image = images[idx].permute(1, 2, 0).numpy()
        image = (image * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
        image = np.clip(image, 0, 1)

        # Color ground truth
        gt_mask = ground_truth[idx].numpy()
        gt_color = color_map[gt_mask]

        # Color predictions
        pred_mask = predictions[idx].numpy()
        pred_color = color_map[pred_mask]

        # Plot
        plt.subplot(len(selected_indices), 3, i * 3 + 1)
        plt.title(f'Original Image ({performance} mIoU: {image_mious[idx]:.4f})')
        plt.imshow(image)
        plt.axis('off')

        plt.subplot(len(selected_indices), 3, i * 3 + 2)
        plt.title('Ground Truth')
        plt.imshow(image)
        plt.imshow(gt_color, alpha=0.5)
        plt.axis('off')

        plt.subplot(len(selected_indices), 3, i * 3 + 3)
        plt.title(f'Prediction ({mode})')
        plt.imshow(image)
        plt.imshow(pred_color, alpha=0.5)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(f'segmentation_results_{mode}.png')
    plt.close()


def test_model(args):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = load_model(args.mode).to(device)
    model.eval()

    # Setup dataset
    transform = CustomTransform()
    test_dataset = VOCSegmentation(root='./data', year='2012', image_set='val',
                                   download=False, transforms=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Metrics tracking
    total_miou = 0.0
    num_batches = 0
    total_inference_time = 0.0
    total_images = 0

    # Storage for visualization
    all_images = []
    all_ground_truth = []
    all_predictions = []

    # Test loop
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc='Testing'):
            images, targets = images.to(device), targets.to(device)

            # Measure inference time
            start_time = time.time()
            outputs = model(images)
            end_time = time.time()

            # Compute predictions
            _, predicted = torch.max(outputs, 1)

            # Compute metrics
            batch_miou = compute_miou(predicted.cpu().numpy(), targets.cpu().numpy())
            total_miou += batch_miou
            num_batches += 1

            # Accumulate inference times
            batch_inference_time = end_time - start_time
            total_inference_time += batch_inference_time
            total_images += images.size(0)

            # Store for visualization
            all_images.append(images.cpu())
            all_ground_truth.append(targets.cpu())
            all_predictions.append(predicted.cpu())

    # Compute final metrics
    avg_miou = total_miou / num_batches
    avg_inference_time_per_image = total_inference_time / total_images

    # Print results
    print(f"\nTesting Results for {args.mode} mode:")
    print(f"Mean Intersection over Union (mIoU): {avg_miou:.5f}")
    print(f"Average Inference Time per Image: {avg_inference_time_per_image:.5f} seconds")

    # Visualize results
    visualize_segmentation_results(
        torch.cat(all_images),
        torch.cat(all_ground_truth),
        torch.cat(all_predictions),
        args.mode
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test semantic segmentation model')
    parser.add_argument('--mode', type=str,
                        choices=['vanilla', 'knowledge_distillation', 'feature_distillation', 'resnet50'],
                        default='vanilla',
                        help='Model mode to test')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for testing')

    args = parser.parse_args()
    test_model(args)