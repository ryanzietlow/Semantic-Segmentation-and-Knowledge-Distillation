import os
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader
from torchvision.models.segmentation import fcn_resnet50
import numpy as np
from torchvision.transforms import functional as TF
from tqdm import tqdm


# Step 1: Define a function to compute mIoU
def compute_miou(pred, target, num_classes=21):
    pred = pred.flatten()
    target = target.flatten()
    iou = []

    for i in range(num_classes):
        intersection = np.sum((pred == i) & (target == i))
        union = np.sum((pred == i) | (target == i))
        if union == 0:
            iou.append(np.nan)
        else:
            iou.append(intersection / union)

    return np.nanmean(iou)


# Modified CustomTransform class for resnet50.py
class CustomTransform:
    def __init__(self, size=(520, 520)):
        self.size = size

    def __call__(self, img, target):
        # Resize the image and target (segmentation mask)
        img = TF.resize(img, self.size)
        target = TF.resize(target, self.size, interpolation=transforms.InterpolationMode.NEAREST)

        # Convert image to tensor and normalize
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Convert target to tensor (segmentation mask should be long type)
        target = torch.as_tensor(np.array(target), dtype=torch.long)

        # Ensure target values are within valid range [0, num_classes-1]
        target = torch.clamp(target, min=0, max=20)  # For 21 classes (0-20)

        return img, target


# Step 3: Set up the teacher model (pre-trained FCN-ResNet50) and dataset
def evaluate_teacher_model():
    # Define data directory
    data_dir = './data/VOCdevkit/VOC2012'

    # Check if dataset is already extracted
    if not os.path.exists(data_dir):
        # If not extracted, download and extract
        print("Downloading and extracting VOC dataset...")
        dataset = VOCSegmentation(root='./data', year='2012', image_set='val', download=True)
    else:
        print("Using existing VOC dataset...")
        dataset = VOCSegmentation(root='./data', year='2012', image_set='val',
                                  download=False, transforms=CustomTransform())

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the pretrained FCN-ResNet50 model from torchvision
    model = fcn_resnet50(weights=torchvision.models.segmentation.FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)
    model.to(device)  # Move the entire model to the device
    model.eval()

    # Create the transform
    transform = CustomTransform()

    # If transform wasn't applied during dataset creation, apply it now
    if dataset.transforms is None:
        dataset.transforms = transform

    # DataLoader to load the dataset in batches
    dataloader = DataLoader(dataset, batch_size=44, shuffle=False)

    total_miou = 0
    num_batches = 0

    with torch.no_grad():
        # Add a progress bar with tqdm
        for images, targets in tqdm(dataloader, desc="Evaluating Teacher Model"):
            # Move images and targets to device
            images = images.to(device)
            targets = targets.to(device)

            # Get model output
            outputs = model(images)['out']
            _, predicted = torch.max(outputs, 1)  # Predicted classes

            # Compute mIoU for the current batch
            batch_miou = compute_miou(predicted.cpu().numpy(), targets.cpu().numpy())
            total_miou += batch_miou
            num_batches += 1

    # Calculate and print mean mIoU over all batches
    mean_miou = total_miou / num_batches
    print(f'Mean IoU for Teacher Model: {mean_miou:.4f}')

# Function to load the pre-trained ResNet50 model
def load_resnet50_model():
    # Load pre-trained FCN-ResNet50 model from torchvision
    model = fcn_resnet50(weights=torchvision.models.segmentation.FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)
    return model


# Run the evaluation function for the teacher model
if __name__ == "__main__":
    evaluate_teacher_model()
