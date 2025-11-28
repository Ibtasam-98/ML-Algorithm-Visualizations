import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import cv2
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torchvision import transforms
import os
from datetime import datetime


# Simple CNN model in PyTorch
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Grad-CAM implementation for PyTorch
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.forward_hook = target_layer.register_forward_hook(self.save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()

    def generate_heatmap(self, input_tensor, target_class=None):
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Zero gradients and backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1.0
        output.backward(gradient=one_hot)

        # Get gradients and activations
        gradients = self.gradients[0].cpu().numpy()
        activations = self.activations[0].cpu().numpy()

        # Global average pooling of gradients
        weights = np.mean(gradients, axis=(1, 2))

        # Weight the activations
        heatmap = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            heatmap += w * activations[i]

        # Apply ReLU and normalize
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)

        return heatmap, target_class, torch.softmax(output, dim=1)[0][target_class].item()


# Create sample medical images with proper data types
def create_sample_medical_images(num_samples=200, img_size=128):
    """Create synthetic medical images for demonstration"""
    images = []
    labels = []

    print("Generating synthetic medical images...")
    for i in range(num_samples):
        if i % 2 == 0:
            # Benign - smoother patterns
            img = np.random.normal(0.3, 0.1, (img_size, img_size, 3)).astype(np.float32)
            # Add circular structures
            center = np.random.randint(30, 98, 2)
            radius = np.random.randint(10, 25)
            y, x = np.ogrid[:img_size, :img_size]
            mask = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2
            img[mask] += 0.3
        else:
            # Malignant - more irregular patterns
            img = np.random.normal(0.4, 0.2, (img_size, img_size, 3)).astype(np.float32)
            # Add irregular shapes
            for _ in range(3):
                center = np.random.randint(20, 108, 2)
                radius = np.random.randint(5, 15)
                y, x = np.ogrid[:img_size, :img_size]
                mask = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2
                img[mask] += 0.4

        img = np.clip(img, 0, 1)
        images.append(img)
        labels.append(i % 2)

    return np.array(images), np.array(labels)


# Visualization function
def visualize_gradcam(img, heatmap, alpha=0.4):
    """Visualize Grad-CAM heatmap overlay"""
    # Convert to numpy if tensor
    if torch.is_tensor(img):
        img = img.cpu().numpy()
        if img.shape[0] == 3:  # CHW to HWC
            img = img.transpose(1, 2, 0)

    # Denormalize if needed
    if img.max() <= 1.0:
        img_display = (img * 255).astype(np.uint8)
    else:
        img_display = img.astype(np.uint8)

    # Resize heatmap to match original image
    heatmap_resized = cv2.resize(heatmap, (img_display.shape[1], img_display.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

    # Superimpose heatmap on original image
    superimposed_img = cv2.addWeighted(img_display, 1 - alpha, heatmap_colored, alpha, 0)

    return superimposed_img, heatmap_resized


# Training function with proper data type handling
def train_model(model, train_loader, test_loader, epochs=10, device='cpu'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.to(device)
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            # Ensure data is on correct device and type
            data = data.to(device).float()  # Convert to float32
            target = target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device).float()
                target = target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()

        train_acc = 100 * correct / total
        val_acc = 100 * val_correct / val_total

        print(f'Epoch {epoch + 1}/{epochs}: '
              f'Loss: {running_loss / len(train_loader):.4f}, '
              f'Train Acc: {train_acc:.2f}%, '
              f'Val Acc: {val_acc:.2f}%')

    return model


# Function to save visualization
def save_gradcam_visualization(original_img, heatmap_resized, superimposed_img,
                               true_label, pred_class, confidence, save_path):
    """Save Grad-CAM visualization as an image file"""

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Original image
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title(f'Original Image\nTrue: {"Benign" if true_label == 0 else "Malignant"}', fontsize=12)
    axes[0, 0].axis('off')

    # Heatmap
    im = axes[0, 1].imshow(heatmap_resized, cmap='jet')
    axes[0, 1].set_title(f'Grad-CAM Heatmap\nPredicted: {"Benign" if pred_class == 0 else "Malignant"}', fontsize=12)
    axes[0, 1].axis('off')
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # Superimposed image
    axes[1, 0].imshow(superimposed_img)
    axes[1, 0].set_title('Grad-CAM Overlay', fontsize=12)
    axes[1, 0].axis('off')

    # Mathematical formulation
    axes[1, 1].text(0.1, 0.9, "Grad-CAM Mathematical Formulation:", fontsize=14, weight='bold')
    axes[1, 1].text(0.1, 0.75, "1. Forward pass: A = last conv features", fontsize=11)
    axes[1, 1].text(0.1, 0.65, "2. Compute gradients: ∇y^c = ∂y^c/∂A", fontsize=11)
    axes[1, 1].text(0.1, 0.55, "3. Global average pooling: α_k = 1/Z Σ_iΣ_j ∇y^c_ij", fontsize=11)
    axes[1, 1].text(0.1, 0.45, "4. Weighted combination: L = ReLU(Σ_k α_k A^k)", fontsize=11)
    axes[1, 1].text(0.1, 0.35, "5. Heatmap: H = normalize(L)", fontsize=11)
    axes[1, 1].text(0.1, 0.2, f"Target Layer: conv3", fontsize=11)
    axes[1, 1].text(0.1, 0.1, f"Confidence: {confidence:.3f}", fontsize=11)
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')

    plt.tight_layout()

    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Visualization saved to: {save_path}")
    plt.close()  # Close the figure to free memory


# Main execution
def main():
    # Create output directory
    output_dir = "gradcam_results"
    os.makedirs(output_dir, exist_ok=True)

    # Set device and random seeds for reproducibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)

    print(f"Using device: {device}")

    # Create sample medical images
    print("Creating sample medical images...")
    X, y = create_sample_medical_images(num_samples=200, img_size=128)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Class distribution - Benign: {sum(y == 0)}, Malignant: {sum(y == 1)}")

    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Convert images to tensors with proper data types
    X_train_tensor = torch.stack([transform(img) for img in X_train]).float()
    X_test_tensor = torch.stack([transform(img) for img in X_test]).float()
    y_train_tensor = torch.LongTensor(y_train)
    y_test_tensor = torch.LongTensor(y_test)

    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Create and train model
    print("Creating model...")
    model = SimpleCNN()

    print("Training model...")
    model = train_model(model, train_loader, test_loader, epochs=10, device=device)

    # Grad-CAM visualization for multiple test images
    print("\nGenerating Grad-CAM heatmaps...")

    # Test multiple images
    test_indices = [0, 5, 10, 15]  # Multiple test cases

    for i, test_idx in enumerate(test_indices):
        if test_idx >= len(X_test):
            continue

        print(f"Processing test image {i + 1}/{len(test_indices)} (index {test_idx})...")

        # Get test image
        model.eval()
        test_img = X_test_tensor[test_idx:test_idx + 1].to(device).float()
        original_img = X_test[test_idx]  # Original image for visualization

        # Initialize Grad-CAM
        grad_cam = GradCAM(model, model.conv3)

        # Generate heatmap
        heatmap, pred_class, confidence = grad_cam.generate_heatmap(test_img)
        grad_cam.remove_hooks()

        # Create visualization
        superimposed_img, heatmap_resized = visualize_gradcam(original_img, heatmap)

        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save visualization
        save_path = os.path.join(output_dir, f"gradcam_visualization_{timestamp}_case_{i + 1}.png")
        save_gradcam_visualization(
            original_img,
            heatmap_resized,
            superimposed_img,
            y_test[test_idx],
            pred_class,
            confidence,
            save_path
        )

        # Also save individual components
        # Save original image
        cv2.imwrite(os.path.join(output_dir, f"original_case_{i + 1}.png"),
                    cv2.cvtColor((original_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

        # Save heatmap
        plt.figure(figsize=(6, 5))
        plt.imshow(heatmap_resized, cmap='jet')
        plt.title('Grad-CAM Heatmap')
        plt.axis('off')
        plt.colorbar()
        plt.savefig(os.path.join(output_dir, f"heatmap_case_{i + 1}.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # Save overlay
        cv2.imwrite(os.path.join(output_dir, f"overlay_case_{i + 1}.png"),
                    cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))

    # Print detailed mathematical formulation
    print("\n" + "=" * 70)
    print("GRAD-CAM MATHEMATICAL FORMULATION FOR MEDICAL IMAGE ANALYSIS")
    print("=" * 70)
    print("Key Equations:")
    print("1. α_k^c = 1/Z * Σ_i Σ_j [∂y^c/∂A_ij^k]  (Global Average Pooling)")
    print("2. L_Grad-CAM^c = ReLU(Σ_k α_k^c A^k)     (Weighted Combination)")
    print("3. H = (L - min(L)) / (max(L) - min(L))   (Normalization)")
    print("\nWhere:")
    print("- A^k: k-th feature map from last convolutional layer")
    print("- y^c: classification score for class c (benign/malignant)")
    print("- α_k^c: importance weight for feature map k towards class c")
    print("- Z: number of pixels in the feature map (H × W)")
    print("- ReLU: rectified linear unit (retains positive influences)")
    print("- H: final heatmap showing discriminative regions")
    print("=" * 70)

    # Save mathematical formulation as text file
    math_formula = """
    GRAD-CAM MATHEMATICAL FORMULATION

    1. Feature Extraction: 
       A^k = f_conv(X)  [Convolutional feature maps]

    2. Gradient Computation: 
       ∇y^c = ∂y^c/∂A^k  [Class-specific gradients]

    3. Global Average Pooling: 
       α_k^c = (1/Z) * Σ_i Σ_j (∂y^c/∂A_ij^k)

    4. Weighted Combination: 
       L_Grad-CAM^c = ReLU(Σ_k α_k^c A^k)

    5. Heatmap Generation: 
       H = (L - min(L)) / (max(L) - min(L))

    Where:
    - A^k: k-th feature map activation
    - y^c: classification score for class c
    - α_k^c: importance weight for feature map k
    - Z: spatial dimensions of feature map (H × W)
    - ReLU: retains only positive contributions
    - H: final heatmap showing discriminative regions
    """

    with open(os.path.join(output_dir, "gradcam_mathematical_formulation.txt"), "w") as f:
        f.write(math_formula)

    print(f"\nAll results saved to directory: {output_dir}")
    print("Files created:")
    print("- gradcam_visualization_*.png (complete visualizations)")
    print("- original_case_*.png (original images)")
    print("- heatmap_case_*.png (heatmap only)")
    print("- overlay_case_*.png (overlay only)")
    print("- gradcam_mathematical_formulation.txt (mathematical formulas)")


if __name__ == "__main__":
    main()