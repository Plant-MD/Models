import os
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# ==== CONFIG (should match your training config) ====
MODEL_PATH = "resnet34_plant_disease.pth"
DATA_DIR = r"C:\Users\suyog\tests\plant-detection-training-model-resnet34\plantvillage dataset\color"
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== TRANSFORMS (same as training) ====
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==== LOAD MODEL ====
def load_model():
    """Load the trained model"""
    # First load dataset to get class names
    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    class_names = dataset.classes
    
    # Load model architecture
    model = models.resnet34(weights=None)  # Don't load pretrained weights
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    
    # Load trained weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model = model.to(DEVICE)
    model.eval()
    
    print(f"Model loaded from {MODEL_PATH}")
    print(f"Classes: {class_names}")
    return model, class_names

# ==== TEST SINGLE IMAGE ====
def test_single_image(model, class_names, image_path):
    """Test a single image and return prediction"""
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_idx = torch.max(probabilities, 0)
            
        predicted_class = class_names[predicted_idx.item()]
        confidence_score = confidence.item()
        
        print(f"\nImage: {os.path.basename(image_path)}")
        print(f"Predicted Class: {predicted_class}")
        print(f"Confidence: {confidence_score:.4f}")
        
        # Show top 3 predictions
        top3_prob, top3_idx = torch.topk(probabilities, 3)
        print("\nTop 3 predictions:")
        for i in range(3):
            class_name = class_names[top3_idx[i].item()]
            prob = top3_prob[i].item()
            print(f"  {i+1}. {class_name}: {prob:.4f}")
            
        return predicted_class, confidence_score
        
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None, None

# ==== TEST DIRECTORY ====
def test_directory(model, class_names, test_dir):
    """Test all images in a directory"""
    results = []
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    for filename in os.listdir(test_dir):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_path = os.path.join(test_dir, filename)
            predicted_class, confidence = test_single_image(model, class_names, image_path)
            if predicted_class:
                results.append({
                    'filename': filename,
                    'predicted_class': predicted_class,
                    'confidence': confidence
                })
    
    return results

# ==== EVALUATE ON TEST SET ====
def evaluate_test_set(model, test_loader, class_names):
    """Evaluate model on a test dataset"""
    model.eval()
    correct = 0
    total = 0
    class_correct = list(0. for i in range(len(class_names)))
    class_total = list(0. for i in range(len(class_names)))
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-class accuracy
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    # Overall accuracy
    overall_accuracy = 100 * correct / total
    print(f'\nOverall Test Accuracy: {overall_accuracy:.2f}%')
    
    # Per-class accuracy
    print('\nPer-class accuracy:')
    for i in range(len(class_names)):
        if class_total[i] > 0:
            accuracy = 100 * class_correct[i] / class_total[i]
            print(f'{class_names[i]}: {accuracy:.2f}% ({int(class_correct[i])}/{int(class_total[i])})')

# ==== VISUALIZE PREDICTIONS ====
def visualize_predictions(model, class_names, test_loader, num_images=8):
    """Visualize predictions on test images"""
    model.eval()
    images_shown = 0
    
    plt.figure(figsize=(12, 8))
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(inputs.size()[0]):
                if images_shown >= num_images:
                    break
                    
                # Denormalize image for display
                img = inputs[i].cpu()
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img = img + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                img = torch.clamp(img, 0, 1)
                
                plt.subplot(2, 4, images_shown + 1)
                plt.imshow(np.transpose(img, (1, 2, 0)))
                
                actual = class_names[labels[i]]
                pred = class_names[predicted[i]]
                color = 'green' if actual == pred else 'red'
                
                plt.title(f'Actual: {actual}\nPred: {pred}', color=color, fontsize=8)
                plt.axis('off')
                
                images_shown += 1
                
            if images_shown >= num_images:
                break
    
    plt.tight_layout()
    plt.show()

# ==== MAIN TESTING FUNCTIONS ====
def main():
    """Main testing function"""
    print("Loading model...")
    model, class_names = load_model()
    
    # Test single image (replace with your image path)
    test_single_image(model, class_names, r"C:\Users\suyog\tests\plant-detection-training-model-resnet34\test.jpg")
    
    # Test directory of images
    # test_results = test_directory(model, class_names, "path/to/test/directory")
    
    # If you have a separate test dataset, evaluate it
    # test_dataset = datasets.ImageFolder("path/to/test/dataset", transform=transform)
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    # evaluate_test_set(model, test_loader, class_names)
    # visualize_predictions(model, class_names, test_loader)

if __name__ == "__main__":
    main()

# ==== USAGE EXAMPLES ====
"""
Example usage:

1. Test a single image:
   model, class_names = load_model()
   test_single_image(model, class_names, "test_image.jpg")

2. Test all images in a directory:
   results = test_directory(model, class_names, "test_images_folder")
   
3. Create a separate test dataset and evaluate:
   test_dataset = datasets.ImageFolder("test_data", transform=transform)
   test_loader = DataLoader(test_dataset, batch_size=32)
   evaluate_test_set(model, test_loader, class_names)
"""