import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image
import os
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom image filter transform
class ImageFilter:
    def __init__(self, filter_type='sobel'):
        self.filter_type = filter_type

    def apply_sobel_filter(self, image):
        if image.shape[0] == 3:
            image = transforms.functional.rgb_to_grayscale(image)
        
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        edge_x = F.conv2d(image.unsqueeze(0), sobel_x, padding=1)
        edge_y = F.conv2d(image.unsqueeze(0), sobel_y, padding=1)
        
        edge_magnitude = torch.sqrt(edge_x ** 2 + edge_y ** 2)
        return edge_magnitude.squeeze(0)

    def __call__(self, image):
        if self.filter_type == 'sobel':
            return self.apply_sobel_filter(image)
        return image

# Dataset class
class LineFollowerDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Model Architecture
class LineFollowerCNN(nn.Module):
    def __init__(self):
        super(LineFollowerCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 3)  # Changed to 3 outputs for GO, LEFT, RIGHT
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=20):
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total

        scheduler.step(val_loss)

        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
        print('-' * 10)

    return model

# Main execution
if __name__ == "__main__":
    # Load your data
    data_dir = '/kaggle/input/data-1/data'
    go_images = [os.path.join(data_dir, 'GO', img) for img in os.listdir(os.path.join(data_dir, 'GO'))]
    left_images = [os.path.join(data_dir, 'LEFT', img) for img in os.listdir(os.path.join(data_dir, 'LEFT'))]
    right_images = [os.path.join(data_dir, 'RIGHT', img) for img in os.listdir(os.path.join(data_dir, 'RIGHT'))]
    
    image_paths = go_images + left_images + right_images
    labels = [0] * len(go_images) + [1] * len(left_images) + [2] * len(right_images)

    # Split the data
    train_paths, val_paths, train_labels, val_labels = train_test_split(image_paths, labels, test_size=0.2, random_state=42)

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        ImageFilter(filter_type='sobel')
    ])

    # Create datasets
    train_dataset = LineFollowerDataset(train_paths, train_labels, transform=transform)
    val_dataset = LineFollowerDataset(val_paths, val_labels, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Create model, loss function, optimizer, and scheduler
    model = LineFollowerCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # Train the model
    print("Training with lr=0.001, weight_decay=0.0001")
    trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=20)

    # Save the trained model
    torch.save(trained_model.state_dict(), 'line_follower_model.pth')

    # Evaluate the model
    trained_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = trained_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    print(f"Final Validation Accuracy: {accuracy:.4f}")

    # Function to visualize model predictions
    def visualize_predictions(model, dataloader, num_images=5):
        import matplotlib.pyplot as plt
        
        model.eval()
        images, labels = next(iter(dataloader))
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        fig, axs = plt.subplots(1, num_images, figsize=(20, 4))
        class_names = ['GO', 'LEFT', 'RIGHT']
        for i in range(num_images):
            axs[i].imshow(images[i].cpu().squeeze(), cmap='gray')
            axs[i].set_title(f'Pred: {class_names[preds[i]]}, True: {class_names[labels[i]]}')
            axs[i].axis('off')
        plt.show()

    # Visualize some predictions
    visualize_predictions(trained_model, val_loader)

    # Function to test the model on a single image
    def predict_single_image(model, image_path, transform):
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        
        model.eval()
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output.data, 1)
        
        class_names = ['GO', 'LEFT', 'RIGHT']
        return class_names[predicted.item()]

    # Test the model on a single image
    test_image_path = '/kaggle/input/data-1/data/GO/GO_1053.jpg'
    prediction = predict_single_image(trained_model, test_image_path, transform)
    print(f"Prediction for test image: {prediction}")

    print("Training and evaluation complete.")
