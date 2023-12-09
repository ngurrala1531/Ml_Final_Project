import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from scnet import scnet50  # Assuming SCNet implementation is available in the scnet module

# Define the model
model = scnet50(pretrained=False)  # Assuming you're not using a pre-trained model
# Modify the last layer for binary classification (with_mask, without_mask)
model.fc = nn.Linear(model.fc.in_features, 2)

# Set up data transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Load the dataset
train_dataset = datasets.ImageFolder(r'C:\Users\reshm\Downloads\SCNet\dataset', transform=transform)


# Split the dataset into training and validation sets
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


num_epochs = 10
for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')

    
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader, 1):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 10 == 0:  
            print(f'  Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}')

    average_train_loss = running_loss / len(train_loader)
    print(f'Training Loss: {average_train_loss:.4f}')

    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        correct = 0
        total = 0
        for i, (inputs, labels) in enumerate(val_loader, 1):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if i % 10 == 0:  
                print(f'  Validation Batch {i}/{len(val_loader)}')

        average_val_loss = val_loss / len(val_loader)
        accuracy = correct / total
        print(f'Validation Loss: {average_val_loss:.4f}, Accuracy: {accuracy * 100:.2f}%')

# Save the trained model
torch.save(model.state_dict(), 'mask_detection_model.pth')
print('Model saved.')
