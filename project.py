
import os

import numpy as np
import seaborn as sns
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from os.path import join
from torch.utils.data import Dataset, SubsetRandomSampler
from PIL import Image
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix, classification_report



BASE = "./Dataset"
# BASE = "gdrive/MyDrive/Dataset"
cloth_folder = join(BASE, "Cloth Mask")
n95_folder = join(BASE, "N95")
n95valve_folder = join(BASE, "N95 mask with valve")
surgical_folder = join(BASE, "Surgical Mask")
without_mask_folder = join(BASE, "No Mask")
root_folder = ("./")

# predefined classes
classes = {
    "cloth": 0,
    "n95": 1,
    "n95valve": 2,
    "surgical": 3,
    "without_mask": 4
}


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


class FaceMaskDataset(Dataset):
    dataset = []
    conversion = None

    def __init__(self, images, indexes, conversion=transforms.ToTensor()):
        self.conversion = conversion
        self.dataset = images[int(indexes[0]):int(indexes[1])]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index]['image']
        return image, self.dataset[index]['target']


class FaceMaskClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        scores, predictions = torch.max(out.data, 1)
        train_correct = (predictions == labels).sum().item()
        return loss, train_correct

    def validation_step(self, batch):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        scores, predictions = torch.max(out.data, 1)
        valid_correct = (predictions == labels).sum().item()
        return loss, valid_correct

    def epoch_end(self, epoch, result):
        print(
            "Epoch [{}], train_loss: {:.4f}, train_accuracy: {:.4f}, valid_loss: {:.4f}, valid_accuracy: {:.4f}".format(
                epoch + 1, result['train_loss'], result['train_accuracy']
                , result['valid_loss'], result['valid_accuracy']))


class CNN(FaceMaskClassificationBase):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(246016, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 5),
        )

    def forward(self, x):
        x = self.conv_layer(x)  # convolution layer
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc_layer(x)  # fully connected layer
        return x


@torch.no_grad()
def evaluate_model(model, val_loader):
    model.eval()
    valid_losses = []
    valid_correct = 0

    for batch in val_loader:
        loss, correct = model.validation_step(batch)
        valid_losses.append(loss)
        valid_correct += correct

    return valid_losses, valid_correct


def fit(epochs, lr, model, train_loader, valid_loader, opt_func=torch.optim.SGD):
    results = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        model.train()
        train_losses = []
        train_correct = 0
        for batch in train_loader:
            images, labels = batch
            # forward-pass: compute-predicted-outputs-by-passing-inputs-to-the-model
            loss, correct = model.training_step(batch)
            # update-training-loss
            train_losses.append(loss)
            train_correct += correct
            # backward-pass: compute-gradient-of-the-loss-wrt-model-parameters
            loss.backward()
            # perform-a-single-optimization-step (parameter-update)
            optimizer.step()
            # clear-the-gradients-of-all-optimized-variables
            optimizer.zero_grad()

        valid_losses, valid_correct = evaluate_model(model, valid_loader)

        train_acc = (train_correct / (len(train_split) * 0.9)) * 100
        valid_acc = (valid_correct / (len(train_split) * 0.1)) * 100
        result = {}
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['train_accuracy'] = train_acc
        result['valid_loss'] = torch.stack(valid_losses).mean().item()
        result['valid_accuracy'] = valid_acc
        model.epoch_end(epoch, result)
        results.append(result)

    return results

transform = transforms.Compose(
    [transforms.Resize((250, 250)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])])

# Load images from the dataset
def load_images():
  dat = []
  cnt = 0
  for filename in os.listdir(cloth_folder):
    try:
        sample = {}
        img = Image.open(os.path.join(cloth_folder, filename)).convert('RGB')
        img = transform(img)
        sample['image'] = img
        sample['target'] = classes['cloth']
        dat.append(sample)
        cnt += 1
    except:
        continue
  print(cnt)

  for filename in os.listdir(n95_folder):
    try:
        sample = {}
        img = Image.open(os.path.join(n95_folder, filename)).convert('RGB')
        img = transform(img)
        sample['image'] = img
        sample['target'] = classes['n95']
        dat.append(sample)
        cnt += 1
    except:
        continue
  print(cnt)

  for filename in os.listdir(n95valve_folder):
    try:
        sample = {}
        img = Image.open(os.path.join(n95valve_folder, filename)).convert('RGB')
        img = transform(img)
        sample['image'] = img
        sample['target'] = classes['n95valve']
        dat.append(sample)
        cnt += 1
    except:
        continue
  print(cnt)

  for filename in os.listdir(surgical_folder):
    try:
        sample = {}
        img = Image.open(os.path.join(surgical_folder, filename)).convert('RGB')
        img = transform(img)
        sample['image'] = img
        sample['target'] = classes['surgical']
        dat.append(sample)
        cnt += 1
    except:
        continue
  print(cnt)

  for filename in os.listdir(without_mask_folder):
    try:
        sample = {}
        img = Image.open(os.path.join(without_mask_folder, filename)).convert('RGB')
        img = transform(img)
        sample['image'] = img
        sample['target'] = classes['without_mask']
        dat.append(sample)
        cnt += 1
    except:
        continue
  print(cnt)
  return dat


def evaluate():
    y_true = torch.tensor([])
    y_true = y_true.to(device)
    y_preds = torch.tensor([])
    y_preds = y_preds.to(device)

    # test-the-model
    model.eval()  # it-disables-dropout
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            y_true = torch.cat((y_true, labels))
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_preds = torch.cat((y_preds, predicted))

        print('\nTest Accuracy of the model: {} %'.format(100 * correct / total))

    y_true = y_true.to('cpu')
    y_preds = y_preds.to('cpu')

    precision, recall, fscore, support = score(y_true, y_preds)
    print(classification_report(y_true, y_preds))


# Plot the losses in each epoch
def plot_losses(results):
    train_losses = [result.get('train_loss') for result in results]
    plt.plot(train_losses, '-bx')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Training Loss'])
    plt.title('Loss vs. No. of epochs');


# show confusion matrix
def show_confusion_matrix(y_true, y_preds):
    matrix = confusion_matrix(y_true, y_preds)
    plt.figure(figsize=(10, 7))
    ax = sns.heatmap(matrix, fmt='', annot=True, cmap='Blues')
    ax.set_title('Confusion Matrix with labels!!');
    ax.set_xlabel('Predicted Mask Type')
    ax.set_ylabel('Actual Mask Type')
    ax.xaxis.set_ticklabels([i for i in classes.keys()])
    ax.yaxis.set_ticklabels([i for i in classes.keys()])
    plt.show()

if __name__ == "__main__":


    # Load images into memory
    images = load_images()
    random.shuffle(images)
    size_of_the_dataset = len(images)
    print("size_of_the_dataset", size_of_the_dataset)

    num_of_classes = len(classes.keys())

    data = FaceMaskDataset(images, [0, size_of_the_dataset], conversion=transform)

    print(len(data))
    TEST_SIZE = 0.15

    train_indices, test_indices = train_test_split(range(len(data)), test_size=TEST_SIZE, random_state=42)
    print("Train split : {} test split : {} ".format(len(train_indices), len(test_indices)))

    train_split = Subset(data, train_indices)
    test_split = Subset(data, test_indices)

    model = CNN()
    model = model.to(device)

    num_epochs = 10
    batch_size = 25
    k = 10
    splits = KFold(n_splits=k, shuffle=True, random_state=42)
    foldperf = {}
    final_results = []

    for fold, (train_idx, valid_idx) in enumerate(splits.split(np.arange(len(train_split)))):
        print('Fold {}'.format(fold + 1))
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        train_loader = DataLoader(train_split, batch_size=batch_size, sampler=train_sampler)
        valid_loader = DataLoader(train_split, batch_size=batch_size, sampler=valid_sampler)

        results = fit(num_epochs, 0.0001, model, train_loader, valid_loader, torch.optim.ASGD)
        evaluate()
        final_results.append(results)
        foldperf['fold{}'.format(fold + 1)] = results

        # Save model
    print("Saving model to ", join(BASE, "model.pth"))
    torch.save(model.state_dict(), join(BASE, "model.pth"))

    # Plotting loss vs epoch chart
    plot_losses(results)

    #Testing new images
    test_loader = torch.utils.data.DataLoader(dataset=test_split, batch_size=25, shuffle=True, num_workers=0)
    y_true = torch.tensor([])
    y_preds = torch.tensor([])

    # Test the model
    model.eval()  # turn off layers like Dropout layers, BatchNorm Layers
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            y_true = torch.cat((y_true, labels))

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_preds = torch.cat((y_preds, predicted))

        print('Test Accuracy of the model: {} %'.format(100 * correct / total))


    # Display Confusion matrix and other important parameters
    show_confusion_matrix(y_true, y_preds)

    precision, recall, fscore, support = score(y_true, y_preds)
    print(classification_report(y_true, y_preds))
