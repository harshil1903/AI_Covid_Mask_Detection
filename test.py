import os
import torch
from PIL import Image
import torchvision.transforms as transforms

from project import CNN

root_folder = ("./")

BASE = "./Dataset"

classes = {
    "cloth": 0,
    "n95": 1,
    "n95valve": 2,
    "surgical": 3,
    "without_mask": 4
}

transform = transforms.Compose(
    [transforms.Resize((250, 250)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def label_to_classname(label):
  for classname in classes.keys():
    if classes[classname] == label:
      return classname
  return 'NULL'



# Put new images at the 'test' directory to classify them
new_images_path = os.path.join(root_folder, "test")

# get new images
new_images = os.listdir(new_images_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CNN()
model.load_state_dict(torch.load(os.path.join(BASE, "model.pth")))
model.eval()

with torch.no_grad():
    for image in new_images:
      file_name = image
      image = transform(Image.open(os.path.join(new_images_path, image)).convert('RGB'))
      image = image.unsqueeze(0)
      labels = model(image)
      _, predicted = torch.max(labels.data, 1)
      print(f'{file_name} file is {label_to_classname(predicted[0])}')