# https://sempioneer.com/python-for-seo/how-to-compress-images-in-python/

from PIL import Image
import os
from tqdm.notebook import tqdm

def make_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

root = "./Dataset"
image_dir1 = os.path.join(root, "./cloth")
image_dir2 = os.path.join(root, "./n95")
image_dir3 = os.path.join(root, "./n95valve")
image_dir4 = os.path.join(root, "./surgical")
image_dir5 = os.path.join(root, "./without_mask")
compressed_dir = os.path.join(root, 'images')
make_dir(compressed_dir)

image_dir = [image_dir1, image_dir2, image_dir3, image_dir4, image_dir5]

# images = [file for file in os.listdir(image_dir) if file.endswith(('jpeg'))]

for dir in image_dir:
    for image in tqdm(os.listdir(dir)):
        img = Image.open(os.path.join(dir, image))
        img = img.convert('RGB')
        # 2. Compressing the image
        img.save(os.path.join(compressed_dir, image),
                optimize=True,
                quality=10)

# for image in tqdm(os.listdir(image_dir1)):
#     # 1. Open the image
#     img = Image.open(os.path.join(image_dir1, image))
#     # 2. Compressing the image
#     img.save(os.path.join(compressed_dir, image),
#              optimize=True,
#              quality=10)