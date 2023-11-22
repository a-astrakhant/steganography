import os.path

import matplotlib.image as img
import numpy as np
from PIL import Image
from pathlib import Path

lsb_lenna_path: Path = Path(__file__).parent / "Encoded_image" / "lsb_lenna.png"
dct_lenna_path: Path = Path(__file__).parent / "Encoded_image" / "dct_lenna.png"

lsb_noised_path: Path = Path(__file__).parent / "Encoded_image" / "lsb_noised.png"
dct_noised_path: Path = Path(__file__).parent / "Encoded_image" / "dct_noised.png"

# Andrii's test
# test_img1 = img.imread(lsb_noised_path)
# test_img2 = img.imread(dct_noised_path)

test_img1 = img.imread(lsb_lenna_path)
test_img2 = img.imread(dct_lenna_path)

print(test_img1[0][0])
# Create empty array for storing signal with noise:
img_with_zavada1 = np.empty((len(test_img1), len(test_img1), 3))
img_with_zavada2 = np.empty((len(test_img2), len(test_img2), 3))

# ===================
# Channel model AWGN     #0.005 - дисперсія завади
# ===================
zavada = np.random.uniform(0, 0.005, size=(len(test_img1), len(test_img1)))
for i in range(len(test_img1)):
    for j in range(len(test_img1[0])):
        img_with_zavada1[i, j] = test_img1[i, j] + zavada[i, j]
        img_with_zavada2[i, j] = test_img2[i, j] + zavada[i, j]

for i in range(len(test_img1)):
    for j in range(len(test_img1[0])):
        for k in range(3):
            if img_with_zavada1[i, j, k] > 1:
                img_with_zavada1[i, j, k] = 1
            if img_with_zavada2[i, j, k] > 1:
                img_with_zavada2[i, j, k] = 1

print(img_with_zavada1[0][0])
print(img_with_zavada2[0][0])

# matplotlib always saves as RGBA png image:
# https://stackoverflow.com/a/45594478/8463690
pil_img_with_zavada1 = Image.fromarray((img_with_zavada1 * 255).astype(np.uint8))
pil_img_with_zavada1.save(lsb_noised_path)
pil_img_with_zavada2 = Image.fromarray((img_with_zavada2 * 255).astype(np.uint8))
pil_img_with_zavada2.save(dct_noised_path)
