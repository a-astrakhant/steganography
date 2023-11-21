import numpy as np
import matplotlib.image as img

test_img1 = img.imread(r"E:\cis_field_support\trening\steganography-master\Original_image\lenna.png")

print(test_img1[0][0])
# Create empty array for storing signal with noise:
img_with_zavada = np.empty((len(test_img1),len(test_img1), 3))

#===================
# Channel model      #0.5 - дисперсія завади
#===================
zavada = np.random.uniform(0, 0.5, size=(len(test_img1),len(test_img1)))
for i in range(len(test_img1)):
    for j in range(len(test_img1[0])):
        img_with_zavada[i,j] = test_img1[i,j] + zavada[i,j]

for i in range(len(test_img1)):
    for j in range(len(test_img1[0])):
        for k in range(3):
            if img_with_zavada[i,j,k] > 1:
                img_with_zavada[i, j, k] = 1

print(img_with_zavada[0][0])
img.imsave(r"E:\cis_field_support\trening\steganography-master\noised2.png", img_with_zavada)
