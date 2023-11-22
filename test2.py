import matplotlib.image as img
import numpy as np

test_img1 = img.imread(r"E:\cis_field_support\trening\steganography-master\Encoded_image\lsb_lenna.png")
test_img2 = img.imread(r"E:\cis_field_support\trening\steganography-master\Encoded_image\dct_lenna.png")

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
img.imsave(r"E:\cis_field_support\trening\steganography-master\Encoded_image\lsb_noised.png", img_with_zavada1)
img.imsave(r"E:\cis_field_support\trening\steganography-master\Encoded_image\dct_noised.png", img_with_zavada2)
