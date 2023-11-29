import math
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import sewar
import skimage.metrics
import xlwt
from PIL import Image
from scipy import signal
from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

from dct import DCT
from dwt import DWT
from lsb import LSB

original_image_dir_path: Path = Path("Original_image").resolve()

encoded_image_dir_path: Path = Path("Encoded_image").resolve()
decoded_output_dir_path: Path = Path("Decoded_output").resolve()
comparison_result_dir_path: Path = Path("Comparison_result").resolve()

'''def show(im):
    im_resized = cv2.resize(im, (500, 500), interpolation=cv2.INTER_LINEAR)
    plt.imshow(cv2.cvtColor(im_resized, cv2.COLOR_BGR2RGB))
    plt.show()'''


class Compare:
    def mean_square_error(self, img1, img2):        # READY
        error = np.sum((img1.astype('float') - img2.astype('float')) ** 2)
        error /= float(img1.shape[0] * img1.shape[1])
        return error

    def psnr(self, img1, img2):    # READY
        mse = self.mean_square_error(img1, img2)
        if mse == 0:
            return 100
        PIXEL_MAX = 255.0
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

    def Visual_Information_Fidelity(self, img1, img2):
        vif = sewar.vifp(img1, img2)
        return vif

    def Spatial_Correlation_Coefficient(self, img1, img2):
        scc = sewar.scc(img1, img2)
        return scc

    def Universal_Quality_Image_Index(self, img1, img2):    # READY
        # https://pyimagesearch.com/2014/09/15/python-compare-two-images/
        # https://towardsdatascience.com/measuring-similarity-in-two-images-using-python-b72233eb53c6
        #ssim = skimage.metrics.structural_similarity(img1,img2)

        uqi = sewar.uqi(img1, img2)
        return uqi


# driver part :

# deleting previous folders :
if encoded_image_dir_path.exists():
    shutil.rmtree(encoded_image_dir_path)
if decoded_output_dir_path.exists():
    shutil.rmtree(decoded_output_dir_path)
if comparison_result_dir_path.exists():
    shutil.rmtree(comparison_result_dir_path)

# creating new folders :
encoded_image_dir_path.mkdir()
decoded_output_dir_path.mkdir()
comparison_result_dir_path.mkdir()

# input processed file
input_original_image_filename: str = input("Enter the name of the file with extension : ")
original_image_path: Path = original_image_dir_path / input_original_image_filename
lsb_encoded_image_path: Path = encoded_image_dir_path / ("lsb_" + input_original_image_filename)
dct_encoded_image_path: Path = encoded_image_dir_path / ("dct_" + input_original_image_filename)
dwt_encoded_image_path: Path = encoded_image_dir_path / ("dwt_" + input_original_image_filename)

first_run = True
while True:
    input_number = input("To encode press '1', to decode press '2', to compare press '3', press any other button to close: ")

    if input_number == "1":

        input_secret_msg: str = input("Enter the message you want to hide: ")
        print(f"The message length is: {len(input_secret_msg)}")

        lsb_img: Image.Image = Image.open(original_image_path)
        print(f"Description : {lsb_img}")
        print(f"Mode : {lsb_img.mode}")
        lsb_img_encoded = LSB().encode_image(lsb_img, input_secret_msg)
        lsb_img_encoded.save(lsb_encoded_image_path)

        dct_img: np.ndarray = cv2.imread(str(original_image_path), cv2.IMREAD_UNCHANGED)
        dct_img_encoded = DCT().encode_image(dct_img, input_secret_msg)
        cv2.imwrite(str(dct_encoded_image_path), dct_img_encoded)

        # dwt_img: np.ndarray = cv2.imread(str(original_image_path), cv2.IMREAD_UNCHANGED)
        # dwt_img_encoded = DWT().encode_image(dwt_img, input_secret_msg)
        # cv2.imwrite(str(dwt_encoded_image_path), dwt_img_encoded)  # saving the image with the hidden text

        print("Encoded images were saved!")

    elif input_number == "2":
        lsb_img: Image.Image = Image.open(lsb_encoded_image_path)
        lsb_hidden_text: str = LSB().decode_image(lsb_img)
        lsb_decoded_text_path: Path = decoded_output_dir_path / "lsb_hidden_text.txt"
        lsb_decoded_text_path.write_text(lsb_hidden_text)  # saving hidden text as text file

        dct_img: np.ndarray = cv2.imread(str(dct_encoded_image_path), cv2.IMREAD_UNCHANGED)
        dct_hidden_text = DCT().decode_image(dct_img)
        dct_decoded_text_path: Path = decoded_output_dir_path / "dct_hidden_text.txt"
        dct_decoded_text_path.write_text(dct_hidden_text)  # saving hidden text as text file

        # dwt_img = cv2.imread(str(dwt_encoded_image_path), cv2.IMREAD_UNCHANGED)
        # dwt_hidden_text = DWT().decode_image(dwt_img)
        # dwt_decoded_text_path: Path = decoded_output_dir_path / "dwt_hidden_text.txt"
        # dwt_decoded_text_path.write_text(dwt_hidden_text)  # saving hidden text as text file

        print("Hidden texts were saved as text file!")

    elif input_number == "3":
        # comparison calls
        original = cv2.imread(str(original_image_path))
        lsb_encoded = cv2.imread(str(lsb_encoded_image_path))
        dct_encoded = cv2.imread(str(dct_encoded_image_path))
        # dwt_encoded = cv2.imread(str(dwt_encoded_image_path))

        original_img = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        lsb_encoded_img = cv2.cvtColor(lsb_encoded, cv2.COLOR_BGR2RGB)  # ???????
        dct_encoded_img = cv2.cvtColor(dct_encoded, cv2.COLOR_BGR2RGB)
        # dwt_encoded_img = cv2.cvtColor(dwt_encoded, cv2.COLOR_BGR2RGB)

        book = xlwt.Workbook()
        sheet1 = book.add_sheet("Sheet 1")
        style_string = "font: bold on , color red; borders: bottom dashed"
        style = xlwt.easyxf(style_string)
        sheet1.write(0, 0, "Original vs", style=style)
        sheet1.write(0, 1, "MSE", style=style)
        sheet1.write(0, 2, "PSNR", style=style)
        sheet1.write(0, 3, "VIF", style=style)  # new one
        sheet1.write(0, 4, "Correlation", style=style)  # new one
        sheet1.write(0, 5, "Quality_Image", style=style)  # new one

        sheet1.write(1, 0, "LSB")
        sheet1.write(1, 1, Compare().mean_square_error(original_img, lsb_encoded_img))
        sheet1.write(1, 2, Compare().psnr(original_img, lsb_encoded_img))
        sheet1.write(1, 3, Compare().Visual_Information_Fidelity(original_img, lsb_encoded_img))
        sheet1.write(1, 4, Compare().Spatial_Correlation_Coefficient(original_img, lsb_encoded_img))
        sheet1.write(1, 5, Compare().Universal_Quality_Image_Index(original_img, lsb_encoded_img))

        sheet1.write(2, 0, "DCT")
        sheet1.write(2, 1, Compare().mean_square_error(original_img, dct_encoded_img))
        sheet1.write(2, 2, Compare().psnr(original_img, dct_encoded_img))
        sheet1.write(2, 3, Compare().Visual_Information_Fidelity(original_img, dct_encoded_img))
        sheet1.write(2, 4, Compare().Spatial_Correlation_Coefficient(original_img, dct_encoded_img))
        sheet1.write(2, 5, Compare().Universal_Quality_Image_Index(original_img, dct_encoded_img))

        sheet1.write(3, 0, "DWT")
        # sheet1.write(3, 1, Compare().meanSquareError(original, dwt_encoded_img))
        # sheet1.write(3, 2, Compare().psnr(original, dwt_encoded_img))

        book.save(comparison_result_dir_path / "Comparison.xls")
        print("Comparison Results were saved as xls file!")

    else:
        print("Closed!")
        break
