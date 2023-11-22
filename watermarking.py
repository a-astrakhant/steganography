import itertools
import math
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import xlwt
from PIL import Image
from scipy import signal

quant = np.array([[16, 11, 10, 16, 24, 40, 51, 61],  # QUANTIZATION TABLE
                  [12, 12, 14, 19, 26, 58, 60, 55],  # required for DCT
                  [14, 13, 16, 24, 40, 57, 69, 56],
                  [14, 17, 22, 29, 51, 87, 80, 62],
                  [18, 22, 37, 56, 68, 109, 103, 77],
                  [24, 35, 55, 64, 81, 104, 113, 92],
                  [49, 64, 78, 87, 103, 121, 120, 101],
                  [72, 92, 95, 98, 112, 100, 103, 99]])
'''def show(im):
    im_resized = cv2.resize(im, (500, 500), interpolation=cv2.INTER_LINEAR)
    plt.imshow(cv2.cvtColor(im_resized, cv2.COLOR_BGR2RGB))
    plt.show()'''


class DWT:
    # encoding part :
    def encode_image(self, img, secret_msg):
        secret = secret_msg
        # show(img)
        # get size of image in pixels
        row, col = img.shape[:2]
        # addPad
        if row % 8 != 0 or col % 8 != 0:
            img = cv2.resize(img, (col + (8 - col % 8), row + (8 - row % 8)))
        b_img, g_img, r_img = cv2.split(img)
        b_img = self.iwt2(b_img)
        # get size of padded image in pixels
        height, width = b_img.shape[:2]
        index = 0
        for row in range(height):
            for col in range(width):
                if img.mode == 'RGB':
                    r, g, b = img.getpixel((col, row))
                else:  # if img.mode != 'RGB':
                    r, g, b, a = img.getpixel((col, row))

                # first value is length of msg
                if row == 0 and col == 0 and index < len(secret):
                    asc = len(secret)
                elif index <= len(secret):
                    c = secret_msg[index - 1]
                    asc = ord(c)
                else:
                    asc = r
                s_img.putpixel((col, row), (asc, g, b))  # TODO need to check 56-62
                s_img = np.array(s_img).reshape(row, col)
                # converted from type float32
                s_img = np.uint8(s_img)
                # show(s_img)
                s_img = cv2.merge((s_img, g_img, r_img))

                index += 1
        return s_img

    # decoding part :
    def decode_image(self, img):
        msg = ""
        # get size of image in pixels
        row, col = img.shape[:2]
        b_img, g_img, r_img = cv2.split(img)

        return msg

    """Helper function to 'stitch' new image back together"""

    def _iwt(self, array):
        output = np.zeros_like(array)
        nx, ny = array.shape
        x = nx // 2
        for j in range(ny):
            output[0:x, j] = (array[0::2, j] + array[1::2, j]) // 2
            output[x:nx, j] = array[0::2, j] - array[1::2, j]
        return output

    def _iiwt(self, array):
        output = np.zeros_like(array)
        nx, ny = array.shape
        x = nx // 2
        for j in range(ny):
            output[0::2, j] = array[0:x, j] + (array[x:nx, j] + 1) // 2
            output[1::2, j] = output[0::2, j] - array[x:nx, j]
        return output

    def iwt2(self, array):
        return self._iwt(self._iwt(array.astype(int)).T).T

    def iiwt2(self, array):
        return self._iiwt(self._iiwt(array.astype(int)).T).T


class DCT:
    def __init__(self):  # Constructor
        self.message = None
        self.bitMess = None
        self.oriCol = 0
        self.oriRow = 0
        self.numBits = 0
        # encoding part :

    def encode_image(self, img, secret_msg):
        # show(img)
        secret = secret_msg
        self.message = str(len(secret)) + '*' + secret
        self.bitMess = self.to_bits()
        # get size of image in pixels
        row, col = img.shape[:2]
        # col, row = img.size
        self.oriRow, self.oriCol = row, col
        if (col / 8) * (row / 8) < len(secret):
            print("Error: Message too large to encode in image")
            return False
        # make divisible by 8x8
        if row % 8 != 0 or col % 8 != 0:
            img = self.add_padd(img, row, col)

        row, col = img.shape[:2]
        # col, row = img.size
        # split image into RGB channels
        b_img, g_img, r_img = cv2.split(img)
        # message to be hid in blue channel so converted to type float32 for dct function
        b_img = np.float32(b_img)
        # break into 8x8 blocks
        img_blocks = [np.round(b_img[j:j + 8, i:i + 8] - 128) for (j, i) in itertools.product(range(0, row, 8),
                                                                                              range(0, col, 8))]
        # Blocks are run through DCT function
        dct_blocks = [np.round(cv2.dct(img_Block)) for img_Block in img_blocks]
        # blocks then run through quantization table
        quantized_dct = [np.round(dct_Block / quant) for dct_Block in dct_blocks]
        # set LSB in dc value corresponding bit of message
        mess_index = 0
        letter_index = 0
        for quantizedBlock in quantized_dct:
            # find LSB in dc coeff and replace with message bit
            dc = quantizedBlock[0][0]
            dc = np.uint8(dc)
            dc = np.unpackbits(dc)
            dc[7] = self.bitMess[mess_index][letter_index]
            dc = np.packbits(dc)
            dc = np.float32(dc)
            dc = dc - 255
            quantizedBlock[0][0] = dc
            letter_index = letter_index + 1
            if letter_index == 8:
                letter_index = 0
                mess_index = mess_index + 1
                if mess_index == len(self.message):
                    break
        # blocks run inversely through quantization table
        s_img_blocks = [quantizedBlock * quant + 128 for quantizedBlock in quantized_dct]
        # blocks run through inverse DCT
        # s_img_blocks = [cv2.idct(B)+128 for B in quantized_dct]
        # puts the new image back together
        s_img = []
        for chunkRowBlocks in self.chunks(s_img_blocks, col / 8):
            for rowBlockNum in range(8):
                for block in chunkRowBlocks:
                    s_img.extend(block[rowBlockNum])
        s_img = np.array(s_img).reshape(row, col)
        # converted from type float32
        s_img = np.uint8(s_img)
        # show(s_img)
        s_img = cv2.merge((s_img, g_img, r_img))
        return s_img

    # decoding part :
    def decode_image(self, img):
        row, col = img.shape[:2]
        mess_size = None
        message_bits = []
        buff = 0
        # split image into RGB channels
        temp = cv2.split(img)
        b_img, g_img, r_img = temp
        # message hid in blue channel so converted to type float32 for dct function
        b_img = np.float32(b_img)
        # break into 8x8 blocks
        img_blocks = [b_img[j:j + 8, i:i + 8] - 128 for (j, i) in itertools.product(range(0, row, 8),
                                                                                    range(0, col, 8))]
        # blocks run through quantization table
        # quantized_dct = [dct_Block/ (quant) for dct_Block in dctBlocks]
        quantized_dct = [img_Block / quant for img_Block in img_blocks]
        i = 0
        # message extracted from LSB of dc coeff
        for quantizedBlock in quantized_dct:
            dc = quantizedBlock[0][0]
            dc = np.uint8(dc)
            dc = np.unpackbits(dc)
            if dc[7] == 1:
                buff += (0 & 1) << (7 - i)
            elif dc[7] == 0:
                buff += (1 & 1) << (7 - i)
            i = 1 + i
            if i == 8:
                message_bits.append(chr(buff))
                buff = 0
                i = 0
                if message_bits[-1] == '*' and mess_size is None:
                    try:
                        mess_size = int(''.join(message_bits[:-1]))
                    except:
                        pass
            if len(message_bits) - len(str(mess_size)) - 1 == mess_size:
                return ''.join(message_bits)[len(str(mess_size)) + 1:]
        # blocks run inversely through quantization table
        s_img_blocks = [quantized_block * quant + 128 for quantized_block in quantized_dct]
        # blocks run through inverse DCT
        # s_img_blocks = [cv2.idct(B)+128 for B in quantized_dct]
        # puts the new image back together
        s_img = []
        for chunkRowBlocks in self.chunks(s_img_blocks, col / 8):
            for rowBlockNum in range(8):
                for block in chunkRowBlocks:
                    s_img.extend(block[rowBlockNum])
        s_img = np.array(s_img).reshape(row, col)
        # converted from type float32
        s_img = np.uint8(s_img)
        s_img = cv2.merge((s_img, g_img, r_img))
        # s_img.save(img)
        # dct_decoded_image_file = "dct_" + original_image_file
        # cv2.imwrite(dct_decoded_image_file,s_img)
        return ''

    """Helper function to 'stitch' new image back together"""

    def chunks(self, l, n):
        m = int(n)
        for i in range(0, len(l), m):
            yield l[i:i + m]

    def add_padd(self, img, row, col):
        img = cv2.resize(img, (col + (8 - col % 8), row + (8 - row % 8)))
        return img

    def to_bits(self):
        bits = []
        for char in self.message:
            binval = bin(ord(char))[2:].rjust(8, '0')
            bits.append(binval)
        self.numBits = bin(len(bits))[2:].rjust(8, '0')
        return bits


class LSB:
    # encoding part :
    def encode_image(self, img, msg):
        length = len(msg)
        if length > 255:
            print("text too long! (don't exeed 255 characters)")
            return False
        encoded = img.copy()
        width, height = img.size
        index = 0
        for row in range(height):
            for col in range(width):
                if img.mode == 'RGB':
                    r, g, b = img.getpixel((col, row))
                else:  # if img.mode != 'RGB':
                    r, g, b, a = img.getpixel((col, row))

                # first value is length of msg
                if row == 0 and col == 0 and index < length:
                    asc = length
                elif index <= length:
                    c = msg[index - 1]
                    asc = ord(c)
                else:
                    asc = b
                encoded.putpixel((col, row), (r, g, asc))
                index += 1
        return encoded

    # decoding part :
    def decode_image(self, img):
        width, height = img.size
        msg = ""
        index = 0
        for row in range(height):
            for col in range(width):
                if img.mode == 'RGB':
                    r, g, b = img.getpixel((col, row))
                else:  # if img.mode != 'RGB':
                    r, g, b, a = img.getpixel((col, row))

                # first pixel r value is length of message
                if row == 0 and col == 0:
                    length = b
                elif index <= length:
                    msg += chr(b)
                index += 1
        lsb_decoded_image_file = "lsb_" + original_image_file
        # img.save(lsb_decoded_image_file)
        # print("Decoded image was saved!")
        return msg


class Compare:
    def correlation(self, img1, img2):
        return signal.correlate2d(img1, img2)

    def mean_square_error(self, img1, img2):
        # NMSE
        for i in range(len(img1)):
            for j in range(len(img1[0])):
                error = (np.sum(img1[i, j] - img2[i, j])) ** 2 / (np.sum(img1[i, j]) ** 2)
        # error = np.sum((img1.astype('float') - img2.astype('float')) ** 2)
        # error /= float(img1.shape[0] * img1.shape[1])
        return error

    def psnr(self, img1, img2):
        mse = self.mean_square_error(img1, img2)
        if mse == 0:
            return 100
        PIXEL_MAX = 255.0
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

    def nad(self, img1, img2):
        # nad = np.sum(np.abs(img1.astype('float') - img2.astype('float')))/np.sum(np.abs(img1.astype('float')))
        for i in range(len(img1)):
            for j in range(len(img1[0])):
                nad = abs(np.sum(img1[i, j] - img2[i, j])) / abs(np.sum(img1[i, j]))
        return nad

    def normalized_correlation(self, img1, img2):
        for i in range(len(img1)):
            for j in range(len(img1[0])):
                nc = (np.sum(img1[i, j] * img2[i, j])) / (np.sum(img1[i, j]) ** 2)
        return nc


# driver part :
# deleting previous folders :
if os.path.exists("Encoded_image/"):
    shutil.rmtree("Encoded_image/")
if os.path.exists("Decoded_output/"):
    shutil.rmtree("Decoded_output/")
if os.path.exists("Comparison_result/"):
    shutil.rmtree("Comparison_result/")
# creating new folders :
os.makedirs("Encoded_image/")
os.makedirs("Decoded_output/")
os.makedirs("Comparison_result/")
original_image_file = ""  # to make the file name global variable
lsb_encoded_image_file = ""
dct_encoded_image_file = ""
dwt_encoded_image_file = ""

while True:
    input_number = input("To encode press '1', to decode press '2', to compare press '3', press any other button to close: ")

    if input_number == "1":
        os.chdir("Original_image/")
        original_image_file = input("Enter the name of the file with extension : ")
        lsb_img = Image.open(original_image_file)
        dct_img = cv2.imread(original_image_file, cv2.IMREAD_UNCHANGED)
        #        dwt_img = cv2.imread(original_image_file, cv2.IMREAD_UNCHANGED)
        print("Description : ", lsb_img, "\nMode : ", lsb_img.mode)
        secret_msg_ = input("Enter the message you want to hide: ")
        print("The message length is: ", len(secret_msg_))
        os.chdir("..")
        os.chdir("Encoded_image/")
        lsb_img_encoded = LSB().encode_image(lsb_img, secret_msg_)
        dct_img_encoded = DCT().encode_image(dct_img, secret_msg_)
        #        dwt_img_encoded = DWT().encode_image(dwt_img, secret_msg_)
        lsb_encoded_image_file = "lsb_" + original_image_file
        lsb_img_encoded.save(lsb_encoded_image_file)
        dct_encoded_image_file = "dct_" + original_image_file
        cv2.imwrite(dct_encoded_image_file, dct_img_encoded)
        #        dwt_encoded_image_file = "dwt_" + original_image_file
        #        cv2.imwrite(dwt_encoded_image_file,dwt_img_encoded) # saving the image with the hidden text
        print("Encoded images were saved!")
        os.chdir("..")

    elif input_number == "2":
        os.chdir("Encoded_image/")
        lsb_img = Image.open(Path(lsb_encoded_image_file))
        dct_img = cv2.imread(dct_encoded_image_file, cv2.IMREAD_UNCHANGED)
        #        dwt_img = cv2.imread(dwt_encoded_image_file, cv2.IMREAD_UNCHANGED)
        os.chdir("..")  # going back to parent directory
        os.chdir("Decoded_output/")
        lsb_hidden_text = LSB().decode_image(lsb_img)
        dct_hidden_text = DCT().decode_image(dct_img)
        #        dwt_hidden_text = DWT().decode_image(dwt_img)
        file = open("lsb_hidden_text.txt", "w")
        file.write(lsb_hidden_text)  # saving hidden text as text file
        file.close()
        file = open("dct_hidden_text.txt", "w")
        file.write(dct_hidden_text)  # saving hidden text as text file
        file.close()
        #        file = open("dwt_hidden_text.txt","w")
        #        file.write(dwt_hidden_text) # saving hidden text as text file
        #        file.close()
        print("Hidden texts were saved as text file!")
        os.chdir("..")
    elif input_number == "3":
        # comparison calls
        os.chdir("Original_image/")
        original = cv2.imread(original_image_file)
        os.chdir("..")
        os.chdir("Encoded_image/")
        lsbEncoded = cv2.imread(lsb_encoded_image_file)
        dctEncoded = cv2.imread(dct_encoded_image_file)
        #        dwtEncoded = cv2.imread(dwt_encoded_image_file)
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        lsb_encoded_img = cv2.cvtColor(lsbEncoded, cv2.COLOR_BGR2RGB)  # ???????
        dct_encoded_img = cv2.cvtColor(dctEncoded, cv2.COLOR_BGR2RGB)
        #        dwt_encoded_img = cv2.cvtColor(dwtEncoded, cv2.COLOR_BGR2RGB)
        os.chdir("..")
        os.chdir("Comparison_result/")

        book = xlwt.Workbook()
        sheet1 = book.add_sheet("Sheet 1")
        style_string = "font: bold on , color red; borders: bottom dashed"
        style = xlwt.easyxf(style_string)
        sheet1.write(0, 0, "Original vs", style=style)
        sheet1.write(0, 1, "Correlation", style=style)  # new one
        sheet1.write(0, 2, "NMSE", style=style)
        sheet1.write(0, 3, "PSNR", style=style)
        sheet1.write(0, 4, "NAD", style=style)  # new one
        sheet1.write(0, 5, "Normalized Cross-Correlation", style=style)  # new one

        sheet1.write(1, 0, "LSB")
        # sheet1.write(1, 1, Compare().correlation(original, lsb_encoded_img))
        sheet1.write(1, 2, Compare().mean_square_error(original, lsb_encoded_img))
        sheet1.write(1, 3, Compare().psnr(original, lsb_encoded_img))
        sheet1.write(1, 4, Compare().nad(original, lsb_encoded_img))
        sheet1.write(1, 5, Compare().normalized_correlation(original, lsb_encoded_img))

        sheet1.write(2, 0, "DCT")
        # sheet1.write(1, 1, Compare().correlation(original, dct_encoded_img))
        sheet1.write(2, 2, Compare().mean_square_error(original, dct_encoded_img))
        sheet1.write(2, 3, Compare().psnr(original, dct_encoded_img))
        sheet1.write(2, 4, Compare().nad(original, dct_encoded_img))
        sheet1.write(2, 5, Compare().normalized_correlation(original, dct_encoded_img))

        sheet1.write(3, 0, "DWT")
        # sheet1.write(3, 1, Compare().meanSquareError(original, dwt_encoded_img))
        # sheet1.write(3, 2, Compare().psnr(original, dwt_encoded_img))

        book.save("Comparison.xls")
        print("Comparison Results were saved as xls file!")
        os.chdir("..")
    else:
        print("Closed!")
        break
