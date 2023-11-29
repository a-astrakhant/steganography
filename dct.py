import itertools

import cv2
import numpy as np

quant = np.array([[16, 11, 10, 16, 24, 40, 51, 61],  # QUANTIZATION TABLE
                  [12, 12, 14, 19, 26, 58, 60, 55],  # required for DCT
                  [14, 13, 16, 24, 40, 57, 69, 56],
                  [14, 17, 22, 29, 51, 87, 80, 62],
                  [18, 22, 37, 56, 68, 109, 103, 77],
                  [24, 35, 55, 64, 81, 104, 113, 92],
                  [49, 64, 78, 87, 103, 121, 120, 101],
                  [72, 92, 95, 98, 112, 100, 103, 99]])


class DCT:
    def __init__(self):  # Constructor
        self.message = None
        self.bits_bin: list[str] = []
        self.ori_col = 0
        self.ori_row = 0

    def encode_image(self, img: np.ndarray, secret_msg: str):
        # show(img)
        self.message = str(len(secret_msg)) + '*' + secret_msg
        self.bits_bin: list[str] = self.to_bits()
        # get size of image in pixels
        rows, columns = img.shape[:2]

        self.ori_row, self.ori_col = rows, columns
        if (columns / 8) * (rows / 8) < len(secret_msg):
            raise ValueError("Error: Message too large to encode in image")
        # add padding to make img divisible by 8x8
        if rows % 8 != 0 or columns % 8 != 0:
            img = cv2.resize(img, (columns + (8 - columns % 8), rows + (8 - rows % 8)))

        rows, columns = img.shape[:2]
        # columns, rows = img.size
        # split image into RGB channels
        b_img, g_img, r_img = cv2.split(img)

        # message to be hid in blue channel so converted to type float32 for dct function
        b_img = np.float32(b_img)
        # break into 8x8 blocks
        img_blocks = [np.round(b_img[j:j + 8, i:i + 8] - 128) for (j, i) in itertools.product(range(0, rows, 8),
                                                                                              range(0, columns, 8))]
        # Blocks are run through DCT function
        dct_blocks = [np.round(cv2.dct(img_block)) for img_block in img_blocks]
        # blocks then run through quantization table
        quantized_dct = [np.round(dct_block / quant) for dct_block in dct_blocks]
        # set LSB in dc value corresponding bit of message
        mess_index = 0
        letter_index = 0
        for quantized_block in quantized_dct:
            # find LSB in dc coeff and replace with message bit
            dc = quantized_block[0][0]
            dc = np.uint8(dc)
            dc = np.unpackbits(dc)
            dc[7] = self.bits_bin[mess_index][letter_index]
            dc = np.packbits(dc)
            dc = np.float32(dc)
            dc = dc - 255
            quantized_block[0][0] = dc
            letter_index = letter_index + 1
            if letter_index == 8:
                letter_index = 0
                mess_index = mess_index + 1
                if mess_index == len(self.message):
                    break

        # blocks run inversely through quantization table
        s_img_blocks = [quantized_block * quant + 128 for quantized_block in quantized_dct]
        # blocks run through inverse DCT
        # s_img_blocks = [cv2.idct(B)+128 for B in quantized_dct]
        # puts the new image back together
        s_img = []
        for chunk_row_blocks in self.chunks(s_img_blocks, columns / 8):
            for row_block_num in range(8):
                for block in chunk_row_blocks:
                    s_img.extend(block[row_block_num])
        s_img = np.array(s_img).reshape(rows, columns)
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
        quantized_dct = [img_block / quant for img_block in img_blocks]
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
        for chunk_row_blocks in self.chunks(s_img_blocks, col / 8):
            for row_block_num in range(8):
                for block in chunk_row_blocks:
                    s_img.extend(block[row_block_num])
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

    def to_bits(self) -> list[str]:
        bits_bin: list[str] = []
        for char in self.message:
            binval = bin(ord(char))[2:].rjust(8, '0')
            bits_bin.append(binval)

        return bits_bin
