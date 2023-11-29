import cv2
import numpy as np


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
