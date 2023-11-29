from PIL import Image


class LSB:
    # encoding part :
    @staticmethod
    def encode_image(img: Image.Image, msg: str) -> Image.Image:
        msg_length = len(msg)
        if len(msg) > 255:
            raise ValueError(f"text too long! (don't exceed 255 characters): {msg}")

        encoded: Image.Image = img.copy()
        width, height = img.size
        index = 0
        for row in range(height):
            for col in range(width):
                if img.mode == 'RGB':
                    r, g, b = img.getpixel((col, row))
                else:  # if img.mode != 'RGB':
                    r, g, b, a = img.getpixel((col, row))

                # first value is length of msg
                if row == 0 and col == 0 and index < msg_length:
                    asc = msg_length
                elif index <= msg_length:
                    c = msg[index - 1]
                    asc = ord(c)
                else:
                    asc = b
                encoded.putpixel((col, row), (r, g, asc))
                index += 1
        return encoded

    # decoding part :
    @staticmethod
    def decode_image(img: Image.Image) -> str:
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

        return msg
