
import numpy as np
import cv2

"""
## python packages used
#  opencv-python
#  numpy

## references
#  http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_core/py_basic_ops/
#  http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html 
"""


class ImageOps(object):

    def __init__(self):
        self.image_file = 'data/images/P_20160708_173326_DF.jpg'
        self.image_loc = 'data/output/'

    """
    # Basic Image Operations
    # 1 Read Image
    # 2 Get Image Size
    # 3 Display Image
    # 4 Get Greyscale Image
    # 5 Split Image Channels
    # 6 Merge Image Channels
    # 7 Display Single Channel using split image results
    # 8 Image fit - square fit
    # 9 square fit with different types of images vertical, horizontal and square
    """

    def read_image(self, user_image=None, ops_image=None):
        """

        :param user_image:
        :param ops_image:
        :return:
        """
        if user_image is None and ops_image is None:
            img = cv2.imread(self.image_file)
        elif user_image is not None and ops_image is None:
            img = cv2.imread(user_image)
        elif user_image is None and ops_image is not None:
            img = ops_image
        return img

    def write_image(self, file_name=None,image_date=None):
        file_name = self.image_loc + file_name
        cv2.imwrite(file_name, image_date)


    def get_size(self, user_image=None, ops_image=None):
        """

        :param user_image:
        :param ops_image:
        :return:
        """
        data = {}
        img = self.read_image(user_image=user_image, ops_image=ops_image)
        if img is not None:
            height, width, channel = img.shape
            data = {"height": height, "width": width, "channel": channel}
        else:
            data = {"height": 0, "width": 0, "channel": 0}
        return data

    def display_image(self, image_title="image", wait_key=0, user_image=None, ops_image=None):
        """
        This method is used to Display the provided image

        :param image_title:
        :param wait_key:
        :param user_image:
        :param ops_image:
        :return:
        """
        img = self.read_image(user_image=user_image, ops_image=ops_image)
        cv2.imshow(image_title, img)
        cv2.waitKey(wait_key)
        cv2.destroyAllWindows()

    def get_grey_scale(self, user_image=None, ops_image=None, display_image=True, write_data=False):
        """
        Convert the given image into a Grey-scale Image

        :param user_image:
        :param ops_image:
        :param display_image:
        :return:
        """
        img = self.read_image(user_image=user_image, ops_image=ops_image)
        grey_scale_image = np.invert(img)
        if write_data is True:
            self.write_image(file_name="greyscale",image_date=grey_scale_image)
        if display_image is True:
            self.display_image(image_title="Greyscale Image", ops_image=grey_scale_image)
            self.write_image(file_name="greyscale.jpg", image_date=grey_scale_image)
        else:
            return grey_scale_image

    def split_image(self, user_image=None, ops_image=None):
        """
        This method splits the original given image into different channels R|G|B

        :param user_image:
        :param ops_image:
        :return:
        """

        img = self.read_image(user_image=user_image, ops_image=ops_image)
        b, g, r = cv2.split(img)
        return b, g, r

    def merge_image(self, b=None, g=None, r=None, display_image=True):
        """
        this method merges different channel red, green and blue to form the original image

        :param b:
        :param g:
        :param r:
        :param display_image:
        :return:
        """
        if b is not None and g is not None and r is not None:
            img = cv2.merge((b, g, r))
            if display_image is True:
                self.display_image(image_title="merge_image", ops_image=img)
            else:
                return img
        else:
            print("unable to display error in input")

    @staticmethod
    def display_single_channel(b=None, g=None, r=None):
        """
        duplicate function of display just for understanding channels

        :param b:
        :param g:
        :param r:
        :return:
        """

        if g is None and r is None and b is not None:
            cv2.imshow('b', b)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif r is None and b is None and g is not None:
            cv2.imshow('g', g)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif b is None and g is None and r is not None:
            cv2.imshow('r', r)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("invalid input")

    def square_fit(self, user_image=None, ops_image=None, display_image=True):
        """
        source code taken from
        https://stackoverflow.com/questions/44720580/resize-image-canvas-to-maintain-square-aspect-ratio-in-python-opencv
        modified and altered based on needs
        :param user_image:
        :param ops_image:
        :param display_image:
        :return:
        """

        img = self.read_image(user_image=user_image,ops_image=ops_image)
        height, width = img.shape[:2]
        img_sq_fit = cv2.resize(img, (height, height), interpolation=cv2.INTER_CUBIC)
        if display_image is True:
            self.display_image(image_title="square fit", ops_image=img_sq_fit)
            # self.vignette(ops_image=img_sq_fit)
        else:
            return img_sq_fit

    def square_fit_formation(self, user_image=None, ops_image=None, size=(200, 200), pad_color=127):
        """
        source code taken from
        https://stackoverflow.com/questions/44720580/resize-image-canvas-to-maintain-square-aspect-ratio-in-python-opencv

        :param user_image:
        :param ops_image:
        :param size:
        :param pad_color:
        :return:
        """

        img = self.read_image(user_image=user_image, ops_image=ops_image)
        h, w = img.shape[:2]
        sh, sw = size

        # interpolation method
        if h > sh or w > sw:  # shrinking image
            interp = cv2.INTER_AREA
        else:  # stretching image
            interp = cv2.INTER_CUBIC

        # aspect ratio of image
        aspect = w / h

        # compute scaling and pad sizing
        if aspect > 1:  # horizontal image
            new_w = sw
            new_h = np.round(new_w / aspect).astype(int)
            pad_vert = (sh - new_h) / 2
            pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
            pad_left, pad_right = 0, 0
        elif aspect < 1:  # vertical image
            new_h = sh
            new_w = np.round(new_h * aspect).astype(int)
            pad_horz = (sw - new_w) / 2
            pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
            pad_top, pad_bot = 0, 0
        else:  # square image
            new_h, new_w = sh, sw
            pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

        # set pad color
        if len(img.shape) is 3 and not isinstance(pad_color,
                                                  (list, tuple, np.ndarray)):  # color image but only one color provided
            pad_color = [pad_color] * 3

        # scale and pad
        scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
        scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right,
                                        borderType=cv2.BORDER_CONSTANT, value=pad_color)

        return scaled_img


    """
    # Image Transformations 
    # 1 Rotation
    # 2 Scaling
    # 3 Translation 
    """

    def rotate_image(self, user_image=None, ops_image=None, degrees=90, display_image=True):
        """

        :param user_image:
        :param ops_image:
        :param degrees:
        :param display_image:
        :return:
        """
        data = {}
        img = self.read_image(user_image=user_image, ops_image=ops_image)
        data = self.get_size(ops_image=img)
        cols = data['width']
        rows = data['height']
        m = cv2.getRotationMatrix2D((cols / 2, rows / 2), degrees, 1)
        dst = cv2.warpAffine(img, m, (cols, rows))
        if display_image is True:
            title = "Rotate Image" + str(degrees) + " degrees"
            self.display_image(image_title=title, ops_image=dst)
        else:
            return data

    def scale_image(self, user_image=None, ops_image=None, scale_vertical=2, scale_horizontal=4, display_image=True):
        """

        :param user_image:
        :param ops_image:
        :param scale_vertical:
        :param scale_horizontal:
        :param display_image:
        :return:
        """
        img = self.read_image(user_image=user_image, ops_image=ops_image)
        # alternate way to get height and width self.get_size
        height, width = img.shape[:2]
        res = cv2.resize(img, ( scale_horizontal * width, scale_vertical * height), interpolation=cv2.INTER_CUBIC)
        if display_image is True:
            title = "scale image vertical " + str(scale_vertical) + "horizontal" + str(scale_horizontal)
            self.display_image(image_title=title, ops_image=res)
        else:
            return res

    def translate_image(self):
        pass

    """
    ## filter
    # 1 vignette
    """
    def vignette(self, user_image=None, ops_image=None):

        """
        source code taken from
        https://www.packtpub.com/mapt/book/application_development/9781785283932/2/ch02lvl1sec25/creating-a-vignette-filter

        :param user_image:
        :param ops_image:
        :return:
        """

        img = self.read_image(user_image=user_image, ops_image=ops_image)
        height, width = img.shape[:2]
        img = cv2.resize(img, (height, height), interpolation=cv2.INTER_CUBIC)
        rows, cols = img.shape[:2]

        # generating vignette mask using Gaussian kernels
        kernel_x = cv2.getGaussianKernel(cols, 200)
        kernel_y = cv2.getGaussianKernel(rows, 200)
        kernel = kernel_y * kernel_x.T
        mask = 255 * kernel / np.linalg.norm(kernel)
        output = np.copy(img)

        # applying the mask to each channel in the input image
        for i in range(3):
            output[:, :, i] = output[:, :, i] * mask
        self.display_image(image_title="Original", ops_image=img)
        self.display_image(image_title="Vignette", ops_image=output)


if __name__ == '__main__':
    im = ImageOps()

    img_data = im.get_size()
    print(img_data)
    img_data = im.get_size(user_image="data/images/P_20160708_173326_DF.jpg")
    print(img_data)
    im.display_image()
    im.display_image(user_image="data/images/P_20160708_173326_DF.jpg")
    im.get_grey_scale()
    b, g, r = im.split_image(user_image="data/images/P_20160708_173326_DF.jpg")
    im.merge_image(r=r, b=b, g=g)
    im.display_single_channel(b=b)
    im.display_single_channel(g=g)
    im.display_single_channel(r=r)
    im.display_single_channel()
    im.rotate_image()
    im.rotate_image(degrees=120)
    im.scale_image()
    im.square_fit(user_image='data/images/P_20160708_173326_DF.jpg')
    scaled_v_img = im.square_fit_formation(user_image='data/images/P_20160708_173326_DF.jpg', size=(200, 200),
                                           pad_color=127)
    im.display_image(image_title="sq fit formation" ,ops_image=scaled_v_img)
    im.vignette(user_image='data/images/P_20160708_173326_DF.jpg')
