import cv2
import matplotlib.pyplot as plt
import numpy as np
from frank_model import FrankNet
from tensorflow.keras.models import Model


class CNNVisualizer:
    def __init__(self, path, img_path):
        self.model = FrankNet.build(200, 150)
        self.model.load_weights(path)
        self.model.summary()
        self.current_image = cv2.imread(img_path)
        self.illustrative_model = None
        self.processed_img = self.preprocess_img()
        print("Setup Complete!")

    def image_resize(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = cv2.resize(image, dim, interpolation=inter)

        # return the resized image
        return resized

    def preprocess_img(self):
        input_image = self.image_resize(self.current_image, width=200)
        input_image = input_image[0:150, 0:200]
        input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2YUV)
        return np.expand_dims(input_image, axis=0)

    def redefine_model(self, layer):
        outputs = [self.model.layers[i].output for i in layer]
        self.illustrative_model = Model(inputs=self.model.inputs, outputs=outputs)
        print("Current setup:")
        self.illustrative_model.summary()


    def visualize(self):
        to_visualize = []
        layer_name = []
        for i in range(len(self.model.layers)):
            layer = self.model.layers[i]
            if 'conv' not in layer.name:
                continue
            to_visualize.append(i)
            layer_name.append(""+str(i)+str(layer.name)+str(layer.output.shape))
            print(i, layer.name, layer.output.shape)
        self.redefine_model(to_visualize)
        feature_maps = self.illustrative_model.predict(self.processed_img)
        square = 8
        for fmap in feature_maps:
            ix = 1
            title_id = 0
            fig = plt.figure()
            for _ in range(square):
                for _ in range(square):
                    ax = fig.add_subplot(square, square, ix)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    # plot filter channel in grayscale
                    try:
                        plt.imshow(fmap[0, :, :, ix-1], cmap='gray')
                    except Exception:
                        continue
                    ix += 1
            title_id += 1

            plt.show()


if __name__ == '__main__':
    node = CNNVisualizer("cbcNet.h5", "curve.jpg")
    node.visualize()
