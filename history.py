import numpy as np
import random
import cv2
try:
    from PIL import Image
except:
    import Image

from utils import *


class History():
    """
        Manages frame history
    """

    def __init__(self, length, im_size, black_and_white=True):
        """
            Args:
                length: How many frames should be stored in the history
                im_size: Target size to crop to, either im_size=WIDTH=HEIGHT or im_size = (WIDTH,HEIGHT)
        """
        self.black_and_white = black_and_white
        self.dimensions = 1 if self.black_and_white else 3

        if type(im_size) == int:
            self.im_size = (im_size, im_size)
        else:
            self.im_size = tuple(im_size)
            assert len(self.im_size) == 2
        self.im_shape = self.im_size + (self.dimensions, )

        self.history = np.zeros((length, ) + self.im_shape, dtype=np.uint8)
        self.shape = self.history.shape


    def fill_raw_frame(self, raw_frame):
        """Fill the history with a raw frame
        
        Arguments:
            raw_frame {np.array} -- Pixel array
        
        Returns:
            np.array -- The new state
        """

        self.add_raw_frame(raw_frame)
        return self.reset()

    def add_raw_frame(self, raw_frame):
        """
            Adds a new frame to the history
        """
        next_frame = self.process_frame(raw_frame)
        self.history = np.roll(self.history, -1, axis=0)
        self.history[-1] = next_frame
        return self.history

    def reset(self):
        """
            Fills the state with the latest experienced frame
        """
        for i in range(len(self.history)-1):
            self.history[i] = self.history[-1]
        return self.history


    def process_frame(self, raw_frame, save=False, filename=None):
        """Processes a frame by resizing and cropping as necessary and then
        converting to grayscale
        
        Arguments:
            raw_frame {np.array} -- Raw pixels
        
        Keyword Arguments:
            save {bool} -- Whether to save the converted frame to disk (default: {False})
            filename {str} -- Filename to save it to (default: {None})
        
        Returns:
            np.array -- The processed frame
        """

        if self.black_and_white:
            raw_frame = cv2.cvtColor(raw_frame,cv2.COLOR_RGB2GRAY)
       
        cropped = cv2.resize(raw_frame, dsize=self.im_size, interpolation=cv2.INTER_AREA)
        if save:
            img = Image.fromarray(cropped, mode='L')
            img.save(filename)

        return  cropped.reshape(self.im_size+(self.dimensions,))
      

    def save_image(self, frame, filename=None):
        if filename is None:
            filename = "img/"+str(random.random())+".png"
        if self.black_and_white:
            frame = frame.reshape(self.im_size)
            img = Image.fromarray(frame, mode='L')
            img.save(filename)
        else:
            img = Image.fromarray(frame, 'RGB')
            img.save(filename)