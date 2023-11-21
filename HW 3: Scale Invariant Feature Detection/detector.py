import cv2 as cv
import numpy
import numpy as np
import scipy.ndimage
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter

class FeatureDetector:
    def __init__(self, sigma=1, nblur=10, thresh=0.05):
        """ Initializes the FeatureDetector object.

            The feature detector detects difference- of-Gaussian (DOG) features.

            Features are detected by finding local minima / maxima in the
            DOG response stack.
        
            Arguments:
              sigma: base sigma value for Gaussian filters
              nblur: number of Gaussian filters
              thresh: minimum absolute response value for a feature detection
            """
        self.sigma = sigma
        self.nblur = nblur
        self.thresh = thresh
  
    def get_dog_response_stack(self, image):
        """ Build the DOG response stack.
        
        The image is first converted to grayscale, floating point on [0 1] range.
        Then a difference-of-Gaussian response stack is built.

        Let I be the original (grayscale) image.
        Let G[i] be the result of applying a Gaussian with sigma s * ((1.5)^i) to I,
        where s is the base sigma value.

        Layer i in the stack is computed as G[i+1]-G[i].
         
        Arguments:
            image: 8-bit BGR input image
        Returns:
            DOG response stack [nblur,H,W]
        """
        # gray scaling image
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # converting to 0 to 1 floating point
        float_im = gray_image / 255.0
        # initializing dog stack
        dog_stack = []
        for i in range(0, self.nblur):
            init_blur = scipy.ndimage.gaussian_filter(float_im, self.sigma * (1.5 ** (i+1)))
            dif_blur = scipy.ndimage.gaussian_filter(float_im, self.sigma * (1.5 ** i))
            # computing difference of adjacent gaussians
            dog_stack.append(init_blur - dif_blur)
        return dog_stack

    def find_features(self, responses):
        """ Find features in the DOG response stack.

        Features are detected using non-minima / non-maxima supression
        over a 3x3x3 window.
        
        To do this, compute the local minimum / maximum at each location using
        skimage.ndimage.minimum_filter / maximum_filter.

        Then find locations where the response value is equal to the local minimum/
        maximum, and the absolute response value exceeds thresh.
        
        See np.argwhere for a fast way to do this.
        
        Arguments:
            response: DOG response stack
        Returns:
            List of features (level,y,x)
        """
        # creating min array
        min = scipy.ndimage.minimum_filter(responses, (3, 3, 3))
        # creating max array
        max = scipy.ndimage.maximum_filter(responses, (3, 3, 3))
        # casting responses to numpy array, in order to abs function
        arr_responses = numpy.array(responses)
        # checking that value is greater than abs of thresh and equal to min or max
        feature_loc = np.argwhere((abs(arr_responses) > self.thresh) & ((responses == min) | (responses == max)))
        return feature_loc

    def draw_features(self, image, features, color=[0, 0, 255]):
        """ Draw features on an image.

        For each feature, draw it as a dot and a circle.
        
        The radius of the circle should be equal to the sigma value at that level.
        
        Arguments:
          image: input 8-bit BGR image
          features: list of (level,y,x) features
          color: color in which to draw
        Returns:
          Image with features drawn
        """
        for i in range(0, len(features)):
            cv.circle(image, (features[i][2], features[i][1]), int(round(self.sigma * (1.5 ** features[i][0]), 0)), color, 2)
            cv.circle(image, (features[i][2], features[i][1]), 0, color, 1)
        return image