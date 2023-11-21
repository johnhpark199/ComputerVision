import cv2
from detector import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input')
args = parser.parse_args()

im = cv2.imread(args.input)

detector = FeatureDetector()

responses = detector.get_dog_response_stack(im)
features = detector.find_features(responses)
image_out = detector.draw_features(im,features)

cv2.imwrite('features.png',image_out)

