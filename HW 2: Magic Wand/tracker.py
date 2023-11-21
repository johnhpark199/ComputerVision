import cv2

from magicwand import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('video',help='path to input video file')
parser.add_argument('--output',help='path to output video file (optional)')
parser.add_argument('--calibration',default='iphone_calib.txt',help='path to calibration file')
parser.add_argument('--ball_radius',type=float,default=3,help='radius of ball in cm')
args = parser.parse_args()

wand = MagicWand(calibration_path=args.calibration,R=args.ball_radius)

cap = cv2.VideoCapture(args.video)
ball_pos = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    ball_pos.append(wand.process_frame(frame))

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) == ord('q'):
        break
cv.destroyWindow('frame')
fig = plt.figure()
ax = fig.add_axes(111, projection='3d')
i = 0
while i + 1 < len(ball_pos):
    # do nothing if no balls found
    if np.sum(ball_pos[i]) == 0:
        None
    else:
        # creating lists inbetween data points to better represent path of ball
        x = np.linspace(ball_pos[i][0][0], ball_pos[i + 1][0][0])
        y = np.linspace(ball_pos[i][0][1], ball_pos[i + 1][0][1])
        z = np.linspace(ball_pos[i][0][2], ball_pos[i + 1][0][2])
        ax.plot(x, y, z)
    i += 1
plt.show()