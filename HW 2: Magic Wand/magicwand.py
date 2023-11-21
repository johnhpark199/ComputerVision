import cv2 as cv
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt


class MagicWand:
    def __init__(self, calibration_path, R):
        """ Loads calibration from file and stores ball radius.
            Arguments:
                calibration_path: path to calibration file
                R: ball radius in cm
        """
        self.focal, self.centerx, self.centery = np.loadtxt(calibration_path, delimiter=' ')
        self.R = R

    def detect_ball(self, image):
        """ Detect one or more balls in image.
            Arguments:
                image: RGB image in which to detect balls
            Returns:
                list of tuples (x, y, radius)
        """
        gray_im = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        blurred_im = cv.GaussianBlur(gray_im, (9, 9), 2)
        circle_arr = cv.HoughCircles(blurred_im, cv.HOUGH_GRADIENT, dp=2, minDist=10)
        # if no balls found return an empty array
        if circle_arr is None:
            return [[0, 0, 0]]
        return circle_arr[0]

    def calculate_ball_position(self, x, y, r):
        """ Calculate ball's (X,Y,Z) position in world coordinates
            Arguments:
                x,y: 2D position of ball in image
                r: radius of ball in image
            Returns:
                X,Y,Z position of ball in world coordinates (cm)
        """
        world_coords = [0, 0, 0]
        # R = ball radius in cm (self.R)
        # r = ball radius in pixels
        # f = focal length (self.focal)
        # Z = Z position of ball (cm)
        # determining Z coordinate
        # Z = f * ( R / r )
        world_coords[2] = self.focal * (self.R / r)
        Z = world_coords[2]

        # x = (f* (X/Z)) + c(x)
        # x - cx = f (X/Z)
        # (x- cx) / f = X/Z
        # Z/f * (x -cx) = X
        world_coords[0] = (Z/self.focal) * (x - self.centerx)

        # (y-cy) = f * (Y/Z)
        # (y-cy)/f = Y/Z
        # (y-cy) * (Z/f) = Y
        world_coords[1] = (Z/self.focal) * (y - self.centery)
        # (X, Y, Z)
        return world_coords

    def draw_ball(self, image, x, y, r, Z):
        """ Draw circle on ball and write depth estimate in center
            Arguments:
                image: image on which to draw
                x,y,r: 2D position and radius of ball
                Z: estimated depth of ball
        """
        cv.circle(image, (int(x), int(y)), int(r), (0, 0, 255), 2)
        cv.putText(image, str(int(Z)) + ' cm', (int(x),int(y)), cv.FONT_HERSHEY_PLAIN, 1, (0,0,255))
    
    def project(self, X, Y, Z):
        """ Pinhole projection.
            Arguments:
                X,Y,Z: 3D point
            Returns:    
                (x,y) 2D location of projection in image
        """
        # calculating 2d pixel location of point
        # casting to an int as only int locations exist in a image
        x = int((self.focal * (X / Z)) + self.centerx)
        y = int((self.focal * (Y / Z)) + self.centery)
        loc_2d = [x, y]
        return loc_2d

    def draw_line(self,image,X1,Y1,Z1,X2,Y2,Z2):
        """ Draw a 3D line
            Arguments:
                image: image on which to draw
                X1,Y1,Z1: 3D position of first line endpoint
                X2,Y2,Z2: 3D position of second line endpoint
        """
        # casting to 2D points and drawing
        start_point = self.project(X1, Y1, Z1)
        end_point = self.project(X2, Y2, Z2)
        cv.line(image, (start_point[0], start_point[1]), (end_point[0], end_point[1]), (0, 0, 255), 1)

    def draw_bounding_cube(self, image, X, Y, Z):
        """ Draw bounding cube around 3D point, with radius R
            Arguments:
                image: image on which to draw
                X,Y,Z: 3D center point of cube
        """
        # calculating different corners of ball, according to radius
        bottom_right_front = (X + 3, Y + 3, Z + 3)
        bottom_left_front = (X - 3, Y + 3, Z + 3)
        top_right_front = (X + 3, Y - 3, Z + 3)
        top_left_front = (X - 3, Y - 3, Z + 3)
        top_right_back = (X + 3, Y - 3, Z - 3)
        top_left_back = (X - 3, Y - 3, Z - 3)
        bottom_right_back = (X + 3, Y + 3, Z - 3)
        bottom_left_back = (X - 3, Y + 3, Z - 3)

        # drawing lines between each of the calculated corners surrounding the ball
        self.draw_line(image, bottom_left_front[0], bottom_left_front[1], bottom_left_front[2], bottom_right_front[0],
                       bottom_right_front[1], bottom_right_front[2])
        self.draw_line(image, bottom_right_front[0], bottom_right_front[1], bottom_right_front[2], top_right_front[0],
                       top_right_front[1], top_right_front[2])
        self.draw_line(image, bottom_left_front[0], bottom_left_front[1], bottom_left_front[2],top_left_front[0],
                       top_left_front[1], top_left_front[2])
        self.draw_line(image, top_left_front[0], top_left_front[1], top_left_front[2], top_right_front[0],
                       top_right_front[1], top_right_front[2])
        self.draw_line(image, top_right_front[0], top_right_front[1], top_right_front[2], top_right_back[0],
                       top_right_back[1], top_right_back[2])
        self.draw_line(image, top_left_front[0], top_left_front[1], top_left_front[2], top_left_back[0],
                       top_left_back[1], top_left_back[2])
        self.draw_line(image, top_left_back[0], top_left_back[1], top_left_back[2], top_right_back[0],
                       top_right_back[1], top_right_back[2])
        self.draw_line(image, bottom_right_front[0], bottom_right_front[1], bottom_right_front[2], bottom_right_back[0],
                       bottom_right_back[1], bottom_right_back[2])
        self.draw_line(image, top_right_back[0], top_right_back[1], top_right_back[2], bottom_right_back[0],
                       bottom_right_back[1], bottom_right_back[2])
        self.draw_line(image, bottom_left_front[0], bottom_left_front[1], bottom_left_front[2], bottom_left_back[0],
                       bottom_left_back[1], bottom_left_back[2])
        self.draw_line(image, bottom_left_back[0], bottom_left_back[1], bottom_left_back[2], bottom_right_back[0],
                       bottom_right_back[1], bottom_right_back[2])
        self.draw_line(image, bottom_left_back[0], bottom_left_back[1], bottom_left_back[2], top_left_back[0],
                       top_left_back[1], top_left_back[2])
        pass
    
    def process_frame(self, image):
        """ Detect balls in frame, estimate 3D positions, and draw on image
            Arguments:
                image: image to be processed
            Returns:
                list of (X,Y,Z) 3D points of detected balls
        """
        circle_cords = self.detect_ball(image)
        # checking to make sure a ball is found
        if np.sum(circle_cords) == 0:
            return circle_cords
        else:
            num_circles = len(circle_cords)
            circle_pos = []
            # iterating for each ball found
            for i in range(0, num_circles):
                # determining 3D ball position and appending to final ball position array locations
                circle_pos.append(self.calculate_ball_position(circle_cords[i][0], circle_cords[i][1], circle_cords[i][2]))
                # calculating 2D location of ball
                circle_xy = self.project(circle_pos[i][0], circle_pos[i][1], circle_pos[i][2])
                # drawing surrounding ball
                self.draw_ball(image, circle_xy[0], circle_xy[1], circle_cords[i][2], circle_pos[i][2])
                # drawing bounding cube
                self.draw_bounding_cube(image, circle_pos[i][0], circle_pos[i][1], circle_pos[i][2])
        return circle_pos


# normal magic wand method
# used for testing
def magicwand_mov(path):

    wand = MagicWand('iphone_calib.txt', 3)
    cap = cv.VideoCapture(path)
    ball_pos = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        ball_pos.append(wand.process_frame(frame))
        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break
    # destroying window to show plot path
    cv.destroyWindow('frame')
    # initializing 3d plot
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


# method used for calculating and processing the length of the wand
# connecting the balls
def wand_mov(path):
    wand = MagicWand('iphone_calib.txt', 3)
    cap = cv.VideoCapture(path)
    ball_pos = []
    i = 0
    distance = []
    # normal image processing and displaying
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        ball_pos.append(wand.process_frame(frame))

        # if there are two balls calculate the distance
        if len(ball_pos[i]) == 2:
            delta_x = abs((ball_pos[i][0][0] - ball_pos[i][1][0]) ** 2)
            delta_y = abs((ball_pos[i][0][1] - ball_pos[i][1][1]) ** 2)
            delta_z = abs((ball_pos[i][0][2] - ball_pos[i][1][2]) ** 2)
            distance.append((delta_x + delta_y + delta_z) ** 0.5)
        cv.imshow('frame.jpg', frame)
        if cv.waitKey(1) == ord('q'):
            break
        i += 1
    # calculating and displaying average wand length
    print(sum(distance)/len(distance))
