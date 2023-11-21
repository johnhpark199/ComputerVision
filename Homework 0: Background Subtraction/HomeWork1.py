import os
import cv2
import imageio
import numpy as np
import cv2 as cv


def p1():
    # reading and print image size
    image = cv.imread('frames/000000.jpg')
    cv.imshow('original_image.png', image)

    # printing image shape
    print(image.shape)

    # printing actual image
    print(image)

    # converting to and showing grayscale
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imshow('grey_image', gray_image)
    cv.waitKey(0)

    # saving greyscale image
    save_path = r'C:\Users\johnh\OneDrive\CAL POLY\EE428\lab0\grey_image.png'
    cv.imwrite(save_path, gray_image)


def p2():
    # setting up video capture
    capture = cv.VideoCapture('frames/%06d.jpg')
    # while loop runs while the capture still has pngs to iterate through
    # first index of array is 1 when there are remaining images to be processed
    images_exist = 1
    traffic_arr = []

    # could be optimized more if number of images within zip file are known
    while images_exist == 1:
        data_arr = capture.read()
        traffic_image = data_arr[1]
        images_exist = data_arr[0]
        if images_exist == 1:
            gray_im = cv.cvtColor(traffic_image, cv.COLOR_BGR2GRAY)
            # appending data from each frame in order to compute mean of array later
            traffic_arr.append(gray_im)
            # displaying each image within the video
            cv.imshow('images.jpg', gray_im)
            cv.waitKey(25)
        else:
            pass
    # casting images to array to ensure they are formatted correctly
    images = np.array(traffic_arr)
    # computing mean of array in order to find average image
    background_image = np.mean(images, axis=0).astype('uint8')
    # displaying average image
    cv.imshow('background.jpg', background_image)
    cv.waitKey(0)
    background_path = r'C:\Users\johnh\OneDrive\CAL POLY\EE428\lab0\background.png'
    # saving background image
    cv.imwrite(background_path, background_image)


def p3():
    # loading images from first two parts
    grayscale_first = cv.imread(r'C:\Users\johnh\OneDrive\CAL POLY\EE428\lab0\grey_image.jpg', cv.IMREAD_GRAYSCALE)
    background_im = cv.imread(r'C:\Users\johnh\OneDrive\CAL POLY\EE428\lab0\background.jpg', cv.IMREAD_GRAYSCALE)

    # computing absolute difference of images
    diff_im = cv.absdiff(background_im, grayscale_first)
    cv.imshow('diff_im.jpg', diff_im)
    cv.waitKey(0)

    # normal filtering method, 35 and 255 were found to be most optimal parameters
    ret1, thresh_im = cv.threshold(diff_im, 35, 255, cv.THRESH_BINARY)
    cv.imshow('thresh_im.jpg', thresh_im)
    cv.waitKey(0)

    # otsu thresholding method
    ret2, otsu_image = cv.threshold(diff_im, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    cv.imshow('otsu_im.jpg', otsu_image)
    cv.waitKey(0)


# completed half of bonus I am still struggling to get bounding boxes to display correctly
def bonus():
    # loading video capture again to read each frame
    capture2 = cv.VideoCapture('frames/%06d.jpg')
    # loading background image to find absolute difference of each frame
    background_im2 = cv.imread(r'C:\Users\johnh\OneDrive\CAL POLY\EE428\lab0\background.jpg', cv.IMREAD_GRAYSCALE)
    i = 0
    while i < 50:
        # reading each image, gray scaling it, finding the absolute difference, and displaying the thresholded image
        color_image = capture2.read()[1]
        gray_image = cv.cvtColor(color_image, cv.COLOR_BGR2GRAY)
        abs_im = cv.absdiff(background_im2, gray_image)
        ret3, otsu_filter = cv.threshold(abs_im, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        cv.imshow('filtered_vid.jpg', otsu_filter)
        cv.waitKey(35)

        # contouring image
        contours, ret4 = cv.findContours(abs_im, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        for j in contours:
            # iterating through each contour found in image
            rect = cv.boundingRect(j)
            if rect[2] < 100 or rect[3] < 100: continue
            x, y, w, h = rect
            # inserting attempted bounding boxes into images
            cv.rectangle(otsu_filter, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # displaying each image with attempted bounding box, waitkey 0 to examine each image and debug
        cv.imshow('bounding_box.jpg', otsu_filter)
        cv.waitKey(0)
        i += 1

# main function
if __name__ == '__main__':
    p1()
    p2()
    p3()
    bonus()