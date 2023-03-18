from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import cv2
import numpy as np
import operator
import os
from gtts import gTTS
from BusRecognitionFast import *

# module level variables
MIN_CONTOUR_AREA = 40

RESIZED_IMAGE_WIDTH = 28
RESIZED_IMAGE_HEIGHT = 28
language = 'en'

class ContourWithData():

    # member variables
    npaContour = None           # contour
    boundingRect = None         # bounding rect for contour
    intRectX = 0                # bounding rect top left corner x location
    intRectY = 0                # bounding rect top left corner y location
    intRectWidth = 0            # bounding rect width
    intRectHeight = 0           # bounding rect height
    fltArea = 0.0               # area of contour

    def calculateRectTopLeftPointAndWidthAndHeight(self):               # calculate bounding rect info
        [intX, intY, intWidth, intHeight] = self.boundingRect
        self.intRectX = intX
        self.intRectY = intY
        self.intRectWidth = intWidth
        self.intRectHeight = intHeight

    def checkIfContourIsValid(self):                            # this is oversimplified, for a production grade program
        if self.fltArea < MIN_CONTOUR_AREA: return False        # much better validity checking would be necessary
        return True

###################################################################################################
def main():
    if brightness is True:
        allContoursWithData = []  # declare empty lists,
        validContoursWithData = []  # we will fill these shortly
        kernel = np.ones((1, 1), np.uint8)

        eroded = cv2.erode(croppedmask, kernel, iterations=1)
        MLModel = tf.keras.models.load_model('my_model.h5')

        npaContours, npaHierarchy = cv2.findContours(eroded,
                                                     # input image, make sure to use a copy since the function will modify this image in the course of finding contours
                                                     cv2.RETR_EXTERNAL,  # retrieve the outermost contours only
                                                     cv2.CHAIN_APPROX_SIMPLE)  # compress horizontal, vertical, and diagonal segments and leave only their end points

        for npaContour in npaContours:  # for each contour
            contourWithData = ContourWithData()  # instantiate a contour with data object
            contourWithData.npaContour = npaContour  # assign contour to contour with data
            contourWithData.boundingRect = cv2.boundingRect(contourWithData.npaContour)  # get the bounding rect
            contourWithData.calculateRectTopLeftPointAndWidthAndHeight()  # get bounding rect info
            contourWithData.fltArea = cv2.contourArea(contourWithData.npaContour)  # calculate the contour area
            allContoursWithData.append(contourWithData)  # add contour with data object to list of all contours with data
        # end for

        for contourWithData in allContoursWithData:  # for all contours
            if contourWithData.checkIfContourIsValid():  # check if valid
                validContoursWithData.append(contourWithData)  # if so, append to valid contour list
            # end if
        # end for

        validContoursWithData.sort(key= operator.attrgetter("intRectX"))  # sort contours from left to right

        strFinalString = "Bus Number: "  # declare final string, this will have the final number sequence by the end of the program

        rectangle = eroded

        for contourWithData in validContoursWithData:  # for each contour
            # draw a green rect around the current char

            cv2.rectangle(rectangle,  # draw rectangle on original testing image
                          (contourWithData.intRectX, contourWithData.intRectY),  # upper left corner
                          (contourWithData.intRectX + contourWithData.intRectWidth,
                           contourWithData.intRectY + contourWithData.intRectHeight),  # lower right corner
                          (0, 255, 0),  # green
                          1)  # thickness
            space = 0
            imgROI = eroded[ contourWithData.intRectY - space : contourWithData.intRectY + contourWithData.intRectHeight + space ,
                     contourWithData.intRectX -space: contourWithData.intRectX + contourWithData.intRectWidth +space ]

            imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH,
                                                RESIZED_IMAGE_HEIGHT))  # resize image, this will be more consistent for recognition and storage

            npaROIResized = imgROIResized.reshape(
                (1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))  # flatten image into 1d numpy array

            npaROIResized = np.float32(npaROIResized)  # convert from 1d numpy array of ints to 1d numpy array of floats

            predict = MLModel.predict(npaROIResized.reshape(1, RESIZED_IMAGE_WIDTH,RESIZED_IMAGE_HEIGHT ))

            print(predict)
            strCurrentChar = str(np.argmax(predict))  # get character from results
            strFinalString = strFinalString + strCurrentChar  # Append characters

        # end for
        print("Full time of the system: --- %.2f seconds ---" % (time.perf_counter() - start_time))
        print("\n" + strFinalString + "\n")  # show the full string

        # Audio Feedback to the user
        #myobj = gTTS(text=strFinalString, lang=language, slow=False)
        #myobj.save("busnumber.wav")
        #os.system("busnumber.wav")
        cv2.imshow('rectimage', rectimage)
        cv2.imshow('image', img)
        toshow = cv2.resize(eroded, (200, 200))
        cv2.imshow("eroded", toshow)
        print(orig.shape)

        cv2.waitKey(0)  # wait for user key press
        cv2.destroyAllWindows()  # remove windows from memory

        return

###################################################################################################
if __name__ == "__main__":
    main()
# end if





