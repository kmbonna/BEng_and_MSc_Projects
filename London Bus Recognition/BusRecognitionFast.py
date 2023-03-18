import cv2
import numpy as np
import time
start_time = time.perf_counter()

# this scale method was only created for easier prototyping on my laptop
def scale(a):
    if a.shape[0] > 650 or a.shape[1] > 1500:
        if a.shape[1] < 2 * a.shape[0]:
            scale_percent = 650 / a.shape[0]  # percent of original size
            width = int(a.shape[1] * scale_percent)
            height = int(a.shape[0] * scale_percent)
            dim = (width, height)
            # resize image
            a = cv2.resize(a, dim, interpolation=cv2.INTER_AREA)
        else:
            scale_percent = 1500 / a.shape[1]
            width = int(a.shape[1] * scale_percent)
            height = int(a.shape[0] * scale_percent)
            dim = (width, height)
            # resize image
            a = cv2.resize(a, dim, interpolation=cv2.INTER_AREA)
    return a


raw = cv2.imread('bus4.jpg')

orig = scale(raw)

img = scale(raw)
brightness = False
gamma = 1


while brightness is False:
    # Convert BGR to HSV  (hue, saturation, value)
    # this for the final image to get mask of the original image and not the gamma corrected one
    forfinal = cv2.cvtColor(orig, cv2.COLOR_BGR2HSV)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # define range of red color in HSV
    lower_red = np.array([0, 50, 90])
    upper_red = np.array([10, 255, 255])

    lower_red2 = np.array([170, 50, 90])
    upper_red2 = np.array([180, 255, 255])

    # define range of white color in HSV

    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 45, 255])

    # define range of yellow color in hsv

    lower_yellow = np.array([25, 30, 190])
    upper_yellow = np.array([50, 255, 255])

    # have a mask that only shows red colors in an image (grayscale output)
    red_mask = cv2.inRange(hsv, lower_red, upper_red)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    finalred_mask = cv2.bitwise_or(red_mask, red_mask2)
    
    # Bitwise-AND mask and original image ("AND"s the mask and the original image)
    redcrop = cv2.bitwise_and(img, img, mask=finalred_mask)

    # have a mask that only shows white colors in an image (grayscale output)
    white_mask = cv2.inRange(forfinal, lower_white, upper_white)

    # have a mask that only shows yellow colors in an image (grayscale output)
    yellow_mask = cv2.inRange(forfinal, lower_yellow, upper_yellow)

    # have a mask that shows both white or yellow for both bus number types
    busnumber_mask = cv2.bitwise_or(white_mask, yellow_mask)

    # Canny edge detection and adding further blur to the photo
    edges = cv2.Canny(finalred_mask, 1000, 1500)
    blurredmask = cv2.GaussianBlur(edges, (5, 5), 0)

    # Finding all contours in edged photo
    contours, hierarchy = cv2.findContours(blurredmask, 1, 2)
    # defining rectimage
    rectimage = redcrop

    # Going through all the contours and finding the rectangular one that is big enough to be the bus number strip
    for cont in contours:
        if cv2.contourArea(cont) > 5000:    #5000 pixels

            arc_len = cv2.arcLength(cont, True)
            # approxPolyDP approximates the found contours to closed shapes for easier identification
            approx = cv2.approxPolyDP(cont,  0.1*arc_len, True)

            if len(approx) == 4:
                # Draw a rectangle around the found contour (for demonstration only)
                brightness = True
                x, y, w, h = cv2.boundingRect(approx)
                rectimage = cv2.rectangle(redcrop, (x, y), (x + w, y + h), (0, 255, 0), 2)
                croppedimage = img[y:y + h, x:x + w]  # cropping the image based on the coordinates of that contour
                croppedmask = busnumber_mask[y:y + h, x:x + w]  # Puts the bus number mask back on (white or yellow)
            # the y-values are actually read from top to bottom
                if croppedmask.shape[0] > 2 * croppedmask.shape[1]:
                    print("Not a bus")
                    brightness = False
                    gamma = 0.1
                elif croppedmask.shape[0] > croppedmask.shape[1]:
                    # Old double deckers
                    loweryvalue = int(croppedmask.shape[0]/3)
                    higheryvalue = int(1.6*croppedmask.shape[0]/3)
                    croppedmask = croppedmask[loweryvalue:higheryvalue, int(croppedmask.shape[1]/1.7): x + w]
                    print("Time to recognize bus: --- %.2f seconds ---" % (time.perf_counter() - start_time))
                    cv2.imshow('rectimage', rectimage)
                elif croppedmask.shape[1] > 2*croppedmask.shape[0]:
                    # New double decker buses
                    croppedmask = busnumber_mask[y:y + h, x + int(croppedmask.shape[1]/1.7): x + w]
                    print("new bus detected")
                    print("Time to recognize bus: --- %.2f seconds ---" % (time.perf_counter() - start_time))
                    cv2.imshow('rectimage', rectimage)
                    cv2.imshow('image', img)
                else:
                    # Single Decker buses
                    higheryvalue = int(croppedmask.shape[0]/2.5)
                    croppedmask = croppedmask[0:higheryvalue, int(croppedmask.shape[1]/1.8): croppedmask.shape[1]]
                    print("Time to recognize bus: --- %.2f seconds ---" % (time.perf_counter() - start_time))
                    cv2.imshow('rectimage', rectimage)

    # the looping gamma correction code snippet so that it keeps making it brighter (further work can be made for improvement)
    if brightness is False:
        if gamma < 0.12:
            print("Image couldn't be processed")
            break
        gamma = gamma - 0.1
        lookUpTable = np.empty((1, 256), np.uint8)
        for i in range(256):
            lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
            img = cv2.LUT(orig, lookUpTable)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.waitKey()
cv2.destroyAllWindows()
print(gamma)
