import cv2
import numpy as np

#Draws keyPoints on image (img) at locations (keypoints)
def drawKeyPoints(img,keypoints):
    for marker in keypoints:
        img = cv2.drawMarker(img, tuple(int(i) for i in marker.pt), color=(244, 56, 244), markerSize = 50,thickness = 5)
    return img

#returns red mask of image
def redMask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, (0,100,100), (10,255,255))
    mask2 = cv2.inRange(hsv, (160,100,100), (180,255,255))
    mask = mask1 + mask2
    return mask

#returns green mask of image
def greenMask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (36,0,0), (86,255,255))
    return mask

#returns blue mask of image
def blueMask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (110,0,0), (130,255,255))
    return mask

#returns black mask of image
def blackMask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0,0,0), (180,255,50))
    return mask

#returns set of keypoints on image given a mask. Based on blob detection, so one can set the blob to be bigger to get rid of just pixels of a certain color
def blobDetectMask(img, mask):
    params=cv2.SimpleBlobDetector_Params()
    params.minArea=1
    params.filterByArea = True
    detector=cv2.SimpleBlobDetector_create(params)
    keypoints=detector.detect(255-mask)
    return keypoints


#MAIN
imgName = "img2.JPG"
imgProc = cv2.imread(imgName)

masks = []

masks.append(redMask(imgProc))
#masks.append(greenMask(imgProc))
masks.append(blueMask(imgProc))
#masks.append(blackMask(imgProc))

for mask in masks:
    kp = blobDetectMask(imgProc, mask) #KeyPoint for mask
    imgProc = drawKeyPoints(imgProc, kp);

scale = 0.15
x = int(imgProc.shape[:2][0] * scale)
y = int(imgProc.shape[:2][1] * scale)
resizeImgProc = cv2.resize(imgProc,(y,x))

img = cv2.imread(imgName)
img = cv2.resize(img,(y,x))

while True:
    bothImgTogether = np.hstack((img, resizeImgProc))
    cv2.imshow("IMGS",bothImgTogether)
    if cv2.waitKey(1) & 0xFF == ord('q'):   #if you hit 'q' it will exit
            cv2.destroyAllWindows()
            break
