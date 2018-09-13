# USAGE
# python3 ncs_realtime_objectdetection_tracking_opt.py --graph Graphs/graphycroom30000 --confidence 0.5 --display 1 --save 1 --fps 5

# import the necessary packages
from itertools import count

from mvnc import mvncapi as mvnc
from imutils.video import FPS
import argparse
import numpy as np
import time
import cv2
from os.path import basename
from os.path import splitext
import sys, os
from picamera import PiCamera
from picamera.array import PiRGBArray
import logging

logging.basicConfig(filename='fps6_5.log', level=logging.INFO)
#logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
fsock = open('fps6_5.log', 'a')
sys.stderr = fsock

# ************************************* prediction logic ****************************************
# initialize the list of class labels our network was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ("background", "person")
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# frame dimensions should be sqaure
PREPROCESS_DIMS = (300, 300)
DISPLAY_DIMS = (300, 300)
CAM_RESOLUTION = (300, 300)
# calculate the multiplier needed to scale the bounding boxes
DISP_MULTIPLIER = DISPLAY_DIMS[0] // PREPROCESS_DIMS[0]


def preprocess_image(input_image):
    # preprocess the image
    #preprocessed = cv2.resize(input_image, PREPROCESS_DIMS)
    preprocessed = input_image

    preprocessed = preprocessed - 127.5
    preprocessed = preprocessed * 0.007843
    preprocessed = preprocessed.astype(np.float16)

    # return the image to the calling function
    return preprocessed


def predict(image, graph):
    # preprocess the image
    image = preprocess_image(image)

    # send the image to the NCS and run a forward pass to grab the
    # network predictions
    graph.LoadTensor(image, None)
    (output, _) = graph.GetResult()

    # grab the number of valid object predictions from the output,
    # then initialize the list of predictions
    num_valid_boxes = output[0]
    predictions = []

    # loop over results
    for box_index in range(int(num_valid_boxes)):
        # calculate the base index into our array so we can extract
        # bounding box information
        base_index = 7 + box_index * 7

        # boxes with non-finite (inf, nan, etc) numbers must be ignored
        if (not np.isfinite(output[base_index]) or
            not np.isfinite(output[base_index + 1]) or
            not np.isfinite(output[base_index + 2]) or
            not np.isfinite(output[base_index + 3]) or
            not np.isfinite(output[base_index + 4]) or
            not np.isfinite(output[base_index + 5]) or
            not np.isfinite(output[base_index + 6])):
            continue

        # extract the image width and height and clip the boxes to the
        # image size in case network returns boxes outside of the image
        # boundaries
        (h, w) = image.shape[:2]
        x1 = max(0, int(output[base_index + 3] * w))
        y1 = max(0, int(output[base_index + 4] * h))
        x2 = min(w, int(output[base_index + 5] * w))
        y2 = min(h, int(output[base_index + 6] * h))

        # grab the prediction class label, confidence (i.e., probability),
        # and bounding box (x, y)-coordinates
        pred_class = int(output[base_index + 1])
        pred_conf = output[base_index + 2]
        pred_boxpts = ((x1, y1), (x2, y2))

        # create prediciton tuple and append the prediction to the
        # predictions list
        prediction = (pred_class, pred_conf, pred_boxpts)
        predictions.append(prediction)

    # return the list of predictions to the calling function
    return predictions


# *****************************main program******************************************************
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-g", "--graph", required=True, help="path to input graph file")
ap.add_argument("-c", "--confidence", default=.5, help="confidence threshold")
ap.add_argument("-d", "--display", type=int, default=0, help="switch to display image on screen")
ap.add_argument("-s", "--save", type=int, default=0, help="save the detection video")
ap.add_argument("-f", "--fps", type=float, default=5.0, help="fps of cam")
args = vars(ap.parse_args())

# grab a list of all NCS devices plugged in to USB
logger.info("finding NCS devices...")
devices = mvnc.EnumerateDevices()

# if no devices found, exit the script
if len(devices) == 0:
    logger.info("No devices found. Please plug in a NCS")
    quit()

# use the first device since this is a simple test script
# (you'll want to modify this is using multiple NCS devices)
logger.info("found {} devices. device0 will be used. " "opening device0...".format(len(devices)))
device = mvnc.Device(devices[0])
device.OpenDevice()

# open the CNN graph file
logger.info("loading the graph file into RPi memory...")
with open(args["graph"], mode="rb") as f:
    graph_in_memory = f.read()

# load the graph into the NCS
logger.info("allocating the graph on the NCS...")
graph = device.AllocateGraph(graph_in_memory)

# open a pointer to the video stream thread and allow the buffer to
# start to fill, then start the FPS counter
logger.info("starting the video stream and FPS counter...")


# **********************people counting logic****************************************************
pX      = DISPLAY_DIMS[0]
pY      = DISPLAY_DIMS[1]
shift   = (120 * DISPLAY_DIMS[0]) // 900 
gate    = (100 * DISPLAY_DIMS[0]) // 900

cY = 0
cX = 0

lineOut = int(pX / 2 + shift - gate )
lineIn = int(pX / 2 + shift + gate)
lineMiddle = int(pX / 2 + shift)

# *********************Box point object**********************************************************
class BoxObject:

    def __init__(self, personId, boxPts, centerNow, roi, hist):
        self.personIdSelf    = personId
        self.boxPtsSelf      = boxPts
        self.centerNowSelf   = centerNow
        self.centerPrevSelf  = (0,0)
        self.roiSelf         = roi
        self.hasPrevMatch    = False
        self.histSelf        = hist
        self.direction       = 0
        
    def isInsideLines(self):
        if abs(self.centerNowSelf[0] - lineMiddle) < gate:
            return True
            
    def getDirection(self):
        if (self.centerNowSelf[0] >= lineMiddle) and (self.centerPrevSelf[0] < lineMiddle):
            return 1
        elif (self.centerNowSelf[0] < lineMiddle) and (self.centerPrevSelf[0] >= lineMiddle):
            return -1
        else:
            return 0
            
    def getLocation(self):
        if (self.centerNowSelf[0] >= lineMiddle):
            return 1
        else:
            return -1

#*********************************************************************************************************
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera(resolution=CAM_RESOLUTION, framerate=args["fps"])

camera.rotation = 180
time.sleep(2)

'''
# Set ISO to the desired value
camera.iso = 100
# Wait for the automatic gain control to settle
# Now fix the values
camera.shutter_speed = camera.exposure_speed
camera.exposure_mode = 'off'
g = camera.awb_gains
camera.awb_mode = 'off'
camera.awb_gains = g
# Turn the camera's LED off
camera.led = False
'''

rawCapture = PiRGBArray(camera, size=CAM_RESOLUTION)
camera.capture('firstFrame.jpg')
firstFrame = cv2.imread('firstFrame.jpg')

i = 0
while 10*i < CAM_RESOLUTION[0]:
    cv2.line(firstFrame, (10*i, 0), (10*i, CAM_RESOLUTION[1]), (0, 255, 0), 1)
    i = i+1
# cv2.imshow("firstFrame", firstFrame)
cv2.imwrite('DataGather/firstFrame_line.jpg', firstFrame)

if args["save"] > 0:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('DetectionVideos/RealtimeDetections_{}_{}.avi'.format(splitext(basename(args["graph"]))[0],
                         time.strftime("%Y%m%d-%H%M%S")), fourcc, 5, DISPLAY_DIMS)

#************************************************************************************************************

goInCount = 0
goOutCount = 0
numPeopleInside = 0

prevBoxObjects = [None] * 10
currentBoxObjects = [None] * 10

fps = FPS().start()
ticks1 = time.time()
logger.info(time.asctime( time.localtime(ticks1)))

try:
    # loop over frames from the video file stream
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        # grab the raw NumPy array representing the image, then initialize the timestamp
        # and occupied/unoccupied text
        frame = frame.array
        #cv2.transpose(frame, frame)
        #frame = cv2.flip(frame, +1)
        #logger.info(frame.shape)
        
        # use the NCS to acquire predictions, after this frame. 
        predictions = predict(frame, graph)
        
        #image_for_show = cv2.resize(frame, DISPLAY_DIMS)
        image_for_show = frame
        #logger.info(frame.shape)

        #clear the list of BoxObject instances for this frame
        prevBoxObjects = currentBoxObjects.copy()
        currentBoxObjects = [None] * 10

        pplInsideImage = 0
        pplInsideLines = 0

        # loop over our predictions
        for (i, pred) in enumerate(predictions):
            # extract prediction data for readability
            (pred_class, pred_conf, pred_boxpts) = pred

            # filter out weak detections by ensuring the `confidence`
            # is greater than the minimum confidence
            if pred_conf > float(args["confidence"]):
                # print prediction to terminal
                logger.debug("Prediction #{}: class={}, confidence={}, "
                        "boxpoints={}".format(i, CLASSES[pred_class], pred_conf, pred_boxpts))

                # extract information from the prediction boxpoints
                (ptA, ptB) = (pred_boxpts[0], pred_boxpts[1])
                ptA = (ptA[0] * DISP_MULTIPLIER, ptA[1] * DISP_MULTIPLIER)
                ptB = (ptB[0] * DISP_MULTIPLIER, ptB[1] * DISP_MULTIPLIER)
                (startX, startY) = (ptA[0], ptA[1])
                y = startY - 15 if startY - 15 > 15 else startY + 15

                cX = int(ptA[0] + (ptB[0] - ptA[0])/2)
                cY = int(ptA[1] + (ptB[1] - ptA[1])/2)
                
                roi = image_for_show[ptA[1]:ptB[1], ptA[0]:ptB[0]]
                hist = cv2.calcHist([roi], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                #cv2.imshow("roi", roi)
                currentObject = BoxObject(0, pred_boxpts, (cX,cY), roi, hist)
                
                if currentObject.isInsideLines():
                    currentBoxObjects[pplInsideLines] = currentObject
                    pplInsideLines = pplInsideLines + 1     

                pplInsideImage = pplInsideImage + 1
                
                if args["display"] > 0 or args["save"] > 0:                
                    # display the rectangle and label text
                    # build a label consisting of the predicted class and associated probability
                    label = "{}: {:.2f}%".format(CLASSES[pred_class], pred_conf * 100)
                    cv2.rectangle(image_for_show, ptA, ptB, COLORS[pred_class], 2)
                    cv2.putText(image_for_show, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS[pred_class], 3)
                    
                    # draw the contour and center of the detected object
                    cv2.circle(image_for_show, (cX, cY), 5, (255, 255, 255), -1)
                
                    cv2.putText(image_for_show, "cX:"+str(cX), (10, image_for_show.shape[0] - 50),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.putText(image_for_show, "cY:"+str(cY), (10, image_for_show.shape[0] - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        #logger.debug("before")
        #logger.debug(currentBoxObjects)
        #logger.debug(prevBoxObjects)
   
                   
        # ***************************************************** histogram traking model **************************************************
        matchingIndex = 0;
        for (i,obj) in enumerate(currentBoxObjects):
            if obj != None:
                matchFound = False
                minHistVal = 1.0;
                for (j,objCompare) in enumerate(prevBoxObjects):
                    if objCompare != None:
                        compareVal = cv2.compareHist(obj.histSelf, objCompare.histSelf, cv2.HISTCMP_BHATTACHARYYA)
                        if compareVal < minHistVal:
                            minHistVal = compareVal
                            matchingIndex = j
                            matchFound = True
                if minHistVal < 0.7 and matchFound == True:
                    obj.hasPrevMatch = True        
                    obj.centerPrevSelf = prevBoxObjects[matchingIndex].centerNowSelf
                    logger.debug("Object:{} tracked with previous object:{} with score:{}".format(i,matchingIndex,minHistVal))
                    del prevBoxObjects[matchingIndex]
                    
                #logger.debug("after")
                #logger.debug(currentBoxObjects)
                #logger.debug(prevBoxObjects)
                            
                
        for obj in currentBoxObjects:
            if obj != None:
                if obj.hasPrevMatch == True:
                    if obj.getDirection() == 1:
                        goInCount += 1
                    elif obj.getDirection() == -1:
                        goOutCount += 1
                    
        numPeopleInside = goInCount - goOutCount
        
        logger.info("Go In:{} Go Out:{} People Inside:{}".format(goInCount, goOutCount, numPeopleInside))
        
        #*********************************************************************************************************************************
        rawCapture.seek(0)
        rawCapture.truncate()
        
        if args["display"] > 0 or args["save"] > 0:
            cv2.line(image_for_show, (lineOut, 0), (lineOut, pY), (255, 0, 0), 3)
            cv2.line(image_for_show, (lineIn, 0), (lineIn, pY), (0, 0, 255), 3)
            cv2.line(image_for_show, (lineMiddle, 0), (lineMiddle, pY), (0, 255, 0), 3)        
            
            cv2.putText(image_for_show, "PeopleInside:"+str(numPeopleInside), (pX-150, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(image_for_show, "goInCount:"+str(goInCount), (pX-150, 70),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(image_for_show, "goOutCount:"+str(goOutCount), (pX-150, 90),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(image_for_show, "PPL"+str(pplInsideImage), (30, 60),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        if args["display"] > 0:
            # can achieve faster FPS if you do not output to the screen display the frame to the screen
            cv2.imshow("Output", image_for_show)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
                
        if args["save"] > 0:
            out.write(image_for_show)            

        # update the FPS counter
        fps.update()
        
# if "ctrl+c" is pressed in the terminal, break from the loop
except KeyboardInterrupt as k:
    logger.warning("KeyboardInterrupt Occurred")

# if there's a problem reading a frame, break gracefully
except Exception as e:
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    logger.warning(exc_type, fname, exc_tb.tb_lineno)


# ***********stop and clean up resources**************************************************************************************************
fps.stop()
ticks2 = time.time()
logger.info(time.asctime( time.localtime(ticks2)))

if args["display"] > 0:
    cv2.destroyAllWindows()

graph.DeallocateGraph()
device.CloseDevice()

# display FPS information
logger.info("elapsed time: {:.3f}".format(fps.elapsed()))
logger.info("approx. FPS: {:.3f}".format(fps.fps()))

logger.info("Go In Count Final: {}".format(goInCount))
logger.info("Go Out Count Final: {}".format(goOutCount))
logger.info("People Inside Final: {}".format(numPeopleInside))

