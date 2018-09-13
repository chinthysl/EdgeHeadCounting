# USAGE
# python3 ncs_video_count_tracking.py --video TestVideos/room2.avi --graph Graphs/graphycroom30000 --confidence 0.5 --display 1 --save 1

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

# ************************************* prediction logic ****************************************
# initialize the list of class labels our network was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ("background", "person")
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# frame dimensions should be sqaure
PREPROCESS_DIMS = (300, 300)
DISPLAY_DIMS = (900, 900)
# calculate the multiplier needed to scale the bounding boxes
DISP_MULTIPLIER = DISPLAY_DIMS[0] // PREPROCESS_DIMS[0]


def preprocess_image(input_image):
    # preprocess the image
    preprocessed = cv2.resize(input_image, PREPROCESS_DIMS)
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
ap.add_argument("-v", "--video", required=True, help="the detection video")
ap.add_argument("-g", "--graph", required=True, help="path to input graph file")
ap.add_argument("-c", "--confidence", default=.5, help="confidence threshold")
ap.add_argument("-d", "--display", type=int, default=0, help="switch to display image on screen")
ap.add_argument("-s", "--save", type=int, default=0, help="save the detection video")
args = vars(ap.parse_args())

# grab a list of all NCS devices plugged in to USB
print("[INFO] finding NCS devices...")
devices = mvnc.EnumerateDevices()

# if no devices found, exit the script
if len(devices) == 0:
    print("[INFO] No devices found. Please plug in a NCS")
    quit()

# use the first device since this is a simple test script
# (you'll want to modify this is using multiple NCS devices)
print("[INFO] found {} devices. device0 will be used. " "opening device0...".format(len(devices)))
device = mvnc.Device(devices[0])
device.OpenDevice()

# open the CNN graph file
print("[INFO] loading the graph file into RPi memory...")
with open(args["graph"], mode="rb") as f:
    graph_in_memory = f.read()

# load the graph into the NCS
print("[INFO] allocating the graph on the NCS...")
graph = device.AllocateGraph(graph_in_memory)

# open a pointer to the video stream thread and allow the buffer to
# start to fill, then start the FPS counter
print("[INFO] starting the video stream and FPS counter...")


# **********************people counting logic****************************************************
pX      = DISPLAY_DIMS[0]
pY      = DISPLAY_DIMS[1]
shift   = 120
gate    = 100

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

#************************************************************************************************

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture(args["video"])

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

ret_,firstFrame= cap.read()
firstFrame = firstFrame.copy()

i = 0
while 10*i < pX:
    cv2.line(firstFrame, (10*i, 0), (10*i, pY), (0, 255, 0), 1)
    i = i+1
# cv2.imshow("firstFrame", firstFrame)
cv2.imwrite('DataGather/firstFrame_line.jpg', firstFrame)

if args["save"] > 0:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('DetectionVideos/{}Detections_{}_{}.avi'.format(splitext(basename(args["video"]))[0], 
                           splitext(basename(args["graph"]))[0], time.strftime("%Y%m%d-%H%M%S")), fourcc, 5, DISPLAY_DIMS)


goInCount = 0
goOutCount = 0
numPeopleInside = 0

prevBoxObjects = [None] * 10
currentBoxObjects = [None] * 10

fps = FPS().start()

# loop over frames from the video file stream
while cap.isOpened():
    try:
        # grab the frame from the threaded video stream
        # make a copy of the frame and resize it for display/video purposes
        ret, frame = cap.read()
        frame = cv2.resize(frame, PREPROCESS_DIMS)
        cv2.transpose(frame, frame)
        frame = cv2.flip(frame, +1)
        image_for_result = frame.copy()
        image_for_show   = frame.copy()
        image_for_result = cv2.resize(image_for_result, DISPLAY_DIMS)
        image_for_show = cv2.resize(image_for_result, DISPLAY_DIMS)

        if args["display"] > 0 or args["save"] > 0:
            cv2.line(image_for_show, (lineOut, 0), (lineOut, pY), (255, 0, 0), 3)
            cv2.line(image_for_show, (lineIn, 0), (lineIn, pY), (0, 0, 255), 3)
            cv2.line(image_for_show, (lineMiddle, 0), (lineMiddle, pY), (0, 255, 0), 3)

        # use the NCS to acquire predictions
        predictions = predict(frame, graph)
        
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
                print("[INFO] Prediction #{}: class={}, confidence={}, "
                        "boxpoints={}".format(i, CLASSES[pred_class], pred_conf, pred_boxpts))

                # extract information from the prediction boxpoints
                (ptA, ptB) = (pred_boxpts[0], pred_boxpts[1])
                ptA = (ptA[0] * DISP_MULTIPLIER, ptA[1] * DISP_MULTIPLIER)
                ptB = (ptB[0] * DISP_MULTIPLIER, ptB[1] * DISP_MULTIPLIER)
                (startX, startY) = (ptA[0], ptA[1])
                y = startY - 15 if startY - 15 > 15 else startY + 15

                cX = int(ptA[0] + (ptB[0] - ptA[0])/2)
                cY = int(ptA[1] + (ptB[1] - ptA[1])/2)
                
                roi = image_for_result[ptA[1]:ptB[1], ptA[0]:ptB[0]]
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

        #print("before")
        #print(currentBoxObjects)
        #print(prevBoxObjects)
   
                   
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
                    print("Object:{} tracked with previous object:{} with score:{}".format(i,matchingIndex,minHistVal))
                    del prevBoxObjects[matchingIndex]
                    
                #print("after")
                #print(currentBoxObjects)
                #print(prevBoxObjects)
                            
                
        for obj in currentBoxObjects:
            if obj != None:
                if obj.hasPrevMatch == True:
                    if obj.getDirection() == 1:
                        goInCount += 1
                    elif obj.getDirection() == -1:
                        goOutCount += 1
                    
        numPeopleInside = goInCount - goOutCount
        
        #*********************************************************************************************************************************
        
        if args["display"] > 0 or args["save"] > 0:    
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
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        break

    # if there's a problem reading a frame, break gracefully
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        break


# ***********stop and clean up resources**************************************************************************************************
fps.stop()
cap.release()

if args["display"] > 0:
    cv2.destroyAllWindows()

graph.DeallocateGraph()
device.CloseDevice()

# display FPS information
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

