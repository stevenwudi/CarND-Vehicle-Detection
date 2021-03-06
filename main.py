#!/usr/bin/python
import argparse
import os
import re
import sys
import cv2

from lib.diagManager import DiagManager
from lib.roadManager import RoadManager
from lib.cameraCal import CameraCal
from matplotlib import pyplot as plt


# process_road_image handles rendering a single image through the pipeline
def process_road_image(img, roadMgr, diagMgr, scrType=0):
    # Run the functions
    roadMgr.findLanes(img)
    roadMgr.findVehicles()

    # debug/diagnostics requested
    if False:
    #if debug:
        # offset for text rendering overlay
        offset = 0
        color = (192, 192, 0)
        # default - full diagnostics
        if scrType & 5 == 5:
            diagScreen = diagMgr.projectionHD()
            offset = 30
        if scrType & 4 == 4:
            diagScreen = diagMgr.projectionHD()
            offset = 30
        elif scrType & 3 == 3:
            diagScreen = diagMgr.fullDiag()
            offset = 30
        elif scrType & 3 == 2:
            diagScreen = diagMgr.projectionDiag()
            offset = 30
        elif scrType & 3 == 1:
            diagScreen = diagMgr.filterDiag()
            offset = 30
            color = (192, 192, 192)
        if scrType & 8 == 8:
            diagScreen = diagMgr.textOverlay(diagScreen, offset=offset, color=color)
        result = diagScreen
    else:
        # if scrType & 8 == 8:
        #     roadMgr.drawLaneStats()
        result = roadMgr.final
        #result = roadMgr.projMgr.curImgRoad
    return result


# process_image handles rendering a single image through the pipeline
# within the moviepy video rendering context
def process_image(image):
    result = process_road_image(image, roadMgr, diagMgr)
    current_frame = (int(roadMgr.curFrame)+startFrame)
    im_name = './test_videos_out/images/' + '%06d.png' % current_frame
    cv2.imwrite(im_name, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    return result


# our main CLI code.  use --help to get full options list
if __name__ == "__main__":

    global roadMgr
    global diagMgr
    global debug
    global scrType
    global startFrame

    # initialize argparse to parse the CLI
    usage = 'python %(prog)s [options] infilename outfilename'
    desc = 'DiWu\'s Udacity SDC Project 5: Vehicle Detection and Tracking'
    diagHelp = 'display diagnostics: [0=off], 1=filter, 2=proj 3=full '
    diagHelp += '4=projHD,complete 5=projHD,sentinal'
    defaultInput = 'project_video.mp4'
    inputHelp = 'input image or video file to process'
    defaultOutput = 'project_video_out.mp4'
    outputHelp = 'output image or video file'

    # set default - final/no diagnostics
    parser = argparse.ArgumentParser(prog='main.py', usage=usage, description=desc)
    parser.add_argument('--scrType', type=int, default=0, help=diagHelp)
    parser.add_argument('--notext', action='store_true', default=False, help='do not render text overlay')
    #parser.add_argument('--infilename', type=str, default='./test_images/test6.jpg', help=inputHelp)
    parser.add_argument('--infilename', type=str, default='./test_videos/project_video.mp4', help=inputHelp)

    args = parser.parse_args()

    file_dir = '/'.join(args.infilename.split('/')[:-1])
    out_file_dir = file_dir + '_out'
    if not os.path.exists(out_file_dir):
        os.mkdir(out_file_dir)
    args.outfilename = os.path.join(out_file_dir, args.infilename.split('/')[-1])

    debug = True

    videopattern = re.compile("^.+\.mp4$")
    imagepattern = re.compile("^.+\.(jpg|jpeg|JPG|png|PNG)$")
    image = None
    videoin = None

    # set up pipeline processing options
    pleaseCheck = "Please check and try again."
    pleaseRemove = "Please remove and try again."
    invalidExt = "Invalid %s filename extension for output.  %s"
    validImageExt = "Must end with one of [jpg,jpeg,JPG,png,PNG]"
    validVideoExt = "Must end with '.mp4'"

    # if video - set up in/out videos
    if videopattern.match(args.infilename):
        if videopattern.match(args.outfilename):
            if not os.path.exists(args.infilename):
                print("Video input file: %s does not exist. %s" % (args.infilename, pleaseCheck))
                sys.exit(1)
            else:
                videoin = args.infilename
                videoout = args.outfilename
                valid = True
        else:
            print(invalidExt % ("video", validVideoExt))
            sys.exit(3)

    # if image - set up image processing options
    elif imagepattern.match(args.infilename):
        if imagepattern.match(args.outfilename):
            if not os.path.exists(args.infilename):
                print("Image input file: %s does not exist. %s" % (args.infilename, pleaseCheck))
                sys.exit(4)
            else:
                image = cv2.cvtColor(cv2.imread(args.infilename), cv2.COLOR_BGR2RGB)
                valid = True
        else:
            print(invalidExt % ("image", validImageExt))
            sys.exit(6)

    # set up diagnostic pipeline options if requested
    if valid:
        # initialization
        # load or perform camera calibrations
        camCal = CameraCal('camera_cal', 'camera_cal/calibrationdata.p')

        # override camCal image size
        if image is not None:
            camCal.setImageSize(image.shape)

        # initialize road manager and its managed pipeline components/modules
        roadMgr = RoadManager(camCal)

        # initialize diag manager and its managed diagnostics components
        diagMgr = DiagManager(roadMgr)

        # Image only?
        if image is not None:
            print("image processing %s..." % args.infilename)
            imageout = process_image(image)
            cv2.imwrite(args.outfilename, cv2.cvtColor(imageout, cv2.COLOR_RGB2BGR))
            print("done image processing %s..." % args.infilename)

        # Full video pipeline
        elif videoin is not None and videoout is not None:
            print("video processing %s..." % videoin)
            cap = cv2.VideoCapture(videoin)
            # Set the start frame number,
            # see: https://stackoverflow.com/questions/11420748/setting-camera-parameters-in-opencv-python
            startFrame = 200
            cap.set(1, startFrame)
            while cap.isOpened():
                ret, frame = cap.read()
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                process_image(image)
            cap.release()
            cv2.destroyAllWindows()
            print("done video processing %s..." % videoin)
    else:
        print("error detected.  exiting.")
