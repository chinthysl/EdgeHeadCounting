# EgdeHeadCounting

This repo contains the source code to run the head counting program on a rasberry pi with Intels NCS setup.
You can find the current detection and object counting state of the project looking at the two videos provided.

Usage is as follows:

If your Pi doesn't have OpenCV and Movidius API installed

1) Install OpenCV-3.4 to PI as described in https://www.pyimagesearch.com/2015/10/26/how-to-install-opencv-3-on-raspbian-jessie/
2) Install Intels Movidius API-only mode - Install only the APi framework on Pi as described in https://movidius.github.io/blog/ncs-apps-on-rpi/

After that,

1) git clone https://github.com/chinthysl/EgdeHeadCounting.git
2) run python3 ncs_video_count.py --video TestVideos/VIDEOFILE --graph graphs/GRAPH --display DISPLAY --confidence CONFIDENTNUM --save SAVEVIDEO
    *VIDEOFILE = video file you want to do the detection inside TestVideos location
    *GRAPH = graph file you compiled from mvNCCompiler, exists examples in graphs path
    *DISPLAY = 0 if not display detection relatime, 1 to display, will reduce FPS (default = 0.5)
    *CONFIDENTNUM = 0.0 - 1.0 (default = 0.5)
    *SAVEVIDEO = 0 or 1 , if you want to save the detection results to a videofile
