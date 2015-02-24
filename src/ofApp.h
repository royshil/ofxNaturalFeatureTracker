#pragma once

#include "ofMain.h"
#include "ofxiOS.h"
#include "ofxiOSExtras.h"

#include <opencv2/opencv.hpp>
#import <opencv2/highgui/cap_ios.h>
using namespace cv;


#include "ImageTrackerLib.h"

using namespace ImageTrackerLib;

class ofApp : public ofxiOSApp{
    ofPtr<ImageTracker> trackers;
    
    ofVideoGrabber grabber;
    
    public:
        void setup();
        void update();
        void draw();
        void exit();
	
        void touchDown(ofTouchEventArgs & touch);
        void touchMoved(ofTouchEventArgs & touch);
        void touchUp(ofTouchEventArgs & touch);
        void touchDoubleTap(ofTouchEventArgs & touch);
        void touchCancelled(ofTouchEventArgs & touch);

        void lostFocus();
        void gotFocus();
        void gotMemoryWarning();
        void deviceOrientationChanged(int newOrientation);

};


