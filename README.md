# ofxNaturalFeatureTracker
A natural features image tracker, detector and Augmented Reality engine. An alternatve to the proprietary AR engine from Qualcomm used in ofxQCAR.

## Building and Using

Import the ImageTrackerLib.h and ImageTrackerLib.mm files to your oF project.

See the example in ofApp.mm to understand how the tracker works.
The API is rather similar to ofxQCAR.

Remember to also add (drag) opencv2.framework to your project. It is compiled only for armv7 iOS devices, and is version 2.4.3 (version 2.4.10 did not compile properly for iOS).
OpenCV can be compiled for all iOS version including the simulator by downloading the desired 2.4.X from the repo (for example https://github.com/Itseez/opencv/tree/2.4.3) and running the iOS framework build script in ‘ios/build_framework.py’.