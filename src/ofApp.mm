#include "ofApp.h"


using namespace cv;

//--------------------------------------------------------------
void ofApp::setup(){
//    ofSetOrientation(OF_ORIENTATION_90_RIGHT);//Set iOS to Orientation Landscape Right
    
    ofSetFrameRate(60);
        
    grabber.initGrabber(ofGetWidth(), ofGetHeight(), OF_PIXELS_BGRA);
    trackers = ofPtr<ImageTracker>(new ImageTracker(&grabber));
    trackers->setup();
    trackers->setDebug(false);
    trackers->startThread();
}

//--------------------------------------------------------------
void ofApp::update(){
    trackers->update();
}

//--------------------------------------------------------------
void ofApp::draw(){
    ofDisableDepthTest();
    ofDisableLighting();
    
    ofSetHexColor(0xffffff);
//    grabber.draw(0, 0, ofGetViewportWidth(), ofGetViewportHeight());
    trackers->draw(ofGetViewportWidth(), ofGetViewportHeight());

    const vector<ofPtr<Tracker> >& trackersList = trackers->getTrackers();
    if(trackersList.size() > 0) {
        glMatrixMode( GL_PROJECTION );
        glPushMatrix();
        glLoadMatrixf((float*)trackers->persp.data);
        
        for (int i=0; i<trackersList.size(); i++) {
            if(trackersList[i]->canCalcModelViewMatrix()) {
                glMatrixMode(GL_MODELVIEW);
                glPushMatrix();
                glLoadMatrixf((float*)(trackersList[i]->getModelViewMatrix().data));
                
                ofPushMatrix();
                ofScale(0.01, 0.01);
                stringstream ss; ss <<trackersList[i]->getTrackedFeatures().size();
                ofDrawBitmapString(ss.str(), ofPoint(-10,-10));
                ofPopMatrix();

                ofSetColor(255, 0, 0);
                ofNoFill();
                
                glTranslatef(0, 0, -0.25);
                ofSetLineWidth(2.0);
                ofDrawBox(.5, .5, .5);

                glPopMatrix();
            }
        }
        
        glMatrixMode( GL_PROJECTION );
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);
    }

    ofPushMatrix();
    ofSetColor(255, 0, 0);
    ofDrawBitmapString("fps = " + ofToString((int)ofGetFrameRate()), ofGetViewportWidth() - 180, 20);
    ofPopMatrix();
    
    ofEnableDepthTest();
    ofEnableLighting();
}

//--------------------------------------------------------------
void ofApp::exit(){
}

//--------------------------------------------------------------
void ofApp::touchDown(ofTouchEventArgs & touch){

}

//--------------------------------------------------------------
void ofApp::touchMoved(ofTouchEventArgs & touch){

}

//--------------------------------------------------------------
void ofApp::touchUp(ofTouchEventArgs & touch){

}

//--------------------------------------------------------------
void ofApp::touchDoubleTap(ofTouchEventArgs & touch){

}

//--------------------------------------------------------------
void ofApp::touchCancelled(ofTouchEventArgs & touch){
    
}

//--------------------------------------------------------------
void ofApp::lostFocus(){

}

//--------------------------------------------------------------
void ofApp::gotFocus(){

}

//--------------------------------------------------------------
void ofApp::gotMemoryWarning(){

}

//--------------------------------------------------------------
void ofApp::deviceOrientationChanged(int newOrientation){

}

