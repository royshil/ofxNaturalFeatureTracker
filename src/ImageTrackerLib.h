/*
 * ImageTrackerLib.h
 *
 *  Created on: Feb 17, 2015
 *      Author: roy_shilkrot
 *
 *  The MIT License (MIT)
 *
 *  Copyright (c) <year> <copyright holders>
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in
 *  all copies or substantial portions of the Software.
 
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *  THE SOFTWARE. *
 */

#ifndef IMAGETRACKERLIB_H_
#define IMAGETRACKERLIB_H_

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <vector>
#include <ofThread.h>
#include <ofVideoGrabber.h>

namespace ImageTrackerLib {

using namespace cv;
using namespace std;

class Tracker : public ofThread  {
private:
    Ptr<FeatureDetector>        detector;
    Ptr<DescriptorExtractor>    extractor;
    Ptr<DescriptorMatcher>      matcher;
    Mat                         marker_frame, marker_desc;
    vector<KeyPoint>            marker_kp;
    vector<Point2f>             obj_bb;
    bool                        bootstrap;
    Mat_<float>                 camMat;

    vector<KeyPoint>            trackedFeatures;
    vector<int>                 trackedFeaturesOnMarker;
    Mat                         prevGray;
    Mat                         toProcessFrame;
    Mat                         raux,taux;
    Mat                         homography;
    Mat                         cvToGl;
    
    bool                        tracking;
    bool                        debug;
    bool                        newFrame;
    
    virtual void                threadedFunction();
public:
    Mat                         outputFrame;
    Mat                         hmask;
    Mat_<float>                 modelViewMatrix;

    Tracker(Mat_<float> cam, Ptr<FeatureDetector>, Ptr<DescriptorExtractor>);
    void update();
    int setMarker(const Mat& marker);
    Mat getMarkerMask();
    void bootstrapTracking(const Mat& frame, const Mat& useHomography = Mat(), const Mat& mask = Mat());
    void track(const Mat& frame);
    Mat process(const Mat& frame, const Mat& mask = Mat());
    void calcModelViewMatrix(Mat_<float>& modelview_matrix, Mat_<float>& camMat);

    const Ptr<Feature2D>& getDetector() const { return detector; }
    const vector<KeyPoint>& getTrackedFeatures() const { return trackedFeatures; }
    bool isTracking() const { return tracking || bootstrap; }
    bool canCalcModelViewMatrix() const;
    void setDebug(bool b) { debug = b; }
    void setToProcessFrame(const Mat& f) { lock(); toProcessFrame = f; unlock(); newFrame = true; }
    Mat_<float> getModelViewMatrix() {
        if(modelViewMatrix.empty()) return Mat_<float>::eye(4,4);
        Mat_<float> tmp; lock(); modelViewMatrix.copyTo(tmp); unlock(); return tmp; }

    void reset() {
        trackedFeatures.clear();
        trackedFeaturesOnMarker.clear();
        tracking = true;
        bootstrap = true;
        raux.release();
        taux.release();
    }
};

class MarkerDetector {
    Ptr<BOWKMeansTrainer>            bowtrainer;
    Ptr<BOWImgDescriptorExtractor>   bowextractor;
    Ptr<DescriptorMatcher>           matcher;
    Ptr<FeatureDetector>             detector;
    Ptr<DescriptorExtractor>         extractor;
    Mat                              vocabulary;
    vector<Mat>                      markers;
    vector<string>                   marker_files;
    PCA                              descriptorPCA;
    Mat                              descriptorsBeforePCA;
    Mat                              descriptorsAfterPCA;
    CvKNearest                       classifier;
    Mat                              training;
    vector<string>                   training_labels;
    vector<string>                   training_labelsUniq;

public:
    MarkerDetector();
    void readFromFiles();
    void saveToFiles();
    void addMarker(const string& marker_file);
    void addMarker(const Mat& marker, const string& marker_file);
    void cluster();
    void extractBOWdescriptor(const Mat& img, Mat& imgDescriptor, const Mat& mask = Mat());
    void addImageToTraining(const Mat& img,const string& label);
    string detectMarkerInImage(const Mat& img, const Mat& mask = Mat());
    Mat getMarker(const string& label);
    void setVocabulary(const Mat& vocabulary);

    const Mat& getVocabulary() const {return vocabulary;}
    const Mat& getTraining() const { return training; }
    void setTraining(const Mat& training) { this->training = training;  }
    vector<string> getTrainingLabels() const { return training_labels;  }
    void setTrainingLabels(vector<string> trainingLabels) { training_labels = trainingLabels; }
    const PCA& getDescriptorPca() const {  return descriptorPCA; }
    void setDescriptorPca(const PCA& descriptorPca) { descriptorPCA = descriptorPca;  }
    vector<string> getMarkerFiles() const {  return marker_files; }
    void setMarkerFiles(vector<string> markerFiles) {  marker_files = markerFiles; }
};

    
class ImageTracker : public ofThread {
    std::vector<ofPtr<Tracker> >    trackers;
    MarkerDetector                  markerDetector;
    ofVideoGrabber*                 grabber;
    Mat                             toProcessFrame;
    bool                            debug;
    Mat_<float>                     camMat;
    ofTexture                       tex;
    Ptr<FeatureDetector>            detector;
    Ptr<DescriptorExtractor>        extractor;
    
    virtual void                    threadedFunction();
public:
    Mat_<float>                     persp;

    ImageTracker(ofVideoGrabber* g);
    virtual ~ImageTracker();
    
    void setup();
    void update();
    void setDebug(bool b) { debug = b; }

    void draw(int w, int h) {
        tex.loadData(toProcessFrame.data, toProcessFrame.cols, toProcessFrame.rows, GL_RGB);
        tex.draw(0, 0, w, h);
    }

    const vector<ofPtr<Tracker> >& getTrackers() const { return trackers; }
};

} /* namespace ImageTrackerLib */

#endif /* IMAGETRACKERLIB_H_ */
