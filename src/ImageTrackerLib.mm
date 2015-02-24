/*
 * ImageTrackerLib.cpp
 *
 *  Created on: Feb 17, 2015
 *      Author: roy_shilkrot
 *
 *  The MIT License (MIT)
 *
 *  Copyright (c) 2015 Roy Shilkrot and Valentin Heun
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

#include <ImageTrackerLib.h>
#include <ofLog.h>
#include <ofImage.h>

namespace ImageTrackerLib {

ImageTracker::ImageTracker(ofVideoGrabber* g):grabber(g) {
    assert(g != NULL);
    
    detector = new ORB(2000);
    extractor = new OpponentColorDescriptorExtractor(new FREAK);

}

ImageTracker::~ImageTracker() {
    // TODO Auto-generated destructor stub
}
    
void ImageTracker::update() {
    grabber->update();
    if(grabber->isFrameNew())
    {
        lock();
        if(debug) ofLog() << "new frame";
        Mat(grabber->getHeight(),grabber->getWidth(),CV_8UC3,grabber->getPixels()).copyTo(toProcessFrame);
        unlock();
    }
}
    
void ImageTracker::setup() {
    tex.allocate(grabber->getWidth(), grabber->getHeight(), GL_RGB);
    
    //    int w = ofGetWidth(), h = ofGetHeight();
    int w = grabber->getWidth(), h = grabber->getHeight();
    ofLog() << "size " << w << "x" << h;
    ofLog() << "grabber size " << grabber->getWidth() << "x"<< grabber->getHeight();
    float f = std::max(w,h); // * 1.1;
    camMat = (Mat_<float>(3,3) <<
              f,       0,       w/2,
              0,       f,       h/2,
              0,       0,       1);
    
    persp.create(4,4); persp.setTo(0);
    
    // http://kgeorge.github.io/2014/03/08/calculating-opengl-perspective-matrix-from-opencv-intrinsic-matrix/
    double fx = camMat.at<float>(0,0);
    double fy = camMat.at<float>(1,1);
    double cx = camMat.at<float>(0,2);
    double cy = camMat.at<float>(1,2);
    double near = 1, far = 1000.0;
    persp(0,0) = fx/cx;
    persp(1,1) = fy/cy;
    persp(2,2) = -(far+near)/(far-near);
    persp(2,3) = -2.0*far*near / (far-near);
    persp(3,2) = -1.0;
    
    ofLog() << "perspective m \n" << persp << endl;
    
    persp = persp.t(); //to col-major
    
    glViewport(0, 0, w, h);
    
    markerDetector.readFromFiles();

    
    ofPtr<Tracker> t(new Tracker(camMat,detector,extractor));
    trackers.push_back(t);
    
    string sFile = ofToDataPath("targetA1_480.jpg");
    ofImage img; img.loadImage(sFile);
    Mat im(img.height,img.width,CV_8UC3,img.getPixels());
//    markerDetector.addMarker(im,sFile);
    ofLog() << "marker size " << im.size() << ", found " << t->setMarker(im) << " keypoints";
    
    t->setDebug(false);
    t->startThread();

    ofPtr<Tracker> t1(new Tracker(camMat,detector,extractor));
    trackers.push_back(t1);
//
    sFile = ofToDataPath("targetA2_480.jpg");
    img.loadImage(sFile);
    im.create(img.height,img.width,CV_8UC3); im.data = img.getPixels();
//    markerDetector.addMarker(im,sFile);
    ofLog() << "marker size " << im.size() << ", found " << t1->setMarker(im) << " keypoints";
    
    t1->setDebug(false);
    t1->startThread();

//    markerDetector.cluster();
}
    
void ImageTracker::threadedFunction() {
    while (isThreadRunning()) {
        for (int i=0; i<trackers.size(); i++) {
            trackers[i]->setToProcessFrame(toProcessFrame);
        }
//        markerDetector.detectMarkerInImage(toProcessFrame);
    }
}


const double ransac_thresh = 20.0f; // RANSAC inlier threshold
const double nn_match_ratio = 0.9f; // Nearest-neighbour matching ratio
const int min_inliers = 80; // Minimal number of inliers to draw bounding box

void imshow(const std::string& name, const cv::Mat& img, double upscale) {
    cv::Mat tmp; cv::resize(img,tmp,cv::Size(),upscale,upscale);
    imshow(name,tmp);
}
Mat imread(const std::string& file, double upscale) {
    Mat im = cv::imread(file); cv::resize(im,im,cv::Size(),upscale,upscale);
    return im;
}
template<typename T>
    vector<cv::Point> Pointsi(const vector<cv::Point_<T> >& points) {
    vector<cv::Point> res;
    for(unsigned i = 0; i < points.size(); i++) {
        res.push_back(cv::Point(points[i].x,points[i].y));
    }
    return res;
}
vector<Point2f> Points(const vector<KeyPoint>& keypoints)
{
    vector<Point2f> res;
    for(unsigned i = 0; i < keypoints.size(); i++) {
        res.push_back(keypoints[i].pt);
    }
    return res;
}
void drawBoundingBox(Mat& image, const vector<Point2f>& bb, const Scalar& color = Scalar(0,0,255))
{
    for(unsigned i = 0; i < bb.size(); i++) {
        line(image, bb[i], bb[(i + 1) % bb.size()], color, 2);
    }
}

Tracker::Tracker(Mat_<float> cam, Ptr<FeatureDetector> d, Ptr<DescriptorExtractor> e):
    bootstrap(true),
    tracking(false),
    debug(false),
    newFrame(false),
    camMat(cam),
    detector(d),
    extractor(e)
{

    //flip y and z axes to match OpenGL renderer's frame
    cvToGl = Mat::zeros(4, 4, CV_64F);
    cvToGl.at<double>(0, 0) = 1.0f;
    cvToGl.at<double>(1, 1) = -1.0f;
    cvToGl.at<double>(2, 2) = -1.0f;
    cvToGl.at<double>(3, 3) = 1.0f;
    
    matcher = new BFMatcher(NORM_HAMMING);

}
    
bool Tracker::canCalcModelViewMatrix() const { return trackedFeatures.size() > min_inliers * 0.75; }

int Tracker::setMarker(const Mat& marker) {
    assert(!marker.empty() && marker.channels() == 3);
    marker.copyTo(marker_frame);

    detector->detect(marker_frame,marker_kp);
    extractor->compute(marker_frame,marker_kp,marker_desc);
    obj_bb.push_back(Point2f(0,0));
    obj_bb.push_back(Point2f(marker_frame.cols,0));
    obj_bb.push_back(Point2f(marker_frame.cols,marker_frame.rows));
    obj_bb.push_back(Point2f(0,marker_frame.rows));

    vector<Mat> descriptors; descriptors.push_back(marker_desc);
    matcher->add(descriptors);

    tracking = true;
    return marker_kp.size();
}

Mat Tracker::getMarkerMask() {
    if(!homography.empty() && trackedFeatures.size() > (min_inliers * 0.5)) {
        vector<Point2f> new_bb;
        perspectiveTransform(obj_bb, new_bb, homography);
        Mat mask(prevGray.size(),CV_8UC1,Scalar(0));
        fillConvexPoly(mask,&(Pointsi(new_bb)[0]),4,Scalar::all(255));
        return mask;
    } else {
        return Mat();
    }
}

void Tracker::bootstrapTracking(const Mat& frame, const Mat& useHomography, const Mat& mask) {
    vector<KeyPoint> kp;

    Mat desc;
    if(!useHomography.empty()) {
        //use a mask if already know area of marker
//        vector<Point2f> new_bb;
//        perspectiveTransform(obj_bb, new_bb, useHomography);
//        if(hmask.empty())
//            hmask.create(frame.rows,frame.cols,CV_8UC1);
//        hmask.setTo(0);
//        fillConvexPoly(hmask,&(Pointsi(new_bb)[0]),4,Scalar::all(255));
//        detector->detect(frame,kp,hmask);

        //use the homography to warp the frame and find features in it
        Mat warpedFrame; warpPerspective(frame,warpedFrame,useHomography,marker_frame.size(),WARP_INVERSE_MAP);
        if(!mask.empty())
            detector->detect(warpedFrame,kp,mask);
        else
            detector->detect(warpedFrame,kp);
        extractor->compute(warpedFrame,kp,desc);
    } else {
        if(!mask.empty())
            detector->detect(frame,kp,mask);
        else
            detector->detect(frame,kp);
        extractor->compute(frame,kp,desc);
    }

    vector< vector<DMatch> > matches;
    vector<KeyPoint> matched_on_marker, matched_on_frame;
    vector<int> matched_on_marker_idx;
    matcher->knnMatch(desc, matches, 2);

    set<int> trackedFeaturesOnMarkerSet(trackedFeaturesOnMarker.begin(),trackedFeaturesOnMarker.end());

    for(unsigned i = 0; i < matches.size(); i++) {
        if(matches[i][0].distance < nn_match_ratio * matches[i][1].distance) {
            if(trackedFeaturesOnMarkerSet.find(matches[i][0].trainIdx) == trackedFeaturesOnMarkerSet.end()) {
                matched_on_marker.push_back(marker_kp[matches[i][0].trainIdx]);
                matched_on_frame.push_back(       kp[matches[i][0].queryIdx]);
                matched_on_marker_idx.push_back(matches[i][0].trainIdx);
            }
        }
    }
    if(debug) ofLog() << kp.size() << " found " << matched_on_frame.size() << " matched new (" << trackedFeatures.size() << " old)";

    if(!useHomography.empty() && matched_on_frame.size() > 0) {
        //if we used the homography to warp the frame, we need to unwarp the keypoints
        vector<Point2f> kpPts;
        perspectiveTransform(Points(matched_on_frame),kpPts,useHomography);
        for (int i = 0; i < kpPts.size(); ++i) {
            matched_on_frame[i].pt = kpPts[i];
        }
    }

    assert(trackedFeaturesOnMarker.size() == trackedFeatures.size());
    for (int i = 0; i < trackedFeaturesOnMarker.size(); ++i) {
        matched_on_marker.push_back(marker_kp[trackedFeaturesOnMarker[i]]);
        matched_on_marker_idx.push_back(trackedFeaturesOnMarker[i]);
        matched_on_frame.push_back(trackedFeatures[i]);
    }
//    drawKeypoints(outputFrame,matched_on_frame, outputFrame, Scalar(0,0,255));

    if(matched_on_marker.size() >= min_inliers / 4) {
        if(debug) ofLog() << "try for homography with "<<matched_on_marker.size()<<" features";
        Mat inlier_mask, homography;
        homography = findHomography(Points(matched_on_marker), Points(matched_on_frame),
                                    RANSAC, ransac_thresh, inlier_mask);

        int survivors = countNonZero(inlier_mask);
        if(debug) ofLog() << survivors << " features  survived";
        vector<DMatch> inlier_matches;
        trackedFeatures.clear();
        trackedFeaturesOnMarker.clear();
        for(unsigned i = 0; i < matched_on_marker.size(); i++) {
            if(inlier_mask.at<uchar>(i)) {
                int new_i = static_cast<int>(trackedFeatures.size());
                trackedFeaturesOnMarker.push_back(matched_on_marker_idx[i]);
                trackedFeatures.push_back(matched_on_frame[i]);
                inlier_matches.push_back(DMatch(new_i, new_i, 0));
            }
        }

        if(survivors >= min_inliers/4) {
            if(debug && !homography.empty()) {
                vector<Point2f> new_bb;
                perspectiveTransform(obj_bb, new_bb, homography);
                drawBoundingBox(outputFrame, new_bb, Scalar(255,0,0));
            }

            if(bootstrap) {
                //if we were bootstrapping, disable bootstrapping
                // (we could also come here from the tracking looking for more features)
                bootstrap = false;
                cvtColor(frame, prevGray, CV_BGR2GRAY);
            }
        }
    }
}


void Tracker::track(const Mat& frame) {
    if(!prevGray.empty()) {
        vector<Point2f> corners;

        vector<uchar> status; vector<float> errors;
        Mat currGray; cvtColor(frame, currGray, CV_BGR2GRAY);
        calcOpticalFlowPyrLK(prevGray,currGray,Points(trackedFeatures),corners,status,errors,cv::Size(11,11));
//        cvtColor(currGray, outputFrame, CV_GRAY2BGR);
        currGray.copyTo(prevGray);

        if(countNonZero(status) < status.size() * 0.8) {
            if(debug) ofLog() << "tracking failed";
            reset();
            return;
        }

        trackedFeatures.clear();
        vector<int> oldFeaturesOnMarker = trackedFeaturesOnMarker;
        vector<KeyPoint> trackedFeaturesOnMarkerKP;
        trackedFeaturesOnMarker.clear();
        for (int i = 0; i < status.size(); ++i) {
            if(status[i])
            {
                trackedFeatures.push_back(KeyPoint(corners[i],1.0,0.0));
                trackedFeaturesOnMarker.push_back(oldFeaturesOnMarker[i]);
                trackedFeaturesOnMarkerKP.push_back(marker_kp[oldFeaturesOnMarker[i]]);
            }
        }
        
        Mat inlier_mask;
        if(trackedFeaturesOnMarkerKP.size() >= 4) {
            homography = findHomography(Points(trackedFeaturesOnMarkerKP),
                                        Points(trackedFeatures),
                                        0,
                                        ransac_thresh,
                                        inlier_mask);
        }

        //This will occur only if the histogram method is RANSAC or LMEDS - so usually not.
        int inliers_num = countNonZero(inlier_mask);
        if(inliers_num != trackedFeatures.size() && inliers_num >= 4 && !homography.empty()) {
            if(debug) ofLog() << "tracking homography resulted in " << inliers_num << "inliers";
            vector<DMatch> inlier_matches;
            vector<KeyPoint> oldTrackedFeatures = trackedFeatures;
            vector<KeyPoint> oldTrackedFeaturesOnMarkerKP = trackedFeaturesOnMarkerKP;
            oldFeaturesOnMarker = trackedFeaturesOnMarker;
            trackedFeatures.clear();
            trackedFeaturesOnMarker.clear();
            trackedFeaturesOnMarkerKP.clear();
            for(unsigned i = 0; i < oldTrackedFeaturesOnMarkerKP.size(); i++) {
                if(inlier_mask.at<uchar>(i)) {
                    inlier_matches.push_back(DMatch(trackedFeatures.size(), trackedFeatures.size(), 0));
                    trackedFeatures.push_back(oldTrackedFeatures[i]);
                    trackedFeaturesOnMarkerKP.push_back(oldTrackedFeaturesOnMarkerKP[i]);
                    trackedFeaturesOnMarker.push_back(oldFeaturesOnMarker[i]);
                }
            }
        }

        if(debug) {
            vector<Point2f> new_bb;
            perspectiveTransform(obj_bb, new_bb, homography);
            drawBoundingBox(outputFrame, new_bb);
        }

        if(trackedFeatures.size() < min_inliers) {
            if(debug) ofLog() << "getting more features " << (trackedFeatures.size() > (min_inliers * 0.75) ? "use homography" : "");
            bootstrapTracking(frame,trackedFeatures.size() > (min_inliers * 0.75) ? homography : Mat());
            if(debug) ofLog() << "now we have " << trackedFeatures.size() << " features";
            if(trackedFeatures.size() > min_inliers * 1.25) {
                //too many features - it's going to slow things down, so trim
                //TODO: we should keep the strong features over the weaker ones...
                trackedFeatures.resize(min_inliers * 1.25);
                trackedFeaturesOnMarker.resize(min_inliers * 1.25);
            }
        }

        if(trackedFeatures.size() < min_inliers / 4) {
            if(debug) ofLog() << "tracking failed";
            reset();
        }

    }
}

Mat Tracker::process(const Mat& frame, const Mat& mask)
{
    if(!newFrame) return Mat();
    newFrame = false;
    
    if(debug) ofLog() << "Tracker: new frame";
    
    frame.copyTo(outputFrame);
    
    if(bootstrap) {
        if(debug) ofLog() << "bootstrapping";
        bootstrapTracking(frame,Mat(),mask);
    } else {
        if(debug) ofLog() << "tracking (" << trackedFeatures.size() << " features)";
        track(frame);
    }

    if(debug) drawKeypoints(outputFrame, trackedFeatures, outputFrame, Scalar(0,0,255));
    
    return outputFrame;
}

void Tracker::calcModelViewMatrix(Mat_<float>& modelview_matrix, Mat_<float>& camMat) {
    if (trackedFeaturesOnMarker.size() < min_inliers / 4) {
        return;
    }
    vector<Point3f> ObjPoints;
    vector<Point2f> ImagePoints;
    for (int i = 0; i < trackedFeaturesOnMarker.size(); ++i) {
        Point2f p = marker_kp[trackedFeaturesOnMarker[i]].pt;
        ObjPoints.push_back(Point3f(p.x - marker_frame.cols/2,p.y - marker_frame.rows/2,0) * (1.0/marker_frame.cols));
    }

    cv::Mat Rvec,Tvec;
    cv::solvePnP(ObjPoints, Points(trackedFeatures), camMat, Mat(), raux, taux, !raux.empty());
    raux.convertTo(Rvec,CV_32F);
    taux.convertTo(Tvec ,CV_64F);

    Mat Rot(3,3,CV_32FC1);
    Rodrigues(Rvec, Rot);

    // [R | t] matrix
    Mat_<double> para = Mat_<double>::eye(4,4);
    Rot.convertTo(para(cv::Rect(0,0,3,3)),CV_64F);
    Tvec.copyTo(para(cv::Rect(3,0,1,3)));
    para = cvToGl * para;

    lock();
    Mat(para.t()).convertTo(modelview_matrix,CV_32FC1); // transpose to col-major for OpenGL
    unlock();
}
    
void Tracker::threadedFunction() {
    while(isThreadRunning()) {
        ofResetElapsedTimeCounter();
        if(isTracking() && !toProcessFrame.empty()) {
            lock();
            Mat tmp; toProcessFrame.copyTo(tmp);
            unlock();
            process(tmp);
            if(canCalcModelViewMatrix()) {
                calcModelViewMatrix(modelViewMatrix,camMat);
            }
        }
        ofLog() << "tracker: " << ofGetElapsedTimeMillis() << "ms";
    }
}

MarkerDetector::MarkerDetector() {
    detector = new SurfFeatureDetector(1000);
    extractor = new OpponentColorDescriptorExtractor(new SurfDescriptorExtractor);
    matcher = DescriptorMatcher::create("BruteForce");
    bowextractor = new BOWImgDescriptorExtractor(extractor,matcher);
    bowtrainer = new BOWKMeansTrainer(50);
}

void MarkerDetector::readFromFiles() {
    ofLog() << "marker detector reading from files...";
//    system_clock::time_point start = system_clock::now();

    FileStorage fs(ofToDataPath("vocab.yml"),FileStorage::READ);
    Mat vocab; fs["vocabulary"] >> vocabulary;
    ofLog() << "vocabulary " << vocabulary.size();
    fs.release();
    matcher->clear();
    matcher->add( std::vector<Mat>(1, vocabulary) );
    fs.open(ofToDataPath("pca.yml"),FileStorage::READ);
    fs["eigenvalues"] >> descriptorPCA.eigenvalues;
    fs["eigenvectors"] >> descriptorPCA.eigenvectors;
    fs["mean"] >> descriptorPCA.mean;
    ofLog() << "PCA eigenvectors " << descriptorPCA.eigenvectors.size();
    ofLog() << "PCA eigenvalues " << descriptorPCA.eigenvalues.size();
    ofLog() << "PCA mean " << descriptorPCA.mean.size();

    fs.release();
    fs.open(ofToDataPath("training.yml"),FileStorage::READ);
    fs["training"] >> training;
    fs["training_labels"] >> training_labels;
    ofLog() << "training " << training.size();
    fs.release();
    fs.open(ofToDataPath("markers.yml"),FileStorage::READ);
    fs["marker_files"] >> marker_files;
    ofLog() << "markers " << marker_files.size();
    fs.release();

    //read markers
    for (int i = 0; i < marker_files.size(); ++i) {
        Mat marker = cv::imread(marker_files[i]);
//            resize(marker,marker,Size(),0.5,0.5);
        markers.push_back(marker);
    }


    // creating labels
    Mat_<int> responses(training.rows,1);
    training_labelsUniq = training_labels;
    unique(training_labelsUniq.begin(),training_labelsUniq.end());
    for (int i = 0; i < training_labels.size(); ++i) {
        responses(i) = find(training_labelsUniq.begin(),training_labelsUniq.end(),training_labels[i]) - training_labelsUniq.begin();
    }
    // train classifier
    classifier.train(training,responses);

//    duration<double> sec = system_clock::now() - start;
//    ofLog() << "ended in " << sec.count() << " seconds\n";
}

void MarkerDetector::saveToFiles() {
    FileStorage fs("vocab.yml",FileStorage::WRITE);
    fs << "vocabulary" << getVocabulary();
    fs.release();
    fs.open("pca.yml",FileStorage::WRITE);
    fs << "eigenvalues" << getDescriptorPca().eigenvalues;
    fs << "eigenvectors" << getDescriptorPca().eigenvectors;
    fs << "mean" << getDescriptorPca().mean;
    fs.release();
    fs.open("training.yml",FileStorage::WRITE);
    fs << "training" << getTraining();
    fs << "training_labels" << getTrainingLabels();
    fs.release();
    fs.open("markers.yml",FileStorage::WRITE);
    fs << "marker_files" << marker_files;
    fs.release();
}

void MarkerDetector::addMarker(const string& marker_file) {
    Mat marker = cv::imread(marker_file);
    resize(marker,marker,cv::Size(),0.5,0.5);
    addMarker(marker,marker_file);
}
    
void MarkerDetector::addMarker(const Mat& marker, const string& marker_file) {

    marker_files.push_back(marker_file);
    markers.push_back(marker);

    vector<KeyPoint> kp;
    detector->detect(marker,kp);
    ofLog() << "detected " << kp.size() << " keypoints\n";

    Mat desc;
    extractor->compute(marker,kp,desc);
    ofLog() << "computed " << desc.rows << " descriptors\n";

    descriptorsBeforePCA.push_back(desc);
}

void MarkerDetector::cluster() {
    ofLog() << "calculating PCA..";
    descriptorPCA(descriptorsBeforePCA, // pass the data
            Mat(), // there is no pre-computed mean vector, so let the PCA engine to compute it
            CV_PCA_DATA_AS_ROW, // indicate that the vectors are stored as matrix rows
            32 // specify how many principal components to retain
            );
    descriptorPCA.project(descriptorsBeforePCA,descriptorsAfterPCA);

    ofLog() << "done with PCA\n";

    ofLog() << "marker detector clustering.. ";
    bowtrainer->add(descriptorsAfterPCA);
    vocabulary = bowtrainer->cluster();
    ofLog() << "done\n";

    matcher->clear();
    matcher->add( std::vector<Mat>(1, vocabulary) );
}

void MarkerDetector::extractBOWdescriptor(const Mat& img, Mat& imgDescriptor, const Mat& mask) {
    // Compute descriptors for the image.
    vector<KeyPoint> kp;
    detector->detect(img,kp,mask);

//        Mat out;
//        drawKeypoints(img,kp,out,Scalar(0,0,255));
//        imshow("kp",out); waitKey(1);

    Mat desc;
    extractor->compute(img,kp,desc);

    Mat descAfterPCA = descriptorPCA.project(desc);
//        ofLog() << descAfterPCA << endl;

    int clusterCount = vocabulary.rows;

    // Match keypoint descriptors to cluster center (to vocabulary)
    std::vector<DMatch> matches;
    matcher->match( descAfterPCA, matches );

    // Compute image descriptor
    imgDescriptor.create(1, clusterCount, CV_32FC1);
    imgDescriptor.setTo(Scalar::all(0));

    float *dptr = imgDescriptor.ptr<float>();
    for( size_t i = 0; i < matches.size(); i++ )
    {
        int queryIdx = matches[i].queryIdx;
        int trainIdx = matches[i].trainIdx; // cluster index
        CV_Assert( queryIdx == (int)i );

        dptr[trainIdx] = dptr[trainIdx] + 1.f;
    }

    // Normalize image descriptor.
    imgDescriptor /= descAfterPCA.rows;

//        ofLog() << imgDescriptor << endl;
}

void MarkerDetector::addImageToTraining(const Mat& img,const string& label) {
    Mat imgDescriptor;
    extractBOWdescriptor(img,imgDescriptor);
    training.push_back(imgDescriptor);
    training_labels.push_back(label);
}

string MarkerDetector::detectMarkerInImage(const Mat& img, const Mat& mask) {
    Mat imgDescriptor;
    extractBOWdescriptor(img,imgDescriptor, mask);
    Mat results,dists;
    float p = classifier.find_nearest(imgDescriptor,10,&results,0,0,&dists);
    ofLog() << training_labelsUniq[(int)p] << " " << results << " " << dists << endl;
    return training_labelsUniq[(int)p];
}

Mat MarkerDetector::getMarker(const string& label) {
    int markerIdx = find(training_labelsUniq.begin(),training_labelsUniq.end(),label) - training_labelsUniq.begin();
    ofLog() << "found " <<  label << " in idx " << markerIdx << endl;
    return markers[markerIdx];
}
void MarkerDetector::setVocabulary(const Mat& vocabulary) {
    this->vocabulary = vocabulary;
    matcher->clear();
    matcher->add( std::vector<Mat>(1, vocabulary) );
}

} /* namespace SequentialReader */
