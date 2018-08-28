#include<opencv2/opencv.hpp>
#include<opencv/cv.h>
#include<iostream>

using namespace std;
using namespace cv;

void computeHist(Mat img_in, Mat& histogram);
void showHist(Mat histogram);
