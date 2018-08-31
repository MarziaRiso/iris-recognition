#pragma once
#include <opencv2/opencv.hpp>
#include <opencv/cv.h>
#include <iostream>

using namespace std;
using namespace cv;

struct subject {
	Mat input;
	Mat mask;
};

subject* subject_create();
void subject_free(subject* eye);