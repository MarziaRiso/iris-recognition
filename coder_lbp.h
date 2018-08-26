#pragma once
#include <opencv2/opencv.hpp>
#include <opencv/cv.h>
#include <iostream>

using namespace std;
using namespace cv;

struct coder_LBP {
	Mat input;
	Mat mask;
	Mat output;
	Mat histogram;
};

coder_LBP* coder_lbp_create();
void coder_lbp_encode(coder_LBP* coder);
void coder_lbp_free(coder_LBP* coder);

