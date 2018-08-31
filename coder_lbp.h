#pragma once
#include <opencv2/opencv.hpp>
#include <opencv/cv.h>
#include <iostream>
#include "subject.h"

using namespace std;
using namespace cv;

struct coder_LBP {
	Mat output;
	Mat histogram;
};

coder_LBP* coder_lbp_create();
void coder_lbp_encode(subject* sub, coder_LBP* coder);
double coder_lbp_match(coder_LBP* coder1, coder_LBP* coder2);
void coder_lbp_free(coder_LBP* coder);

