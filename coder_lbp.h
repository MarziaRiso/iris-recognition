#pragma once
#include <opencv2/opencv.hpp>
#include <opencv/cv.h>
#include <iostream>
#include "subject.h"

using namespace std;
using namespace cv;

#define NUM_ZONE 5

struct coder_LBP {
	Mat input[NUM_ZONE];
	Mat mask[NUM_ZONE];
	Mat output[NUM_ZONE];
	Mat histogram[NUM_ZONE];
};

coder_LBP* coder_lbp_create();
void coder_lbp_encode(subject* sub, coder_LBP* coder);
double coder_lbp_match(coder_LBP* coder1, coder_LBP* coder2);
void coder_lbp_free(coder_LBP* coder);

