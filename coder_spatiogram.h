#pragma once
#include<opencv2/opencv.hpp>
#include<opencv/cv.h>
#include<iostream>
#include "subject.h"

using namespace std;
using namespace cv;

struct spatiogram {
	Mat histogram;
	Mat mu;
	Mat sigma_x;
	Mat sigma_y;
};

struct coder_spatiogram {
	Mat output;
	spatiogram* spatiogram;
};

spatiogram* spatiogram_create();
void spatiogram_free(spatiogram* spatio);
coder_spatiogram* coder_spatiogram_create();
void coder_spatiogram_encode(subject* sub, coder_spatiogram* coder);
double coder_spatiogram_match(coder_spatiogram* coder1, coder_spatiogram* coder2);
void coder_spatiogram_free(coder_spatiogram* coder);
