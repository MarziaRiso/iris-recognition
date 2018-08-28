#include<opencv2/opencv.hpp>
#include<opencv/cv.h>
#include<iostream>

using namespace std;
using namespace cv;

struct spatiogram {
	Mat histogram;
	Mat mean_vector;
	Mat covariance_matrix;
};

struct coder_spatiogram {
	Mat input;
	Mat mask;
	Mat output;
	spatiogram* spatiogram;
};

spatiogram* spatiogram_create();
void spatiogram_free(spatiogram* spatio);

coder_spatiogram* coder_spatiogram_create();
void coder_spatiogram_encode(coder_spatiogram* coder);
void coder_spatiogram_free(coder_spatiogram* coder);
