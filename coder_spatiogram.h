#include<opencv2/opencv.hpp>
#include<opencv/cv.h>
#include<iostream>

using namespace std;
using namespace cv;

struct spatiogram_entry {
	int histogram;
	int mean_vector;
	int covariance_matrix;
};

struct coder_spatiogram {
	Mat input;
	Mat mask;
	Mat output;
	spatiogram_entry spatiogram[256];
};

coder_spatiogram* coder_spatiogram_create();
void coder_spatiogram_encode(coder_spatiogram* coder);
void coder_spatiogram_free(coder_spatiogram* coder);
