#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv/cv.h>
#include <iostream>

using namespace std;
using namespace cv;

#define MAX_NUM_KERNEL 4

struct coder_blob {
	Mat input;
	Mat mask;
	Mat log_imgs[MAX_NUM_KERNEL];
	Mat log_bin_imgs[MAX_NUM_KERNEL];
	Mat log_merge;
	Mat log_bin_merge;
};

coder_blob* coder_blob_create();
void coder_blob_init();
void coder_blob_encode(coder_blob* coder);
void coder_blob_free(coder_blob* coder);

