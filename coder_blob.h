#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv/cv.h>
#include <iostream>

using namespace std;
using namespace cv;

#define MAX_NUM_KERNEL 4
#define SHIFT 10
#define MOD(x,y) ((x) >= 0 ? ((x)%(y)) : ((y) + (x)%(y)))

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
double coder_blob_match(coder_blob* coder1, coder_blob* coder2);
void shift_image(Mat img, Mat shifted_image, int shift);
double hamming_distance(Mat img1, Mat mask1, Mat img2, Mat mask2);
double shifted_hamming_distance(Mat img1, Mat mask1, Mat img2, Mat mask2, int shift);
void coder_blob_free(coder_blob* coder);

