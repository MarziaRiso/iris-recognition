#include<opencv2/opencv.hpp>
#include<opencv/cv.h>
#include<iostream>
#include "coder_lbp.h"

using namespace std;
using namespace cv;

int standard_lbp_pixel(Mat img_in, int row, int column);
void calc_standard_lbp(subject* sub, coder_LBP *coder);
float contrast_lbp_pixel(Mat img_in, int row, int column);
float difference_lbp_pixel(Mat img_in, int row, int column);
void calc_contrast_lbp(subject* sub, coder_LBP *coder);
//void create_equalized_image(coder_LBP *coder);