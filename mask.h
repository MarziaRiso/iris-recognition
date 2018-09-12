#include<opencv2/opencv.hpp>
#include<opencv/cv.h>

using namespace std;
using namespace cv;

void adjust_mask_single(Mat input, Mat old_mask, Mat new_mask);
void adjust_mask(Mat input, Mat old_mask, Mat new_mask);
