#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv/cv.h>
#include <iostream>

using namespace std;
using namespace cv;

int get_kernel_side(double sigma);
double LoG(double x, double y, double sigma);
void create_kernel_LoG(Mat &mat, double sigma);
void binarize_LoG(Mat &logImage, Mat &binImage);
void merge_LoG(Mat* log_imgs, Mat &img_out);