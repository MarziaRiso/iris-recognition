#include<opencv2/opencv.hpp>
#include<opencv/cv.h>
#include<iostream>

using namespace std;
using namespace cv;

void convert_lightness(Mat &img_in, Mat&img_out);
void convert_average(Mat &img_in, Mat&img_out);
void convert_luminosity(Mat &img_in, Mat&img_out);

int calc_posterize_pixel(Mat &img_in, int row, int column, int windowSize);
void calc_posterize_gray(Mat &img_in, Mat &img_out, int windowSize);
void calc_posterize(Mat &img_in, Mat &img_out, int windowSize);
void delete_sclera_gray(Mat &img_in, Mat &img_out, int threshold);
void delete_sclera(Mat &img_in, Mat &img_out, int threshold);
void calc_equalized(Mat& img_in, Mat& img_out);
