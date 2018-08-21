#include<opencv2/opencv.hpp>
#include<opencv/cv.h>
#include<iostream>

using namespace std;
using namespace cv;

void convert_lightness(Mat &img_in, Mat&img_out);
void convert_average(Mat &img_in, Mat&img_out);
void convert_luminosity(Mat &img_in, Mat&img_out);
void convert_clahe(Mat &img_in, Mat &img_out);
void convert_reduce_saturation(Mat &img_in, Mat &img_out, int thresh);

int calcPixelPosterize(Mat &img_in, int row, int column, int windowSize);
void calcPosterizeGrayscale(Mat &img_in, Mat &img_out, int windowSize);
void calcPosterizeColor(Mat &img_in, Mat &img_out, int windowSize);
void delete_sclera_gray(Mat &img_in, Mat &img_out, int threshold);
void delete_sclera(Mat &img_in, Mat &img_out);
void hough_something();