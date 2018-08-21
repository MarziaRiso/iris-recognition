#include <opencv2/opencv.hpp>
#include <opencv/cv.h>
#include <iostream>

using namespace std;
using namespace cv;

struct coder_LBP {
	Mat input;
	Mat mask;
	Mat output;
};

int standardLBP(Mat img_in, int row, int column);
void calcStandardLBP(coder_LBP *coder);
float contrastLBP(Mat img_in, int row, int column);
float differenceLBP(Mat img_in, int row, int column);
void calcContrastLBP(coder_LBP *coder);
void createImageEqualized(coder_LBP *coder);