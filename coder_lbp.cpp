#include "coder_lbp.h"
#include "LBP.h"
#include "hist.h"

coder_LBP* coder_lbp_create() {
	coder_LBP* coder = (coder_LBP*) calloc(1, sizeof(coder_LBP));
	if (coder == NULL) return NULL;

	coder->input = NULL;
	coder->mask = NULL;
	coder->output = NULL;
	coder->histogram = Mat(256, 1, CV_32FC1);

	return coder;
}

void coder_lbp_encode(coder_LBP* coder) {
	calc_standard_lbp(coder);
	//calc_contrast_lbp(coder);
}

void coder_lbp_free(coder_LBP* coder) {

	coder->input.release();
	coder->mask.release();
	coder->output.release();
	coder->histogram.release();

	free(coder);
}











//int main() {
//	Mat img = imread("green.jpg");
//	Mat out = Mat(img.rows, img.cols, CV_8UC1);
//
//	createImageEqualized(img, out);
//
//	namedWindow("LBP", WINDOW_NORMAL);
//	imshow("LBP", out);
//	imwrite("green_contrast.jpg", out);
//	waitKey(0);
//}

//int main() {
//	Mat img = imread("green.jpg");
//	Mat out = Mat(img.rows, img.cols, CV_8UC1);
//
//	img.convertTo(out,-1,1.3,0);
//
//	namedWindow("LBP", WINDOW_NORMAL);
//	imshow("LBP", out);
//	imwrite("green_contrast2.jpg", out);
//	waitKey(0);
//}

//int main() {
//	Mat img = imread("green.jpg");
//	Mat out = Mat(3,3,CV_8SC1);
//
//	//medianBlur(img, out, 5);
//	//stylization(img, out);
//	//detailEnhance(img, out);
//	posterize(img, out);
//
//	namedWindow("LBP", WINDOW_NORMAL);
//	imshow("LBP", out);
//	imwrite("green_contrast3.jpg", out);
//
//	waitKey(0);
//}

//int main() {
//	Mat img = imread("iris_norm.png", IMREAD_GRAYSCALE);
//	Mat out = Mat(img.rows, img.cols, CV_8UC1);
//
//	createImageLBP(img, out);
//
//	namedWindow("LBP", WINDOW_AUTOSIZE);
//	imshow("LBP", out);
//	waitKey(0);
//
//	imwrite("iris_norm_LBP.png", out);
//}

//int main() {
//	//Crea e mostra immagine istogramma
//	int windowSize = 15;
//	Mat img = imread("eye.jpg");
//	Mat out = Mat(img.rows, img.cols, CV_8UC1);
//
//	calcPosterizeColor(img, out, windowSize);
//
//	namedWindow("Posterize", WINDOW_AUTOSIZE);
//	imshow("Posterize", out);
//	imwrite("eye_posterize_15.png", out);
//
/*printf("rows: %d cols: %d\n", img.rows, img.cols);
//Quindi il for deve partire da windowSize/2 compreso e arrivare a width-windowSize/2
printf("Row: %d, Col: %d -> %d", 7, 7, img.at<uchar>(7, 7));
Mat img_out = Mat(windowSize, windowSize, CV_8UC1);
img(Range(468 - windowSize/2, 468 + windowSize/2 +1), Range(400- windowSize/2, 400 + windowSize/2 + 1)).copyTo(img_out);

cout << "M = " << endl << " " << img_out << endl << endl;
*/

//	printf("%d\n", calcPixelPosterize(img, 7, 7, 15));
/*posterize(img, out, windowSize);

Mat img_out = Mat(windowSize, windowSize, CV_8UC1);
img(Range(1, 4), Range(1, 4)).copyTo(img_out);

Mat hist;
computeHist(img_out, hist);

//for (int i = 0; i < hist.rows; i++)
//	printf("%i -> %f\n", i, hist.at<float>(i));

double min, max;
int minLoc, maxLoc;
minMaxIdx(hist, &min, &max, &minLoc, &maxLoc);

printf("min: %f, max: %f, minLoc: %i, maxLoc: %i", min, max, minLoc, maxLoc);

//showHist(hist);
cin.get();
}*/

/*int main() {
//Crea e mostra immagine istogramma
Mat img = imread("eye.jpg",IMREAD_GRAYSCALE);

int histSize = 256; //from 0 to 255
float range[] = { 0, 256 }; //the upper boundary is exclusive
const float* histRange = { range };

Mat out;
calcHist(&img, 1, 0, Mat(), out, 1, &histSize, &histRange);

int hist_w = 1024;
int hist_h = 400;
int bin_w = cvRound((double)hist_w / histSize);
Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

normalize(out, out, 0, histImage.rows, NORM_MINMAX, -1, Mat());

for (int i = 1; i < histSize; i++){
line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(out.at<float>(i - 1))),
Point(bin_w*(i), hist_h - cvRound(out.at<float>(i))),
Scalar(0, 0, 255), 2, 8, 0);

line(histImage, Point(2 * bin_w*(i), hist_h),
Point(2 * bin_w*(i), hist_h - cvRound(out.at<float>(i))),
Scalar(0, 0, 255), 2, 8, 0);
}

namedWindow("calcHist Demo", WINDOW_AUTOSIZE);
imshow("calcHist Demo", histImage);
waitKey(0);

cin.get();
}*/