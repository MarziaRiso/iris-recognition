#include "coder_lbp.h"
#include "LBP.h"
#include "hist.h"

coder_LBP* coder_lbp_create() {
	coder_LBP* coder = (coder_LBP*) calloc(1, sizeof(coder_LBP));
	if (coder == NULL) return NULL;

	coder->out = NULL;

	for (int i = 0; i < NUM_ZONE; i++) {
		coder->histogram[i] = NULL;
	}

	return coder;
}

void coder_lbp_load(string subject_name, coder_LBP* coder) {
	coder->out = imread("dataset/" + subject_name.substr(9, 3)
		+ "/output/" + subject_name.substr(14) + ".output.png", IMREAD_GRAYSCALE);

	int sub_rows = coder->out.rows / NUM_ZONE;

	for (int k = 0; k < NUM_ZONE; k++) {
		coder->histogram[k] = Mat::zeros(256, 1, CV_32FC1);
		computeHist(coder->out(Range(sub_rows*k, sub_rows*(k + 1)), Range(0, coder->out.cols)), coder->histogram[k]);
	}
}

void coder_lbp_encode(subject* sub, coder_LBP* coder) {
	coder->out = Mat::zeros(sub->input.size(), CV_8UC1);

	for (int i = 0; i < NUM_ZONE; i++)
		coder->histogram[i] = Mat::zeros(256, 1, CV_32FC1);

	calc_standard_lbp(sub, coder);
	//calc_contrast_lbp(sub, coder);
}

double coder_lbp_match(subject* sub1, coder_LBP* coder1,subject* sub2, coder_LBP* coder2) {

	int sub_rows = sub1->input.rows / NUM_ZONE;

	Mat norm_hist1 = Mat::zeros(256, 1, CV_32FC1);
	Mat norm_hist2 = Mat::zeros(256, 1, CV_32FC1);
	Mat mask1 = Mat::zeros(sub_rows, sub1->input.cols, CV_8UC1);
	Mat mask2 = Mat::zeros(sub_rows, sub2->input.cols, CV_8UC1);

	double similarity = 0.0;

	for (int i = 0; i < NUM_ZONE; i++) {
		normalize(coder1->histogram[i], norm_hist1);
		normalize(coder2->histogram[i], norm_hist2);

		sub1->mask(Range(sub_rows*i, sub_rows*(i + 1)), Range(0, sub1->input.cols)).copyTo(mask1);
		sub2->mask(Range(sub_rows*i, sub_rows*(i + 1)), Range(0, sub2->input.cols)).copyTo(mask2);

		similarity += compareHist(norm_hist1, norm_hist2, CV_COMP_CHISQR)
			* (1.0 - (((2*mask1.rows*mask1.cols) - (countNonZero(mask1) + countNonZero(mask2))) 
				/ (double)(2 * mask1.rows*mask1.cols)));
	}

	return similarity / NUM_ZONE;
}

void coder_lbp_free(coder_LBP* coder) {
	coder->out.release();
	for (int i = 0; i < NUM_ZONE; i++) {
		coder->histogram[i].release();
	}

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