#include "hist.h"

void computeHist(Mat img_in, Mat &histogram) {
	int histSize = 256; //from 0 to 255
	float range[] = { 0, 256 }; //the upper boundary is exclusive
	const float* histRange = { range };
	calcHist(&img_in, 1, 0, Mat(), histogram, 1, &histSize, &histRange);
}


void showHist(Mat &histogram) {
	int hist_w = 1024;
	int hist_h = 400;
	int bin_w = cvRound((double)hist_w / 256);
	Mat histImage = Mat(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

	normalize(histogram, histogram, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	for (int i = 1; i < 256; i++) {
		//line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(out.at<float>(i - 1))),
		//	Point(bin_w*(i), hist_h - cvRound(out.at<float>(i))),
		//	Scalar(0, 0, 255), 2, 8, 0);*/

		line(histImage, Point(2 * bin_w*(i), hist_h),
			Point(2 * bin_w*(i), hist_h - cvRound(histogram.at<float>(i))),
			Scalar(0, 0, 255), 2, 8, 0);
	}

	namedWindow("calcHist Demo", WINDOW_AUTOSIZE);
	imshow("calcHist Demo", histImage);
	imwrite("histogram_standard_lbp.jpg", histImage);
	waitKey(0);
}
