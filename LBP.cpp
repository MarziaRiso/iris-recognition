#include<opencv2/opencv.hpp>
#include<opencv/cv.h>
#include<iostream>

using namespace std;
using namespace cv;

//Crea il valore LBP associato al pixel (row, column) dell'immagine img_in
int createPixelLBP(Mat& img_in, int row, int column){
	int LBP = 0;
	int exp = 7;
	uchar rgba = 0;
	uchar thresh = img_in.at<uchar>(row, column);
	vector<Vec2i> pos =	{Vec2i(-1,-1),Vec2i(-1,0),Vec2i(-1,1),
						Vec2i(0,1),Vec2i(1,1),
						Vec2i(1,0),Vec2i(1,-1),Vec2i(0,-1)};

	for (int i = 0; i < pos.size(); i++) {
		rgba = img_in.at<uchar>(row + pos[i][0], column + pos[i][1]);
		LBP += (rgba >= thresh ? 1 : 0) << exp--;
	}
	return LBP;
}

//Crea l'immagine LBP di img_in
void createImageLBP(Mat& img_in, Mat& img_out) {
	for (int i = 1; i < img_in.rows-1; i++) {
		for (int j = 1; j < img_in.cols-1; j++) {
			int LBP = createPixelLBP(img_in, i, j);
			img_out.at<uchar>(i, j) = LBP;
		}
	}
}

void createImageEqualized(Mat& img_in, Mat& img_out) {
	Mat channel_in[3] =	{	Mat(img_in.rows, img_in.cols, CV_8UC1),
							Mat(img_in.rows, img_in.cols, CV_8UC1),
							Mat(img_in.rows, img_in.cols, CV_8UC1) };

	split(img_in, channel_in);

	for (int i = 0; i < 3; i++)
		equalizeHist(channel_in[i], channel_in[i]);

	merge(channel_in, 3, img_out);
}

void computeHist(Mat& img_in, Mat& histogram) {
	int histSize = 256; //from 0 to 255
	float range[] = { 0, 256 }; //the upper boundary is exclusive
	const float* histRange = { range };
	calcHist(&img_in, 1, 0, Mat(), histogram, 1, &histSize, &histRange);
}

void showHist(Mat& histogram) {
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
	waitKey(0);
}


int calcPixelPosterize(Mat& img_in, int row, int column, int windowSize){

	//Mat img_out = Mat(windowSize, windowSize, CV_8UC1);
	//img_in(Range(a, b), Range(c, d)).copyTo(img_out);
	//La matrice che viene creata prende le righe da "a" a "b-1" e da "c" a "d-q"
	Mat img_out = Mat(windowSize, windowSize, CV_8UC1);
	img_in(Range(row- windowSize / 2, row + windowSize / 2 +1 ), Range(column- windowSize / 2, column + windowSize / 2 + 1 )).copyTo(img_out);

	//cout << "M = " << endl << " " << img_out << endl << endl;

	Mat hist;
	computeHist(img_out, hist);

	int maxLoc = 0;
	for (int i = 0; i < hist.size().height-1; i++)
		if (hist.at<float>(i) < hist.at<float>(i+1))
			maxLoc = i;

	//printf("%d\n", maxLoc);
	//cin.get();

	//Con questo showHist l'istogramma in hist viene anche normalizzato
	//Ricordatelo oppure cambia il metodo creando un altro istogramma
	//showHist(hist);

	return maxLoc;
}

void posterize(Mat& img_in, Mat& img_out, int windowSize) {
	int dim = windowSize / 2;
	printf("rows: %d cols: %d\n", img_in.rows, img_in.cols);
	for (int i = dim; i < img_in.rows - dim; i++) {
		for (int j = dim; j < img_in.cols - dim; j++) {
			int post = calcPixelPosterize(img_in, i, j, windowSize);
			img_out.at<uchar>(i, j) = post;
			//printf("%ROW: %d, COL: %d\n", i, j);
		}
	}

	namedWindow("Posterize", WINDOW_NORMAL);
	imshow("Posterize", img_out);
	imwrite("posterize_15.jpg", img_out);

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

int main() {
	//Crea e mostra immagine istogramma
	int windowSize = 15;
	Mat img = imread("tulip.png", IMREAD_GRAYSCALE);
	Mat out = Mat(img.rows, img.cols, CV_8UC1);

	posterize(img, out, windowSize);

	/*printf("rows: %d cols: %d\n", img.rows, img.cols);
	//Quindi il for deve partire da windowSize/2 compreso e arrivare a width-windowSize/2
	printf("Row: %d, Col: %d -> %d", 7, 7, img.at<uchar>(7, 7));
	Mat img_out = Mat(windowSize, windowSize, CV_8UC1);
	img(Range(468 - windowSize/2, 468 + windowSize/2 +1), Range(400- windowSize/2, 400 + windowSize/2 + 1)).copyTo(img_out);

	cout << "M = " << endl << " " << img_out << endl << endl;
	*/
	cin.get();
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
	cin.get();*/
}

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