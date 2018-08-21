#include "coder_lbp.h"
#include "hist.h"

//Crea il valore LBP associato al pixel (row, column) dell'immagine img_in
int standardLBP(Mat img_in, int row, int column) {
	int LBP = 0;
	int exp = 7;
	uchar rgba = 0;
	uchar thresh = img_in.at<uchar>(row, column);
	//printf("THRESHOLD: %d\n", thresh);
	vector<Vec2i> pos = { Vec2i(-1,-1),Vec2i(-1,0),Vec2i(-1,1),
		Vec2i(0,1),Vec2i(1,1),
		Vec2i(1,0),Vec2i(1,-1),Vec2i(0,-1) };

	for (int i = 0; i < pos.size(); i++) {
		rgba = img_in.at<uchar>(row + pos[i][0], column + pos[i][1]);
		//printf("PIXEL H:%d W:%d COLOR: %d\n", row+pos[i][0], column + pos[i][1], rgba);
		LBP += (rgba >= thresh ? 1 : 0) << exp--;
	}
	return LBP;
}

//Crea l'immagine LBP di img_in
void calcStandardLBP(coder_LBP *coder) {
	for (int i = 1; i < coder->input.rows - 1; i++) {
		for (int j = 1; j < coder->input.cols - 1; j++) {
			int LBP = standardLBP(coder->input, i, j);
			printf("Row: %d Col: %d -> LBP: %f\n", i, j, LBP);
			coder->output.at<uchar>(i, j) = LBP;
		}
	}

	Mat hist;
	computeHist(coder->output, hist);
	showHist(hist);
}

//Utilizza la formula trovata nel paper
float contrastLBP(Mat img_in, int row, int column) {
	float LBP = 0;
	int exp = 7;
	uchar rgba = 0;
	vector<Vec2i> pos = {Vec2i(-1,-1),Vec2i(-1,0),Vec2i(-1,1),
		Vec2i(0,1),Vec2i(1,1),
		Vec2i(1,0),Vec2i(1,-1),Vec2i(0,-1) };

	float mu = 0;
	for (int i = 0; i < pos.size(); i++)
		mu += img_in.at<uchar>(row + pos[i][0], column + pos[i][1]);
	mu /= pos.size();

	for (int i = 0; i < pos.size(); i++) {
		rgba = img_in.at<uchar>(row + pos[i][0], column + pos[i][1]);
		//printf("PIXEL H:%d W:%d COLOR: %d\n", row + pos[i][0], column + pos[i][1], rgba);
		LBP += powf(rgba - mu, 2.0f);
	}

	LBP /= pos.size();
	//printf("R: %d C:%d -> MU: %f LBP: %f\n",row, column, mu, LBP);

	return LBP;
}

//Utilizza la formula delle differenze positive e negative
float differenceLBP(Mat img_in, int row, int column) {
	float positive = 0.0f;
	float negative = 0.0f;
	int n_positive = 0;
	int n_negative = 0;
	uchar rgba = 0;
	uchar threshold = img_in.at<uchar>(row, column);
	//printf("THRESHOLD : %d\n", threshold);


	vector<Vec2i> pos = { Vec2i(-1,-1),Vec2i(-1,0),Vec2i(-1,1),
		Vec2i(0,1),Vec2i(1,1),
		Vec2i(1,0),Vec2i(1,-1),Vec2i(0,-1) };

	for (int i = 0; i < pos.size(); i++) {
		rgba = img_in.at<uchar>(row + pos[i][0], column + pos[i][1]);
		//printf("R: %d C:%d -> COLOR: %d\n", row + pos[i][0], column + pos[i][1], rgba);
		if (img_in.at<uchar>(row + pos[i][0], column + pos[i][1]) >= threshold) {
			positive += rgba;
			n_positive++;
		} else {
			negative += rgba;
			n_negative++;
		}
	}

	if (!n_positive) return -(negative / n_negative);
	else if (!n_negative) return positive / n_positive;
	return positive/n_positive - negative/n_negative;
}

//Crea l'immagine LBP di img_in
void calcContrastLBP(coder_LBP *coder) {
	float max = 0;
	float min = 0;
	Mat tmp = Mat(coder->input.rows, coder->input.cols, CV_32FC1);
	for (int i = 1; i < coder->input.rows - 1; i++) {
		for (int j = 1; j < coder->input.cols - 1; j++) {
			float LBP =contrastLBP(coder->input, i, j);
			printf("Row: %d Col: %d -> LBP: %f\n", i, j, LBP);
			max = (max >= LBP ? max : LBP);
			min = (min <= LBP ? min : LBP);
			tmp.at<float>(i, j) = LBP;
		}
	}

	printf("MIN : %f MAX:%f", min, max);

	min = abs(min);
	max = max + min;

	for (int i = 1; i < tmp.rows - 1; i++) {
		for (int j = 1; j < tmp.cols - 1; j++) {
			float color_f = tmp.at<float>(i, j);
			int color_u = (int) ((color_f + min) * 255 / max);
			//printf("PIXEL R:%d C:%d PREV: %f NOW: %d\n",i,j, color_f, color_u);
			coder->output.at<uchar>(i, j) = color_u;
		}
	}

	//printf("MIN : %f MAX:%f", min, max);

	Mat hist;
	computeHist(coder->output, hist);
	showHist(hist);
}


void createImageEqualized(coder_LBP *coder) {
	Mat channel_in[3] = { Mat(coder->input.rows, coder->input.cols, CV_8UC1),
		Mat(coder->input.rows, coder->input.cols, CV_8UC1),
		Mat(coder->input.rows, coder->input.cols, CV_8UC1) };

	split(coder->input, channel_in);

	for (int i = 0; i < 3; i++)
		equalizeHist(channel_in[i], channel_in[i]);

	merge(channel_in, 3, coder->output);
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