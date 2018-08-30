#include "preprocessing.h"
#include "hist.h"

void convert_lightness(Mat &img_in, Mat&img_out) {
	printf("ROWS: %d COLS: %d\n", img_in.rows, img_in.cols);
	for (int i = 0; i < img_in.rows; i++) {
		for (int j = 0; j < img_in.cols; j++) {
			Vec3b bgr = img_in.at<Vec3b>(i, j);

			uchar channel_max = max(max(bgr[0], bgr[1]), bgr[2]);
			uchar channel_min = min(min(bgr[0], bgr[1]), bgr[2]);

			uchar gray = (channel_max + channel_min) / 2;
			img_out.at<Vec3b>(i, j) = Vec3b(gray,gray,gray);
		}
	}
}

void convert_average(Mat &img_in, Mat&img_out) {
	printf("ROWS: %d COLS: %d\n", img_in.rows, img_in.cols);
	for (int i = 0; i < img_in.rows; i++) {
		for (int j = 0; j < img_in.cols; j++) {
			Vec3b bgr = img_in.at<Vec3b>(i, j);
			uchar gray = (bgr[0] + bgr[1] + bgr[2])/3;
			img_out.at<Vec3b>(i, j) = Vec3b(gray, gray, gray);
		}
	}
}

void convert_luminosity(Mat &img_in, Mat &img_out) {
	printf("ROWS: %d COLS: %d\n", img_in.rows, img_in.cols);
	for (int i = 0; i < img_in.rows; i++) {
		for (int j = 0; j < img_in.cols; j++) {
			Vec3b bgr = img_in.at<Vec3b>(i, j);
			//uchar gray = (uchar)(bgr[0] * 0.07 + bgr[1] * 0.72 + bgr[2] * 0.21);
			uchar gray = (uchar)(bgr[0] * 0.114 + bgr[1] * 0.587 + bgr[2] * 0.299);
			img_out.at<Vec3b>(i, j) = Vec3b(gray, gray, gray);
			//printf("RED: %d GREEN: %d BLUE: %d -> GRAY: %d\n", bgr[2], bgr[1], bgr[0], gray);
		}
	}
}

void convert_reduce_saturation(Mat &img_in, Mat &img_out, int thresh)
{
	Mat channel_in[3] = { Mat(img_in.rows, img_in.cols, CV_8UC1),
		Mat(img_in.rows, img_in.cols, CV_8UC1),
		Mat(img_in.rows, img_in.cols, CV_8UC1)};

	Mat channel_out[3] = { Mat(img_in.rows, img_in.cols, CV_8UC1),
		Mat(img_in.rows, img_in.cols, CV_8UC1),
		Mat(img_in.rows, img_in.cols, CV_8UC1)};

	split(img_in, channel_in);

	for (int k = 0; k < 3; k++) {
		for (int i = 0; i < channel_in->rows; i++) {
			for (int j = 0; j < channel_in->cols; j++) {
				uchar color = channel_in[k].at<uchar>(i, j) - thresh;
				color = (color < 0) ? 0 : color;
				channel_out[k].at<uchar>(i, j) = color;
			}
		}
	}


	imwrite("preprocessing/sat_blue.png", channel_out[0]);
	imwrite("preprocessing/sat_green.png", channel_out[1]);
	imwrite("preprocessing/sat_red.png", channel_out[2]);

	merge(channel_out, 3, img_out);
}

void convert_whitening(Mat &img_in, Mat &img_out, int thresh)
{
	for (int i = 0; i < img_in.rows; i++) {
		for (int j = 0; j < img_in.cols; j++) {
			Vec3b bgr = img_in.at<Vec3b>(i, j);
			for (int k = 0; k < 3; k++) {
				bgr[k] += (uchar)(((255 - bgr[k])*thresh) / 100);
			}
			img_out.at<Vec3b>(i, j) = bgr;
		}
	}
}



void convert_clahe(Mat &img_in, Mat &img_out)
{
	Mat channel_in[3] = { Mat(img_in.rows, img_in.cols, CV_8UC1),
		Mat(img_in.rows, img_in.cols, CV_8UC1),
		Mat(img_in.rows, img_in.cols, CV_8UC1) };

	Mat channel_out[3] = { Mat(img_in.rows, img_in.cols, CV_8UC1),
		Mat(img_in.rows, img_in.cols, CV_8UC1),
		Mat(img_in.rows, img_in.cols, CV_8UC1) };

	split(img_in, channel_in);

	Ptr<CLAHE> clahe = createCLAHE();
	clahe->setClipLimit(-2);

	for (int i = 0; i < 3; i++) {
		clahe->apply(channel_in[i], channel_out[i]);
	}

	imwrite("Clahe_blue.png", channel_out[0]);
	imwrite("Clahe_green.png", channel_out[1]);
	imwrite("Clahe_red.png", channel_out[2]);

	merge(channel_out, 3, img_out);

}

int calcPixelPosterize(Mat &img_in, int row, int column, int windowSize) {

	//Mat img_out = Mat(windowSize, windowSize, CV_8UC1);
	//img_in(Range(a, b), Range(c, d)).copyTo(img_out);
	//La matrice che viene creata prende le righe da "a" a "b-1" e da "c" a "d-q"
	Mat img_out = Mat(windowSize, windowSize, CV_8UC1);
	img_in(Range(row - windowSize / 2, row + windowSize / 2 + 1), Range(column - windowSize / 2, column + windowSize / 2 + 1)).copyTo(img_out);

	//cout << "M = " << endl << " " << img_out << endl << endl;

	Mat hist;
	//computeHist(img_out, hist);

	int maxLoc = 0;
	for (int i = 0; i < hist.size().height - 1; i++)
		if (hist.at<float>(i) < hist.at<float>(i + 1))
			maxLoc = i;

	//printf("%d\n", maxLoc);
	//cin.get();

	//Con questo showHist l'istogramma in hist viene anche normalizzato
	//Ricordatelo oppure cambia il metodo creando un altro istogramma
	//showHist(hist);

	return maxLoc;
}

void calcPosterizeGrayscale(Mat &img_in, Mat &img_out, int windowSize) {
	int dim = windowSize / 2;
	printf("rows: %d cols: %d\n", img_in.rows, img_in.cols);
	for (int i = dim; i < img_in.rows - dim; i++) {
		for (int j = dim; j < img_in.cols - dim; j++) {
			int post = calcPixelPosterize(img_in, i, j, windowSize);
			img_out.at<uchar>(i, j) = post;
			//printf("%ROW: %d, COL: %d\n", i, j);
		}
	}
}

void calcPosterizeColor(Mat &img_in, Mat &img_out, int windowSize) {
	Mat channel_in[3] = { Mat(img_in.rows, img_in.cols, CV_8UC1),
		Mat(img_in.rows, img_in.cols, CV_8UC1),
		Mat(img_in.rows, img_in.cols, CV_8UC1) };

	Mat channel_out[3] = { Mat(img_in.rows, img_in.cols, CV_8UC1),
		Mat(img_in.rows, img_in.cols, CV_8UC1),
		Mat(img_in.rows, img_in.cols, CV_8UC1) };

	split(img_in, channel_in);

	for (int i = 0; i < 3; i++) {
		calcPosterizeGrayscale(channel_in[i], channel_out[i], windowSize);
		/*namedWindow("Posterize", WINDOW_AUTOSIZE);
		imshow("Posterize", channel_out[i]);
		imwrite("Posterize_15_colors.png", channel_out[i]);*/
	}

	imwrite("Posterize_BB.png", channel_out[0]);
	imwrite("Posterize_GG.png", channel_out[1]);
	imwrite("Posterize_RR.png", channel_out[2]);

	merge(channel_out, 3, img_out);

}

void delete_sclera_gray(Mat &img_in, Mat &img_out, int threshold) {

	for (int i = 0; i < img_in.rows; i++) {
		for (int j = 0; j < img_in.cols; j++) {
			uchar pixel = img_in.at<uchar>(i, j);
			if (pixel < threshold) img_out.at<uchar>(i, j) = pixel;
			else img_out.at<uchar>(i, j) = 255;
		}
	}
}

void delete_sclera(Mat &img_in, Mat &img_out) {
	Mat channel_in[3] = { Mat(img_in.rows, img_in.cols, CV_8UC1),
		Mat(img_in.rows, img_in.cols, CV_8UC1),
		Mat(img_in.rows, img_in.cols, CV_8UC1) };

	Mat channel_out[3] = { Mat(img_in.rows, img_in.cols, CV_8UC1),
		Mat(img_in.rows, img_in.cols, CV_8UC1),
		Mat(img_in.rows, img_in.cols, CV_8UC1) };

	split(img_in, channel_in);

	delete_sclera_gray(channel_in[0], channel_out[0], 200);
	delete_sclera_gray(channel_in[1], channel_out[1], 200);
	delete_sclera_gray(channel_in[2], channel_out[2], 230); //Il rosso deve essere mantenuto alto

	//imwrite("Posterize_BB.png", channel_out[0]);
	//imwrite("Posterize_GG.png", channel_out[1]);
	//imwrite("Posterize_RR.png", channel_out[2]);

	merge(channel_out, 3, img_out);

}