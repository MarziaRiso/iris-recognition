#include<opencv2/opencv.hpp>
#include<opencv/cv.h>
#include<iostream>

#include "log.h"
#include "coder_lbp.h"
#include "coder_blob.h"
#include "coder_spatiogram.h"
#include "mask.h"
#include "hist.h"
#include "preprocessing.h"

using namespace std;
using namespace cv;

int main() {

	//MAIN DEL MATCHING CON LBP
	coder_LBP* coder = coder_lbp_create();
	coder->input = imread("matching/IMG_006_L_1.iris.norm.png",IMREAD_GRAYSCALE);
	coder->mask = imread("matching/IMG_006_L_1.defectsnorm.png", IMREAD_GRAYSCALE);
	coder->output = Mat(coder->input.rows, coder->input.cols, CV_8UC1);
	coder_lbp_encode(coder);

	coder_LBP* coder_2 = coder_lbp_create();
	coder_2->input = imread("matching/IMG_006_L_1.iris.norm.png", IMREAD_GRAYSCALE);
	coder_2->mask = imread("matching/IMG_006_L_1.defectsnorm.png", IMREAD_GRAYSCALE);
	coder_2->output = Mat(coder->input.rows, coder->input.cols, CV_8UC1);
	coder_lbp_encode(coder_2);

	//showHist(coder->histogram);
	showHist(coder_2->histogram);

	double result = compareHist(coder->histogram, coder_2->histogram, CV_COMP_CHISQR);
	printf("Result: %f\n", result);

	/*namedWindow("MASK", WINDOW_NORMAL);
	imshow("MASK",coder->output);
	imwrite("eyes/lbp_coder.png", coder->output);*/

	coder_lbp_free(coder);
	coder_lbp_free(coder_2);

	waitKey(0);
	

	//MAIN DEL CODER DEGLI SPATIOGRAM
	/*coder_spatiogram* coder = coder_spatiogram_create();
	Mat input = Mat(4, 5, CV_8UC1);
	input.at<uchar>(0, 0) = 0;
	input.at<uchar>(0, 1) = 1;
	input.at<uchar>(0, 2) = 2;
	input.at<uchar>(0, 3) = 0;
	input.at<uchar>(0, 4) = 2;
	input.at<uchar>(1, 0) = 4;
	input.at<uchar>(1, 1) = 3;
	input.at<uchar>(1, 2) = 2;
	input.at<uchar>(1, 3) = 2;
	input.at<uchar>(1, 4) = 3;
	input.at<uchar>(2, 0) = 1;
	input.at<uchar>(2, 1) = 1;
	input.at<uchar>(2, 2) = 5;
	input.at<uchar>(2, 3) = 3;
	input.at<uchar>(2, 4) = 5;
	input.at<uchar>(3, 0) = 7;
	input.at<uchar>(3, 1) = 9;
	input.at<uchar>(3, 2) = 1;
	input.at<uchar>(3, 3) = 3;
	input.at<uchar>(3, 4) = 5;

	coder->input = input;

	cout << coder->input;
	cout << "\n\n\n";

	//coder->input = imread("006/IMG_006_L_1.iris.norm.png", IMREAD_GRAYSCALE);

	coder_spatiogram_encode(coder);

	for (int i = 0; i < 10; i++) {
		printf("%d -> h: %f\t\t mv: %f\t\t cm: %f\t\t\n", i,
			coder->spatiogram->histogram.at<float>(i,0),
			coder->spatiogram->mean_vector.at<float>(i, 0), 
			coder->spatiogram->covariance_matrix.at<float>(i, 0));
	}

	showHist(coder->spatiogram->histogram);
	coder_spatiogram_free(coder);

	cin.get();
	waitKey(0);*/

	//MAIN DEL CODER DEI BLOB
	/*coder_blob* coder = coder_blob_create();
	coder->input = imread("006/IMG_006_L_1.iris.norm.png", IMREAD_GRAYSCALE);

	coder_blob_init();

	coder_blob_encode(coder);

	for (int i = 0; i < MAX_NUM_KERNEL; i++)
	{
		namedWindow("LoG", WINDOW_NORMAL);
		imshow("LoG", coder->log_bin_imgs[i]);
		waitKey(0);
	}

	imwrite("006/log_bin_1.png", coder->log_bin_imgs[0]);
	imwrite("006/log_bin_2.png", coder->log_bin_imgs[1]);
	imwrite("006/log_bin_4.png", coder->log_bin_imgs[2]);
	imwrite("006/log_bin_8.png", coder->log_bin_imgs[3]);

	namedWindow("LoG", WINDOW_NORMAL);
	imshow("LoG", coder->log_bin_merge);
	imwrite("006/log_bin_merge.png", coder->log_bin_merge);

	coder_blob_free(coder);

	waitKey(0);*/


	//MAIN DEL PREPROCESSING
	/*Mat img_in = imread("preprocessing/IMG_070_L_3.jpg");
	Mat img_out = Mat(img_in.rows, img_in.cols, CV_8UC3);

	coder_spatiogram* coder = coder_spatiogram_create();
	calc_spatiogram(coder);

	//convert_whitening(img_in, img_out, 20);
	//hough_something();
	//convert_clahe(img_in,img_out);
	//convert_reduce_saturation(img_in, img_out, 20);
	//convert_luminosity(img_in, img_out);
	//cvtColor(img_in, img_out, COLOR_BGR2GRAY);

	namedWindow("RGB", WINDOW_NORMAL);
	imshow("RGB", img_out);
	imwrite("preprocessing/IMG_070_L_3_white20.png", img_out);

	waitKey(0);*/


	//MAIN DEL CODER DEI LBP
	/*coder_LBP* coder = coder_lbp_create();
	coder->input = imread("eyes/iris_norm.png",IMREAD_GRAYSCALE);
	//coder->mask = imread("006/IMG_006_L_1.defects.png", IMREAD_GRAYSCALE);
	coder->output = Mat(coder->input.rows, coder->input.cols, CV_8UC1);

	coder_lbp_encode(coder);
	
	namedWindow("MASK", WINDOW_NORMAL);
	imshow("MASK",coder->output);
	imwrite("eyes/lbp_coder.png", coder->output);

	coder_lbp_free(coder);

	waitKey(0);*/
}



//----------------------------------------------------------------------------
//Questo è come si usa l'iteratore
/*int main() {
	Mat img = imread("tulip.png");

	//int histogram[255] = {};
	MatIterator_<Vec3b> start = img.begin<Vec3b>();
	MatIterator_<Vec3b> end = img.end<Vec3b>();
	for (start; start < end; start++) {
		printf("%d, %d, %d\n", (*start)[0], (*start)[1], (*start)[2]);
	}

	cin.get();	
	namedWindow("image!", WINDOW_NORMAL);
	imshow("image!", img);
	waitKey(0);
	return 0;

}*/

//Questo è come si scorre su righe e colonne
/*int main() {
	Mat img = imread("tulip.png");

	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			Vec3b rgba = img.at<Vec3b>(i, j);
			printf("%d, %d, %d\n", rgba[0], rgba[1], rgba[2]);
		}
	}

	cin.get();
	namedWindow("image!", WINDOW_NORMAL);
	imshow("image!", img);
	waitKey(0);
	return 0;
}*/

//void equalize(Mat img_in, Mat& img_out) {
//	int histogram[256] = {};
//	int eq_histogram[256] = {};
//
//	//Crea l'istogramma per l'immagine di input imgin
//	for (int i = 0; i < img_in.rows; ++i)
//		for (int j = 0; j < img_in.cols; ++j)
//			histogram[img_in.at<uchar>(i, j)] += 1;
//
//	//Crea l'istogramma equalizzato eq_histogram dall' istogramma histogram
//	for (int k = 1; k < 256; k++) {
//		histogram[k] += histogram[k - 1];
//		eq_histogram[k] = (int)round((histogram[k]) * 255 / (img_in.rows*img_in.cols - 1.0f));
//	}
//
//	//Crea l'immagine di output img_out usando l'istogramma equalizzato
//	for (int i = 0; i < img_in.rows; ++i)
//		for (int j = 0; j < img_in.cols; ++j)
//			img_out.at<uchar>(i, j) = eq_histogram[img_in.at<uchar>(i, j)];
//
//}
//
//void equalizeColors(Mat img_in, Mat& img_out) {
//	Mat channel_arr[3] = {	Mat(img_in.rows, img_in.cols, CV_8UC1),
//							Mat(img_in.rows, img_in.cols, CV_8UC1),
//							Mat(img_in.rows, img_in.cols, CV_8UC1)};
//
//	//Divido l'immagine RGB in canali singoli
//	split(img_in, channel_arr);
//
//	//Ogni canale viene equalizzato
//	for (int i = 0; i < 3; i++)
//		equalize(channel_arr[i], channel_arr[i]);
//
//	//Ricompongo i canali equalizzati per creare l'immagine finale
//	merge(channel_arr, 3, img_out);
//}

//int main() {
//	////Main per l'equalizzazione di immagine in scala di grigi
//	//Mat img = imread("tulip.png",IMREAD_GRAYSCALE);
//	//Mat out(img.rows, img.cols,CV_8UC1);
//
//	//namedWindow("image!", WINDOW_NORMAL);
//	//imshow("image!", img);
//	//waitKey(0);
//
//	//out = equalize(img, out);
//
//	////Salva l'immagine di output img_out
//	//vector<int> compression_params;
//	//compression_params.push_back(IMWRITE_PNG_COMPRESSION);
//	//compression_params.push_back(9);
//	//imwrite("tulip_eq_2.png", out, compression_params);
//
//
//	////Main per l'equalizzazione di immagine a color
//	Mat img = imread("eye.jpg",IMREAD_ANYCOLOR);
//	Mat output(img.rows, img.cols,CV_8UC1);
//
//	equalizeColors(img, output);
//
//	imwrite("eye_eq.jpg", output);
//
//
//}