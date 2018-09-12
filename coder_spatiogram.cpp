#include "coder_spatiogram.h"
#include "coder.h"
#include <math.h>

#define _USE_MATH_DEFINES

spatiogram* spatiogram_create()
{
	spatiogram* spatio = (spatiogram*)calloc(1, sizeof(spatiogram));
	spatio->histogram = Mat::zeros(256, 1, CV_64FC1);
	spatio->mu = Mat::zeros(256, 2, CV_64FC1);
	spatio->sigma_x = Mat::zeros(256, 1, CV_64FC1);
	spatio->sigma_y = Mat::zeros(256, 1, CV_64FC1);
	return spatio;

	/*spatiogram* spatio = (spatiogram*)calloc(1, sizeof(spatiogram));
	spatio->histogram = Mat::zeros(256, 1, CV_32FC1);
	spatio->mean_vector = Mat::zeros(256, 1, CV_32FC1);
	spatio->covariance_matrix = Mat::zeros(256, 1, CV_32FC1);
	return spatio;*/
}

void spatiogram_free(spatiogram* spatio)
{
	spatio->histogram.release();
	spatio->mu.release();
	spatio->sigma_x.release();
	spatio->sigma_y.release();
	free(spatio);
}

coder_spatiogram* coder_spatiogram_create() 
{
	coder_spatiogram* coder = (coder_spatiogram*)calloc(1, sizeof(coder_spatiogram));
	if (coder == NULL) return NULL;

	coder->output = NULL;
	coder->spatiogram = spatiogram_create();
	return coder;
}

void coder_spatiogram_encode(subject* sub, coder_spatiogram* coder) 
{
	int z = sub->input.channels();
	int xs = sub->input.cols;
	int ys = sub->input.rows;
	int bins = 256;

	Mat binno = Mat::zeros(sub->input.size(), CV_32FC1);
	Mat channels[3];
	Mat new_channel;

	split(sub->input, channels);

	int f = 1;
	for (int k = 0; k < z; k++)
	{
		new_channel = Mat::zeros(channels[k].size(), CV_32FC1);
		for (int i = 0; i < new_channel.rows; i++) {
			for (int j = 0; j < new_channel.cols; j++) {
				binno.at<float>(i, j) += (float)(f*floor(channels[k].at<uchar>(i, j) * bins / 256));
			}
		}
		f *= bins;
	}

	float xf = 2.0f / (xs - 1);
	float yf = 2.0f / (ys - 1);

	Mat grid_x = Mat::zeros(binno.size(), CV_32FC1);
	for (int i = 0; i < binno.rows; i++) {
		float value = -1;
		for (int j = 0; j < binno.cols; j++) {
			grid_x.at<float>(i, j) = value;
			value += xf;
		}
	}

	Mat grid_y = Mat::zeros(binno.size(), CV_32FC1);
	for (int j = 0; j < binno.cols; j++) {
		float value = -1;
		for (int i = 0; i < binno.rows; i++) {
			grid_y.at<float>(i, j) = value;
			value += yf;
		}
	}

	Mat kdist = Mat::ones(ys, xs, CV_32FC1) / (xs*ys);
	for (int i = 0; i < kdist.rows; i++) {
		for (int j = 0; j < kdist.cols; j++) {
			kdist.at<float>(i, j) *= sub->mask.at<double>(i, j);
		}
	}

	float value_sum = sum(kdist)[0];

	for (int i = 0; i < kdist.rows; i++) {
		for (int j = 0; j < kdist.cols; j++) {
			kdist.at<float>(i, j) /= value_sum;
		}
	}

	double min_mk;
	minMaxLoc(kdist, &min_mk);

	Mat histogram = Mat::zeros(f, 1, CV_64FC1);
	Mat wsum = Mat::zeros(f, 1, CV_64FC1);
	Mat mu = Mat::zeros(f, 2, CV_64FC1);

	Mat sigma_x = Mat::zeros(f, 1, CV_64FC1);
	Mat sigma_y = Mat::zeros(f, 1, CV_64FC1);

	//Funzione accumarray
	for (int j = 0; j < binno.cols; j++) {
		for (int i = 0; i < binno.rows; i++) {
			histogram.at<double>((int)binno.at<float>(i, j)) += kdist.at<float>(i, j);
			wsum.at<double>((int)binno.at<float>(i, j)) += kdist.at<float>(i, j);
		}
	}

	for (int i = 0; i < wsum.rows; i++) {
		if (wsum.at<double>(i) == 0.0) wsum.at<double>(i) = 1.0;
	}

	//// Creazione del mean vector
	for (int j = 0; j < binno.cols; j++) {
		for (int i = 0; i < binno.rows; i++) {
			mu.at<double>((int)binno.at<float>(i, j), 0) += grid_x.at<float>(i, j)*kdist.at<float>(i, j);
			mu.at<double>((int)binno.at<float>(i, j), 1) += grid_y.at<float>(i, j)*kdist.at<float>(i, j);
		}
	}

	//// Creazione delle matrici di covarianza
	for (int j = 0; j < binno.cols; j++) {
		for (int i = 0; i < binno.rows; i++) {
			sigma_x.at<double>((int)binno.at<float>(i, j)) += pow(grid_x.at<float>(i, j), 2.0)*kdist.at<float>(i, j);
			sigma_y.at<double>((int)binno.at<float>(i, j)) += pow(grid_y.at<float>(i, j), 2.0)*kdist.at<float>(i, j);
		}
	}

	for (int i = 0; i < sigma_x.rows; i++) {
		sigma_x.at<double>(i) /= wsum.at<double>(i);
		sigma_x.at<double>(i) -= pow(mu.at<double>(i, 0) / wsum.at<double>(i), 2.0);
		sigma_x.at<double>(i) += (min_mk - sigma_x.at<double>(i))*(sigma_x.at<double>(i)<min_mk);

		sigma_y.at<double>(i) /= wsum.at<double>(i);
		sigma_y.at<double>(i) -= pow(mu.at<double>(i, 1) / wsum.at<double>(i), 2.0);
		sigma_y.at<double>(i) += (min_mk - sigma_y.at<double>(i))*(sigma_y.at<double>(i)<min_mk);
	}

	////Normalizzazione del mean vector
	for (int i = 0; i < mu.rows; i++) {
		mu.at<double>(i, 0) /= wsum.at<double>(i);
		mu.at<double>(i, 1) /= wsum.at<double>(i);
	}
}

double coder_spatiogram_match(coder_spatiogram* coder1, coder_spatiogram* coder2)
{
	return 0.0;
}

void coder_spatiogram_free(coder_spatiogram* coder) 
{
	spatiogram_free(coder->spatiogram);
	coder->output.release();
	free(coder);
}