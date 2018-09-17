#include "coder_spatiogram.h"
#include "coder.h"

#define PI 3.1415

spatiogram* spatiogram_create() {
	spatiogram* spatio = (spatiogram*)calloc(1, sizeof(spatiogram));
	spatio->histogram = Mat::zeros(256, 1, CV_64FC1);
	spatio->mu = Mat::zeros(256, 2, CV_64FC1);
	spatio->sigma_x = Mat::zeros(256, 1, CV_64FC1);
	spatio->sigma_y = Mat::zeros(256, 1, CV_64FC1);
	return spatio;
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
			kdist.at<float>(i, j) *= (double)sub->mask.at<uchar>(i, j);
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

	Mat wsum = Mat::zeros(f, 1, CV_64FC1);

	//Funzione accumarray
	for (int j = 0; j < binno.cols; j++) {
		for (int i = 0; i < binno.rows; i++) {
			coder->spatiogram->histogram.at<double>((int)binno.at<float>(i, j)) += kdist.at<float>(i, j);
			wsum.at<double>((int)binno.at<float>(i, j)) += kdist.at<float>(i, j);
		}
	}

	//cout << histogram << endl;

	for (int i = 0; i < wsum.rows; i++) {
		if (wsum.at<double>(i) == 0.0) wsum.at<double>(i) = 1.0;
	}

	//cout << wsum << endl;

	//// Creazione del mean vector
	for (int j = 0; j < binno.cols; j++) {
		for (int i = 0; i < binno.rows; i++) {
			coder->spatiogram->mu.at<double>((int)binno.at<float>(i, j), 0) += grid_x.at<float>(i, j)*kdist.at<float>(i, j);
			coder->spatiogram->mu.at<double>((int)binno.at<float>(i, j), 1) += grid_y.at<float>(i, j)*kdist.at<float>(i, j);
		}
	}

	//cout << mu;

	//// Creazione delle matrici di covarianza
	for (int j = 0; j < binno.cols; j++) {
		for (int i = 0; i < binno.rows; i++) {
			coder->spatiogram->sigma_x.at<double>((int)binno.at<float>(i, j)) += pow(grid_x.at<float>(i, j), 2.0)*kdist.at<float>(i, j);
			coder->spatiogram->sigma_y.at<double>((int)binno.at<float>(i, j)) += pow(grid_y.at<float>(i, j), 2.0)*kdist.at<float>(i, j);
		}
	}

	for (int i = 0; i < coder->spatiogram->sigma_x.rows; i++) {
		coder->spatiogram->sigma_x.at<double>(i) /= wsum.at<double>(i);
		coder->spatiogram->sigma_x.at<double>(i) -= pow(coder->spatiogram->mu.at<double>(i, 0) / wsum.at<double>(i), 2.0);
		coder->spatiogram->sigma_x.at<double>(i) += (min_mk - coder->spatiogram->sigma_x.at<double>(i))*(coder->spatiogram->sigma_x.at<double>(i)<min_mk);

		coder->spatiogram->sigma_y.at<double>(i) /= wsum.at<double>(i);
		coder->spatiogram->sigma_y.at<double>(i) -= pow(coder->spatiogram->mu.at<double>(i, 1) / wsum.at<double>(i), 2.0);
		coder->spatiogram->sigma_y.at<double>(i) += (min_mk - coder->spatiogram->sigma_y.at<double>(i))*(coder->spatiogram->sigma_y.at<double>(i)<min_mk);
	}

	////Normalizzazione del mean vector
	for (int i = 0; i < coder->spatiogram->mu.rows; i++) {
		coder->spatiogram->mu.at<double>(i, 0) /= wsum.at<double>(i);
		coder->spatiogram->mu.at<double>(i, 1) /= wsum.at<double>(i);
	}
}

double coder_spatiogram_match(coder_spatiogram* coder1, coder_spatiogram* coder2)
{
	Mat qx = Mat::zeros(coder1->spatiogram->sigma_x.size(), CV_64FC1);
	Mat qy = Mat::zeros(coder1->spatiogram->sigma_x.size(), CV_64FC1);
	Mat q = Mat::zeros(coder1->spatiogram->sigma_x.size(), CV_64FC1);

	double C = 2 * sqrt(2 * PI);
	double C2 = 1 / (2 * PI);

	for (int i = 0; i < coder1->spatiogram->sigma_x.rows; i++) {
		qx.at<double>(i) = coder1->spatiogram->sigma_x.at<double>(i) + coder2->spatiogram->sigma_x.at<double>(i);
		qy.at<double>(i) = coder1->spatiogram->sigma_y.at<double>(i) + coder2->spatiogram->sigma_y.at<double>(i);

		q.at<double>(i) = C * pow((qx.at<double>(i) * qy.at<double>(i)), 1 / 4.0);
	}

	Mat sigmai_x = Mat::zeros(coder1->spatiogram->sigma_x.size(), CV_64FC1);
	Mat sigmai_y = Mat::zeros(coder1->spatiogram->sigma_x.size(), CV_64FC1);

	for (int i = 0; i < coder2->spatiogram->sigma_x.rows; i++) {
		sigmai_x.at<double>(i) = 1.0 / (1.0 / (coder1->spatiogram->sigma_x.at<double>(i)
			+ (coder1->spatiogram->sigma_x.at<double>(i) == 0)) + 1.0 /
			(coder2->spatiogram->sigma_x.at<double>(i) + (coder2->spatiogram->sigma_x.at<double>(i) == 0)));

		sigmai_y.at<double>(i) = 1.0 / (1.0 / (coder1->spatiogram->sigma_y.at<double>(i)
			+ (coder1->spatiogram->sigma_y.at<double>(i) == 0)) + 1.0 /
			(coder2->spatiogram->sigma_y.at<double>(i) + (coder2->spatiogram->sigma_y.at<double>(i) == 0)));
	}

	Mat Q = Mat::zeros(coder1->spatiogram->sigma_x.size(), CV_64FC1);
	for (int i = 0; i < coder1->spatiogram->sigma_x.rows; i++) {
		Q.at<double>(i) = C * pow(sigmai_x.at<double>(i)*sigmai_y.at<double>(i), 1 / 4.0);
	}

	Mat x = Mat::zeros(coder1->spatiogram->mu.rows, 1, CV_64FC1);
	Mat y = Mat::zeros(coder1->spatiogram->mu.rows, 1, CV_64FC1);

	for (int i = 0; i < coder1->spatiogram->mu.rows; i++) {
		x.at<double>(i) = coder1->spatiogram->mu.at<double>(i, 0) - coder2->spatiogram->mu.at<double>(i, 0);
		y.at<double>(i) = coder1->spatiogram->mu.at<double>(i, 1) - coder2->spatiogram->mu.at<double>(i, 1);
	}

	for (int i = 0; i < qx.rows; i++) {
		//uso qx e qy come i sigmax del codice originale
		qx.at<double>(i) *= 2.0;
		qy.at<double>(i) *= 2.0;
	}

	Mat isigma_x = Mat::zeros(coder1->spatiogram->sigma_x.size(), CV_64FC1);
	Mat isigma_y = Mat::zeros(coder1->spatiogram->sigma_x.size(), CV_64FC1);
	for (int i = 0; i < qx.rows; i++) {
		isigma_x.at<double>(i) = 1.0 / (qx.at<double>(i) + (qx.at<double>(i) == 0));
		isigma_y.at<double>(i) = 1.0 / (qy.at<double>(i) + (qy.at<double>(i) == 0));
	}

	Mat detsigmax = Mat::zeros(coder1->spatiogram->sigma_x.size(), CV_64FC1);
	for (int i = 0; i < qx.rows; i++) {
		detsigmax.at<double>(i) = qx.at<double>(i) * qy.at<double>(i);
	}

	Mat z = Mat::zeros(isigma_x.size(), CV_64FC1);
	for (int i = 0; i < z.rows; i++) {
		z.at<double>(i) = C2 / sqrt(detsigmax.at<double>(i))*exp(-0.5 *
			(isigma_x.at<double>(i) *pow(x.at<double>(i), 2.0) +
				isigma_y.at<double>(i) *pow(y.at<double>(i), 2.0)));
	}

	Mat dist = Mat::zeros(z.size(), CV_64FC1);
	for (int i = 0; i < z.rows; i++) {
		dist.at<double>(i) = q.at<double>(i)*Q.at<double>(i)*z.at<double>(i);
	}

	Mat s = Mat::zeros(coder1->spatiogram->histogram.size(), CV_64FC1);
	for (int i = 0; i < coder1->spatiogram->histogram.rows; i++) {
		s.at<double>(i) = sqrt(coder1->spatiogram->histogram.at<double>(i))*
			sqrt(coder2->spatiogram->histogram.at<double>(i))*
			dist.at<double>(i);
	}

	double sum_s = 0.0;
	for (int i = 0; i < s.rows; i++) {
		sum_s += isnan(s.at<double>(i)) ? 0 : s.at<double>(i);
	}

	return sum_s;
}

void coder_spatiogram_free(coder_spatiogram* coder) 
{
	spatiogram_free(coder->spatiogram);
	coder->output.release();
	free(coder);
}