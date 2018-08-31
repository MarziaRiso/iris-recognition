#include "coder_spatiogram.h"
#include "coder.h"

spatiogram* spatiogram_create()
{
	spatiogram* spatio = (spatiogram*)calloc(1, sizeof(spatiogram));
	spatio->histogram = Mat::zeros(256, 1, CV_32FC1);
	spatio->mean_vector = Mat::zeros(256, 1, CV_32FC1);
	spatio->covariance_matrix = Mat::zeros(256, 1, CV_32FC1);
	return spatio;
}

void spatiogram_free(spatiogram* spatio)
{
	spatio->histogram.release();
	spatio->mean_vector.release();
	spatio->covariance_matrix.release();
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
	for (int i = 0; i < sub->input.rows; i++) {
		for (int j = 0; j < sub->input.cols; j++) {
			uchar color = sub->input.at<uchar>(i, j);
			coder->spatiogram->histogram.at<float>(color,0) += 1;
			coder->spatiogram->mean_vector.at<float>(color,0) += color;
			coder->spatiogram->covariance_matrix.at<float>(color,0) += powf(color, 2);
		}
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