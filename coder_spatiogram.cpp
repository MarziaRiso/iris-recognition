#include "coder_spatiogram.h"

coder_spatiogram* coder_spatiogram_create() 
{
	coder_spatiogram* coder = (coder_spatiogram*)calloc(1, sizeof(coder_spatiogram));
	if (coder == NULL) return NULL;

	coder->input = NULL;
	coder->mask = NULL;
	coder->output = NULL;
	return coder;
}

void coder_spatiogram_encode(coder_spatiogram* coder) 
{
	for (int i = 0; i < coder->input.rows; i++) {
		for (int j = 0; j < coder->input.cols; j++) {
			uchar color = coder->input.at<uchar>(i, j);
			coder->spatiogram[color].histogram += 1;
			coder->spatiogram[color].mean_vector += color;
			coder->spatiogram[color].covariance_matrix += (int)pow(color, 2);
		}
	}
}

void coder_spatiogram_free(coder_spatiogram* coder) 
{
	coder->input.release();
	coder->mask.release();
	coder->output.release();

	free(coder);
}