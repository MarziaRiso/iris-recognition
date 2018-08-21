#include "log.h"
#include "coder_blob.h"

Mat kernel[MAX_NUM_KERNEL];
int sigma[MAX_NUM_KERNEL] = { 1, 2, 4, 8 };

coder_blob* coder_blob_create() 
{
	coder_blob* coder = (coder_blob*)calloc(1, sizeof(coder_blob));
	if (coder == NULL) return NULL;

	coder->input = NULL;
	coder->log_bin_merge = NULL;
	coder->log_merge = NULL;
	coder->mask = NULL;

	for (int i = 0; i < MAX_NUM_KERNEL; i++) {
		coder->log_bin_imgs[i] = NULL;
		coder->log_imgs[i] = NULL;
	}

	return coder;
}


void coder_blob_init()
{
	for (int i = 0; i < MAX_NUM_KERNEL; i++) {
		int kernel_side = get_kernel_side(sigma[i]);
		kernel[i] = Mat(kernel_side, kernel_side, CV_32FC1);
		create_kernel_LoG(kernel[i], sigma[i]);
	}
}

void coder_blob_encode(coder_blob* coder)
{
	for (int i = 0; i < MAX_NUM_KERNEL; i++)
	{
		coder->log_imgs[i] = Mat(coder->input.rows, coder->input.cols, CV_32FC1);
		coder->log_bin_imgs[i] = Mat(coder->input.rows, coder->input.cols, CV_8UC1);
		
		filter2D(coder->input, coder->log_imgs[i], CV_32FC1, kernel[i]);

		coder->log_imgs[i].convertTo(coder->log_imgs[i], -1, sigma[i], 0);
		binarize_LoG(coder->log_imgs[i], coder->log_bin_imgs[i]);
	}

	coder->log_merge = Mat(coder->input.rows, coder->input.cols, CV_32FC1);
	coder->log_bin_merge = Mat(coder->input.rows, coder->input.cols, CV_8UC1);

	merge_LoG(coder->log_imgs, coder->log_merge);

	binarize_LoG(coder->log_merge, coder->log_bin_merge);
}


void coder_blob_free(coder_blob* coder)
{
	coder->input.release();
	coder->log_bin_merge.release();
	coder->log_merge.release();
	coder->mask.release();

	for (int i = 0; i < MAX_NUM_KERNEL; i++) {
		coder->log_bin_imgs[i].release();
		coder->log_imgs[i].release();
	}

	free(coder);
}