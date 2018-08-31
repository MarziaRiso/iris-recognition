#include "log.h"
#include "coder_blob.h"
#include "coder.h"

Mat kernel[MAX_NUM_KERNEL];
int sigma[MAX_NUM_KERNEL] = { 1, 2, 4, 8 };

coder_blob* coder_blob_create() 
{
	coder_blob* coder = (coder_blob*)calloc(1, sizeof(coder_blob));
	if (coder == NULL) return NULL;

	coder->log_bin_merge = NULL;
	coder->log_merge = NULL;

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

void coder_blob_encode(subject* sub, coder_blob* coder)
{
	for (int i = 0; i < MAX_NUM_KERNEL; i++)
	{
		coder->log_imgs[i] = Mat(sub->input.rows, sub->input.cols, CV_32FC1);
		coder->log_bin_imgs[i] = Mat(sub->input.rows, sub->input.cols, CV_8UC1);
		
		filter2D(sub->input, coder->log_imgs[i], CV_32FC1, kernel[i]);

		coder->log_imgs[i].convertTo(coder->log_imgs[i], -1, sigma[i], 0);
		binarize_LoG(coder->log_imgs[i], coder->log_bin_imgs[i]);
	}

	coder->log_merge = Mat(sub->input.rows, sub->input.cols, CV_32FC1);
	coder->log_bin_merge = Mat(sub->input.rows, sub->input.cols, CV_8UC1);

	merge_LoG(coder->log_imgs, coder->log_merge);

	binarize_LoG(coder->log_merge, coder->log_bin_merge);
}

double coder_blob_match(subject* sub1, coder_blob* coder1, subject* sub2, coder_blob* coder2)
{
	return shifted_hamming_distance(coder1->log_bin_merge, sub1->mask,
		coder2->log_bin_merge, sub2->mask, SHIFT);
}

void shift_image(Mat img, Mat shifted_image, int shift)
{
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			shifted_image.at<uchar>(i, MOD(j+shift,img.cols)) = img.at<uchar>(i, j);
		}
	}
}

double hamming_distance(Mat img1, Mat mask1, Mat img2, Mat mask2)
{
	Mat xor_img = Mat(img1.size(), CV_8UC1);
	Mat or_mask = Mat(mask1.size(), CV_8UC1);
	Mat not_or_mask = Mat(mask1.size(), CV_8UC1);
	Mat and_img = Mat(img1.size(), CV_8UC1);

	bitwise_xor(img1, img2, xor_img);
	bitwise_or(mask1, mask2, or_mask);
	bitwise_not(or_mask, not_or_mask);
	bitwise_and(xor_img, not_or_mask, and_img);

	return countNonZero(and_img) / ((double)(img1.cols * img1.rows - countNonZero(or_mask)));
}

double shifted_hamming_distance(Mat img1, Mat mask1, Mat img2, Mat mask2, int shift)
{
	double min_hd = 1.0;
	double cur_hd;

	Mat shifted_img = Mat(img2.size(), CV_8UC1);
	Mat shifted_mask = Mat(mask2.size(), CV_8UC1);

	for (int i = -shift; i <= shift; i++) {

		shift_image(img2, shifted_img, i);
		shift_image(mask2, shifted_mask, i);

		cur_hd = hamming_distance(img1, mask1, shifted_img, shifted_mask);
		min_hd = (cur_hd < min_hd) ? cur_hd : min_hd;
	}
	return min_hd;
}


void coder_blob_free(coder_blob* coder)
{
	coder->log_bin_merge.release();
	coder->log_merge.release();

	for (int i = 0; i < MAX_NUM_KERNEL; i++) {
		coder->log_bin_imgs[i].release();
		coder->log_imgs[i].release();
	}

	free(coder);
}