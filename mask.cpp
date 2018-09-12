#include "mask.h"

void adjust_mask_single(Mat input, Mat old_mask, Mat new_mask) {

	uchar thresh_black = 180;
	old_mask.copyTo(new_mask);

	for (int i = 0; i < input.rows; ++i) {
		for (int j = 0; j < input.cols; ++j) {
			uchar rgba = input.at<uchar>(i, j);
			if (old_mask.at<uchar>(i, j) == 255) continue;
			if (rgba < thresh_black)
				new_mask.at<uchar>(i, j) = 0; //Se è troppo nero non lo considero

			/*else if (rgba > thresh_white)
				new_mask.at<uchar>(i, j) = 0;	//Se è troppo bianco non lo considero*/

			else new_mask.at<uchar>(i, j) = 255;
			//printf("R:%d C:%d -> %d\n", i, j, rgba);
		}
	}

}


void adjust_mask(Mat input, Mat old_mask, Mat new_mask) {

	Vec3i average = (0,0,0);
	int n_average = 0;

	for (int i = 0; i < input.rows; ++i) {
		for (int j = 0; j < input.cols; ++j) {
			if (old_mask.at<uchar>(i, j) == 0) continue;

			Vec3b rgba = input.at<Vec3b>(i, j);
			average += rgba;
			n_average++;
		}
	}

	average /= n_average;
	printf("Average color: r->%d, g->%d, b->%d\n", average[2], average[1], average[0]);

	Vec3b thresh_black = (70, 70, 70);
	Vec3b thresh_white = (200,200,200);

	for (int i = 0; i < input.rows; ++i) {
		for (int j = 0; j < input.cols; ++j) {
			Vec3b rgba = input.at<Vec3b>(i, j);

			if (rgba[0] < thresh_black[0] && rgba[1] < thresh_black[1] && rgba[2] < thresh_black[2])
				new_mask.at<uchar>(i, j) = 0; //Se è troppo nero non lo considero

			else if ((rgba[0] > thresh_white[0] && rgba[1] > thresh_white[1] && rgba[2] > thresh_white[2]))
				new_mask.at<uchar>(i, j) = 0;	//Se è troppo bianco non lo considero

			else new_mask.at<uchar>(i, j) = 255;
			//printf("R:%d C:%d -> %d\n", i, j, rgba);
		}
	}
			
}
