#pragma once
#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv/cv.h>
#include <iostream>
#include "subject.h"

using namespace std;
using namespace cv;

#define MAX_NUM_KERNEL 4
#define SHIFT 5
#define MOD(x,y) ((x) >= 0 ? ((x)%(y)) : ((y) + (x)%(y)))

struct coder_blob {
	Mat log_imgs[MAX_NUM_KERNEL];		//Array delle immagini ottenute applicando filtri LoG con diverse scale
	Mat log_bin_imgs[MAX_NUM_KERNEL];	//Array delle immagini ottenute applicando filtri LoG con diverse scale binarizzate
	Mat log_merge;						//Immagine ottenuta dal merge delle immagini LoG
	Mat log_bin_merge;					//Immagine ottenuta dal merge delle immagini LoG binarizzata
};

coder_blob* coder_blob_create();
void coder_blob_init();
void coder_blob_load(string subject, coder_blob* coder);
void coder_blob_encode(subject* sub, coder_blob* coder);
double coder_blob_match(subject* sub1, coder_blob* coder1, subject* sub2, coder_blob* coder2);
void shift_image(Mat img, Mat shifted_image, int shift);
double hamming_distance(Mat img1, Mat mask1, Mat img2, Mat mask2);
double shifted_hamming_distance(Mat img1, Mat mask1, Mat img2, Mat mask2, int shift);
void coder_blob_free(coder_blob* coder);

