#include "log.h"
#include "coder_blob.h"

//Conoscendo il sigma del filtro gaussiano calcola la dimensione del kernel adatta
int get_kernel_side(double sigma)
{
	int side = (int)round(4.783 * sigma + 7.044);
	if (side % 2 == 0) side+= 1;
	return side;
}

//Applica il Laplaciano del gaussiamo al pixel (x,y) usando sigma per il filtro gaussiano
double LoG(double x, double y, double sigma)
{
	return (1 / (2 * 3.14*sigma))
		*((x*x + y * y - 4 * sigma) / (4 * pow(sigma, 2)))
		*exp(-((x*x + y * y) / (4 * sigma)));
}

//Crea il kernel per l'applicazione del laplaciano del gaussiano
void create_kernel_LoG(Mat &mat, double sigma)
{
	for (int i = 0; i < mat.cols; i++) {
		for (int j = 0; j < mat.rows; j++) {
			mat.at<float>(i, j) = (float)LoG(i - mat.rows / 2, j - mat.cols / 2, sigma);
		}
	}
}

//Si occupa di binarizzare il risultato. Se è negativo si mette zero altrimenti 255.
void binarize_LoG(Mat &logImage, Mat &binImage)
{
	for (int i = 0; i < logImage.rows; i++) {
		for (int j = 0; j < logImage.cols; j++) {
			if (logImage.at<float>(i, j) <= 0) binImage.at<uchar>(i, j) = 0;
			else binImage.at<uchar>(i, j) = 255;
		}
	}
}

//Effettua il merge delle maschere ottenute applicando il laplaciano del gaussiano 
void merge_LoG(Mat* log_imgs, Mat &img_out)
{
	for (int i = 0; i < MAX_NUM_KERNEL; i++)
		threshold(log_imgs[i], log_imgs[i], 0, 0, THRESH_TOZERO);

	for (int i = 0; i < img_out.rows; i++) {
		for (int j = 0; j < img_out.cols; j++) {

			float max = 0.0f;
			for (int k = 0; k < MAX_NUM_KERNEL; k++) {
				if (max < log_imgs[k].at<float>(i, j))
					max = log_imgs[k].at<float>(i, j);
			}

			img_out.at<float>(i, j) = max;
		}
	}
}