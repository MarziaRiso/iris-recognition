#include "log.h"
#include "coder_blob.h"

/**
* Metodo che calcola la dimensione adeguata del kernel del LoG
* @param sigma è il valore della sigma del filtro gaussiano considerato
* @return intero che rappresenta la dimensione del kernel (sempre dispari!)
**/
int get_kernel_side(double sigma) {
	int side = (int)round(4.783 * sigma + 7.044);
	if (side % 2 == 0) side+= 1;
	return side;
}



/**
* Metodo che calcola il valore del LoG per il pixel (x,y) usando sigma per il filtro gaussiano
* @param x,y coordinate del pixel
* @param sigma del filtro gaussiano
* @return valore del LoG associato al pixel (x,y)
**/
double LoG(double x, double y, double sigma) {
	return (1 / (2 * 3.14*sigma))
		*((x*x + y * y - 4 * sigma) / (4 * pow(sigma, 2)))
		*exp(-((x*x + y * y) / (4 * sigma)));
}



/**
* Metodo che crea e calcola il kernel per l'applicazione del LoG
* @param mat matrice che rappresenta un kernel
* @param sigma valore della sigma del filtro gaussiano
**/
void create_kernel_LoG(Mat &mat, double sigma) {
	for (int i = 0; i < mat.cols; i++) {
		for (int j = 0; j < mat.rows; j++) {
			mat.at<float>(i, j) = (float)LoG(i - mat.rows / 2, j - mat.cols / 2, sigma);
		}
	}
}



/**
* Metodo che si occupa di binarizzare il risultato. Se il valore di un pixel è negativo si mette zero altrimenti 255.
* @param logImage Immagine a cui è stato applicato il filtro LoG
* @param binImage Immagine che conterrà la binarizzazione dell'immagine di input
**/
void binarize_LoG(Mat &logImage, Mat &binImage) {
	for (int i = 0; i < logImage.rows; i++) {
		for (int j = 0; j < logImage.cols; j++) {
			if (logImage.at<float>(i, j) <= 0) binImage.at<uchar>(i, j) = 0;
			else binImage.at<uchar>(i, j) = 255;
		}
	}
}



/**
* Metodo che effettua il merge delle immagini a cui è stato applicato il LoG
* @param log_imgs puntatore all'array che contiene le immagini a cui è stato applicato il LoG
* @param img_out Immagine che conterrà l'immagine di output
**/
void merge_LoG(Mat* log_imgs, Mat &img_out) {
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