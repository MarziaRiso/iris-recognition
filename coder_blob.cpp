#include "log.h"
#include "coder_blob.h"
#include "coder.h"

Mat kernel[MAX_NUM_KERNEL];
int sigma[MAX_NUM_KERNEL] = { 1, 2, 4, 8 };


/**
* Metodo che crea e inizializza un coder per i blob
* @return puntatore al coder
**/
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


/**
* Metodo che inizializza i kernel di convoluzione dell'operatore LoG
**/
void coder_blob_init()
{
	for (int i = 0; i < MAX_NUM_KERNEL; i++) {
		int kernel_side = get_kernel_side(sigma[i]);
		kernel[i] = Mat(kernel_side, kernel_side, CV_32FC1);
		create_kernel_LoG(kernel[i], sigma[i]);
	}
}


/**
* Metodo che carica dalla memoria l'immmagine generata precedentemente
* dall'applicazione dell'operatore LoG su un soggetto
* @params subject soggetto la cui immagine deve essere caricata
* @params coder coder blob su cui l'immagine deve essere salvata
**/
void coder_blob_load(string subject, coder_blob* coder) {
	coder->log_bin_merge = imread("dataset/" + subject.substr(9, 3)
		+ "/blob/" + subject.substr(14) +
		".log.bin.merge.png", IMREAD_GRAYSCALE);
}


/**
* Metodo che effettua la codifica dei blob tramite l'operatore LoG
* @param sub soggetto a cui la codifica è riferita
* @param coder puntatore al coder blob in cui vengono salvare le informazioni
**/
void coder_blob_encode(subject* sub, coder_blob* coder)
{
	for (int i = 0; i < MAX_NUM_KERNEL; i++)
	{
		coder->log_imgs[i] = Mat(sub->input.rows, sub->input.cols, CV_32FC1);
		coder->log_bin_imgs[i] = Mat(sub->input.rows, sub->input.cols, CV_8UC1);

		filter2D(sub->input, coder->log_imgs[i], CV_32FC1, kernel[i]);

		coder->log_imgs[i].convertTo(coder->log_imgs[i], -1, sigma[i], 0);
		imwrite("Blob_" + to_string(i) + ".png", coder->log_imgs[i]);

		binarize_LoG(coder->log_imgs[i], coder->log_bin_imgs[i]);
		imwrite("Blob_bin" + to_string(i) + ".png", coder->log_bin_imgs[i]);
	}

	coder->log_merge = Mat(sub->input.rows, sub->input.cols, CV_32FC1);
	coder->log_bin_merge = Mat(sub->input.rows, sub->input.cols, CV_8UC1);

	merge_LoG(coder->log_imgs, coder->log_merge);
	binarize_LoG(coder->log_merge, coder->log_bin_merge);
}


/**
* Metodo che effettua il match tra due codifiche blob utilizzando la distanza di Hamming
* @param sub1 Soggetto_1 in input
* @param coder1 Codifica_1 relativa al Soggetto_1 in input
* @param sub2 Soggetto_2 in input
* @param coder2 Codifica_2 relativa al Soggetto_2 in input
* @return distanza tra le due codifiche in input
**/
double coder_blob_match(subject* sub1, coder_blob* coder1, subject* sub2, coder_blob* coder2)
{
	return shifted_hamming_distance(coder1->log_bin_merge, sub1->mask,
		coder2->log_bin_merge, sub2->mask, SHIFT);
}


/**
* Metodo "privato" che effettua lo shift circolare dell'immagine fornita in input
* @param img Immagine da shiftare
* @param shifted_image Immagine shiftata Codifica_1 relativa al Soggetto_1 in input
* @param shift numero di pixel di cui effettuare lo shift 
**/
void shift_image(Mat img, Mat shifted_image, int shift)
{
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			shifted_image.at<uchar>(i, MOD(j+shift,img.cols)) = img.at<uchar>(i, j);
		}
	}
}


/**
* Metodo "privato" che calcola la distanza di Hamming tra due codifiche binarie
* (nel nosto caso codifiche di blob) tenendo conto delle maschere binarie
* @param img1 Immagine_1 in input
* @param mask1 Maschera binaria relativa all'Immagine_1
* @param img2 Immagine_2 in input
* @param mask2 Maschera binaria relativa all'Immagine_2
* @return distanza tra le due immagini
**/
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


/**
* Metodo "privato" che calcola la distanza di Hamming shiftata tra due codifiche binarie
* (nel nosto caso codifiche di blob) tenendo conto delle maschere binarie
* @param img1 Immagine_1 in input
* @param mask1 Maschera binaria relativa all'Immagine_1
* @param img2 Immagine_2 in input
* @param mask2 Maschera binaria relativa all'Immagine_2
* @param shift numero di pixel di cui effettuare lo shift (a destra e a sinistra)
* @return distanza tra le due immagini
**/
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


/**
* Metodo che rilascia le risorse impegnate dal coder blob
* @param coder Puntatore al coder blob da rilasciare
**/
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