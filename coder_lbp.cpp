#include "coder_lbp.h"
#include "LBP.h"
#include "hist.h"


/**
* Metodo che crea e inizializza un coder per Local Binary Pattern
* @return puntatore al coder
**/
coder_LBP* coder_lbp_create() {
	coder_LBP* coder = (coder_LBP*) calloc(1, sizeof(coder_LBP));
	if (coder == NULL) return NULL;

	coder->out = NULL;

	for (int i = 0; i < NUM_ZONE; i++) {
		coder->histogram[i] = NULL;
	}

	return coder;
}


/**
* Metodo che carica dalla memoria l'immmagine generata precedentemente
* dall'applicazione dell'operatore LBP su un soggetto e genera gli istrogrammi
* delle fasce.
* @params subject soggetto la cui immagine deve essere caricata
* @params coder coder LBP su cui l'immagine deve essere salvata
**/
void coder_lbp_load(string subject_name, coder_LBP* coder) {
	coder->out = imread("dataset/" + subject_name.substr(9, 3)
		+ "/output/" + subject_name.substr(14) + ".output.png", IMREAD_GRAYSCALE);

	int sub_rows = coder->out.rows / NUM_ZONE;

	for (int k = 0; k < NUM_ZONE; k++) {
		coder->histogram[k] = Mat::zeros(256, 1, CV_32FC1);
		computeHist(coder->out(Range(sub_rows*k, sub_rows*(k + 1)), Range(0, coder->out.cols)), coder->histogram[k]);
	}
}


/**
* Metodo che effettua la codifica tramite l'operatore LBP
* @param sub soggetto a cui la codifica è riferita
* @param coder puntatore al coder LBP in cui vengono salvare le informazioni
**/
void coder_lbp_encode(subject* sub, coder_LBP* coder) {
	coder->out = Mat::zeros(sub->input.size(), CV_8UC1);

	for (int i = 0; i < NUM_ZONE; i++)
		coder->histogram[i] = Mat::zeros(256, 1, CV_32FC1);

	calc_standard_lbp(sub, coder);
}


/**
* Metodo che effettua il match tra due codifiche lbp utilizzando la distanza tra gli istogrammi
* @param sub1 Soggetto_1 in input
* @param coder1 Codifica_1 relativa al Soggetto_1 in input
* @param sub2 Soggetto_2 in input
* @param coder2 Codifica_2 relativa al Soggetto_2 in input
* @return distanza tra le due codifiche in input
**/
double coder_lbp_match(subject* sub1, coder_LBP* coder1, subject* sub2, coder_LBP* coder2) {

	int sub_rows = sub1->input.rows / NUM_ZONE;

	Mat norm_hist1 = Mat::zeros(256, 1, CV_32FC1);
	Mat norm_hist2 = Mat::zeros(256, 1, CV_32FC1);
	Mat mask1 = Mat::zeros(sub_rows, sub1->input.cols, CV_8UC1);
	Mat mask2 = Mat::zeros(sub_rows, sub2->input.cols, CV_8UC1);

	double similarity = 0.0;

	for (int i = 0; i < NUM_ZONE; i++) {
		normalize(coder1->histogram[i], norm_hist1);
		normalize(coder2->histogram[i], norm_hist2);

		sub1->mask(Range(sub_rows*i, sub_rows*(i + 1)), Range(0, sub1->input.cols)).copyTo(mask1);
		sub2->mask(Range(sub_rows*i, sub_rows*(i + 1)), Range(0, sub2->input.cols)).copyTo(mask2);

		similarity += compareHist(norm_hist1, norm_hist2, CV_COMP_BHATTACHARYYA)
			* (1.0 - (((2*mask1.rows*mask1.cols) - (countNonZero(mask1) + countNonZero(mask2))) 
				/ (double)(2 * mask1.rows*mask1.cols)));
	}

	return similarity / NUM_ZONE;
}


/**
* Metodo che rilascia le risorse impegnate dal coder blob
* @param coder Puntatore al coder blob da rilasciare
**/
void coder_lbp_free(coder_LBP* coder) {
	coder->out.release();
	for (int i = 0; i < NUM_ZONE; i++) {
		coder->histogram[i].release();
	}
	free(coder);
}