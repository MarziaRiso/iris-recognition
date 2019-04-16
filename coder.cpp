#include "coder.h"

/**
* Metodo che crea e inizializza un code
* @param subject soggetto a cui il code è riferito
* @return puntatore al code
**/
code* code_create(subject* eye) {
	code* coder = (code*)calloc(1, sizeof(code));
	if (coder == NULL) return NULL;

	coder->eye = eye;
	coder->code_lbp = coder_lbp_create();
	coder->code_blob = coder_blob_create();
	coder->code_spatiogram = coder_spatiogram_create();

	return coder;
}


/**
* Metodo che inizializza i kernel della codifica blob
**/
void code_init() {
	coder_blob_init();
}


/**
* Metodo che effettua la codifica chiamando i metodi di codifica
* dei singoli operatori
* @param coder puntatore al code in cui vengono salvare le informazioni 
**/
void code_encode(code* coder) {
	coder_lbp_encode(coder->eye, coder->code_lbp);
	coder_blob_encode(coder->eye, coder->code_blob);
	coder_spatiogram_encode(coder->eye, coder->code_spatiogram);
}


/**
* Metodo che effettua il match calcolando la distanza tra due codifiche
* @param coder1 Codifica_1 in input
* @param coder2 Codifica_2 in input
* @return distanza tra le due codifiche in input
**/
double code_match(code* coder1, code* coder2) {
	double lbp_dist = coder_lbp_match(coder1->eye, coder1->code_lbp, coder2->eye, coder2->code_lbp);
	double blob_dist = coder_blob_match(coder1->eye, coder1->code_blob, coder2->eye, coder2->code_blob);
	double spatiogram_dist = 1.0 - coder_spatiogram_match(coder1->code_spatiogram, coder2->code_spatiogram);
	return (lbp_dist + blob_dist + spatiogram_dist) / 3.0;
}


/**
* Metodo che rilascia le risorse impegnate dal coder
* @param coder Puntatore al code da rilasciare
**/
void code_free(code* coder) {
	coder_lbp_free(coder->code_lbp);
	coder_blob_free(coder->code_blob);
	coder_spatiogram_free(coder->code_spatiogram);
	free(coder);
}
