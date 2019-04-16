#pragma once
#include<opencv2/opencv.hpp>
#include<opencv/cv.h>
#include<iostream>
#include "subject.h"

using namespace std;
using namespace cv;

struct spatiogram {
	Mat histogram;	//Histogram dell'immagine
	Mat mu;			//Mean vector delle coordinare del pixel
	Mat sigma_x;	//Matrice di covarianza della coordinata x dei pixel
	Mat sigma_y;	//Matrice di convarianza della coordinata y dei pixel
					//NOTA: le sigma_x e sigma_y potrebbero essere unite in una struttura simile a quella usata per mu!
};

struct coder_spatiogram {
	Mat output;				//Per ora non utilizzata, potrebbe contenere l'immagine Spatiogram ottenuta
	spatiogram* spatiogram; //Puntatore alla struttura che mantiene le informazioni sullo Spatiogram calcolato
};

spatiogram* spatiogram_create();
void spatiogram_free(spatiogram* spatio);
coder_spatiogram* coder_spatiogram_create();
void coder_spatiogram_encode(subject* sub, coder_spatiogram* coder);
double coder_spatiogram_match(coder_spatiogram* coder1, coder_spatiogram* coder2);
void coder_spatiogram_free(coder_spatiogram* coder);
