#pragma once
#include <opencv2/opencv.hpp>
#include <opencv/cv.h>
#include <iostream>
#include "subject.h"
#include "coder_lbp.h"
#include "coder_blob.h"
#include "coder_spatiogram.h"

using namespace std;
using namespace cv;

//GALLERY: 79 persone, 10 immagini a persona circa
//quindi in totale 790 immagini 
//considero l'80% per fare il modello e il 20%
//per fare il probe (158 immagini circa, cioè 2 immagini a persona)

struct code {
	subject* eye;
	coder_LBP* code_lbp;
	coder_blob* code_blob;
	coder_spatiogram* code_spatiogram;
};

code* code_create(subject* eye);
void code_init();
void code_encode(code* code);
double code_match(code* code1, code* code2);
void code_free(code* code);
