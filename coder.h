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

struct code {
	subject* eye;						//Puntatore al subject a cui il code è riferito
	coder_LBP* code_lbp;				//Puntatore al codificatore per LBP
	coder_blob* code_blob;				//Puntatore al codificatore per Blob
	coder_spatiogram* code_spatiogram;	//Puntatore al codificatore per Spatiogram
};

code* code_create(subject* eye);
void code_init();
void code_encode(code* code);
double code_match(code* code1, code* code2);
void code_free(code* code);
