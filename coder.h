#pragma once
#include <opencv2/opencv.hpp>
#include <opencv/cv.h>
#include <iostream>
#include "subject.h"
#include "coder_lbp.h"
#include "coder_blob.h"

using namespace std;
using namespace cv;

struct code {
	subject* eye;
	coder_LBP* code_lbp;
	coder_blob* code_blob;
};

code* code_create(subject* eye);
void code_init();
void code_encode(code* code);
double code_match(code* code1, code* code2);
void code_free(code* code);
