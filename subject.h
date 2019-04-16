#pragma once
#include <opencv2/opencv.hpp>
#include <opencv/cv.h>
#include <iostream>

using namespace std;
using namespace cv;

struct subject {
	Mat input; //Immagine dell'iride (normalizzata o no in base a dove è usato)
	Mat mask; //Maschera binaria relativa all'immagine in input 
};

subject* subject_create();
void subject_free(subject* eye);