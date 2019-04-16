#include "hist.h"


/**
* Metodo che nasconde l'implementazione del calcolo degli istrogrammi della libreria OpenCV.
* @params Mat img_in Immagine di cui calcolare l'istogramma
* @params Mat& histogram Matrice su cui viene salvato l'istogramma appena calcolato
*			generalmente viene inizializzata come ...	
**/
void computeHist(Mat img_in, Mat& histogram) {
	int histSize = 256; //from 0 to 255
	float range[] = { 0, 256 }; //the upper boundary is exclusive
	const float* histRange = { range };
	calcHist(&img_in, 1, 0, Mat(), histogram, 1, &histSize, &histRange);
}



/**
* Metodo che crea e salva una rappresentazione grafica dell'istogramma histogram
* @params histogram Matrice che contiene l'istogramma da visualizzare
**/
void showHist(Mat histogram) {
	// I parametri sotto riguardano la dimensione dell'immagine in output
	int hist_w = 512; 
	int hist_h = 400;
	int histSize = 256;
	int bin_w = cvRound((double)hist_w / histSize);
	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

	normalize(histogram, histogram, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	//Rappresentazione a colonne
	/*for (int i = 0; i < histogram.rows; i++) {
		line(histImage, Point(2 * bin_w*(i) + 3, hist_h),
			Point(2 * bin_w*(i) + 3, hist_h - cvRound(histogram.at<float>(i))),
			Scalar(0, 0, 255), 2, 8, 0);
	}*/

	//Rappresentazione a linea spezzata
	for (int i = 1; i < histSize; i++) {
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(histogram.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(histogram.at<float>(i))),
			Scalar(0, 255, 0), 2, 8, 0);
	}

	//Permette di visualizzare e salvare l'istogramma nel file passato alla funzione imwrite
	namedWindow("Histogram", WINDOW_AUTOSIZE);
	imshow("Histogram", histImage);
	//waitKey(0) //se si vuole visualizzare l'istogramma decommentare questa riga!!
	imwrite("histogram_2.jpg", histImage);
}
