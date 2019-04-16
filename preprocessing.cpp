#include "preprocessing.h"
#include "hist.h"

/**
* Metodo che converte l'immagine in scala di grigi usando la metrica "Lightness"
* ovvero img(x,y) = (max(R,G,B) + min(R,G,B))/2
* @param img_in Immagine di input in scala RBG 
* @param img_out Immagine di output in scala di grigi 
*		  (ma ogni pixel è comunque rappresentato da 3 componenti RGB uguali!!)
**/
void convert_lightness(Mat &img_in, Mat&img_out) {
	for (int i = 0; i < img_in.rows; i++) {
		for (int j = 0; j < img_in.cols; j++) {
			Vec3b bgr = img_in.at<Vec3b>(i, j);

			uchar channel_max = max(max(bgr[0], bgr[1]), bgr[2]);
			uchar channel_min = min(min(bgr[0], bgr[1]), bgr[2]);

			uchar gray = (channel_max + channel_min) / 2;
			img_out.at<Vec3b>(i, j) = Vec3b(gray,gray,gray);
		}
	}
}



/**
* Metodo che converte l'immagine in scala di grigi usando la metrica "Average"
* ovvero img(x,y) = (R + G + B)/3
* @param img_in Immagine di input in scala RBG
* @param img_out Immagine di output in scala di grigi
*		  (ma ogni pixel è comunque rappresentato da 3 componenti RGB uguali!!)
**/
void convert_average(Mat &img_in, Mat&img_out) {
	for (int i = 0; i < img_in.rows; i++) {
		for (int j = 0; j < img_in.cols; j++) {
			Vec3b bgr = img_in.at<Vec3b>(i, j);
			uchar gray = (bgr[0] + bgr[1] + bgr[2])/3;
			img_out.at<Vec3b>(i, j) = Vec3b(gray, gray, gray);
		}
	}
}



/**
* Metodo che converte l'immagine in scala di grigi usando la metrica "Luminosity"
* ovvero img(x,y) = (R * peso + G * peso + B * peso)
* @param img_in Immagine di input in scala RBG
* @param img_out Immagine di output in scala di grigi
*		  (ma ogni pixel è comunque rappresentato da 3 componenti RGB uguali!!)
**/
void convert_luminosity(Mat &img_in, Mat &img_out) {
	for (int i = 0; i < img_in.rows; i++) {
		for (int j = 0; j < img_in.cols; j++) {
			Vec3b bgr = img_in.at<Vec3b>(i, j);
			//uchar gray = (uchar)(bgr[0] * 0.07 + bgr[1] * 0.72 + bgr[2] * 0.21);
			uchar gray = (uchar)(bgr[0] * 0.114 + bgr[1] * 0.587 + bgr[2] * 0.299);
			img_out.at<Vec3b>(i, j) = Vec3b(gray, gray, gray);
		}
	}
}


/**
* Metodo "privato" che calcola il valore assunto da ogni pixel utilizzando il filtro di posterizzazione
* @param img_in Immagine di input in scala RBG
* @param row Riga del pixel
* @param column Colonna del pixel
* @param windowSize Dimensione della finestra del filtro
* @return valore da assegnare al pixel
**/
int calc_posterize_pixel(Mat &img_in, int row, int column, int windowSize) {

	//Spiego come funziona il metodo copyTo con Range!
	//Mat img_out = Mat(windowSize, windowSize, CV_8UC1);
	//img_in(Range(a, b), Range(c, d)).copyTo(img_out);
	//La matrice che viene creata prende le righe da "a" a "b-1" e da "c" a "d-q"

	Mat img_out = Mat(windowSize, windowSize, CV_8UC1);
	img_in(Range(row - windowSize / 2, row + windowSize / 2 + 1), Range(column - windowSize / 2, column + windowSize / 2 + 1)).copyTo(img_out);

	Mat hist;
	computeHist(img_out, hist);

	int maxLoc = 0;
	for (int i = 0; i < hist.size().height - 1; i++)
		if (hist.at<float>(i) < hist.at<float>(i + 1))
			maxLoc = i;

	return maxLoc;
}


/**
* Metodo "privato" che applica il filtro di posterizzazione su un singolo canale RGB
* @param img_in Immagine di input in scala di grigi
* @param img_out Immagine di output in scala di grigi
* @param windowSize Dimensione della finestra del filtro
**/
void calc_posterize_gray(Mat &img_in, Mat &img_out, int windowSize) {
	int dim = windowSize / 2;
	for (int i = dim; i < img_in.rows - dim; i++) {
		for (int j = dim; j < img_in.cols - dim; j++) {
			int post = calc_posterize_pixel(img_in, i, j, windowSize);
			img_out.at<uchar>(i, j) = post;
		}
	}
}


/**
* Metodo che applica il filtro di posterizzazione su un' immagine RGB
* @param img_in Immagine di input in scala RGB
* @param img_out Immagine di output in scala RGB
* @param windowSize Dimensione della finestra del filtro
**/
void calc_posterize(Mat &img_in, Mat &img_out, int windowSize) {
	Mat channel_in[3] = { Mat(img_in.rows, img_in.cols, CV_8UC1),
		Mat(img_in.rows, img_in.cols, CV_8UC1),
		Mat(img_in.rows, img_in.cols, CV_8UC1) };

	Mat channel_out[3] = { Mat(img_in.rows, img_in.cols, CV_8UC1),
		Mat(img_in.rows, img_in.cols, CV_8UC1),
		Mat(img_in.rows, img_in.cols, CV_8UC1) };

	split(img_in, channel_in);

	for (int i = 0; i < 3; i++) {
		calc_posterize_gray(channel_in[i], channel_out[i], windowSize);
	}

	merge(channel_out, 3, img_out);
}


/**
* Metodo che applica l'Eliminazione della sclera su un singolo canale RGB
* @param img_in Immagine di input in scala di grigi
* @param img_out Immagine di output in scala di grigi 
* @param threshold Valore di soglia scelto
**/
void delete_sclera_gray(Mat &img_in, Mat &img_out, int threshold) {
	for (int i = 0; i < img_in.rows; i++) {
		for (int j = 0; j < img_in.cols; j++) {
			uchar pixel = img_in.at<uchar>(i, j);
			if (pixel < threshold) img_out.at<uchar>(i, j) = pixel;
			else img_out.at<uchar>(i, j) = 255;
		}
	}
}


/**
* Metodo che applica l'Eliminazione della sclera su un' immagine RGB
* @param img_in Immagine di input in scala RGB
* @param img_out Immagine di output in scala RGB
* @param threshold Valore di soglia scelto
**/
void delete_sclera(Mat &img_in, Mat &img_out, int threshold) {
	Mat channel_in[3] = { Mat(img_in.rows, img_in.cols, CV_8UC1),
		Mat(img_in.rows, img_in.cols, CV_8UC1),
		Mat(img_in.rows, img_in.cols, CV_8UC1) };

	Mat channel_out[3] = { Mat(img_in.rows, img_in.cols, CV_8UC1),
		Mat(img_in.rows, img_in.cols, CV_8UC1),
		Mat(img_in.rows, img_in.cols, CV_8UC1) };

	split(img_in, channel_in);

	delete_sclera_gray(channel_in[0], channel_out[0], threshold);
	delete_sclera_gray(channel_in[1], channel_out[1], threshold);
	delete_sclera_gray(channel_in[2], channel_out[2], threshold);
	merge(channel_out, 3, img_out);

}

/**
* Metodo che applica l'Equalizzazione su un' immagine RGB
* @param img_in Immagine di input in scala RGB
* @param img_out Immagine di output in scala RGB
**/
void calc_equalized(Mat& img_in, Mat& img_out) {
	Mat channel_in[3] = { Mat(img_in.rows, img_in.cols, CV_8UC1),
		Mat(img_in.rows, img_in.cols, CV_8UC1),
		Mat(img_in.rows, img_in.cols, CV_8UC1) };

	split(img_in, channel_in);

	for (int i = 0; i < 3; i++)
		equalizeHist(channel_in[i], channel_in[i]);

	merge(channel_in, 3, img_out);
}