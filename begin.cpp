#include <opencv2/opencv.hpp>
#include <opencv/cv.h>
#include <iostream>
#include <string>
#include <time.h>
#include <thread>
#include <cmath>
#include "log.h"
#include "coder.h"
#include "coder_lbp.h"
#include "coder_blob.h"
#include "coder_spatiogram.h"
#include "mask.h"
#include "hist.h"
#include "preprocessing.h"
#include "subject.h"

#define PI 3.1415
#define MAX_LINE 256
#define NUM_THREADS 8

using namespace std;
using namespace cv;

vector<string> probe_names;
vector<string> gallery_names;
Mat results;

std::vector<std::string> string_split(const std::string& s, char delimiter)
{
	std::vector<std::string> tokens;
	std::string token;
	std::istringstream tokenStream(s);
	while (std::getline(tokenStream, token, delimiter)) {
		tokens.push_back(token);
	}
	return tokens;
}

void create_file()
{
	const char* gallery_file = "gallery_file1.txt";
	const char* probe_file = "probe_file1.txt";
	const char* output_file = "probe_final.txt";

	FILE* gallery = fopen(probe_file, "r");
	FILE* output = fopen(output_file, "w");

	char line[MAX_LINE];
	int gallery_size = atoi(fgets(line, MAX_LINE - 1, gallery));
	string file;

	for (int i = 0; i < gallery_size; i++) {
		file = string(fgets(line, MAX_LINE - 1, gallery));
		std::cout << file;
		if (i % 2 != 0) continue;
			//fprintf(output,"%s\n", ("gallery\\\\" + file.substr(4, 3) + 
			//	"\\\\" + file.substr(0, 11)).c_str());
			std::fprintf(output, "%s\n", ("probe\\\\" + file.substr(0, 11)).c_str());
	}

}

void create_gallery()
{
	const char* gallery_file = "gallery_final.txt";
	FILE* gallery = fopen(gallery_file, "r");

	char line[MAX_LINE];
	int gallery_size = atoi(fgets(line, MAX_LINE - 1, gallery));

	vector<string> gallery_names;
	for (int i = 0; i < gallery_size; i++)
		gallery_names.push_back(string(strtok(fgets(line, MAX_LINE - 1, gallery),"\n")));

	subject* gallery_sub = subject_create();
	code* gallery_code;

	for (int i = 0; i < gallery_size; i++) {
		std::cout << gallery_names.at(i).substr(9,3) << endl;
		gallery_sub->input = imread(gallery_names.at(i) + ".iris.norm.png", IMREAD_GRAYSCALE);
		gallery_sub->mask = imread(gallery_names.at(i) + ".defects.norm.png", IMREAD_GRAYSCALE);
		gallery_code = code_create(gallery_sub);
		
		code_init();
		
		code_encode(gallery_code);

		imwrite("dataset/" + gallery_names.at(i).substr(9, 3)
			+ "/output/" + gallery_names.at(i).substr(14) + ".output.png", gallery_code->code_lbp->out);

		imwrite("dataset/" + gallery_names.at(i).substr(9, 3)
		+ "/blob/" + gallery_names.at(i).substr(14) +
		".log.bin.merge.png", gallery_code->code_blob->log_bin_merge);

		code_free(gallery_code);
	}

	std::cin.get();
}

vector<string> load_matrix(const string path) {
	ifstream input;
	input.open(path);

	string line;
	getline(input, line); // scarto la riga con tutte i nomi delle immagini della gallery
	vector<string> gallery_codes = string_split(line, ';');
	gallery_codes.erase(gallery_codes.begin());

	for (int i = 0; i < results.rows; i++) {
		for (int j = 0; j < results.cols; j++) {
			if (j == 0) getline(input, line, ';');
			getline(input, line, ';');
			results.at<double>(i, j) = stod(line);
		}
	}

	input.close();
	return gallery_codes;
}

void rearrange_matrix(const string path) {
	ifstream input;
	input.open(path);

	string line;
	getline(input, line); // scarto la riga con tutte i nomi delle immagini della gallery

	//Carico la matrice come è stata calcolata e la metto in results
	for (int i = 0; i < results.rows; i++) {
		for (int j = 0; j < results.cols; j++) {
			if (j == 0) getline(input, line, ';');
			getline(input, line, ';');
			results.at<double>(i, j) = stod(line);
		}
	}

	//Calcolo la nuova matrice e la metto nel file
	ofstream output;
	output.open("results_BLOB_MATCHING.csv");
	string firstrow = " ;";
	for (int i = 0; i < results.rows; i++) {
		double min_subject = 0.0;
		string row = probe_names[i] + ";";
		for (int j = 0; j < results.cols; j++) {
			if (j == 0) min_subject = results.at<double>(i, j);
			else if (gallery_names[j].substr(18, 5).compare(gallery_names[j - 1].substr(18, 5)) == 0) {
				min_subject = (results.at<double>(i, j) < min_subject) ? results.at<double>(i, j) : min_subject;
			} else {
				firstrow += to_string(j-1) + ";";
				row += to_string(min_subject) + ";";
				min_subject = results.at<double>(i, j);
			}
			results.at<double>(i, j) = stod(line);
		}
		if (i == 0) output << firstrow << endl;
		output << row << endl;
	}
	input.close();
}

void thread_code(int start, int end) {
	subject* probe_sub = subject_create();
	subject* gallery_sub = subject_create();

	code* probe_code = NULL;
	code* gallery_code = NULL;

	for (int i = start; i < end; i++) {
		std::cout << probe_names[i] + "\n" << endl;
		probe_sub->input = imread(probe_names[i] + ".iris.norm.png", IMREAD_GRAYSCALE);
		probe_sub->mask = imread(probe_names[i] + ".defects.norm.png", IMREAD_GRAYSCALE);

		probe_code = code_create(probe_sub);
		code_encode(probe_code);

		double min_val = DBL_MAX;
		string min_str;
		bool auth = false;

		for (int j = 0; j < gallery_names.size(); j++) {
			//cout << "	" + gallery_names[j] << endl;
			//if (gallery_names[j].substr(9, 3).compare(probe_names[i].substr(11, 3)) != 0) continue;

			gallery_sub->input = imread(gallery_names[j] + ".iris.norm.png", IMREAD_GRAYSCALE);
			gallery_sub->mask = imread(gallery_names[j] + ".defects.norm.png", IMREAD_GRAYSCALE);

			gallery_code = code_create(gallery_sub);

			//coder_lbp_load(gallery_names[j], gallery_code->code_lbp);
			//coder_blob_load(gallery_names[j], gallery_code->code_blob);
			code_encode(gallery_code);

			results.at<double>(i, j) = 1.0-code_match(probe_code, gallery_code);
			//cout << probe_names[i] + " " + gallery_names[j] + to_string(1.0-results.at<double>(i, j)) +"\n" << endl;

			//std::cout << "	" + gallery_names[j] + " -> " + to_string(results.at<float>(i, j)) << endl;

			/*if (results.at<double>(i, j) <= 0.50) {
				auth = true;
				break;
			}

			if (results.at<double>(i, j) < min_val) {
				min_val = results.at<double>(i, j);
				min_str = gallery_names[j];
			}*/

			code_free(gallery_code);
		}

		//if (auth == true)
			//std::cout << "Persona autenticata\n" << endl;
			//count++;
		//if (min_str.substr(9, 3).compare(probe_names[i].substr(11, 3)) == 0) count++;

		/*std::fprintf(output, (probe_names[i] + " -> " + min_str
			+ " : " + to_string(min_val) + "\n\n").c_str());
		std::printf("Totale azzeccati: %d/158\n\n", count);*/
		code_free(probe_code);
	}
}

int main()
{
	const char* gallery_file = "gallery_final.txt";
	const char* probe_file = "probe_final.txt";

	FILE* probe = NULL;
	FILE* gallery = NULL;

	if ((probe = fopen(probe_file, "r")) == NULL ||
		(gallery = fopen(gallery_file, "r")) == NULL) {
		std::cout << "Impossibile aprire file di input\n";
		return EXIT_FAILURE;
	}

	//Viene letta la dimensione della Gallery e del Probe
	char line[MAX_LINE];
	int gallery_size = atoi(fgets(line, MAX_LINE - 1, gallery));
	int probe_size = atoi(fgets(line, MAX_LINE - 1, probe));

	int gallery_single_size = 153;

	//Vengono letti e memorizzati i nomi delle immagini che 
	//costituiscono la Gallery e il Probe.
	for (int i = 0; i < probe_size; i++)
		probe_names.push_back(string(strtok(fgets(line, MAX_LINE - 1, probe), "\n")));

	for (int i = 0; i < gallery_size; i++)
		gallery_names.push_back(string(strtok(fgets(line, MAX_LINE - 1, gallery), "\n")));

	//fclose(gallery);
	//fclose(probe);

	results = Mat::zeros(probe_size, gallery_size, CV_64FC1);
	rearrange_matrix("results_BLOB3.0.csv");

	results = Mat::zeros(probe_size, gallery_single_size, CV_64FC1);
	vector<string> gallery_codes = load_matrix("results_BLOB_MATCHING.csv");

	cout << gallery_codes.size();

	//cout << results;
	cin.get();

	//double thresh=0.5;
	for (double thresh = 0.0; thresh <= 1.0; thresh += 0.1) {
		int TP = 0;
		int TN = 0;
		int FP = 0;
		int FN = 0;

		for (int i = 0; i < probe_size; i++) {
			for (int j = 0; j < gallery_single_size; j++) {
				if (results.at<double>(i, j) <= thresh) {
					if (gallery_names[stoi(gallery_codes.at(j))].substr(18, 5).compare(probe_names[i].substr(11, 5)) == 0) {
						TP++;
					} else {
						FP++;
					}
				} else {
					if (gallery_names[stoi(gallery_codes.at(j))].substr(18, 5).compare(probe_names[i].substr(11, 5)) == 0) {
						FN++;
					} else {
						TN++;
					}
				}
			}
		}

		std::cout << "Thresh utilizzata: " + to_string(thresh) << endl;
		std::cout << "Total: " + to_string(gallery_single_size*probe_size) << endl;
		std::cout << "True positive: " + to_string(TP) << endl;
		std::cout << "True negative: " + to_string(TN) << endl;
		std::cout << "False positive: " + to_string(FP) << endl;
		std::cout << "False negative: " + to_string(FN) << endl;
		std::cout << endl;
		std::cout << "FAR: " + to_string((double)FP / (FP + TN)) << endl;
		std::cout << "FRR: " + to_string((double)FN / (TP + FN)) << endl;
		std::cout << "GAR: " + to_string(1.0 - (double)FN / (TP + FN)) << endl;
		std::cout << "\n\n\n";
		std::cin.get();
	}

	//rearrange_matrix("results_LBP3.0.csv");

	/*Mat input = Mat(3, 4, CV_8UC3);
	input.at<Vec3b>(0, 0) = Vec3b(7,7,7);
	input.at<Vec3b>(0, 1) = Vec3b(2,2,2);
	input.at<Vec3b>(0, 2) = Vec3b(5,5,5);
	input.at<Vec3b>(0, 3) = Vec3b(7,7,7);
	input.at<Vec3b>(1, 0) = Vec3b(1,1,1);
	input.at<Vec3b>(1, 1) = Vec3b(4,4,4);
	input.at<Vec3b>(1, 2) = Vec3b(2,2,2);
	input.at<Vec3b>(1, 3) = Vec3b(3,3,3);
	input.at<Vec3b>(2, 0) = Vec3b(1,1,1);
	input.at<Vec3b>(2, 1) = Vec3b(1,1,1);
	input.at<Vec3b>(2, 2) = Vec3b(6,6,6);
	input.at<Vec3b>(2, 3) = Vec3b(8,8,8);*/

	/*Mat input1 = Mat(3, 4, CV_8UC1);
	input1.at<uchar>(0, 0) = 7;
	input1.at<uchar>(0, 1) = 2;
	input1.at<uchar>(0, 2) = 5;
	input1.at<uchar>(0, 3) = 7;
	input1.at<uchar>(1, 0) = 1;
	input1.at<uchar>(1, 1) = 4;
	input1.at<uchar>(1, 2) = 2;
	input1.at<uchar>(1, 3) = 3;
	input1.at<uchar>(2, 0) = 1;
	input1.at<uchar>(2, 1) = 1;
	input1.at<uchar>(2, 2) = 6;
	input1.at<uchar>(2, 3) = 8;

	Mat input2 = Mat(3, 4, CV_8UC1);
	input2.at<uchar>(0, 0) = 7;
	input2.at<uchar>(0, 1) = 2;
	input2.at<uchar>(0, 2) = 5;
	input2.at<uchar>(0, 3) = 7;
	input2.at<uchar>(1, 0) = 1;
	input2.at<uchar>(1, 1) = 4;
	input2.at<uchar>(1, 2) = 2;
	input2.at<uchar>(1, 3) = 3;
	input2.at<uchar>(2, 0) = 1;
	input2.at<uchar>(2, 1) = 1;
	input2.at<uchar>(2, 2) = 6;
	input2.at<uchar>(2, 3) = 8;

	Mat mask1 = Mat(3, 4, CV_8UC1);
	mask1.at<uchar>(0, 0) = 0;
	mask1.at<uchar>(0, 1) = 1;
	mask1.at<uchar>(0, 2) = 1;
	mask1.at<uchar>(0, 3) = 1;
	mask1.at<uchar>(1, 0) = 1;
	mask1.at<uchar>(1, 1) = 0;
	mask1.at<uchar>(1, 2) = 1;
	mask1.at<uchar>(1, 3) = 1;
	mask1.at<uchar>(2, 0) = 0;
	mask1.at<uchar>(2, 1) = 0;
	mask1.at<uchar>(2, 2) = 0;
	mask1.at<uchar>(2, 3) = 1;

	Mat mask2 = Mat(3, 4, CV_8UC1);
	mask2.at<uchar>(0, 0) = 0;
	mask2.at<uchar>(0, 1) = 1;
	mask2.at<uchar>(0, 2) = 1;
	mask2.at<uchar>(0, 3) = 1;
	mask2.at<uchar>(1, 0) = 1;
	mask2.at<uchar>(1, 1) = 0;
	mask2.at<uchar>(1, 2) = 1;
	mask2.at<uchar>(1, 3) = 1;
	mask2.at<uchar>(2, 0) = 0;
	mask2.at<uchar>(2, 1) = 0;
	mask2.at<uchar>(2, 2) = 0;
	mask2.at<uchar>(2, 3) = 1;*/

	/*Mat input1 = imread("prove/matching/IMG_060_L_1.iris.norm.png", IMREAD_GRAYSCALE);
	Mat mask1 = imread("prove/matching/IMG_060_L_1.defects.norm.png", IMREAD_GRAYSCALE);*/

	/*Mat input2 = imread("prove/matching/IMG_060_L_1.iris.norm.png", IMREAD_GRAYSCALE);
	Mat mask2 = imread("prove/matching/IMG_060_L_1.defects.norm.png", IMREAD_GRAYSCALE);*/

	/*subject* sub1 = subject_create();
	sub1->input = imread("prove/matching/IMG_060_L_1.iris.norm.png", IMREAD_GRAYSCALE);
	sub1->mask = imread("prove/matching/IMG_060_L_1.defects.norm.png", IMREAD_GRAYSCALE);

	subject* sub2 = subject_create();
	sub2->input = imread("prove/matching/IMG_006_L_1.iris.norm.png", IMREAD_GRAYSCALE);
	sub2->mask = imread("prove/matching/IMG_006_L_1.defects.norm.png", IMREAD_GRAYSCALE);

	code* coder1 = code_create(sub1);
	code* coder2 = code_create(sub2);

	code_encode(coder1);
	cout << coder1->code_spatiogram->spatiogram->histogram.t();
	code_encode(coder2);

	cout << code_match(coder1, coder2);
	cin.get();*/

	//create_spatiogram(input1, mask1, spatio1);
	//create_spatiogram(input2, mask2, spatio2);

	//match_spatiogram(spatio1, spatio2);
	//cout << match_spatiogram(spatio1, spatio2);
	//cin.get();


	/*Mat img = imread("prove/matching/IMG_006_L_1.iris.norm.png", IMREAD_GRAYSCALE);
	Mat mask = imread("prove/matching/IMG_006_L_1.defects.norm.png", IMREAD_GRAYSCALE);
	Mat new_mask = Mat(img.size(), CV_8UC1);

	adjust_mask_single(img, mask, new_mask);

	imwrite("prove/matching/new_mask.png", new_mask);
	*/

	/*vector<string> gall = { "ciao", "cia", "ci", "c", "caio" };
	vector<string> prob = { "pippo", "pluto", "topo", "cane"};

	ofstream t;
	t.open("file.csv");

	Mat input = Mat(4, 5, CV_64FC1);
	input.at<double>(0, 0) = 0.0;
	input.at<double>(0, 1) = 1.0;
	input.at<double>(0, 2) = 2.0;
	input.at<double>(0, 3) = 0.0;
	input.at<double>(0, 4) = 2.0;
	input.at<double>(1, 0) = 4.0;
	input.at<double>(1, 1) = 3.0;
	input.at<double>(1, 2) = 2.0;
	input.at<double>(1, 3) = 2.0;
	input.at<double>(1, 4) = 3.0;
	input.at<double>(2, 0) = 1.0;
	input.at<double>(2, 1) = 1.0;
	input.at<double>(2, 2) = 5.0;
	input.at<double>(2, 3) = 3.0;
	input.at<double>(2, 4) = 5.0;
	input.at<double>(3, 0) = 7.0;
	input.at<double>(3, 1) = 9.0;
	input.at<double>(3, 2) = 1.0;
	input.at<double>(3, 3) = 3.0;
	input.at<double>(3, 4) = 5.0;

	string firstrow = " ;";
	for (int i = 0; i < gall.size(); i++)
		firstrow += gall[i] + ";";
	firstrow += "\n";

	t << firstrow;
	int probe_number = 0;

	for (int i = 0; i < input.rows; i++) {
		string row = prob[probe_number++] + ";";
		for (int j = 0; j < input.cols; j++) {
			row += to_string(input.at<double>(i, j)) + ";";
		}
		row += "\n";
		t << row;
	}

	cin.get();*/



	/*FILE* csv = fopen("prova.csv","w");

	cout << "img_01;0.3;0.7;0.9;\n";
	cout << "img_02;0.005;0.003;0.9;\n";*/

	//create_file();

	//create_gallery();


	///MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN
	/*clock_t start, end;
	double tempo;
	start = clock();

	//Riscrivere tutto in modo che la gallery sia caricata e non ricalcolata ogni volta
	const char* gallery_file = "gallery_final.txt";
	const char* probe_file = "probe_final.txt";
	//const char* probe_file = "single_probe.txt";

	FILE* probe = NULL;
	FILE* gallery = NULL;

	if ((probe = fopen(probe_file, "r")) == NULL || 
		(gallery = fopen(gallery_file, "r")) == NULL) {
		std::cout << "Impossibile aprire file di input\n";
		return EXIT_FAILURE;
	}
	
	//Viene letta la dimensione della Gallery e del Probe
	char line[MAX_LINE];
	int gallery_size = atoi(fgets(line, MAX_LINE - 1 , gallery));
	int probe_size = atoi(fgets(line, MAX_LINE - 1 , probe));

	//Vengono letti e memorizzati i nomi delle immagini che 
	//costituiscono la Gallery e il Probe.
	for (int i = 0; i < probe_size; i++)
		probe_names.push_back(string(strtok(fgets(line, MAX_LINE - 1, probe), "\n")));

	for (int i = 0; i < gallery_size; i++)
		gallery_names.push_back(string(strtok(fgets(line, MAX_LINE - 1, gallery), "\n")));

	fclose(gallery);
	fclose(probe);

	//Viene inizializzata la matrice dei risultati.
	//Righe -> Elementi della Gallery.
	//Colonne -> Elementi del Probe.
	results = Mat::zeros(probe_size, gallery_size, CV_64FC1);

	code_init();

	int count = 0;
	int people_per_thread = (int) ceil(probe_size/(double)NUM_THREADS);

	thread thds[NUM_THREADS];
	for (int t = 0; t < NUM_THREADS; t++)
		thds[t] = thread(thread_code, t*people_per_thread, 
			(t+1)*people_per_thread < probe_size ? (t + 1)*people_per_thread : probe_size);

	for (int t = 0; t < NUM_THREADS; t++)
		thds[t].join();

	/*double min_res;
	double max_res;
	minMaxLoc(results, &min_res, &max_res);

	for (int i = 0; i < results.rows; i++) {
		for (int j = 0; j < results.cols; j++) {
			results.at<double>(i, j) /= max_res;
		}
	}*/

	//Stampa dei risultati
	/*ofstream output;
	output.open("results_SPATIOGRAM3.0.csv");

	string firstrow = " ;";
	for (int i = 0; i < gallery_names.size(); i++)
		firstrow += gallery_names[i] + ";";
	firstrow += "\n";

	output << firstrow;

	int probe_number = 0;
	for (int i = 0; i < results.rows; i++) {
		string row = probe_names[probe_number++] + ";";
		for (int j = 0; j < results.cols; j++) {
			row += to_string(results.at<double>(i, j)) + ";";
		}
		row += "\n";
		output << row;
	}

	end = clock();
	tempo = ((double)(end - start)) / CLOCKS_PER_SEC;
	std::cout << to_string(tempo);
	std::cin.get();

	//MAIN MATCHING GENERALE
	/*subject* sub1 = subject_create();
	sub1->input = imread("matching/IMG_006_L_1.iris.norm.png", IMREAD_GRAYSCALE);
	sub1->mask = imread("matching/IMG_006_L_1.defects.norm.png", IMREAD_GRAYSCALE);
	code* coder = code_create(sub1);

	subject* sub2 = subject_create();
	sub2->input = imread("matching/IMG_006_L_1.iris.norm.png", IMREAD_GRAYSCALE);
	sub2->mask = imread("matching/IMG_006_L_1.defects.norm.png", IMREAD_GRAYSCALE);
	code* coder2 = code_create(sub2);

	code_init();

	code_encode(coder);
	code_encode(coder2);

	double result = code_match(coder, coder2);
	printf("Result: %f\n", result);

	code_free(coder);
	code_free(coder2);

	cin.get();*/

	//MAIN DEL MATCHING CON BLOB
	/*coder_blob* coder = coder_blob_create();
	coder->input = imread("matching/IMG_006_L_1.iris.norm.png",IMREAD_GRAYSCALE);
	coder->mask = imread("matching/IMG_006_L_1.defects.norm.png", IMREAD_GRAYSCALE);

	coder_blob* coder_2 = coder_blob_create();
	coder_2->input = imread("matching/IMG_060_L_1.iris.norm.png", IMREAD_GRAYSCALE);
	coder_2->mask = imread("matching/IMG_060_L_1.defects.norm.png", IMREAD_GRAYSCALE);

	coder_blob_init();

	coder_blob_encode(coder);
	coder_blob_encode(coder_2);

	double result = coder_blob_match(coder, coder_2);
	printf("Result: %f\n", result);

	coder_blob_free(coder);
	coder_blob_free(coder_2);

	cin.get();
	waitKey(0);*/


	//MATCHING CON HAMMING DISTANCE SHIFTATA
	/*Mat img1 = imread("006/log_bin_merge.png", IMREAD_GRAYSCALE);
	Mat img2 = imread("006/log_bin_merge.png", IMREAD_GRAYSCALE);

	Mat mask1 = imread("006/IMG_006_L_1.defects.norm.png", IMREAD_GRAYSCALE);
	Mat mask2 = imread("006/IMG_006_L_1.defects.norm.png", IMREAD_GRAYSCALE);

	double hd = shifted_hamming_distance(img1, mask1, img2, mask2, 10);
	printf("hamming: %f\n", hd);

	cin.get();*/


	//MATCHING MAIN
	/*Mat v1 = (Mat_<uchar>(3, 3) << 0, 7, 5, 9, 2, 2, 2, 3, 7);
	Mat v2 = (Mat_<uchar>(3, 3) << 0, 1, 1, 1, 0, 0, 1, 0, 0);
	Mat v3 = (Mat_<uchar>(3, 3) << 0, 7, 7, 7, 5, 9, 2, 2, 3);
	Mat v4 = (Mat_<uchar>(3, 3) << 2, 3, 3, 3, 2, 2, 3, 2, 2);

	Mat h1 = Mat::zeros(10, 1, CV_32FC1);
	Mat h2 = Mat::zeros(10, 1, CV_32FC1); 
	Mat h3 = Mat::zeros(10, 1, CV_32FC1);
	Mat h4 = Mat::zeros(10, 1, CV_32FC1);

	computeHist(v1, h1);
	computeHist(v2, h2);
	computeHist(v3, h3);
	computeHist(v4, h4);

	double h1h1_corr = compareHist(h1, h1, CV_COMP_CORREL);
	double h1h1_chi = compareHist(h1, h1, CV_COMP_CHISQR);
	double h1h1_int = compareHist(h1, h1, CV_COMP_INTERSECT);
	double h1h1_bhat = compareHist(h1, h1, CV_COMP_BHATTACHARYYA);

	printf("h1h1 : %f, %f, %f, %f\n", h1h1_corr, h1h1_chi, h1h1_int, h1h1_bhat);

	h1h1_corr = compareHist(h1, h2, CV_COMP_CORREL);
	h1h1_chi = compareHist(h1, h2, CV_COMP_CHISQR);
	h1h1_int = compareHist(h1, h2, CV_COMP_INTERSECT);
	h1h1_bhat = compareHist(h1, h2, CV_COMP_BHATTACHARYYA);

	printf("h1h2 : %f, %f, %f, %f\n", h1h1_corr, h1h1_chi, h1h1_int, h1h1_bhat);

	h1h1_corr = compareHist(h1, h3, CV_COMP_CORREL);
	h1h1_chi = compareHist(h1, h3, CV_COMP_CHISQR);
	h1h1_int = compareHist(h1, h3, CV_COMP_INTERSECT);
	h1h1_bhat = compareHist(h1, h3, CV_COMP_BHATTACHARYYA);

	printf("h1h3 : %f, %f, %f, %f\n", h1h1_corr, h1h1_chi, h1h1_int, h1h1_bhat);

	h1h1_corr = compareHist(h2, h4, CV_COMP_CORREL);
	h1h1_chi = compareHist(h2, h4, CV_COMP_CHISQR);
	h1h1_int = compareHist(h2, h4, CV_COMP_INTERSECT);
	h1h1_bhat = compareHist(h2, h4, CV_COMP_BHATTACHARYYA);

	printf("h2h4 : %f, %f, %f, %f\n", h1h1_corr, h1h1_chi, h1h1_int, h1h1_bhat);

	cin.get();
	waitKey(0);*/

	//MAIN DEL MATCHING CON LBP
	/*code_init();
	subject* sub1 = subject_create();
	sub1->input = imread("prove/matching/IMG_006_L_1.iris.norm.png", IMREAD_GRAYSCALE);
	sub1->mask = imread("prove/matching/IMG_006_L_1.defects.norm.png", IMREAD_GRAYSCALE);

	code* coder = code_create(sub1);
	coder->code_lbp = coder_lbp_create();
	coder->code_blob = coder_blob_create();
	code_encode(coder);
	*/
	/*subject* sub2 = subject_create();
	sub2->input = imread("matching/IMG_006_L_1.iris.norm.png", IMREAD_GRAYSCALE);
	sub2->mask = imread("matching/IMG_006_L_1.defects.norm.png", IMREAD_GRAYSCALE);

	code* coder2 = code_create(sub2);
	coder2->code_lbp = coder_lbp_create();
	coder2->code_blob = coder_blob_create();
	code_encode(coder2);

	double result = code_match(coder, coder2);
	printf("Result: %f\n", result);

	namedWindow("MASK", WINDOW_NORMAL);
	imshow("MASK",coder->code_lbp->output);
	imwrite("eyes/lbp_coder.png", coder->code_lbp->output);

	code_free(coder);
	code_free(coder2);

	cin.get();
	waitKey(0);*/
	

	//MAIN DEL CODER DEGLI SPATIOGRAM
	/*subject* sub1 = subject_create();
	Mat input = Mat(4, 5, CV_8UC1);
	input.at<uchar>(0, 0) = 0;
	input.at<uchar>(0, 1) = 1;
	input.at<uchar>(0, 2) = 2;
	input.at<uchar>(0, 3) = 0;
	input.at<uchar>(0, 4) = 2;
	input.at<uchar>(1, 0) = 4;
	input.at<uchar>(1, 1) = 3;
	input.at<uchar>(1, 2) = 2;
	input.at<uchar>(1, 3) = 2;
	input.at<uchar>(1, 4) = 3;
	input.at<uchar>(2, 0) = 1;
	input.at<uchar>(2, 1) = 1;
	input.at<uchar>(2, 2) = 5;
	input.at<uchar>(2, 3) = 3;
	input.at<uchar>(2, 4) = 5;
	input.at<uchar>(3, 0) = 7;
	input.at<uchar>(3, 1) = 9;
	input.at<uchar>(3, 2) = 1;
	input.at<uchar>(3, 3) = 3;
	input.at<uchar>(3, 4) = 5;

	sub1->input = input;
	cout << sub1->input;
	cout << "\n\n\n";
	code* coder = code_create(sub1);
	code_init();
	code_encode(coder);

	for (int i = 0; i < 10; i++) {
		printf("%d -> h: %f\t\t mv: %f\t\t cm: %f\t\t\n", i,
			coder->code_spatiogram->spatiogram->histogram.at<float>(i,0),
			coder->code_spatiogram->spatiogram->mean_vector.at<float>(i, 0),
			coder->code_spatiogram->spatiogram->covariance_matrix.at<float>(i, 0));
	}

	showHist(coder->code_spatiogram->spatiogram->histogram);

	cin.get();
	waitKey(0);*/

	//MAIN DEL CODER DEI BLOB
	/*coder_blob* coder = coder_blob_create();
	coder->input = imread("060/IMG_060_L_1.iris.norm.png", IMREAD_GRAYSCALE);

	coder_blob_init();

	coder_blob_encode(coder);

	for (int i = 0; i < MAX_NUM_KERNEL; i++)
	{
		namedWindow("LoG", WINDOW_NORMAL);
		imshow("LoG", coder->log_bin_imgs[i]);
		waitKey(0);
	}

	namedWindow("LoG", WINDOW_NORMAL);
	imshow("LoG", coder->log_bin_merge);
	imwrite("060/log_bin_merge.png", coder->log_bin_merge);

	coder_blob_free(coder);

	waitKey(0);
	*/

	//MAIN DEL PREPROCESSING
	/*Mat img_in = imread("preprocessing/IMG_070_L_3.jpg");
	Mat img_out = Mat(img_in.rows, img_in.cols, CV_8UC3);

	coder_spatiogram* coder = coder_spatiogram_create();
	calc_spatiogram(coder);

	//convert_whitening(img_in, img_out, 20);
	//hough_something();
	//convert_clahe(img_in,img_out);
	//convert_reduce_saturation(img_in, img_out, 20);
	//convert_luminosity(img_in, img_out);
	//cvtColor(img_in, img_out, COLOR_BGR2GRAY);

	namedWindow("RGB", WINDOW_NORMAL);
	imshow("RGB", img_out);
	imwrite("preprocessing/IMG_070_L_3_white20.png", img_out);

	waitKey(0);*/


	//MAIN DEL CODER DEI LBP
	/*coder_LBP* coder = coder_lbp_create();
	coder->input = imread("eyes/iris_norm.png",IMREAD_GRAYSCALE);
	//coder->mask = imread("006/IMG_006_L_1.defects.png", IMREAD_GRAYSCALE);
	coder->output = Mat(coder->input.rows, coder->input.cols, CV_8UC1);

	coder_lbp_encode(coder);
	
	namedWindow("MASK", WINDOW_NORMAL);
	imshow("MASK",coder->output);
	imwrite("eyes/lbp_coder.png", coder->output);

	coder_lbp_free(coder);

	waitKey(0);*/
}


//----------------------------------------------------------------------------
//Questo è come si usa l'iteratore
/*int main() {
	Mat img = imread("tulip.png");

	//int histogram[255] = {};
	MatIterator_<Vec3b> start = img.begin<Vec3b>();
	MatIterator_<Vec3b> end = img.end<Vec3b>();
	for (start; start < end; start++) {
		printf("%d, %d, %d\n", (*start)[0], (*start)[1], (*start)[2]);
	}

	cin.get();	
	namedWindow("image!", WINDOW_NORMAL);
	imshow("image!", img);
	waitKey(0);
	return 0;

}*/

//Questo è come si scorre su righe e colonne
/*int main() {
	Mat img = imread("tulip.png");

	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			Vec3b rgba = img.at<Vec3b>(i, j);
			printf("%d, %d, %d\n", rgba[0], rgba[1], rgba[2]);
		}
	}

	cin.get();
	namedWindow("image!", WINDOW_NORMAL);
	imshow("image!", img);
	waitKey(0);
	return 0;
}*/

//void equalize(Mat img_in, Mat& img_out) {
//	int histogram[256] = {};
//	int eq_histogram[256] = {};
//
//	//Crea l'istogramma per l'immagine di input imgin
//	for (int i = 0; i < img_in.rows; ++i)
//		for (int j = 0; j < img_in.cols; ++j)
//			histogram[img_in.at<uchar>(i, j)] += 1;
//
//	//Crea l'istogramma equalizzato eq_histogram dall' istogramma histogram
//	for (int k = 1; k < 256; k++) {
//		histogram[k] += histogram[k - 1];
//		eq_histogram[k] = (int)round((histogram[k]) * 255 / (img_in.rows*img_in.cols - 1.0f));
//	}
//
//	//Crea l'immagine di output img_out usando l'istogramma equalizzato
//	for (int i = 0; i < img_in.rows; ++i)
//		for (int j = 0; j < img_in.cols; ++j)
//			img_out.at<uchar>(i, j) = eq_histogram[img_in.at<uchar>(i, j)];
//
//}
//
//void equalizeColors(Mat img_in, Mat& img_out) {
//	Mat channel_arr[3] = {	Mat(img_in.rows, img_in.cols, CV_8UC1),
//							Mat(img_in.rows, img_in.cols, CV_8UC1),
//							Mat(img_in.rows, img_in.cols, CV_8UC1)};
//
//	//Divido l'immagine RGB in canali singoli
//	split(img_in, channel_arr);
//
//	//Ogni canale viene equalizzato
//	for (int i = 0; i < 3; i++)
//		equalize(channel_arr[i], channel_arr[i]);
//
//	//Ricompongo i canali equalizzati per creare l'immagine finale
//	merge(channel_arr, 3, img_out);
//}

//int main() {
//	////Main per l'equalizzazione di immagine in scala di grigi
//	//Mat img = imread("tulip.png",IMREAD_GRAYSCALE);
//	//Mat out(img.rows, img.cols,CV_8UC1);
//
//	//namedWindow("image!", WINDOW_NORMAL);
//	//imshow("image!", img);
//	//waitKey(0);
//
//	//out = equalize(img, out);
//
//	////Salva l'immagine di output img_out
//	//vector<int> compression_params;
//	//compression_params.push_back(IMWRITE_PNG_COMPRESSION);
//	//compression_params.push_back(9);
//	//imwrite("tulip_eq_2.png", out, compression_params);
//
//
//	////Main per l'equalizzazione di immagine a color
//	Mat img = imread("eye.jpg",IMREAD_ANYCOLOR);
//	Mat output(img.rows, img.cols,CV_8UC1);
//
//	equalizeColors(img, output);
//
//	imwrite("eye_eq.jpg", output);
//
//
//}