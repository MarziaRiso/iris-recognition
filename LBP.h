#pragma once
#include<opencv2/opencv.hpp>
#include<opencv/cv.h>
#include<iostream>

int createPixelLBP(Mat &img_in, int row, int column);
void createImageLBP(Mat &img_in, Mat &img_out);
void createImageEqualized(Mat &img_in, Mat &img_out);