#include "stdafx.h"

#pragma comment(lib, "opencv_world330d.lib") // use debug library
#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	Mat im = imread("farm.jpg");

// https://stackoverflow.com/questions/23811638/convert-hsv-to-grayscale-in-opencv


	Mat imHSV;
	cvtColor(im, imHSV, CV_BGR2HSV);

	Mat hsv_channels[3];
	split(imHSV, hsv_channels);
	imshow("channel", hsv_channels[0]);

	Mat canny;
	Canny(hsv_channels[0], canny, 50, 150, 3, true);

	// threshold(canny, canny, 127, 255, THRESH_BINARY);

	int dilation_size = 1;
	Mat element = getStructuringElement(MORPH_CROSS,
		Size(2 * dilation_size + 1, 2 * dilation_size + 1),
		Point(dilation_size, dilation_size));

	dilate(canny, canny, element);

	floodFill(canny, Point(0, 0), Scalar(255));
	floodFill(canny, Point(0, canny.rows - 1), Scalar(255));
	floodFill(canny, Point(canny.cols - 1, 0), Scalar(255));
	floodFill(canny, Point(canny.cols - 1, canny.rows - 1), Scalar(255));

	imshow("canny", canny);

	waitKey();

	return 0;
}