#include "stdafx.h"

#pragma comment(lib, "opencv_world330d.lib")

#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <fstream>
using namespace std;
using namespace cv;

int main(void)
{
	namedWindow("Example3", WINDOW_AUTOSIZE);
	VideoCapture cap;
	cap.open(string("test.avi"));

	Mat frame;

	for (;;)
	{
		cap >> frame;

		if (frame.empty())
			break;



		Mat imHSV;
		cvtColor(frame, imHSV, CV_BGR2HSV);

		//imshow("HSV", imHSV);

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

		floodFill(canny, Point(0, 0), Scalar(127));
		floodFill(canny, Point(0, canny.rows - 1), Scalar(127));
		floodFill(canny, Point(canny.cols - 1, 0), Scalar(127));
		floodFill(canny, Point(canny.cols - 1, canny.rows - 1), Scalar(127));

		//		imshow("canny", canny);

		imshow("Example3", canny);

		if (waitKey(33) >= 0)
			break;



	}

	return 0;
}
