// opencvtest.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#pragma comment(lib, "opencv_world330d.lib") // use debug library

#include <opencv2/opencv.hpp>
using namespace cv;

#include <string>
using namespace std;

int main(int argc, char **argv)
{
	namedWindow("Video", WINDOW_AUTOSIZE);
	VideoCapture cap;
	cap.open(string("tree.avi"));

	Mat frame;

	for (;;)
	{
		cap >> frame;

		if (frame.empty())
			break;

		Mat canny;

		GaussianBlur(frame, frame, Size(5, 5), 3, 3);
		GaussianBlur(frame, frame, Size(5, 5), 3, 3);

		Canny(frame, canny, 10, 100, 3, true);

		threshold(canny, canny, 127, 255, THRESH_BINARY);

		floodFill(canny, cv::Point(0, 0), Scalar(255));
		floodFill(canny, cv::Point(0, canny.rows - 1), Scalar(255));
		floodFill(canny, cv::Point(canny.cols - 1, 0), Scalar(255));
		floodFill(canny, cv::Point(canny.cols - 1, canny.rows - 1), Scalar(255));

		int obstacle_count = 0;

		for (int j = 0; j < canny.rows; j++)
		{
			for (int i = 0; i < canny.cols; i++)
			{
				int pixelValue = (int)canny.at<unsigned char>(j, i);

				if (pixelValue == 0)
				{
					floodFill(canny, cv::Point(i, j), Scalar(127));
					obstacle_count++;
					j = 0; i = 0;
					continue;
				}
			}
		}


		imshow("Video", canny);

		if (waitKey(33) >= 0)
			break;



	}

	return 0;
}



