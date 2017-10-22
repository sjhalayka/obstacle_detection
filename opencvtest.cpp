#include "stdafx.h"

#pragma comment(lib, "opencv_world330.lib")

#include <opencv2/opencv.hpp>
using namespace cv;

#include <iostream>
using namespace std;

int main(void)
{
	VideoCapture cap;
	cap.open("test.avi");

	Mat frame;

	for(;;)
	{
		cap >> frame;

		if(frame.empty())
			break;

		Mat imHSV;
		cvtColor(frame, imHSV, CV_BGR2HSV);

		imshow("BGR", frame);

		Mat hsv_channels[3];
		split(imHSV, hsv_channels);
		

		//GaussianBlur(hsv_channels[0], hsv_channels[0], Size(5, 5), 3, 3);
		//GaussianBlur(hsv_channels[0], hsv_channels[0], Size(5, 5), 3, 3);

		imshow("H channel", hsv_channels[0]);

		Mat canny;
		Canny(hsv_channels[0], canny, 50, 150, 3, true);

		// threshold(canny, canny, 127, 255, THRESH_BINARY);

		int dilation_size = 1;
		Mat element = getStructuringElement(MORPH_CROSS,
			Size(2 * dilation_size + 1, 2 * dilation_size + 1),
			Point(dilation_size, dilation_size));

		dilate(canny, canny, element);
		erode(canny, canny, element);

		floodFill(canny, Point(0, 0), Scalar(127));
		floodFill(canny, Point(0, canny.rows - 1), Scalar(127));
		floodFill(canny, Point(canny.cols - 1, 0), Scalar(127));
		floodFill(canny, Point(canny.cols - 1, canny.rows - 1), Scalar(127));

		int obstacle_count = 0;

		for (int j = 0; j < canny.rows; j++)
		{
			for (int i = 0; i < canny.cols; i++)
			{
				int pixelValue = (int)canny.at<unsigned char>(j, i);

				if (pixelValue == 0)
				{
					floodFill(canny, cv::Point(i, j), Scalar(63));
					obstacle_count++;
					j = 0; i = 0;
					continue;
				}
			}
		}

//		cout << obstacle_count << endl;

		imshow("canny", canny);

		if (waitKey(33) >= 0)
			break;
	}

	destroyAllWindows();

	return 0;
}
