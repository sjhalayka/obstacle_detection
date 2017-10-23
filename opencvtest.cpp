#include "stdafx.h"

#pragma comment(lib, "opencv_world330.lib")

#include <opencv2/opencv.hpp>
using namespace cv;

#include <iostream>
#include <map>
#include <set>
using namespace std;


class int_pair
{
public:

	int a, b;
};

bool operator< (const int_pair lhs, const int_pair &rhs)
{
	return lhs.b < rhs.b;
}

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

		int dilation_size = 2;
		Mat dilation_element = getStructuringElement(
			MORPH_CROSS,
			Size(2 * dilation_size + 1, 2 * dilation_size + 1),
			Point(dilation_size, dilation_size));

		int erosion_size = 1;
		Mat erosion_element = getStructuringElement(
			MORPH_CROSS,
			Size(2 * erosion_size + 1, 2 * erosion_size + 1),
			Point(erosion_size, erosion_size));


		dilate(canny, canny, dilation_element);
		erode(canny, canny, erosion_element);

		//floodFill(canny, Point(0, 0), Scalar(127));
		//floodFill(canny, Point(0, canny.rows - 1), Scalar(127));
		//floodFill(canny, Point(canny.cols - 1, 0), Scalar(127));
		//floodFill(canny, Point(canny.cols - 1, canny.rows - 1), Scalar(127));

		Mat black(canny.rows, canny.cols, CV_32F);

		for (int j = 0; j < canny.rows; j++)
		{
			for (int i = 0; i < canny.cols; i++)
			{
				if (canny.at<unsigned char>(j, i) == 255)
					black.at<float>(j, i) = 10000.0f;
				//else
					//black.at<float>(j, i) = canny.at<unsigned char>(j, i) / 255.0f;
			}
		}


		int section_count = 0;
		int colour = 1;

		for(int j = 0; j < canny.rows; j++)
		{
			for(int i = 0; i < canny.cols; i++)
			{
				int pixelValue = canny.at<unsigned char>(j, i);

				if(pixelValue == 0)
				{
					floodFill(canny, cv::Point(i, j), Scalar(colour));
					//floodFill(black, cv::Point(i, j), Scalar(colour) / 255.0f);

					section_count++;
					colour += 10;
					j = 0; i = 0;
					continue;
				}
			}
		}

		map<int, int> section_sizes;

		for(int j = 0; j < canny.rows; j++)
		{
			for(int i = 0; i < canny.cols; i++)
			{
				int pixelValue = (int)canny.at<unsigned char>(j, i);

				if(pixelValue != 255)
					section_sizes[pixelValue]++;
			}
		}

		set<int_pair> sorted_section_sizes;

		for(map<int, int>::const_iterator ci = section_sizes.begin(); ci != section_sizes.end(); ci++)
		{
			int_pair ip;
			ip.a = ci->first;
			ip.b = ci->second;

			sorted_section_sizes.insert(ip);
		}

		int count = 0;

		for(set<int_pair>::reverse_iterator ci = sorted_section_sizes.rbegin(); ci != sorted_section_sizes.rend(); ci++)
		{
			count++;

			if (count < 2 || count > sorted_section_sizes.size() - 2)
				continue;

			for (int j = 0; j < black.rows; j++)
			{
				for (int i = 0; i < black.cols; i++)
				{
					if (canny.at<unsigned char>(j, i) == ci->a)
						black.at<float>(j, i) = 0.75;// (float)canny.at<unsigned char>(j, i) / 255.0f;
				}
			}

			//cout << ci->a << " " << ci->b << endl;
		}


		imshow("canny", canny);
		imshow("black", black);


		if (waitKey(33) >= 0)

			break;
	}

	destroyAllWindows();

	return 0;
}
