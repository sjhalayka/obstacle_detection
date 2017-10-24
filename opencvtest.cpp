#pragma comment(lib, "opencv_world330.lib")

#include <opencv2/opencv.hpp>
using namespace cv;

#include <iostream>
#include <map>
#include <set>
using namespace std;


class float_int_pair
{
public:

	float a;
	int b;
};

bool operator<(const float_int_pair lhs, const float_int_pair &rhs)
{
	return lhs.b < rhs.b;
}

int main(void)
{
	VideoCapture cap;

	cap.open("test.avi");
	Mat frame;

	for (;;)
	{
		cap >> frame;

		if (frame.empty())
			break;

		Mat imHSV;
		cvtColor(frame, imHSV, CV_BGR2HSV);

		imshow("BGR", frame);

		Mat hsv_channels[3];
		split(imHSV, hsv_channels);

		imshow("H channel", hsv_channels[0]);

		Mat canny;
		Canny(hsv_channels[0], canny, 50, 150, 3, true);

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

		Mat black(canny.rows, canny.cols, CV_32F);

		const float flt_white = 10000.0f;

		for(int j = 0; j < canny.rows; j++)
		{
			for(int i = 0; i < canny.cols; i++)
			{
				if(canny.at<unsigned char>(j, i) == 255)
					black.at<float>(j, i) = flt_white;
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
					floodFill(canny, cv::Point(i, j), Scalar(1));
					floodFill(black, cv::Point(i, j), Scalar(colour) / 255.0);

					section_count++;
					colour++;

					j = 0; i = 0;
					continue;
				}
			}
		}

		map<float, int> section_sizes;

		for(int j = 0; j < black.rows; j++)
		{
			for(int i = 0; i < black.cols; i++)
			{
				float pixelValue = black.at<float>(j, i);

				if(pixelValue != flt_white)
					section_sizes[pixelValue]++;
			}
		}

		set<float_int_pair> sorted_section_sizes;

		for(map<float, int>::const_iterator ci = section_sizes.begin(); ci != section_sizes.end(); ci++)
		{
			float_int_pair ip;
			ip.a = ci->first;
			ip.b = ci->second;

			sorted_section_sizes.insert(ip);
		}

		int count = 0;

		Mat output(black.rows, black.cols, CV_32F);

		for(set<float_int_pair>::const_iterator ci = sorted_section_sizes.begin(); ci != sorted_section_sizes.end(); ci++)
		{
			count++;

			if(count > sorted_section_sizes.size() - 1)
				continue;

			for(int j = 0; j < black.rows; j++)
			{
				for(int i = 0; i < black.cols; i++)
				{
					float colour = black.at<float>(j, i);

					if(colour == ci->a)
						output.at<float>(j, i) = 0.9f;
					else if(colour == flt_white)
						output.at<float>(j, i) = 1.0f;
				}
			}
		}

		imshow("output", output);

		if(waitKey(33) >= 0)
			break;
	}

	destroyAllWindows();

	return 0;
}
