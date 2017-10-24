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
		// Get next frame
		cap >> frame;

		if (frame.empty())
			break;

		// Show frame
		imshow("BGR", frame);

		// Convert frame from BGR to HSV
		Mat hsv;
		cvtColor(frame, hsv, CV_BGR2HSV);

		// Get separate H, S, V channels
		Mat hsv_channels[3];
		split(hsv, hsv_channels);

		// Show H channel
		imshow("H channel", hsv_channels[0]);

		// Run edge detector on H channel
		// The edges will show up in white
		// and the sections in black
		Mat canny;
		Canny(hsv_channels[0], canny, 50, 150, 3, true);

		// Dilate and erode the white edges
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

		// Assume that there are less than 100000 sections in the image.
		// If this number is not high enough, change it accordingly.
		const float flt_white = 100000.0f;

		// Mark white edges and black sections on floating point image
		Mat flt_canny(canny.rows, canny.cols, CV_32F);

		for(int j = 0; j < canny.rows; j++)
		{
			for(int i = 0; i < canny.cols; i++)
			{
				if(canny.at<unsigned char>(j, i) == 255)
					flt_canny.at<float>(j, i) = flt_white;
				else
					flt_canny.at<float>(j, i) = 0.0f; // Make sure to initialize the whole image.
			}
		}

		// Find black section in canny, fill both canny and flt_canny using floodFill,
		// reset counters and try again. This loop will end when all sections in 
		// canny/flt_canny have been coloured in.
		int section_count = 0;
		int colour = 1;

		for(int j = 0; j < canny.rows; j++)
		{
			for(int i = 0; i < canny.cols; i++)
			{
				int pixelValue = canny.at<unsigned char>(j, i);

				if(pixelValue == 0)
				{
					floodFill(canny, Point(i, j), Scalar(1));
					floodFill(flt_canny, Point(i, j), Scalar(colour));

					section_count++;
					colour++;

					j = 0; i = 0;
					continue;
				}
			}
		}

		// Get number of pixels per section colour
		map<float, int> section_sizes;

		for(int j = 0; j < flt_canny.rows; j++)
		{
			for(int i = 0; i < flt_canny.cols; i++)
			{
				float pixelValue = flt_canny.at<float>(j, i);

				if(pixelValue != flt_white)
					section_sizes[pixelValue]++;
			}
		}

		// Sort the section sizes in ascending order
		multiset<float_int_pair> sorted_section_sizes;

		for(map<float, int>::const_iterator ci = section_sizes.begin(); ci != section_sizes.end(); ci++)
		{
			float_int_pair ip;
			ip.a = ci->first;
			ip.b = ci->second;

			sorted_section_sizes.insert(ip);
		}

		Mat output(flt_canny.rows, flt_canny.cols, CV_32F);

		// For each colour in flt_canny
		for(multiset<float_int_pair>::const_iterator ci = sorted_section_sizes.begin(); ci != sorted_section_sizes.end(); ci++)
		{
			// Iterate throughout the whole image
			for(int j = 0; j < flt_canny.rows; j++)
			{
				for(int i = 0; i < flt_canny.cols; i++)
				{
					float colour = flt_canny.at<float>(j, i);

					// If colour (ci->a) matches, colour by ci->b
					// Else if white
					if(colour == ci->a)
						output.at<float>(j, i) = (1.0f - ci->b / 10000.0f);
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
