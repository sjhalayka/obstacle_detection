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

bool operator<(const float_int_pair &lhs, const float_int_pair &rhs)
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
		// Get next frame
		cap >> frame;

		if(frame.empty())
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

		// Dilate the white edges
		int dilation_size = 2;
		Mat dilation_element = getStructuringElement(
			MORPH_RECT,
			Size(2 * dilation_size + 1, 2 * dilation_size + 1),
			Point(dilation_size, dilation_size));

		dilate(canny, canny, dilation_element);

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

		// Find black sections in flt_canny, fill using floodFill,
		// reset counters and try again. 
		// This loop will end when all sections in flt_canny have 
		// been coloured in.
		int section_count = 0;
		int fill_colour = 1;

		for(int j = 0; j < flt_canny.rows; j++)
		{
			for(int i = 0; i < flt_canny.cols; i++)
			{
				float colour = flt_canny.at<float>(j, i);

				if(colour == 0)
				{
					floodFill(flt_canny, Point(i, j), Scalar(fill_colour));

					section_count++;
					fill_colour++;

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
				float colour = flt_canny.at<float>(j, i);

				if(colour != flt_white)
					section_sizes[colour]++;
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

		Mat output(flt_canny.rows, flt_canny.cols, CV_32FC3);

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
					// Else if white, colour gray
					if(colour == ci->a)
					{
						float output_colour = powf(1.0f - ci->b / 1000.0f, 1.0f/5.0f);

						if(output_colour < 0)
							output_colour = 0;
						else if(output_colour > 1)
							output_colour = 1;

						// Mark each section in orange
						output.at<Vec3f>(j, i)[0] = 0;
						output.at<Vec3f>(j, i)[1] = output_colour*0.5f;
						output.at<Vec3f>(j, i)[2] = output_colour;
					}
					else if(colour == flt_white)
					{
						output.at<Vec3f>(j, i)[0] = 1;
						output.at<Vec3f>(j, i)[1] = 1;
						output.at<Vec3f>(j, i)[2] = 1;
					}
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
