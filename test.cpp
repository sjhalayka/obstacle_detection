#pragma comment(lib, "opencv_world330.lib")

#include <opencv2/opencv.hpp>
using namespace cv;

#include <iostream>
#include <map>
#include <set>
using namespace std;


// For later use with multiset
class float_int_pair
{
public:
	float a;
	int b;
};

// For later use with multiset
bool operator<(const float_int_pair &lhs, const float_int_pair &rhs)
{
	return lhs.b < rhs.b;
}


int main(void)
{
	VideoCapture vid_in;
	vid_in.open("test.avi");

	int frame_width = static_cast<int>(vid_in.get(CV_CAP_PROP_FRAME_WIDTH));
	int frame_height = static_cast<int>(vid_in.get(CV_CAP_PROP_FRAME_HEIGHT));

	VideoWriter vid_out("output.avi", CV_FOURCC('M', 'J', 'P', 'G'), 30, Size(frame_width, frame_height));

	Mat frame;

	for(;;)	
	{
		// Get a frame
		vid_in >> frame;

		// This happens when the video is over.
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

		// Erode the white edges
		int erosion_size = 1;
		Mat erosion_element = getStructuringElement(
			MORPH_RECT,
			Size(2 * erosion_size + 1, 2 * erosion_size + 1),
			Point(erosion_size, erosion_size));

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

		// No sections (the image was all white)
		if(section_sizes.size() == 0)
			continue;

		// Sort the section sizes in ascending order
		multiset<float_int_pair> sorted_section_sizes;

		for(map<float, int>::const_iterator ci = section_sizes.begin(); ci != section_sizes.end(); ci++)
		{
			float_int_pair ip;
			ip.a = ci->first;
			ip.b = ci->second;

			sorted_section_sizes.insert(ip);
		}

		Mat output(flt_canny.rows, flt_canny.cols, CV_8UC3);

		// Get the largest section size
		int largest_section_size = sorted_section_sizes.rbegin()->b;

		// For each section in flt_canny
		for(multiset<float_int_pair>::const_iterator ci = sorted_section_sizes.begin(); ci != sorted_section_sizes.end(); ci++)
		{
			// Iterate throughout the whole image
			for(int j = 0; j < flt_canny.rows; j++)
			{
				for(int i = 0; i < flt_canny.cols; i++)
				{
					float colour = flt_canny.at<float>(j, i);

					// If colour and ci->a match, colour orange by ci->b
					// Else if white, colour white
					if(colour == ci->a)
					{
						// Some arbitrary function
						float output_colour = 255.0f*pow(1.0f - ci->b / static_cast<float>(largest_section_size), 5.0f);

						// Mark each section in orange
						output.at<Vec3b>(j, i)[0] = 0;
						output.at<Vec3b>(j, i)[1] = static_cast<unsigned char>(output_colour*0.5f);
						output.at<Vec3b>(j, i)[2] = static_cast<unsigned char>(output_colour);
					}
					else if(colour == flt_white)
					{
						output.at<Vec3b>(j, i)[0] = 255;
						output.at<Vec3b>(j, i)[1] = 255;
						output.at<Vec3b>(j, i)[2] = 255;
					}
				}
			}
		}

		imshow("output", output);
		vid_out.write(output);

		if(waitKey(33) >= 0)
			break;
	}

	destroyAllWindows();

	return 0;
}
