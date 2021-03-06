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
	// http://www.dofpro.com/cgigallery.htm
	Mat frame = imread("diet_pepsi.jpg", CV_LOAD_IMAGE_COLOR);
	Mat depth_map = imread("diet_pepsi_depth.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	imshow("frame", frame);
	imshow("depth map", depth_map);

	Mat depth_canny;
	Canny(depth_map, depth_canny, 50, 150, 3, true);

	// Convert frame from BGR to HSV
	Mat hsv;
	cvtColor(frame, hsv, CV_BGR2HSV);

	// Get separate H, S, V channels
	Mat hsv_channels[3];
	split(hsv, hsv_channels);

	Mat h_canny, s_canny, v_canny;
	Canny(hsv_channels[0], h_canny, 50, 150, 3, true);
	Canny(hsv_channels[1], s_canny, 50, 150, 3, true);
	Canny(hsv_channels[2], v_canny, 50, 150, 3, true);
	
	Mat or_result;
	bitwise_or(depth_canny, h_canny, or_result);
	bitwise_or(s_canny, or_result, or_result);
	bitwise_or(v_canny, or_result, or_result);

	// Assume that there are less than 100000 sections in the image.
	// If this number is not high enough, change it accordingly.
	const float flt_white = 100000.0f;

	// Mark white edges and black sections on floating point image
	Mat flt_canny(or_result.rows, or_result.cols, CV_32F);

	for (int j = 0; j < or_result.rows; j++)
	{
		for (int i = 0; i < or_result.cols; i++)
		{
			if (or_result.at<unsigned char>(j, i) == 255)
				flt_canny.at<float>(j, i) = flt_white;
			else
				flt_canny.at<float>(j, i) = 0.0f; // Make sure to initialize the whole image.
		}
	}

	// Find black sections in flt_canny, fill using floodFill.
	int section_count = 0;
	int fill_colour = 1;

	for (int j = 0; j < flt_canny.rows; j++)
	{
		for (int i = 0; i < flt_canny.cols; i++)
		{
			float colour = flt_canny.at<float>(j, i);

			if (colour == 0)
			{
				floodFill(flt_canny, Point(i, j), Scalar(fill_colour));

				section_count++;
				fill_colour++;
			}
		}
	}

	// Get number of pixels per section colour
	map<float, int> section_sizes;

	for (int j = 0; j < flt_canny.rows; j++)
	{
		for (int i = 0; i < flt_canny.cols; i++)
		{
			float colour = flt_canny.at<float>(j, i);

			if (colour != flt_white)
				section_sizes[colour]++;
		}
	}

	// No sections (the image was all white)
	if (section_sizes.size() == 0)
		return 0;

	// Sort the section sizes in ascending order
	multiset<float_int_pair> sorted_section_sizes;

	for (map<float, int>::const_iterator ci = section_sizes.begin(); ci != section_sizes.end(); ci++)
	{
		float_int_pair ip;
		ip.a = ci->first;
		ip.b = ci->second;

		sorted_section_sizes.insert(ip);
	}

	Mat output(flt_canny.rows, flt_canny.cols, CV_8UC3);

	// Get the largest section size
	int largest_section_size = sorted_section_sizes.rbegin()->b;

	// Keep track of all the depth measurements per section.
	int depths_index = 0;
	vector< vector< float > > depths;
	depths.resize(sorted_section_sizes.size());

	// For each section in flt_canny
	for (multiset<float_int_pair>::const_iterator ci = sorted_section_sizes.begin(); ci != sorted_section_sizes.end(); ci++)
	{
		// Iterate throughout the whole image
		for (int j = 0; j < flt_canny.rows; j++)
		{
			for (int i = 0; i < flt_canny.cols; i++)
			{
				float colour = flt_canny.at<float>(j, i);

				// If colour and ci->a match, colour orange by ci->b
				// Else if white, colour white
				if (colour == ci->a)
				{
					// Some arbitrary function
					float output_colour = 255.0f*(1.0f - ci->b / static_cast<float>(largest_section_size));

					// Mark each section in orange
					// The unused blue channel stores the depth
					output.at<Vec3b>(j, i)[0] = depth_map.at<unsigned char>(j, i);
					output.at<Vec3b>(j, i)[1] = static_cast<unsigned char>(output_colour*0.5f);
					output.at<Vec3b>(j, i)[2] = static_cast<unsigned char>(output_colour);

					depths[depths_index].push_back(depth_map.at<unsigned char>(j, i));
				}
				else if (colour == flt_white)
				{
					output.at<Vec3b>(j, i)[0] = 255;
					output.at<Vec3b>(j, i)[1] = 255;
					output.at<Vec3b>(j, i)[2] = 255;
				}
			}
		}

		depths_index++;
	}

	// Get average depth per section.
	vector<float> avg_depths;
	avg_depths.resize(depths.size());

	for (int i = 0; i < avg_depths.size(); i++)
	{
		// Get sum
		for (int j = 0; j < depths[i].size(); j++)
			avg_depths[i] += depths[i][j];

		// Get average
		avg_depths[i] /= static_cast<float>(depths[i].size());

		// Normalize. The higher the average depth, the closer the section is to the camera.
		avg_depths[i] /= 255.0f;
	}

	float proximity_warning = 0.99f;
	depths_index = 0;

	// For each section in flt_canny
	for (multiset<float_int_pair>::const_iterator ci = sorted_section_sizes.begin(); ci != sorted_section_sizes.end(); ci++)
	{
		// Iterate throughout the whole image
		for (int j = 0; j < flt_canny.rows; j++)
		{
			for (int i = 0; i < flt_canny.cols; i++)
			{
				float colour = flt_canny.at<float>(j, i);

				// If colour and ci->a match then colour red if
				// closer than the proximity warning distance
				if (colour == ci->a)
				{
					if (proximity_warning < avg_depths[depths_index])
					{
						output.at<Vec3b>(j, i)[0] = 0;
						output.at<Vec3b>(j, i)[1] = 0;
						output.at<Vec3b>(j, i)[2] = 255;
					}
				}
			}
		}

		depths_index++;
	}

	imshow("or", or_result);
	imshow("output", output);

	waitKey(0);

	destroyAllWindows();

	return 0;
}
