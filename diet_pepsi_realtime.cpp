#pragma comment(lib, "opencv_world330.lib")

#include <opencv2/opencv.hpp>
using namespace cv;

int main(void)
{
	// http://www.dofpro.com/cgigallery.htm
	Mat frame = imread("diet_pepsi.jpg", CV_LOAD_IMAGE_COLOR);
	Mat depth_map = imread("diet_pepsi_depth.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	imshow("frame", frame);
	imshow("depth map", depth_map);

	float proximity_warning = 0.9f;

	Mat output(depth_map.rows, depth_map.cols, CV_8UC3);

	// Iterate throughout the whole image
	for (int j = 0; j < depth_map.rows; j++)
	{
		for (int i = 0; i < depth_map.cols; i++)
		{
			unsigned char colour = depth_map.at<unsigned char>(j, i);

			if (colour / 255.0f > proximity_warning)
			{
				output.at<Vec3b>(j, i)[0] = 0;
				output.at<Vec3b>(j, i)[1] = 0;
				output.at<Vec3b>(j, i)[2] = 255;
			}
			else
			{
				output.at<Vec3b>(j, i)[0] = colour;
				output.at<Vec3b>(j, i)[1] = colour;
				output.at<Vec3b>(j, i)[2] = colour;
			}
		}
	}

	imshow("output", output);

	waitKey(0);

	destroyAllWindows();

	return 0;
}
