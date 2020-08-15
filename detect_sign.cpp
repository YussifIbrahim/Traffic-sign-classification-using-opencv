/*
 * last.cpp
 *
 *  Created on: 11.08.2020
 *      Author:Shen & yibrahim
 */


#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <iterator>
#include <stdio.h>
#include <time.h>
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui_c.h>
#include "numpy/arrayobject.h"



#define PI 3.1415926


using namespace std;
using namespace cv;

bool ContoursSortFun(vector<Point> contour1, vector<Point> contour2)
{
	return (contourArea(contour1) > contourArea(contour2));
}

void getHSV(double red, double green, double blue, double& hue, double& saturation, double& intensity)
{
	double r, g, b;
	double h, s, i;

	double sum;
	double minRGB, maxRGB;
	double theta;

	r = red / 255.0;
	g = green / 255.0;
	b = blue / 255.0;

	minRGB = ((r < g) ? (r) : (g));
	minRGB = (minRGB < b) ? (minRGB) : (b);

	maxRGB = ((r > g) ? (r) : (g));
	maxRGB = (maxRGB > b) ? (maxRGB) : (b);

	sum = r + g + b;
	i = sum / 3.0;

	if (i < 0.001 || maxRGB - minRGB < 0.001)
	{
		//this is a black image or grayscale image
		//in this circumstance, hue is undefined, not zero
		h = 0.0;
		s = 0.0;
		//return ;
	}
	else
	{
		s = 1.0 - 3.0 * minRGB / sum;
		theta = sqrt((r - g) * (r - g) + (r - b) * (g - b));
		theta = acos((r - g + r - b) * 0.5 / theta);
		if (b <= g)
			h = theta;
		else
			h = 2 * PI - theta;
		if (s <= 0.01)
			h = 0;
	}

	hue = (int)(h * 180 / PI);
	saturation = (int)(s * 100);
	intensity = (int)(i * 100);
}

void fillHole(const Mat src, Mat& dst)
{
	Size ImgSize = src.size();
	Mat Temp = Mat::zeros(ImgSize.height + 2, ImgSize.width + 2, src.type());
	//Mat Temp = Mat::zeros(ImgSize.height + 2, ImgSize.width + 2, src.type());

	src.copyTo(Temp(Range(1, ImgSize.height + 1), Range(1, ImgSize.width + 1)));
	floodFill(Temp, Point(0, 0), Scalar(255));

	Mat cutImg;
	Temp(Range(1, ImgSize.height + 1), Range(1, ImgSize.width + 1)).copyTo(cutImg);

	dst = src | (~cutImg);
}

double getAvg(Mat img)//convert to gray image and then get the average of value
{
	Mat gray;
	cvtColor(img, gray, COLOR_RGB2GRAY);
	Scalar scalar = mean(gray);
	//cout << "getAvg is " << scalar.val[0] << endl;
	return scalar.val[0];
}
void detectCircle(Mat srcImg, Mat outImage ) {




}

void hsvExtraction(Mat srcImg, const double aveValue, Mat& rangeImg, int width, int height) {

	rangeImg = Mat::zeros(srcImg.size(), CV_8UC1);
	double H = 0.0, S = 0.0, V = 0.0, R = 0.0, G = 0.0, B = 0.0;
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			B = srcImg.at<Vec3b>(y, x)[0];
			G = srcImg.at<Vec3b>(y, x)[1];
			R = srcImg.at<Vec3b>(y, x)[2];
			getHSV(R, G, B, H, S, V);

			if (aveValue > 120)
			{

				if ((((H >= 300 && H <= 360) || (H >= 0 && H <= 25)) && (S >= 30 && S <= 95) && (V >= 20 && V < 95)) // red range: normally the range of red is 337~360 or 0~25, but in case of the dark enviroment, just expand the range to purple.
				//|| (((H >= 49 && H <= 62.5)) && (S >= 30 && S <= 95) && (V > 40 && V < 85)) //yellow range
					|| ((H >= 190 && H <= 255) && (S >= 30 && S <= 95) && (V > 20 && V < 95))//blue range
					|| (((H >= 0 && H <= 300)) && (S >= 6 && S <= 10) && (V > 37 && V < 50))
					//|| (((H >= 249 && H <= 300) || (H >= 56 && H <= 150)) && (S >= 6 && S <= 10) && (V > 37 && V < 50))

					) //gray range

				{
					rangeImg.at<uchar>(y, x) = 255;
					//all of pixel which in this range will converted to 255(white).
				}
			}
			else if (aveValue <= 120 && aveValue > 90)
			{
				{
					if (
						(
						((H >= 300 && H <= 360) || (H >= 0 && H <= 8)) && (S >= 20 && S <= 85) && (V >= 13 && V < 50)) // red range: normally the range of red is 337~360 or 0~25, but in case of the dark enviroment, just expand the range to purple.
						|| (((H >= 46 && H <= 55)) && (S >= 16 && S <= 100) && (V > 30 && V < 63)) //yellow range
						|| ((H >= 210 && H <= 248) && (S >= 20 && S <= 100) && (V > 10 && V < 50))//blue range
						//|| ((S >= 0 && S <= 1) && (V > 37 && V < 40))//gray range
						)


					{
						rangeImg.at<uchar>(y, x) = 255;
						//all of pixel which in this range will converted to 255(white).
					}
				}
			}
			else if (aveValue <= 90 && aveValue > 60)
			{
				{
					if ((((H >= 300 && H <= 360) || (H >= 0 && H <= 10)) && (S >= 3 && S <= 20) && (V >= 7 && V < 35)) //red
						|| (((H >= 31 && H <= 55)) && (S >= 12 && S <= 100) && (V > 12 && V < 50)) //yellow range
						|| ((H >= 220 && H <= 248) && (S >= 30 && S <= 100) && (V > 16 && V < 50))//blue range
						//|| ((S >= 0 && S <= 4) && (V > 22 && V < 95))
						) //gray range


					{
						rangeImg.at<uchar>(y, x) = 255;
						//all of pixel which in this range will converted to 255(white).
					}
				}
			}
			else if (aveValue <= 60 && aveValue > 30)
			{
				{
					if ((((H >= 300 && H <= 360) || (H >= 0 && H <= 10)) && (S >= 5 && S <= 95) && (V >= 15 && V < 30)) // red range: normally the range of red is 337~360 or 0~25, but in case of the dark enviroment, just expand the range to purple.
						|| (((H >= 31 && H <= 55)) && (S >= 12 && S <= 100) && (V > 10 && V < 50)) //yellow range
						|| ((H >= 220 && H <= 248) && (S >= 30 && S <= 100) && (V > 16 && V < 50))//blue range
						|| ((S >= 0 && S <= 8.5) && (V > 14 && V < 37))) //gray range


					{
						rangeImg.at<uchar>(y, x) = 255;
						//all of pixel which in this range will converted to 255(white).
					}
				}
			}
			else
			{
				if ((((H >= 35 && H <= 100)) && (S >= 10 && S <= 100) && (V > 10 && V < 85)) //yellow range
					) //gray range
				{
					rangeImg.at<uchar>(y, x) = 255;
					//cout << "H " << H << "; S " << S << "; V " << V << endl;//all of pixel which in this range will converted to 255(white).
				}
			}
		}
	}
}

void frameProcess(Mat srcImg, Mat& original, vector<Rect>& wanted) {
	Mat output;
	Mat downsized;
	//Mat original;
	double ratioDownsize = 1.5;
	const int colDownsize = srcImg.cols / ratioDownsize;
	const int rowDownsize = srcImg.rows / ratioDownsize;
	resize(srcImg, downsized, Size(colDownsize, rowDownsize));


	//Mat cropped = downsized(Rect(0, downsized.rows*0.2, downsized.cols, downsized.rows*0.55));
	//Mat cropped = downsized(Rect(downsized.cols*0.1, downsized.rows*0.2, downsized.cols*0.35, downsized.rows*0.55));
	Mat cropped1 = downsized(Rect(downsized.cols*0.55, downsized.rows*0.2, downsized.cols*0.35, downsized.rows*0.55));
	Mat cropTest = Mat::zeros(downsized.size(), CV_8UC3);
	//Mat cropRoi = cropTest(Rect(0, downsized.rows*0.2, cropTest.cols, cropTest.rows*0.55));
	//Mat cropRoi = cropTest(Rect(downsized.cols*0.1, downsized.rows*0.2, downsized.cols*0.35, downsized.rows*0.55));
	Mat cropRoi_1 = cropTest(Rect(downsized.cols*0.55, downsized.rows*0.2, downsized.cols*0.35, downsized.rows*0.55));
	//cropped.copyTo(cropRoi);
	cropped1.copyTo(cropRoi_1);





	double aveValue = getAvg(downsized); // get the average value of all gray pixels
	std::cout << "aveValue is " << aveValue << std::endl;
	Mat hsvExtracted = Mat::zeros(downsized.size(), CV_8UC1);
	hsvExtraction(cropTest, aveValue, hsvExtracted, colDownsize, rowDownsize);//Feature extraction in HSV color space.

	Mat hsvWhite;



	Mat median;
	medianBlur(hsvExtracted, median, 5);//To remove some noise pixels.


	Mat erodeElement = getStructuringElement(MORPH_ELLIPSE,
		Size(2 * 1 + 1, 2 * 1 + 1),
		Point(1, 1));
	int Element1[5][5] = { { 0,0,1,0,0 },{ 0,1,1,1,0 },{ 1,1,1,1,1 },{ 0,1,1,1,0 },{ 0,0,1,0,0 } };

	Mat dilateElement = getStructuringElement(MORPH_ELLIPSE,
		Size(7 * 1 + 1, 7 * 1 + 1),
		Point(1, 1));
	double AreaImage = median.rows * median.cols;


	Mat after_erode, after_morph;//To remove some noise pixels.
	if (aveValue > 70 && AreaImage > 2500) {
		//because when the aveValue < 70, the image is dark and use erode function will reduce the number of white point.
		erode(median, after_erode, erodeElement);
		dilate(after_erode, after_morph, dilateElement);
		//cv::imshow("dilate", after_morph);

	}
	else {
		dilate(median, after_morph, dilateElement);
	}


	Mat filled = Mat::zeros(downsized.size(), CV_8UC1);;
	fillHole(after_morph, filled);
	//cv::imshow("filled", filled);

	output = filled;



	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(filled, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));



	sort(contours.begin(), contours.end(), ContoursSortFun);

	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());
	vector<Point2f>centerOfCircle(contours.size());
	vector<float>radiusOfCircle(contours.size());
	vector<Rect> oriRect(contours.size());
	//vector<Rect> wanted;

	//Mat original;
	original = downsized.clone();
	for (unsigned int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
		boundRect[i] = boundingRect(Mat(contours_poly[i]));
		minEnclosingCircle(contours_poly[i], centerOfCircle[i], radiusOfCircle[i]);
	}

	Mat drawing = Mat::zeros(downsized.size(), CV_8UC3);
	Mat srcImgCopy;
	downsized.copyTo(srcImgCopy);

	for (unsigned int i = 0; i < contours.size(); i++)
	{
		Rect rect = boundRect[i];

		float ratioOfRect = abs((float)rect.width / (float)rect.height);

		float ContourArea = (float)contourArea(contours[i]);

		if (ContourArea < downsized.cols * downsized.rows / 900 || ContourArea > downsized.cols * downsized.rows / 100) {

			continue;

		}

		else if (ratioOfRect > 1.5 || ratioOfRect < 0.9)
		{

			continue;

		}

		else {



			Scalar color = (0, 0, 255);
			drawContours(drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point());
			rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
			rectangle(original, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
			wanted.push_back(boundRect[i]);

		}




	}

}

int main(int argc, char *argv[])
{
	clock_t startTime, endTime;
	clock_t resizeTime, hsvExtraTime, medianblurTime, dialteTime, fillTime, findConTime, sortConTime, drawTime;
	float duration;

/*This part of the code detects the road signs which sometimes include false positives. All the detections are sent to an ANN 
*written in python using keras to make the predictions. there is where the false positives are discarded. So we embedd python 
*code in this c++ to perform this actions
*/
//Initialise the python module
	Py_Initialize();

	//import_array();


  //Add the file path to system path
	PyObject* sysPath = PySys_GetObject("path");
	PyList_Insert(sysPath, 0, PyUnicode_FromString("/home/yibrahim/new_workspace/New_Car"));
//name the python script to be run
	PyObject *pName = PyUnicode_FromString("speed_test");


	PyObject *pModule = PyImport_Import(pName);
//name the function in the python script to be called
	PyObject *pFunc = PyObject_GetAttrString(pModule, "prediction");



//start capture object, and declare a mat object for the frame containing detections
	VideoCapture capture;
	VideoWriter outputVideo;
	Mat frame;
//capture frame
	frame = capture.open("/home/yibrahim/new_workspace/New_Car/light_output.mp4");
	string outputVideoPath = "/home/yibrahim/new_workspace/New_Car";
	cv::Size S = cv::Size((int)(capture.get(CAP_PROP_FRAME_WIDTH) / 1.5),
		(int)(capture.get(CAP_PROP_FRAME_HEIGHT) / 1.5));
	int codec = VideoWriter::fourcc('X', 'V', 'I', 'D');
	outputVideo.open(outputVideoPath, codec, 30, S, true);

	if (!capture.isOpened())
	{
		printf("can not open ...\n");
		return -1;
	}
	namedWindow("output", CV_WINDOW_AUTOSIZE);
	try {

		while (capture.read(frame))
		{


        //declare mat object to store output and vector to store the bounding box cordinates.
				Mat output;
				vector<Rect> boxes,want;
        //initialise an array to store the image 
				npy_intp dimensions[] = { frame.rows, frame.cols, frame.channels() };
        
        //declare integer to store size of data elements in frame
				int nElem = frame.rows * frame.cols * frame.channels();
        
        //declare character to help in getting the size of data to be paresed to python script
				uchar* m = new uchar[nElem];
        
        //copy frame to m 
				std::memcpy(m, frame.data, nElem * sizeof(uchar));


				std::cout << "dimension is " << frame.size << std::endl;
				std::cout << "number of cols is  " << frame.cols << std::endl;
				std::cout << "number of rows is  " << frame.rows << std::endl;
				std::cout << "number of channels is  " << frame.channels() << std::endl;
				std::cout << "dims  is  " << frame.dims << std::endl;



			//process the frame to make detections
				frameProcess(frame, output, boxes);


			//output = frame;
				//outputVideo << output;

				for (int i = 0; i < int(boxes.size()); i++) {
					int x = boxes[i].x;
					int y = boxes[i].y;
					int w = boxes[i].width;
					int h = boxes[i].height;

					std::cout << "the box cordinates for x is " << boxes[i].x << std::endl;
					std::cout << "the box cordinates for y is " << boxes[i].y << std::endl;
					std::cout << "the box cordinates for width is " << boxes[i].width << std::endl;
					std::cout << "the box cordinates for height is " << boxes[i].height<< std::endl;





					if (pModule)
					{

            //create a tuple which is required to pass arguments to python function
						PyObject* pPosArgs = PyTuple_New(5);
            
            //insert positional arguments into the tuple
						PyObject* Res = PyObject_Repr(pPosArgs);
						std::cout << "This is to check \n" << PyUnicode_AsUTF8(Res) << std::endl;



						PyObject * pVal1 = PyArray_SimpleNewFromData(3, dimensions, NPY_UINT8,m);
						PyErr_Print();
						PyObject* Result = PyObject_Repr(pVal1);
						std::cout << "Py simple Data \n" << PyUnicode_AsUTF8(Result) << std::endl;
						//PyErr_Print();
						std::cin.get();


						int rc = PyTuple_SetItem(pPosArgs, 0, pVal1);

						PyObject* pVal_x = PyLong_FromLong(x);
						rc = PyTuple_SetItem(pPosArgs, 1, pVal_x);

						PyObject* pVal_y = PyLong_FromLong(y);
						rc = PyTuple_SetItem(pPosArgs, 2, pVal_y);

						PyObject* pVal_w = PyLong_FromLong(w);
						rc = PyTuple_SetItem(pPosArgs, 3, pVal_w);

						PyObject* pVal_h = PyLong_FromLong(h);
						rc = PyTuple_SetItem(pPosArgs, 4, pVal_h);



					
            //calls the python method and pass the arguments and recieve the return value back in c++
						if (pFunc && PyCallable_Check(pFunc))
						{
							PyObject* pValue = PyObject_CallObject(pFunc, pPosArgs);

							printf("C: getInteger() = %s\n", PyUnicode_AsUTF8(pValue));


							Py_XDECREF(pVal_x);
							Py_XDECREF(pVal_y);
							Py_XDECREF(pVal_w);
							Py_XDECREF(pVal_h);
							Py_XDECREF(pVal1);
							Py_XDECREF(pValue);
							Py_XDECREF(pPosArgs);


						}
						else
						{
							printf("ERROR: function getInteger()\n");
						}

					}
					else
					{
						printf("ERROR: Module not imported\n");


					}


					Py_XDECREF(sysPath);


					std::cin.get();



				}


				endTime = clock();
				duration = (double)(endTime - startTime) / CLOCKS_PER_SEC;
				std::cout << "duration is " << duration << std::endl;
				std::cout << output.cols << "x" << output.rows << std::endl;

				//cv::imshow("output", output);
				cv::waitKey(20);

				delete[] m;
		}
	}

	catch (cv::Exception&e)
	{
		std::cout << e.what() << std::endl << std::endl;

	};
	Py_XDECREF(pFunc);
	Py_XDECREF(pName);
	Py_XDECREF(pModule);
	Py_Finalize();

	capture.release();


	return 0;
}

