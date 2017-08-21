#include "opencv2/opencv.hpp"
#include "VideoFaceDetector.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string.h>

using namespace cv;

Mat getHandMask(const Mat& image, int minCr = 128, int maxCr = 170, int minCb = 73, int maxCb = 158);
Point getHandCenter(const Mat& mask, double& radius);
int getFingerCount(const Mat& mask, Point center, double radius, double scale = 1.9);

std::vector<Mat>				planes;
std::vector<std::vector<Point>> contours;
const cv::String				CASCADE_FILE("haarcascade_frontalface_default.xml");

int main(int argc, char** argv) {


	VideoCapture cap;
	VideoFaceDetector detector(CASCADE_FILE, cap);
	if (!cap.open(0))
		return 0;

	double fps = 0, time_per_frame;

	for (;;) {

		Mat frame;
		auto start = cv::getCPUTickCount();
		detector >> frame;

		const string& text = "count: ";
		const string& fpstext = ", fps: ";

		if (detector.isFaceFound())
		{
			cv::rectangle(frame, detector.face(), cv::Scalar(0, 0, 0), CV_FILLED);
			//cv::circle(frame, detector.facePosition(), 30, cv::Scalar(0, 255, 0));
		}

		Mat mask = getHandMask(frame);
		Mat handImage(frame.size(), CV_8UC3, Scalar(0));

		erode(mask, mask, Mat(3, 3, CV_8U, Scalar(1)), Point(-1, -1), 2);
		erode(mask, mask, Mat(3, 3, CV_8U, Scalar(1)), Point(-1, -1), 2);

		add(frame, Scalar(0), handImage, mask);

		double radius;

		Point center = getHandCenter(mask, radius);
		int fingerCount = getFingerCount(mask, center, radius);

		auto end = cv::getCPUTickCount();
		time_per_frame = (end - start) / cv::getTickFrequency();
		fps = (15 * fps + (1 / time_per_frame)) / 16;


		std::cout << "�չٴ� �߽��� ��ǥ:" << center << ", ������:" << radius << ", �հ��� ��:" << fingerCount << std::endl;

		//�չٴ� �߽��� �׸���
		circle(frame, center, 2, Scalar(0, 255, 0), -1);

		//�չٴ� ���� �׸���
		circle(frame, center, (int)(radius * 1.9), Scalar(255, 0, 0), 2);


		putText(frame, text + std::to_string(fingerCount) + fpstext + std::to_string(time_per_frame), Point(10, 30), CV_FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 0, 255), 1);

		imshow("mask", mask);
		imshow("Hand Image", handImage);
		imshow("frame", frame);

		if (waitKey(30) >= 0) break;
	}

	return 0;
}

Mat getHandMask(const Mat& image, int minCr, int maxCr, int minCb, int maxCb) {
	//�÷� ���� ��ȯ BGR->YCrCb
	Mat YCrCb;
	cvtColor(image, YCrCb, CV_BGR2YCrCb);

	//�� ä�κ��� �и�
	split(YCrCb, planes);

	//�� ä�κ��� ȭ�Ҹ��� ��
	Mat mask(image.size(), CV_8U, Scalar(0));   //��� ����ũ�� ������ ����
	uchar* data = (uchar*) mask.data;
	int nr = image.rows;    //��ü ���� ��
	int nc = image.cols;

	for (int i = 0; i<nr; i++) {
		uchar* CrPlane = planes[1].ptr<uchar>(i);   //Crä���� i��° �� �ּ�
		uchar* CbPlane = planes[2].ptr<uchar>(i);   //Cbä���� i��° �� �ּ�
		for (int j = 0; j<nc; j++) {
			if ((minCr < CrPlane[j]) && (CrPlane[j] <maxCr) && (minCb < CbPlane[j]) && (CbPlane[j] < maxCb))
				data[i * nc + j] = 255;
		}
	}

	return mask;
}

//�չٴ��� �߽����� ������ ��ȯ
//�Է��� 8bit ���� ä��(CV_8U), �������� ������ double�� ����
Point getHandCenter(const Mat& mask, double& radius) {
	//�Ÿ� ��ȯ ����� ������ ����
	Mat dst;
	distanceTransform(mask, dst, CV_DIST_L2, 5);  //����� CV_32SC1 Ÿ��

												  //�Ÿ� ��ȯ ��Ŀ��� ��(�Ÿ�)�� ���� ū �ȼ��� ��ǥ��, ���� ���´�.
	int maxIdx[2];    //��ǥ ���� ���� �迭(��, �� ������ �����)
	minMaxIdx(dst, NULL, &radius, NULL, maxIdx, mask);   //�ּҰ��� ��� X

	return Point(maxIdx[1], maxIdx[0]);
}

//�ո��� �������� ���� ���¿��� �հ��� ���� ����
int getFingerCount(const Mat& mask, Point center, double radius, double scale) {
	//�հ��� ������ ���� ���� �� �׸���
	Mat cImg(mask.size(), CV_8U, Scalar(0));
	circle(cImg, center, radius*scale, Scalar(255));

	//���� �ܰ����� ������ ����

	findContours(cImg, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	if (contours.size() == 0)   //�ܰ����� ���� �� == �� ���� X
		return -1;

	//�ܰ����� ���� ���� mask�� ���� 0���� 1�� �ٲ�� ���� Ȯ��
	int fingerCount = 0;
	for (int i = 1; i<contours[0].size(); i++) {
		Point p1 = contours[0][i - 1];
		Point p2 = contours[0][i];
		if (mask.at<uchar>(p1.y, p1.x) == 0 && mask.at<uchar>(p2.y, p2.x)>1)
			fingerCount++;
	}
	if (fingerCount > 6)
		return -1;

	//�ո�� ������ ���� 1�� ����
	return fingerCount - 1;
}
