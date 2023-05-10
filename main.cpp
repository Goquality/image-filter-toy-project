#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>


#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <opencv2/imgproc/imgproc.hpp>

#include "ldmarkmodel.h"

using namespace std;
using namespace cv;

void rotcolor(Mat& src, Mat& dst, double deg)
{
    float rad = deg * 3.14 / 180.0;
    float R[2][2] = { {cos(rad), sin(rad)},{-sin(rad),cos(rad)} };
    float rx, ry, fx1, fx2, fy1, fy2, w1, w2, w3, w4;
    int px, py;
    for (int j = 0; j < src.rows; j++) for (int i = 0; i < src.cols; i++)
    {
        rx = (float)((double)R[0][0] * (i - (int)src.cols / 2) + (double)R[0][1] * (j - (int)src.rows / 2));
        ry = (float)((double)R[1][0] * (i - (int)src.cols / 2) + (double)R[1][1] * (j - (int)src.rows / 2));

        rx = rx + src.cols / 2;
        ry = ry + src.rows / 2;

        px = (int)rx;
        py = (int)ry;

        fx1 = (float)rx - (float)px;
        fx2 = 1 - fx1;
        fy1 = (float)ry - (float)py;
        fy2 = 1 - fy1;

        w1 = (float)fx2 * fy2;
        w2 = (float)fx1 * fy2;
        w3 = (float)fx2 * fy1;
        w4 = (float)fx1 * fy1;
        if (rx >= 0 && rx < (float)src.cols - 1 && ry >= 0 && ry < (float)src.rows - 1)
        {
            Vec3b P1 = src.at<Vec3b>(py, px);
            Vec3b P2 = src.at<Vec3b>(py, px + 1);
            Vec3b P3 = src.at<Vec3b>(py + 1, px);
            Vec3b P4 = src.at<Vec3b>(py + 1, px + 1);

            dst.at<Vec3b>(j, i) = w1 * P1 + w2 * P2 + w3 * P3 + w4 * P4;
        }
        else
        {
            dst.at<Vec3b>(j, i) = 0;
        }
    }
}
void rotgray(Mat& src, Mat& dst, double deg)
{
    float rad = deg * 3.14 / 180.0;
    float R[2][2] = { {cos(rad), sin(rad)},{-sin(rad),cos(rad)} };
    float rx, ry, fx1, fx2, fy1, fy2, w1, w2, w3, w4;
    int px, py;
    for (int j = 0; j < src.rows; j++) for (int i = 0; i < src.cols; i++)
    {
        rx = (float)((double)R[0][0] * (i - (int)src.cols / 2) + (double)R[0][1] * (j - (int)src.rows / 2));
        ry = (float)((double)R[1][0] * (i - (int)src.cols / 2) + (double)R[1][1] * (j - (int)src.rows / 2));

        rx = rx + src.cols / 2;
        ry = ry + src.rows / 2;

        px = (int)rx;
        py = (int)ry;

        fx1 = (float)rx - (float)px;
        fx2 = 1 - fx1;
        fy1 = (float)ry - (float)py;
        fy2 = 1 - fy1;

        w1 = (float)fx2 * fy2;
        w2 = (float)fx1 * fy2;
        w3 = (float)fx2 * fy1;
        w4 = (float)fx1 * fy1;
        if (rx >= 0 && rx < (float)src.cols - 1 && ry >= 0 && ry < (float)src.rows - 1)
        {
            uchar P1 = src.at<uchar>(py, px);
            uchar P2 = src.at<uchar>(py, px + 1);
            uchar P3 = src.at<uchar>(py + 1, px);
            uchar P4 = src.at<uchar>(py + 1, px + 1);

            dst.at<uchar>(j, i) = w1 * P1 + w2 * P2 + w3 * P3 + w4 * P4;
        }
        else
        {
            dst.at<uchar>(j, i) = 0;
        }
    }
}

int main()
{
    int flag = 0;
    Mat image = imread("squid.jpg");
    Mat mask = imread("squid2.png");
    Rect rectangle(1, 1, image.cols, image.rows);
    Mat result(image.rows, image.cols, CV_8UC1);
    for (int j = 0; j < image.rows; j++)    for (int i = 0; i < image.cols; i++)
        result.at<uchar>(j, i) = GC_PR_FGD;
    Mat bg, fg;
    for (int j = 0; j < image.rows - 2; j++)    for (int i = 0; i < image.cols - 2; i++) {
        if (mask.at<Vec3b>(j, i)[0] >= 235 && mask.at<Vec3b>(j, i)[1] <= 10 && mask.at<Vec3b>(j, i)[2] <= 10)
            result.at<uchar>(j, i) = GC_BGD;
        else if (mask.at<Vec3b>(j, i)[0] <= 10 && mask.at<Vec3b>(j, i)[1] <= 10 && mask.at<Vec3b>(j, i)[2] >= 235)
            result.at<uchar>(j, i) = GC_FGD;
    }
    printf("second step..\n");
    ldmarkmodel modelt;
    std::string modelFilePath = "roboman-landmark-model.bin";
    while (!load_ldmarkmodel(modelFilePath, modelt)) {
        std::cout << "fail loag ldmarkmodel." << std::endl;
        std::cin >> modelFilePath;
    }

    cv::VideoCapture mCamera(0);
    if (!mCamera.isOpened()) {
        std::cout << "Camera opening failed..." << std::endl;
        system("pause");
        return 0;
    }

    cv::Mat Image;
    cv::Mat current_shape;
    cv::Mat image2(image.rows, image.cols, CV_8UC3);
    cv::Mat result2(result.rows, result.cols, CV_8UC1);
    cv::Mat image3;
    cv::Mat result3;

    for (;;) {
        mCamera >> Image;
        modelt.track(Image, current_shape);
        cv::Vec3d eav;
        modelt.EstimateHeadPose(current_shape, eav);
        modelt.drawPose(Image, current_shape, 50);
        int numLandmarks = current_shape.cols / 2;
        int x1 = current_shape.at<float>(36);
        int x2 = current_shape.at<float>(45);
        int y1 = current_shape.at<float>(36 + numLandmarks);
        int y2 = current_shape.at<float>(45 + numLandmarks);
        int x3 = current_shape.at<float>(8);
        int y3 = current_shape.at<float>(8 + numLandmarks);
        printf("%.2f\n", eav[2]);

        int m = 3 * (x2 - x1), n = (int)2 * (y3 - (y2 + y1) / 2);
        rotcolor(image, image2, eav[2]);
        rotgray(result, result2, eav[2]);
        cv::resize(result2, result3, Size(m, n));
        cv::resize(image2, image3, Size(m, n));
        for (int j = 0; j < n; j++) for (int i = 0; i < m; i++) {
            if (result3.at<uchar>(j, i) == GC_FGD && j + y3 - n >= 0 && j + y3 - n < Image.rows && i + x1 - m / 3 >= 0 && i + x1 - m / 3 < Image.cols) {
                Image.at<Vec3b>(j + y3 - n, i + x1 - m / 3)[0] = image3.at<Vec3b>(j, i)[0];
                Image.at<Vec3b>(j + y3 - n, i + x1 - m / 3)[1] = image3.at<Vec3b>(j, i)[1];
                Image.at<Vec3b>(j + y3 - n, i + x1 - m / 3)[2] = image3.at<Vec3b>(j, i)[2];
            }
        }
        cv::imshow("Camera", Image);
        if (27 == cv::waitKey(5)) {
            mCamera.release();
            cv::destroyAllWindows();
            break;
        }
    }
    system("pause");
    return 0;
}