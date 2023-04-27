//
// Created by rc_qh on 23-4-24.
//
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void find_feature_matches(const Mat &img_1, const Mat &img_2, vector<KeyPoint> &keypoints_1, vector<KeyPoint> &keypoints_2, vector<DMatch> &matches)
{
    Mat description_1, description_2;
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    descriptor->compute(img_1, keypoints_1, description_1);
    descriptor->compute(img_2, keypoints_2, description_2);

    vector<DMatch> match;
    matcher->match(description_1, description_2, match);

    double min_dist = 10000, max_dist = 0;
    for(int i = 0; i < description_1.rows; i++)
    {
        double dist = match[i].distance;
        if(dist < min_dist)
        {
            min_dist = dist;
        }
        if(dist > max_dist)
        {
            max_dist = dist;
        }
    }

    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);

    for(int i = 0; i < description_1.rows; i++)
    {
        if(match[i].distance <= max(2 * min_dist, 30.0))
        {
            matches.push_back(match[i]);
        }
    }
}

void pose_estimation_2d2d(vector<KeyPoint> keypoints_1, vector<KeyPoint> keypoints_2, vector<DMatch> matches, Mat &R, Mat &t)
{
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    vector<Point2f> points1;
    vector<Point2f> points2;

    for(int i = 0; i < (int)matches.size(); i++)
    {
        points1.push_back(keypoints_1[matches[i].queryIdx].pt);
        points2.push_back(keypoints_2[matches[i].queryIdx].pt);
    }

    Mat fundamental_matrix;
    fundamental_matrix = findFundamentalMat(points1, points2);
    cout << "fundamental_matrix = " << endl << fundamental_matrix << endl;

    Point2d principal_point(325.1, 249.7);

    double focal_length = 521;
    Mat essential_matrix;
    essential_matrix = findEssentialMat(points1, points2, focal_length, principal_point);
    cout << "essential_matrix = " << endl << essential_matrix << endl;

    Mat homography_matrix;
    homography_matrix = findHomography(points1, points2, RANSAC, 3);
    cout << "homography_matrix = " << endl << homography_matrix << endl;

    recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
    cout << "R = " << endl << R << endl;
    cout << "t = " << endl << t << endl;
}

Point2d pixel2cam(const Point2d &p, const Mat &K)
{
    return Point2d(
            (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
            (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    );
}

void triangulation(const vector<KeyPoint> &keypoints_1, const vector<KeyPoint> &keypoints_2, const vector<DMatch> &matches, const Mat &R, const Mat &t, vector<Point3d> &points)
{
    Mat T1 = (Mat_<float>(3, 4) <<
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0
    );
    Mat T2 = (Mat_<float>(3, 4) <<
            R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
            R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
            R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0)
    );

    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    vector<Point2f> pts_1, pts_2;
    for(DMatch m : matches)
    {
        pts_1.push_back(pixel2cam(keypoints_1[m.queryIdx].pt, K));
        pts_2.push_back(pixel2cam(keypoints_2[m.queryIdx].pt, K));
    }

    Mat pts_4d;
    triangulatePoints(T1, T2, pts_1, pts_2, pts_4d);

    for(int i = 0; i < pts_4d.cols; i++)
    {
        Mat x = pts_4d.col(i);
        x /= x.at<float>(3, 0);
        Point3d p(
                x.at<float>(0, 0),
                x.at<float>(1, 0),
                x.at<float>(2, 0)
        );
        points.push_back(p);
    }
}

inline Scalar get_color(float depth)
{
    float up_th = 50, low_th = 10, th_range = up_th - low_th;
    if(depth > up_th)
    {
        depth = up_th;
    }
    if(depth < low_th)
    {
        depth = low_th;
    }
    return Scalar(255 * depth / th_range, 0, 255 * (1 - depth / th_range));
}

int main(int argc, char **argv)
{
    if(argc != 3)
    {
        cout << "usage: triangulation img1, img2" << endl;
        return 1;
    }

    Mat img_1 = imread(argv[1]);
    Mat img_2 = imread(argv[2]);
    assert(img_1.data != nullptr && img_2.data != nullptr);

    vector<KeyPoint> key_points_1, key_points_2;
    vector<DMatch> matches;
    find_feature_matches(img_1, img_2, key_points_1, key_points_2, matches);
    cout << "一共找到了" << matches.size() << "组匹配点" << endl;

    Mat R, t;
    pose_estimation_2d2d(key_points_1, key_points_2, matches, R, t);

    vector<Point3d> points;
    triangulation(key_points_1, key_points_2, matches, R, t, points);

    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    Mat img1_plot = img_1.clone();
    Mat img2_plot = img_2.clone();
    for(int i = 0; i < matches.size(); i++)
    {
        float depth1 = points[i].z;
        cout << "depth = " << depth1 << endl;
        Point2d pt1_cam = pixel2cam(key_points_1[matches[i].queryIdx].pt, K);
        circle(img1_plot, key_points_1[matches[i].queryIdx].pt, 2, get_color(depth1), 2);

        Mat pt2_trans = R * (Mat_<double>(3, 1) << points[i].x, points[i].y, points[i].z) + t;
        float depth2 = pt2_trans.at<double>(2, 0);
        circle(img2_plot, key_points_2[matches[i].trainIdx].pt, 2, get_color(depth2), 2);
    }

    imshow("img1", img1_plot);
    imshow("img2", img2_plot);
    waitKey(0);

    return 0;
}