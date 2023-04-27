//
// Created by rc_qh on 23-4-25.
//
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <chrono>
#include <sophus/se3.hpp>

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

Point2d pixel2cam(const Point2d &p, const Mat &K)
{
    return Point2d(
            (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
            (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    );
}

class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    virtual void setToOriginImpl() override{
        _estimate = Sophus::SE3d();
    }

    virtual void oplusImpl(const double *update) override {
        Eigen::Matrix<double, 6, 1> update_eigen;
        update_eigen << update[0], update[1], update[2], update[3], update[4], update[5];
        _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;
    }

    virtual bool read(istream &in) override{}

    virtual bool write(ostream &out) const override {}
};

class EdgeProjectXYZRGBDPoseOnly : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, VertexPose>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeProjectXYZRGBDPoseOnly(const Eigen::Vector3d &point) : _point(point) {}

    virtual void computeError() override {
        const VertexPose *pose = static_cast<const VertexPose *>(_vertices[0]);
        _error = _measurement - pose->estimate() * _point;
    }

    virtual void linearizeOplus() override {
        VertexPose *pose = static_cast<VertexPose *>(_vertices[0]);
        Sophus::SE3d T = pose->estimate();
        Eigen::Vector3d xyz_trans = T * _point;
        _jacobianOplusXi.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();
        _jacobianOplusXi.block<3, 3>(0, 3) = Sophus::SO3d::hat(xyz_trans);
    }

    bool read(istream &in) {}
    bool write(ostream &out) const {}

protected:
    Eigen::Vector3d _point;
};

void pose_estimate_3d3d(const vector<Point3f> &pts_1, const vector<Point3f> &pts_2, Mat &R, Mat &t)
{
    Point3f p1, p2;
    int N = pts_1.size();
    for(int i = 0; i < N; i++)
    {
        p1 += pts_1[i];
        p2 += pts_2[i];
    }
    p1 = Point3f(Vec3f(p1) / N);
    p2 = Point3f(Vec3f(p2) / N);
    vector<Point3f> q1(N), q2(N);
    for(int i = 0; i < N; i++)
    {
        q1[i] = pts_1[i] - p1;
        q2[i] = pts_2[i] - p2;
    }
    Eigen::Matrix3d w = Eigen::Matrix3d::Zero();
    for(int i = 0; i < N; i++)
    {
        w += Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) * Eigen::Vector3d(q2[i].x, q2[i].y, q2[i].z).transpose();
    }
    cout << "w = " << w << endl;
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(w, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    cout << "U = " << U << endl;
    cout << "V = " << V << endl;

    Eigen::Matrix3d R_ = U * V.transpose();
    if(R_.determinant() < 0)
    {
        R_ = -R_;
    }
    Eigen::Vector3d t_ = Eigen::Vector3d(p1.x, p1.y, p1.z) - R_ * Eigen::Vector3d(p2.x, p2.y, p2.z);
    R = (Mat_<double>(3, 3) << R_(0, 0), R_(0, 1), R_(0, 2),
                                          R_(1, 0), R_(1, 1), R_(1, 2),
                                          R_(2, 0), R_(2, 1), R_(2, 2)
    );

    t = (Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));
}

void bundleAdjustment(const vector<Point3f> &pts_1, const vector<Point3f> &pts_2, Mat &R, Mat &t)
{
    typedef g2o::BlockSolverX BlockSolverType;
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;

    auto solver = new g2o::OptimizationAlgorithmLevenberg(
            std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>())
            );
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    VertexPose *pose = new VertexPose();
    pose->setId(0);
    pose->setEstimate(Sophus::SE3d());
    optimizer.addVertex(pose);

    for(size_t i = 0; i < pts_1.size(); i++)
    {
        EdgeProjectXYZRGBDPoseOnly *edge = new EdgeProjectXYZRGBDPoseOnly(
                Eigen::Vector3d(pts_2[i].x, pts_2[i].y, pts_2[i].z)
                );
        edge->setVertex(0, pose);
        edge->setMeasurement(Eigen::Vector3d(pts_1[i].x, pts_1[i].y, pts_1[i].z));
        edge->setInformation(Eigen::Matrix3d::Identity());
        optimizer.addEdge(edge);
    }

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "optimization costs time: " << time_used.count() << "seconds" << endl;
    cout << "T = \n" << pose->estimate().matrix() << endl;

    Eigen::Matrix3d R_ = pose->estimate().rotationMatrix();
    Eigen::Vector3d t_ = pose->estimate().translation();
    R = (Mat_<double>(3, 3) <<
                            R_(0, 0), R_(0, 1), R_(0, 2),
                            R_(1, 0), R_(1, 1), R_(1, 2),
                            R_(2, 0), R_(2, 1), R_(2, 2)
    );
    t = (Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));
}

int main(int argc, char **argv)
{
    if(argc != 5)
    {
        cout << "usage: pose_estimation_3d2d img1 img2 depth1 depth2" << endl;
        return 1;
    }
    Mat img_1 = imread(argv[1]);
    Mat img_2 = imread(argv[2]);
    assert(img_1.data != nullptr && img_2.data != nullptr);

    vector<KeyPoint> key_points_1, key_points_2;
    vector<DMatch> matches;
    find_feature_matches(img_1, img_2, key_points_1, key_points_2, matches);
    cout << "一共找到了" << matches.size() << "组匹配点"  << endl;

    Mat d1 = imread(argv[3]);
    Mat d2 = imread(argv[4]);
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    vector<Point3f> pts_1, pts_2;
    for(DMatch m : matches)
    {
        ushort depth1 = d1.ptr<unsigned short>(int(key_points_1[m.queryIdx].pt.y))[int(key_points_1[m.queryIdx].pt.x)];
        ushort depth2 = d2.ptr<unsigned short>(int(key_points_2[m.queryIdx].pt.y))[int(key_points_2[m.queryIdx].pt.x)];
        if(depth1 == 0 || depth2 == 0)
        {
            continue;
        }
        float dd_1 = depth1 / 5000.0;
        float dd_2 = depth2 / 5000.0;
        Point2d p1 = pixel2cam(key_points_1[m.queryIdx].pt, K);
        Point2d p2 = pixel2cam(key_points_2[m.queryIdx].pt, K);
        pts_1.push_back(Point3f(p1.x * dd_1, p1.y * dd_1, dd_1));
        pts_2.push_back(Point3f(p2.x * dd_2, p2.y * dd_2, dd_2));
    }
    cout << "3d-3d pairs:" << pts_1.size() << endl;

    Mat R, t;
    pose_estimate_3d3d(pts_1, pts_2, R, t);
    cout << "ICP via SVD results:" << endl;
    cout << "R = " << R << endl;
    cout << "t = " << t << endl;
    cout << "R_inv = " << R.t() << endl;
    cout << "t_inv = " << t.t() << endl;
    cout << "calling bundle adjustment " << endl;
    bundleAdjustment(pts_1, pts_2, R, t);

    for(int i = 0; i < 5; i++)
    {
        cout << "p1 = " << pts_1[i] << endl;
        cout << "p2 = " << pts_2[i] << endl;
        cout << "(R * p2 + t) = " << R * (Mat_<double>(3, 1) << pts_2[i].x, pts_2[i].y, pts_2[i].z) + t << endl;
        cout << endl;
    }
    return 0;
}
