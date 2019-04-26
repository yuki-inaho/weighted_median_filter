#include "header.h"
#include "utils.hpp"
#include "ParameterManager.hpp"
#include "weighted_median_filter.hpp"

#include <pcl/filters/statistical_outlier_removal.h>
using namespace pcl;

std::string CFG_PARAM_PATH = "/home/inaho-00/work/cpp/weighted_median_filter/cfg/recognition_parameter.toml";

int
main (int argc, char** argv)
{
    ParameterManager cfg_param(CFG_PARAM_PATH);
    float sigma = cfg_param.ReadFloatData("Param", "sigma");
    float param_R = cfg_param.ReadFloatData("Param", "param_R");
    int param_K = cfg_param.ReadIntData("Param", "param_K");
    std::string DATA_PATH = cfg_param.ReadStringData("Param", "data_path");

    float fx = cfg_param.ReadFloatData("Camera", "fx");
    float fy = cfg_param.ReadFloatData("Camera", "fy");
    float cx = cfg_param.ReadFloatData("Camera", "cx");
    float cy = cfg_param.ReadFloatData("Camera", "cy");

    cv::Mat depth = cv::imread(DATA_PATH, cv::IMREAD_ANYDEPTH);
    cv::Mat depth_denoised = WeightedMedianFilterOMP(depth, 3, 0.0005, 0.001);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    cloud = Depth2Point(depth, fx, fy, cx, cy);
    cloud = removeNan(cloud);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_denoised (new pcl::PointCloud<pcl::PointXYZ>);
    cloud_denoised = Depth2Point(depth_denoised, fx, fy, cx, cy);
    cloud_denoised = removeNan(cloud_denoised);

    // Create the filtering object
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud (cloud);
    sor.setMeanK (10);
    sor.setStddevMulThresh (0.5);
    sor.filter (*cloud);

    arma::mat output;
    cout << cloud->points.size()  << endl;

    pcl::visualization::PCLVisualizer::Ptr viewer;
    viewer = simpleVis();
    viewer->addPointCloud<pcl::PointXYZ> (cloud_denoised, "sample cloud");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "sample cloud");  
    pcl::io::savePCDFileBinary("../data/denoised.pcd", *cloud_denoised);

    while (!viewer->wasStopped ())
    {
        viewer->spinOnce (100);
        boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    }

    return (0);
}