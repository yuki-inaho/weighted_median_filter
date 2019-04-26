#pragma once

#include "header.h"
#include <boost/thread/thread.hpp>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/search/pcl_search.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/extract_indices.h>
#include <iomanip>
#include <fstream>

using namespace std;
using namespace pcl;

//removeNan: NaN要素を点群データから除去するメソッド
//input : target(NaN要素を除去する対象の点群)
//output: cloud(除去を行った点群)
pcl::PointCloud<PointXYZ>::Ptr removeNan(pcl::PointCloud<pcl::PointXYZ>::Ptr target){
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  int n_point = target->points.size();
  for(int i=0;i<n_point; i++){
    pcl::PointXYZ tmp_point;
    if(std::isfinite(target->points[i].x) || std::isfinite(target->points[i].y) || std::isfinite(target->points[i].z)){
      tmp_point.x = target->points[i].x;
      tmp_point.y = target->points[i].y;
      tmp_point.z = target->points[i].z;
      cloud->points.push_back(tmp_point);
    }
  }
  return cloud;
}


pcl::visualization::PCLVisualizer::Ptr simpleVis ()
{
  // --------------------------------------------
  // -----Open 3D viewer and add point cloud-----
  // --------------------------------------------
  pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();
  return (viewer);
}


pcl::PointCloud<pcl::PointXYZ>::Ptr 
Depth2Point(cv::Mat src, float fx, float fy, float cx, float cy) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    for (int h = 0; h < src.rows; h++) {
        for (int w = 0; w < src.cols; w++) {
            unsigned short z_value_short;
            z_value_short = src.at<short>(h, w);

            if (z_value_short > 0) {
                Eigen::Vector3f v;
                v = Eigen::Vector3f::Zero();

                v.z() = (float)(z_value_short)/1000;

                if(v.z() == 0) continue;
                v.x() = v.z() * (w - cx) * (1.0 / fx);
                v.y() = v.z() * (h - cy) * (1.0 / fy);

                pcl::PointXYZ point_tmp;
                point_tmp.x = v.x();
                point_tmp.y = v.y();
                point_tmp.z = v.z();
                cloud->points.push_back(point_tmp);
            }
        }
    }
    return cloud;
}
