#include "header.h"

using namespace std;
using namespace cv;

template <typename InputIterator, typename OutputIterator> 
inline void 
normalize(InputIterator begin, InputIterator end, OutputIterator out) {
    typedef typename std::iterator_traits<InputIterator>::reference ref_t;
    double norm = std::accumulate(begin, end, 0.0);
    std::transform(begin, end, out, [norm](const ref_t t){ return t/norm; });
}

cv::Mat
PaddingDepth(cv::Mat img, int kernel_size)
{

    cv::Mat img_smoothed = cv::Mat::zeros(img.rows+kernel_size-1, img.cols+kernel_size-1, CV_16UC1);
    for(int y=0;y<img.rows;y++){
        for(int x=0;x<img.cols;x++){
            img_smoothed.at<unsigned short>(y+kernel_size/2, x+kernel_size/2) = img.at<unsigned short>(y, x);
        }
    }
    return img_smoothed;
}

unsigned short
_WeightedMedianFilter(const cv::Mat &depthImage_padded, int x, int y, int kernel_size, double sigma_pos, double sigma_depth)
{
    std::vector<double> similarity_vec;
    std::vector<unsigned short> depth_vec;

    double sum_sim = 0;
    for(int k_y=-kernel_size/2; k_y<=kernel_size/2; k_y++){
        for(int k_x=-kernel_size/2; k_x<=kernel_size/2; k_x++){
            unsigned short center_depth = depthImage_padded.at<unsigned short>(y,x);
            unsigned short perf_depth = depthImage_padded.at<unsigned short>(y+k_y,x+k_x);
            
            //double diff_pos = std::sqrt(double(k_x)*double(k_x) + double(k_y)*double(k_y));
            double _diff_depth = double(center_depth)*0.001 - double(perf_depth)*0.001;
            double diff_depth = std::sqrt(_diff_depth*_diff_depth);
            //double kernel_pos = std::exp(-diff_pos*diff_pos/(2*sigma_pos*sigma_pos));
            double kernel_depth = std::exp(-diff_depth*diff_depth/(2*sigma_depth*sigma_depth));
            //double similarity = kernel_pos * kernel_depth;
            double similarity = kernel_depth;
            sum_sim += similarity;
            similarity_vec.push_back(similarity);
            depth_vec.push_back(center_depth);
        }
    }

    if(sum_sim == 0){
        return depthImage_padded.at<unsigned short>(y,x);
    }
    normalize(similarity_vec.begin(), similarity_vec.end(), similarity_vec.begin());

    //argsort(descending order)
    std::vector<int> argsort_indices(similarity_vec.size());
    std::iota(argsort_indices.begin(), argsort_indices.end(), 0);
    std::sort(argsort_indices.begin(), argsort_indices.end(), [&similarity_vec](size_t i1, size_t i2) {
        return similarity_vec[i1] < similarity_vec[i2];
    });

    vector<double> weighted_depth_vec;
    for(auto it = argsort_indices.begin(); it != argsort_indices.end(); ++it ){
        //double k2k = (double)(kernel_size*kernel_size*100);
        /*
        double k2k = 100;
        int weight = (int)(std::floor(similarity_vec[*it]*k2k));
        if(weight==0){
            weighted_depth_vec.push_back(depth_vec[*it]);
        }else{
            for(int w=0 ; w<weight ; w++){
                weighted_depth_vec.push_back(depth_vec[*it]);
            }
        }*/
        weighted_depth_vec.push_back(depth_vec[*it]);
    }

    int med_idx = weighted_depth_vec.size()/2;
    std::nth_element(weighted_depth_vec.begin(), weighted_depth_vec.begin()+med_idx, weighted_depth_vec.end());
    double kernel_var = weighted_depth_vec[med_idx];
    return kernel_var;
}

cv::Mat
WeightedMedianFilterOMP(const cv::Mat &depthImage, int kernel_size, double sigma_pos, double sigma_depth)
{
    cv::Mat depthImage_smoothed = cv::Mat::zeros(depthImage.rows, depthImage.cols, CV_16U);
    cv::Mat depthImage_padded = PaddingDepth(depthImage, kernel_size);
    cv::Mat depthImage_mask = cv::Mat::zeros(depthImage.rows, depthImage.cols, CV_8UC1);

    cv::parallel_for_(cv::Range(0, depthImage.rows*depthImage.cols), [&](const cv::Range& range){
    //for(int r=0; r< depthImage.rows*depthImage.cols; r++){
        for (int r = range.start; r < range.end; r++)
        {
            int y = r / depthImage.cols;
            int x = r % depthImage.cols;
            if(depthImage.at<unsigned short>(y, x) == 0){
                unsigned short kernel_var = _WeightedMedianFilter(depthImage_padded, x+kernel_size/2, y+kernel_size/2, kernel_size, sigma_pos, sigma_depth);
                depthImage_smoothed.at<unsigned short>(y, x) = kernel_var;
            }else{
                depthImage_smoothed.at<unsigned short>(y, x) = depthImage.at<unsigned short>(y, x);
            }
        }
    });
//    }

    return depthImage_smoothed;
}
