#include <vector>
#include <map>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <set>
#include <string>
#include <opencv2/opencv.hpp>
#include "rk_common.h"

std::vector<cv::Rect> getSliceBBoxes(
    int imageHeight,
        int imageWidth,
            int sliceHeight,
                int sliceWidth,
                    bool autoSliceResolution,
                        float overlapHeightRatio,
                            float overlapWidthRatio);

std::map<int, std::vector<int>> greedy_nmm(
    const std::vector<object_detect_result>& object_predictions,
    const std::string& match_metric,
    float match_threshold
);

std::map<int, std::vector<int>> batched_greedy_nmm(
    const std::vector<object_detect_result>& object_predictions,
    const std::string& match_metric,
    float match_threshold
);

