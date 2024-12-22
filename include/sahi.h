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

float compute_iou(const image_rect_t& box1, const image_rect_t& box2);

float compute_ios(const image_rect_t& box1, const image_rect_t& box2);

bool has_match(
    const object_detect_result& pred1,
    const object_detect_result& pred2,
    const std::string& match_metric,
    float match_threshold);

object_detect_result merge_object_prediction_pair(
    const object_detect_result& pred1,
    const object_detect_result& pred2);

std::vector<object_detect_result> greedy_nmm_postprocess(
    const std::vector<object_detect_result>& object_predictions,
    float match_threshold,
    const std::string& match_metric,
    bool class_agnostic
);



