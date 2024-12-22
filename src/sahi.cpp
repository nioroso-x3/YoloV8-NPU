#include "sahi.h"

std::vector<cv::Rect> getSliceBBoxes(
    int imageHeight,
    int imageWidth,
    int sliceHeight = 0,
    int sliceWidth = 0,
    bool autoSliceResolution = true,
    float overlapHeightRatio = 0.2f,
    float overlapWidthRatio = 0.2f
) {
    std::vector<cv::Rect> sliceBBoxes;

    int yMin = 0, yMax = 0;
    int xOverlap = 0, yOverlap = 0;
    int xMax, xMin;
    // Automatically determine slice dimensions if required
    if (sliceHeight > 0 && sliceWidth > 0) {
        yOverlap = static_cast<int>(overlapHeightRatio * sliceHeight);
        xOverlap = static_cast<int>(overlapWidthRatio * sliceWidth);
    } else if (autoSliceResolution) {
        // Implement getAutoSliceParams as needed
        // Example: auto [xOverlap, yOverlap, sliceWidth, sliceHeight] = getAutoSliceParams(imageHeight, imageWidth);
        throw std::runtime_error("Auto slice resolution is not implemented.");
    } else {
        throw std::invalid_argument("Slice dimensions not provided and autoSliceResolution is false.");
    }

    while (yMax < imageHeight) {
        xMin = xMax = 0;
        yMax = yMin + sliceHeight;

        while (xMax < imageWidth) {
            xMax = xMin + sliceWidth;
            if(yMax > imageHeight || xMax > imageWidth){
               int actualXMax = std::min(imageWidth, xMax);
               int actualYMax = std::min(imageHeight, yMax);
               int actualXMin = std::max(0, actualXMax - sliceWidth);
               int actualYMin = std::max(0, actualYMax - sliceHeight);

               sliceBBoxes.emplace_back(cv::Rect(cv::Point(actualXMin, actualYMin), cv::Size(sliceWidth, sliceHeight)));
            }
            else{
               sliceBBoxes.emplace_back(cv::Rect(cv::Point(xMin, yMin), cv::Size(sliceWidth, sliceHeight)));
            }
            xMin = xMax - xOverlap;
        }
        yMin = yMax - yOverlap;
    }

    return sliceBBoxes;
}


std::map<int, std::vector<int>> greedy_nmm(
    const std::vector<object_detect_result>& object_predictions,
    const std::string& match_metric = "IOU",
    float match_threshold = 0.5
) {
    std::map<int, std::vector<int>> keep_to_merge_list;

    size_t num_boxes = object_predictions.size();
    std::vector<float> x1(num_boxes);
    std::vector<float> y1(num_boxes);
    std::vector<float> x2(num_boxes);
    std::vector<float> y2(num_boxes);
    std::vector<float> scores(num_boxes);
    std::vector<float> areas(num_boxes);

    // Extract box coordinates and scores
    for (size_t i = 0; i < num_boxes; ++i) {
        x1[i] = static_cast<float>(object_predictions[i].box.left);
        y1[i] = static_cast<float>(object_predictions[i].box.top);
        x2[i] = static_cast<float>(object_predictions[i].box.right);
        y2[i] = static_cast<float>(object_predictions[i].box.bottom);
        scores[i] = object_predictions[i].prop;
        areas[i] = (x2[i] - x1[i]) * (y2[i] - y1[i]);
    }

    // Sort the boxes according to their confidence scores (ascending order)
    std::vector<size_t> order(num_boxes);
    for (size_t i = 0; i < num_boxes; ++i) {
        order[i] = i;
    }
    std::sort(order.begin(), order.end(), [&scores](size_t i1, size_t i2) {
        return scores[i1] < scores[i2];
    });

    while (!order.empty()) {
        size_t idx = order.back();
        order.pop_back();

        if (order.empty()) {
            keep_to_merge_list[(int)idx] = std::vector<int>();
            break;
        }

        size_t N = order.size();
        std::vector<float> xx1(N);
        std::vector<float> yy1(N);
        std::vector<float> xx2(N);
        std::vector<float> yy2(N);
        for (size_t i = 0; i < N; ++i) {
            size_t ind = order[i];
            xx1[i] = std::max(x1[ind], x1[idx]);
            yy1[i] = std::max(y1[ind], y1[idx]);
            xx2[i] = std::min(x2[ind], x2[idx]);
            yy2[i] = std::min(y2[ind], y2[idx]);
        }

        // Compute widths and heights of the intersections
        std::vector<float> w(N);
        std::vector<float> h(N);
        for (size_t i = 0; i < N; ++i) {
            w[i] = std::max(0.0f, xx2[i] - xx1[i]);
            h[i] = std::max(0.0f, yy2[i] - yy1[i]);
        }

        // Compute intersection areas
        std::vector<float> inter(N);
        for (size_t i = 0; i < N; ++i) {
            inter[i] = w[i] * h[i];
        }

        // Compute remaining areas
        std::vector<float> rem_areas(N);
        for (size_t i = 0; i < N; ++i) {
            rem_areas[i] = areas[order[i]];
        }

        // Compute the match metric values
        std::vector<float> match_metric_value(N);
        if (match_metric == "IOU") {
            for (size_t i = 0; i < N; ++i) {
                float union_area = rem_areas[i] + areas[idx] - inter[i];
                match_metric_value[i] = inter[i] / union_area;
            }
        } else if (match_metric == "IOS") {
            for (size_t i = 0; i < N; ++i) {
                float smaller = std::min(rem_areas[i], areas[idx]);
                match_metric_value[i] = inter[i] / smaller;
            }
        } else {
            throw std::invalid_argument("Invalid match_metric");
        }

        // Identify boxes to keep and boxes to merge
        std::vector<size_t> matched_box_indices;
        std::vector<size_t> unmatched_indices;
        for (size_t i = 0; i < N; ++i) {
            if (match_metric_value[i] < match_threshold) {
                unmatched_indices.push_back(order[i]);
            } else {
                matched_box_indices.push_back(order[i]);
            }
        }

        // Sort unmatched indices again by scores (ascending)
        std::sort(unmatched_indices.begin(), unmatched_indices.end(),
                  [&scores](size_t i1, size_t i2) { return scores[i1] < scores[i2]; });
        order = unmatched_indices;

        // Map current index to indices of boxes to merge
        std::vector<int> merge_list;
        std::reverse(matched_box_indices.begin(), matched_box_indices.end());
        for (size_t idx_to_merge : matched_box_indices) {
            merge_list.push_back((int)idx_to_merge);
        }
        keep_to_merge_list[(int)idx] = merge_list;
    }

    return keep_to_merge_list;
}

std::map<int, std::vector<int>> batched_greedy_nmm(
    const std::vector<object_detect_result>& object_predictions,
    const std::string& match_metric = "IOU",
    float match_threshold = 0.5
) {
    std::vector<int> category_ids;
    for (const auto& prediction : object_predictions) {
        category_ids.push_back(prediction.cls_id);
    }

    // Find unique category IDs
    std::set<int> unique_category_ids(category_ids.begin(), category_ids.end());
    std::map<int, std::vector<int>> keep_to_merge_list;

    for (int category_id : unique_category_ids) {
        // Get indices of predictions with the current category_id
        std::vector<size_t> curr_indices;
        for (size_t i = 0; i < category_ids.size(); ++i) {
            if (category_ids[i] == category_id) {
                curr_indices.push_back(i);
            }
        }

        // Create a subset of predictions for the current category
        std::vector<object_detect_result> curr_object_predictions;
        for (size_t idx : curr_indices) {
            curr_object_predictions.push_back(object_predictions[idx]);
        }

        // Apply greedy NMM to the current category
        std::map<int, std::vector<int>> curr_keep_to_merge_list =
            greedy_nmm(curr_object_predictions, match_metric, match_threshold);

        // Map the indices back to the original predictions
        for (const auto& pair : curr_keep_to_merge_list) {
            int curr_keep = pair.first;
            const std::vector<int>& curr_merge_list = pair.second;

            int keep = (int)curr_indices[curr_keep];
            std::vector<int> merge_list;
            for (int merge_ind : curr_merge_list) {
                merge_list.push_back((int)curr_indices[merge_ind]);
            }
            keep_to_merge_list[keep] = merge_list;
        }
    }

    return keep_to_merge_list;
}


// Helper function to compute Intersection over Union (IoU)
float compute_iou(const image_rect_t& box1, const image_rect_t& box2) {
    int x_left = std::max(box1.left, box2.left);
    int y_top = std::max(box1.top, box2.top);
    int x_right = std::min(box1.right, box2.right);
    int y_bottom = std::min(box1.bottom, box2.bottom);

    int width = std::max(0, x_right - x_left);
    int height = std::max(0, y_bottom - y_top);

    float intersection = width * height;
    float area1 = (box1.right - box1.left) * (box1.bottom - box1.top);
    float area2 = (box2.right - box2.left) * (box2.bottom - box2.top);

    float union_area = area1 + area2 - intersection;

    if (union_area == 0.0f)
        return 0.0f;
    else
        return intersection / union_area;
}

// Helper function to compute Intersection over Smaller Area (IoS)
float compute_ios(const image_rect_t& box1, const image_rect_t& box2) {
    int x_left = std::max(box1.left, box2.left);
    int y_top = std::max(box1.top, box2.top);
    int x_right = std::min(box1.right, box2.right);
    int y_bottom = std::min(box1.bottom, box2.bottom);

    int width = std::max(0, x_right - x_left);
    int height = std::max(0, y_bottom - y_top);

    float intersection = width * height;
    float area1 = (box1.right - box1.left) * (box1.bottom - box1.top);
    float area2 = (box2.right - box2.left) * (box2.bottom - box2.top);

    float smaller_area = std::min(area1, area2);

    if (smaller_area == 0.0f)
        return 0.0f;
    else
        return intersection / smaller_area;
}

// Function to check if two predictions match based on the given metric and threshold
bool has_match(
    const object_detect_result& pred1,
    const object_detect_result& pred2,
    const std::string& match_metric,
    float match_threshold)
{
    float match_value;
    if (match_metric == "IOU")
        match_value = compute_iou(pred1.box, pred2.box);
    else if (match_metric == "IOS")
        match_value = compute_ios(pred1.box, pred2.box);
    else
        throw std::invalid_argument("Invalid match_metric");

    return match_value >= match_threshold;
}

// Function to merge two object predictions
object_detect_result merge_object_prediction_pair(
    const object_detect_result& pred1,
    const object_detect_result& pred2)
{
    object_detect_result merged_pred;

    // Merge the boxes by averaging coordinates
    merged_pred.box.left = (pred1.box.left + pred2.box.left) / 2;
    merged_pred.box.top = (pred1.box.top + pred2.box.top) / 2;
    merged_pred.box.right = (pred1.box.right + pred2.box.right) / 2;
    merged_pred.box.bottom = (pred1.box.bottom + pred2.box.bottom) / 2;

    // Merge scores by averaging
    merged_pred.prop = (pred1.prop + pred2.prop) / 2.0f;

    // Assuming the class IDs are the same; otherwise, additional logic may be needed
    merged_pred.cls_id = pred1.prop > pred2.prop ? pred1.cls_id : pred2.cls_id;

    return merged_pred;
}

// The main function that combines everything
std::vector<object_detect_result> greedy_nmm_postprocess(
    const std::vector<object_detect_result>& object_predictions,
    float match_threshold = 0.5f,
    const std::string& match_metric = "IOU",
    bool class_agnostic = true
)
{
    // Make a copy of object predictions to modify
    std::vector<object_detect_result> object_prediction_list = object_predictions;

    std::map<int, std::vector<int>> keep_to_merge_list;

    if (class_agnostic) {
        keep_to_merge_list = greedy_nmm(
            object_prediction_list,
            match_metric,
            match_threshold
        );
    } else {
        keep_to_merge_list = batched_greedy_nmm(
            object_prediction_list,
            match_metric,
            match_threshold
        );
    }

    std::vector<object_detect_result> selected_object_predictions;

    for (const auto& pair : keep_to_merge_list) {
        int keep_ind = pair.first;
        const std::vector<int>& merge_ind_list = pair.second;

        for (int merge_ind : merge_ind_list) {
            if (has_match(
                object_prediction_list[keep_ind],
                object_prediction_list[merge_ind],
                match_metric,
                match_threshold)) {

                object_prediction_list[keep_ind] = merge_object_prediction_pair(
                    object_prediction_list[keep_ind],
                    object_prediction_list[merge_ind]
                );
            }
        }
        selected_object_predictions.push_back(object_prediction_list[keep_ind]);
    }

    return selected_object_predictions;
}


