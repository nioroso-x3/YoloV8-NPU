// Copyright (c) 2021 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//
// Modified by Q-engineering 4-6-2026
//

/*-------------------------------------------
                Includes
-------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <iostream>
#include <fstream>
#include <cstdlib>                  // for malloc and free
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "postprocess.h"
#include "rk_common.h"
#include "rknn_api.h"
#include "sahi.h"
/*-------------------------------------------
                  COCO labels
-------------------------------------------*/

static const char* labels[] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};
/*-------------------------------------------
                  Main Functions
-------------------------------------------*/
int main(int argc, char** argv)
{
    float f;
    float FPS[16];
    int   i, Fcnt=0;
    std::chrono::steady_clock::time_point Tbegin, Tend;
    const float    nms_threshold      = NMS_THRESH;
    const float    box_conf_threshold = BOX_THRESH;
    int            ret;
    int            N = 6;
    for(i=0;i<16;i++) FPS[i]=0.0;

    if (argc < 3) {
        fprintf(stderr,"Usage: %s [model] [imagepath]\n", argv[0]);
        return -1;
    }

    char*          model_path = argv[1];
    const char*    imagepath = argv[2];

    printf("model: %s\n", model_path);
    printf("post process config: box_conf_threshold = %.2f, nms_threshold = %.2f\n", box_conf_threshold, nms_threshold);

    // fire up the neural network
    printf("Loading mode...\n");

    std::vector<rknn_app_context_t> rknn_app_ctx(N);
    // Create the neural network
    int            model_data_size = 0;
    unsigned char* model_data      = load_model(model_path, model_data_size);
    for (int i = 0; i < N; ++i){

        rknn_context          ctx;
        rknn_input_output_num io_num;

        ret = rknn_init(&ctx, model_data, model_data_size, 0, NULL);

        if(ret < 0){
            printf("rknn_init fail! ret=%d %d\n", ret,i);
            return -1;
        }

        rknn_sdk_version version;
        ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
        if (ret < 0) {
            printf("rknn_init error 0 ret=%d %d\n", ret,i);
            return -1;
        }
        if(i == 0) printf("sdk version: %s driver version: %s\n", version.api_version, version.drv_version);

        // Get Model Input Output Number
        ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d %d\n", ret, i);
            return -1;
        }
        if(i==0)printf("\nmodel input num: %d\n", io_num.n_input);

        rknn_tensor_attr input_attrs[io_num.n_input];
        memset(input_attrs, 0, sizeof(input_attrs));
        for(uint32_t i = 0; i < io_num.n_input; i++) {
            input_attrs[i].index = i;
            ret                  = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
            if (ret < 0) {
                printf("rknn_init error 1 ret=%d %d\n", ret,i);
                return -1;
            }
            if(i==0)dump_tensor_attr(&(input_attrs[i]));
        }

        if(i==0) printf("\nmodel output num: %d\n", io_num.n_output);
        rknn_tensor_attr output_attrs[io_num.n_output];
        memset(output_attrs, 0, sizeof(output_attrs));
        for(uint32_t i = 0; i < io_num.n_output; i++) {
            output_attrs[i].index = i;
            ret                   = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
            if(i==0)dump_tensor_attr(&(output_attrs[i]));
        }
    
        // Set to context
        rknn_app_ctx[i].rknn_ctx = ctx;

        // TODO
        if (output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC && output_attrs[0].type != RKNN_TENSOR_FLOAT16)
        {
            rknn_app_ctx[i].is_quant = true;
        }
        else
        {
            rknn_app_ctx[i].is_quant = false;
        }

        rknn_app_ctx[i].io_num = io_num;
        rknn_app_ctx[i].input_attrs = (rknn_tensor_attr *)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
        memcpy(rknn_app_ctx[i].input_attrs, input_attrs, io_num.n_input * sizeof(rknn_tensor_attr));
        rknn_app_ctx[i].output_attrs = (rknn_tensor_attr *)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
        memcpy(rknn_app_ctx[i].output_attrs, output_attrs, io_num.n_output * sizeof(rknn_tensor_attr));

        if (input_attrs[0].fmt == RKNN_TENSOR_NCHW){
            if(i==0)printf("model input is NCHW\n");
            rknn_app_ctx[i].model_channel = input_attrs[0].dims[1];
            rknn_app_ctx[i].model_height = input_attrs[0].dims[2];
            rknn_app_ctx[i].model_width = input_attrs[0].dims[3];
        }
        else{
            if(i==0)printf("model input is NHWC\n");
            rknn_app_ctx[i].model_height = input_attrs[0].dims[1];
            rknn_app_ctx[i].model_width = input_attrs[0].dims[2];
            rknn_app_ctx[i].model_channel = input_attrs[0].dims[3];
        }
        if(i==0)printf("model input height=%d, width=%d, channel=%d\n",
           rknn_app_ctx[i].model_height, rknn_app_ctx[i].model_width, rknn_app_ctx[i].model_channel);

        if(ret != 0){
            printf("init_yolox_model fail! ret=%d %d model_path=%s\n", ret, i, model_path);
            return ret;
        }
    }
    free(model_data);
    //set the core affinity for each context
    //rknn_core_mask masks[] = {RKNN_NPU_CORE_0,RKNN_NPU_CORE_0,RKNN_NPU_CORE_1,RKNN_NPU_CORE_1,RKNN_NPU_CORE_2,RKNN_NPU_CORE_2};
    for (int i = 0; i < N; ++i){
        ret = rknn_set_core_mask(rknn_app_ctx[i].rknn_ctx, RKNN_NPU_CORE_AUTO);
        if(ret != 0){
          printf("failed to set context npu core affinity error %d %d\n",ret,i);
          return -1;
        }
    }
    //set openmp threads
    omp_set_num_threads(N);

    //Stores the current frame and model width and height
    cv::Mat orig_img;
    int width   = rknn_app_ctx[0].model_width;
    int height  = rknn_app_ctx[0].model_height;

    std::string device = imagepath;
    /*
    std::string pipeline = "shmsrc is-live=true do-timestamp=true socket-path=" + device + " ! "
                           "video/x-raw, format=(string)RGB, width=(int)2592, height=(int)1944, framerate=(fraction)50/1, "
                           "interlace-mode=(string)progressive, multiview-mode=(string)mono, "
                           "multiview-flags=(GstVideoMultiviewFlagsSet)0:ffffffff:/right-view-first/left-flipped/left-flopped/right-flipped/right-flopped/half-aspect/mixed-mono, "
                           "pixel-aspect-ratio=(fraction)1/1, colorimetry=(string)2:1:5:1 ! "
                           "queue ! videorate ! videoconvert ! video/x-raw,framerate=5/1 ! appsink max-buffers=3 drop=true";
    */
    std::string out_pipeline = "appsrc ! queue ! mpph265enc bps=10000000 gop=20 header-mode=1 width=1024 height=768 ! h265parse config-interval=-1 ! rtph265pay ! multiudpsink clients=192.168.1.10:6600";
    std::string pipeline = "videotestsrc ! videoconvert ! appsink"; 
    cv::VideoCapture cap(pipeline,cv::CAP_GSTREAMER);
    cv::VideoWriter out(out_pipeline, cv::CAP_GSTREAMER, 0, 10, cv::Size (2592, 1944), true);
    
    std::vector<object_detect_result_list> results;
    
    bool run = true;
    if (!cap.isOpened()) {
        printf("Failed to open capture\n");
        run = false;
    }
    if (!out.isOpened()) {
        printf("Failed to open output\n");
        run = false;
    }

    std::vector<cv::Rect> slices;

    while(run){
        //load image or frame
        //cap >> orig_img;
        orig_img = cv::imread("./image.png");
        if(orig_img.empty()) {
            printf("Error grabbing\n");
            break;
        }

        Tbegin = std::chrono::steady_clock::now();
        if (slices.size() == 0){
            slices = getSliceBBoxes(orig_img.rows,orig_img.cols,height,width,true,0.2f,0.2f);
            slices.push_back(cv::Rect(cv::Point(0,0),cv::Size(orig_img.cols,orig_img.rows)));
            printf("Num slices: %d\n",slices.size());
            for (int i = 0; i < slices.size(); ++i){
                printf("Slice %d: w:%d h:%d x:%d y:%d\n",i,slices[i].width,slices[i].height,slices[i].x,slices[i].y);
            }
            results.resize(slices.size());
        }

        #pragma omp parallel for
        for(int s = 0; s < slices.size(); ++s){
            int tid = omp_get_thread_num();
            rknn_input inputs[rknn_app_ctx[tid].io_num.n_input];
            rknn_output outputs[rknn_app_ctx[tid].io_num.n_output];
            cv::Mat orig_slice = orig_img(slices[s]).clone();
            cv::Mat slice;
            int img_width = orig_slice.cols;
            int img_height = orig_slice.rows;
            if (img_width != width || img_height != height) {
                cv::resize(orig_slice,slice,cv::Size(width,height),cv::INTER_AREA);
            }
            else slice = orig_slice;
            void *buf = (void*)slice.data;
            //declare your list
            object_detect_result_list od_results;

            // Set Input Data
            inputs[0].index = 0;
            inputs[0].type = RKNN_TENSOR_UINT8;
            inputs[0].fmt = RKNN_TENSOR_NHWC;
            inputs[0].size = rknn_app_ctx[tid].model_width * rknn_app_ctx[tid].model_height * rknn_app_ctx[tid].model_channel;
            inputs[0].buf = buf;

            // allocate inputs
            ret = rknn_inputs_set(rknn_app_ctx[tid].rknn_ctx, rknn_app_ctx[tid].io_num.n_input, inputs);
            if(ret < 0){
                printf("rknn_input_set fail! ret=%d\n", ret);
            }
            // allocate outputs
            memset(outputs, 0, sizeof(outputs));
            for(uint32_t i = 0; i < rknn_app_ctx[tid].io_num.n_output; i++){
                outputs[i].index = i;
                outputs[i].want_float = (!rknn_app_ctx[tid].is_quant);
            }
            // run
            rknn_run(rknn_app_ctx[tid].rknn_ctx, nullptr);
            rknn_outputs_get(rknn_app_ctx[tid].rknn_ctx, rknn_app_ctx[tid].io_num.n_output, outputs, NULL);

            // post process
            float scale_w = (float)width / img_width;
            float scale_h = (float)height / img_height;

            post_process(&rknn_app_ctx[tid], outputs, box_conf_threshold, nms_threshold, scale_w, scale_h, &od_results);
            for (int i = 0; i < od_results.count; i++) {
                object_detect_result* det_result = &(od_results.results[i]);
                det_result->box.left += slices[s].x;
                det_result->box.top += slices[s].y;
                det_result->box.right += slices[s].x;
                det_result->box.bottom += slices[s].y;
            }
            results[s] = od_results;
            ret = rknn_outputs_release(rknn_app_ctx[tid].rknn_ctx, rknn_app_ctx[tid].io_num.n_output, outputs);
        }
        //display results
        for(int s = 0; s < slices.size(); ++s){
            for(int r = 0; r < results[s].count; ++r){
                object_detect_result* det_result = &(results[s].results[r]);
                printf("S %d R %d: %d %d %d %d %d %f\n",s,
                                                        r,
                                                        det_result->box.left,
                                                        det_result->box.top,
                                                        det_result->box.right,
                                                        det_result->box.bottom,
                                                        det_result->cls_id,
                                                        det_result->prop);

            }
        }

        //show output
        //std::cout << "FPS" << f/16 << std::endl;
        Tend = std::chrono::steady_clock::now();
        //calculate frame rate
        f = std::chrono::duration_cast <std::chrono::milliseconds> (Tend - Tbegin).count();
        if(f>0.0) FPS[((Fcnt++)&0x0F)]=1000.0/f;
        for(f=0.0, i=0;i<16;i++){ f+=FPS[i]; }
        run = false;
    }
    for(int i = 0; i < N; ++i){
        if(rknn_app_ctx[i].input_attrs  != NULL) free(rknn_app_ctx[i].input_attrs);
        if(rknn_app_ctx[i].output_attrs != NULL) free(rknn_app_ctx[i].output_attrs);
        if(rknn_app_ctx[i].rknn_ctx != 0) rknn_destroy(rknn_app_ctx[i].rknn_ctx);
    }
    return 0;
}
