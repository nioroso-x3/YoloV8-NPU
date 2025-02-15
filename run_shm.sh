#!/bin/bash
./YoloV8_NPU ./rk3588/yolov8s.rknn "shmsrc is-live=true do-timestamp=true socket-path=${1} ! \
video/x-raw, format=(string)BGR, width=(int)${2}, height=(int)${3}, framerate=(fraction)${4}/1, \
interlace-mode=(string)progressive, multiview-mode=(string)mono, \
multiview-flags=(GstVideoMultiviewFlagsSet)0:ffffffff:/right-view-first/left-flipped/left-flopped/right-flipped/right-flopped/half-aspect/mixed-mono, \
pixel-aspect-ratio=(fraction)1/1, colorimetry=(string)2:1:5:1 ! \
queue leaky=1 ! videoconvert ! appsink max-buffers=2 drop=true" \
"appsrc do-timestamp=true is-live=true ! queue ! videorate ! \
mpph265enc bps=3000000 gop=15 header-mode=1 max-pending=1 ! \
h265parse ! rtph265pay config-interval=-1 aggregate-mode=1 mtu=1472 ! multiudpsink clients=192.168.1.128:6600"
