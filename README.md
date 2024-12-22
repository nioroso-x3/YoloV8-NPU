# YoloV8 Multithreaded NPU
## YoloV8 multithreaded with SAHI for RK3566/68/88 NPU (Rock 5, Orange Pi 5, Radxa Zero 3). <br/>

SAHI Greedy NMM pulled from https://github.com/obss/sahi and partly translated to C++ with the help of ChatGPT o1.

For now it just outputs to console, but should be easy to extend to draw boxes, send results over network or shared memory to another model, etc..

Edit the OpenCV gstreamer pipeline to fit your purposes.

------------

## Model performance benchmark (FPS)

Using a 2048x1536 image that is sliced into 21 slices (Twenty 640x640 subimages plus the scaled full image) it runs at 10FPS using yolov8n on all 3 cores of the rk3588 NPU.


## Dependencies.
To run the application, you have to:
- OpenCV 64-bit installed.
- Optional: Code::Blocks. (```$ sudo apt-get install codeblocks```)

### Installing the dependencies.
Start with the usual 
```
$ sudo apt-get update 
$ sudo apt-get upgrade
$ sudo apt-get install cmake wget curl
```
#### OpenCV
Follow the Raspberry Pi 4 [guide](https://qengineering.eu/install-opencv-on-raspberry-64-os.html).<br>

#### RKNPU2
```
$ git clone https://github.com/airockchip/rknn-toolkit2.git
```
We only use a few files.
```
rknn-toolkit2-master
│      
└── rknpu2
    │      
    └── runtime
        │       
        └── Linux
            │      
            └── librknn_api
                ├── aarch64
                │   └── librknnrt.so
                └── include
                    ├── rknn_api.h
                    ├── rknn_custom_op.h
                    └── rknn_matmul_api.h

$ cd ~/rknn-toolkit2-master/rknpu2/runtime/Linux/librknn_api/aarch64
$ sudo cp ./librknnrt.so /usr/local/lib
$ cd ~/rknn-toolkit2-master/rknpu2/runtime/Linux/librknn_api/include
$ sudo cp ./rknn_* /usr/local/include
```
Save 2 GB of disk space by removing the toolkit. We do not need it anymore.
```
$ cd ~
$ sudo rm -rf ./rknn-toolkit2-master
```

------------

## Installing the app.
To extract and run the network in Code::Blocks <br/>
```
$ mkdir *MyDir* <br/>
$ cd *MyDir* <br/>
$ git clone https://github.com/Qengineering/YoloV8-NPU.git <br/>
```

------------

## Running the app.
You can use **Code::Blocks**.
- Load the project file *.cbp in Code::Blocks.
- Select _Release_, not Debug.
- Compile and run with F9.
- You can alter command line arguments with _Project -> Set programs arguments..._ 

Or use **Cmake**.
```
$ cd *MyDir*
$ mkdir build
$ cd build
$ cmake ..
$ make -j4
```
Make sure you use the model fitting your system.<br><br>

More info or if you want to connect a camera to the app, follow the instructions at [Hands-On](https://qengineering.eu/deep-learning-examples-on-raspberry-32-64-os.html#HandsOn).<br/><br/>
![output image]( https://qengineering.eu/github/YoloV8_Bus_NPU.webp )

