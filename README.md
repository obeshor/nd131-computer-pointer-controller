# Computer Pointer Controller

In this project, you will use a gaze detection model to control the mouse pointer of your computer. You will be using the Gaze Estimation model to estimate the gaze of the user's eyes and change the mouse pointer position accordingly. This project will demonstrate your ability to run multiple models in the same machine and coordinate the flow of data between those models.

04 pre-trained models from the Intel Pre-trained Models Zoo have to be used:
* [Face Detection](https://docs.openvinotoolkit.org/latest/omz_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)
* [Head Pose Estimation](https://docs.openvinotoolkit.org/latest/omz_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
* [Facial LandMarks Detection](https://docs.openvinotoolkit.org/latest/omz_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)
* [Gaze Estimation Model](https://docs.openvinotoolkit.org/latest/omz_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)

The flow of data look like this:
![Gameplay Screenshot](./bin/assets/pipeline.png)


## Project Set Up and Installation
* Download [OpenVINO ToolKit](https://docs.openvinotoolkit.org/latest/index.html) and install it locally.
* Clone the Repository  `git clone https://github.com/obeshor/nd131-computer-pointer-controller.git`
* Create and activate a virtual environment 

   `pip install virtualenv`
   
   `virtualenv venv`
   
   `cd venv/Scripts/`
   
   `activate`
   
* Install dependencies

  `pip install -r requirements.txt`

* Initialize OpenVINO environment

  `cd C:\Program Files (x86)\IntelSWTools\openvino\bin\`
  `setupvars.bat`
  
 * Download models
 
   `python ./src/download_models.py`

Refer below is the project structure:

The bin folder contains the demo video file, the models folder contains all the Intel's Pretrained models needed for execution, and the src folder contains all necessary python files.
```
📦nd131-computer-pointer-controller
 ┣ 📂bin
 ┃ ┣ 📂assets
 ┃ ┃ ┗ 📜pipeline.png
 ┃ ┣ 📜demo.mp4
 ┃
 ┣ 📂venv
 ┃ ┣ 📂bin
 ┃   ┣ 📜activate
 ┃   ┣ 📜activate.fish
 ┃    
 ┣ 📂models
 ┃ ┗ 📂intel
 ┃   ┣ 📂face-detection-adas-0001
 ┃   ┃ ┣ 📂FP16
 ┃   ┃ ┃ ┣ 📜face-detection-adas-0001.bin
 ┃   ┃ ┃ ┗ 📜face-detection-adas-0001.xml
 ┃   ┃ ┗ 📂FP32
 ┃   ┃   ┣ 📜face-detection-adas-0001.bin
 ┃   ┃   ┗ 📜face-detection-adas-0001.xml
 ┃   ┣ 📂face-detection-adas-binary-0001
 ┃   ┃ ┗ 📂INT1
 ┃   ┃   ┣ 📜face-detection-adas-binary-0001.bin
 ┃   ┃   ┗ 📜face-detection-adas-binary-0001.xml
 ┃   ┣ 📂gaze-estimation-adas-0002
 ┃   ┃ ┣ 📂FP16
 ┃   ┃ ┃ ┣ 📜gaze-estimation-adas-0002.bin
 ┃   ┃ ┃ ┗ 📜gaze-estimation-adas-0002.xml
 ┃   ┃ ┗ 📂FP32
 ┃   ┃   ┣ 📜gaze-estimation-adas-0002.bin
 ┃   ┃   ┗ 📜gaze-estimation-adas-0002.xml
 ┃   ┣ 📂head-pose-estimation-adas-0001
 ┃   ┃ ┣ 📂FP16
 ┃   ┃ ┃ ┣ 📜head-pose-estimation-adas-0001.bin
 ┃   ┃ ┃ ┗ 📜head-pose-estimation-adas-0001.xml
 ┃   ┃ ┗ 📂FP32
 ┃   ┃   ┣ 📜head-pose-estimation-adas-0001.bin
 ┃   ┃   ┗ 📜head-pose-estimation-adas-0001.xml
 ┃   ┗ 📂landmarks-regression-retail-0009
 ┃     ┣ 📂FP16
 ┃     ┃ ┣ 📜landmarks-regression-retail-0009.bin
 ┃     ┃ ┗ 📜landmarks-regression-retail-0009.xml
 ┃     ┗ 📂FP32
 ┃       ┣ 📜landmarks-regression-retail-0009.bin
 ┃       ┗ 📜landmarks-regression-retail-0009.xml
 ┣ 📂results
 ┃ ┗ 📜stats.txt
 ┣ 📂src  
 ┃ ┣ 📜face_detection.py
 ┃ ┣ 📜gaze_estimation.py
 ┃ ┣ 📜head_pose_estimation.py
 ┃ ┣ 📜input_feeder.py
 ┃ ┣ 📜landmarks_detection.py
 ┃ ┣ 📜main.py
 ┃ ┗ 📜mouse_controller.py
 ┃ 
 ┣ 📜README.md
 ┣ 📜requirements.txt
 ┗ 📜vtune_cpu_script.sh
```
    
## Demo
Step 1:  Go back to the project directory src folder
 
        `cd path_of_project_directory` 
Step 2: Run below commands to execute the project
 * Run on CPU
 ```
python src/main.py -fd models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001 -hp models/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001 -fl models/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009 -ge models/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002 -d CPU -i bin/demo.mp4
```
* Run on GPU
 ```
python src/main.py -fd <Face detection model name directory> -fl <Facial landmark detection model name directory> -hp <head pose estimation model name directory> -ge <Gaze estimation model name directory> -i <input video directory> -d GPU
```
* Run on FPGA
 ```
python src/main.py -fd <Face detection model name directory> -fl <Facial landmark detection model name directory> -hp <head pose estimation model name directory> -ge <Gaze estimation model name directory> -i <input video directory> -d HETERO:FPGA,CPU
```  
* Run on NSC2
 ```
python src/main.py -fd <Face detection model name directory> -fl <Facial landmark detection model name directory> -hp <head pose estimation model name directory> -ge <Gaze estimation model name directory> -i <input video directory> -d MYRIAD
```     
       
## Documentation
Below are the command line arguments needed and there brief use case.

Argument|Type|Description
| ------------- | ------------- | -------------
-fd | Required | Path to a face detection model xml file with a trained model.
-fl | Required | Path to a facial landmarks detection model xml file with a trained model.
-hp| Required | Path to a head pose estimation model xml file with a trained model.
-ge| Required | Path to a gaze estimation model xml file with a trained model.
-i| Required | Path to image or video file or WEBCAM.
-l| Optional | Absolute path to a shared library with the kernels impl.
-pt  | Optional | Specify confidence threshold which the value here in range(0, 1), default=0.5
-flag  | Optional | for see the visualization of different model outputs of each frame.
-d | Optional | Provide the target device: CPU / GPU / VPU / FPGA


## Benchmarks
 Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.
 The Performance tests were run on HP Laptop with **Intel i7-8565U 1.99Ghz** and **16 GB Ram**

#### CPU

| Properties       | FP32        | FP16        | INT8        |
| ---------------- | ----------- | ----------- | ----------- |
| *Model Loading*  | 2.864384s   | 2.834568s   | 2.881543s   |
| *Inference Time* | 9.084232s   | 9.002353s   | 9.015497s   |
| *Total FPS*      | 1.245678fps | 2.67456fps | 2.134785fps |

## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

As we can see from the results, models with lower precision gives us better inference time but they lose in accuracy. This happens as lower precision model uses less memory and they are less computationally expensive.

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.
