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

    
## Demo
Step 1:  Go back to the project directory src folder
 
        `cd path_of_project_directory
         cd src` 
Step 2: Run below commands to execute the project
      
       
## Documentation
*TODO:* Include any documentation that users might need to better understand your project code. For instance, this is a good place to explain the command line arguments that your project supports.

## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.
