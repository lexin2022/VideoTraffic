# Video_Traffic_Project

The project implements the video traffic identification function in the high-speed network, which mainly includes the following modules:

- Feature extraction module
- Flow labeling module
- Machine Learning Module

## Files

- **main_video_flows.cpp:** Analyze local PCAP files, organize flows and extract traffic characteristics
- **main_flow_HT.cpp:** Extract Server Name Indication (SNI) from the protocol information of each flow, and batch label video flows based on SNI
- **quick_model_train.py: **Training machine learning model and applying the models to video traffic identification in high-speed networks


## Using

### VideoFlows.exe

This program is used to analyze local PCAP files and extract features.

`VideoFlows.exe` needs to be run under Windows. Here we show how to compile and run `VideoFlows.exe`.

#### Compiling Environment

Windows: `Visual Studio Code 64-bit` + `CMake 3.19.3` + `MinGW 8.1.0`

#### Build

```shell
$ make VideoFlows
```

#### Run

##### 1.Modify the settings in `data.cfg`

```C++
//================================VideoFlows.exe=====================================
//the path of the pcap file to be analyzed.
VF_File = "E:/Code Repository/video_traffic/data/video_720p_facebook_mix.pcap.Mixed_rseed-22.pcap";

VF_seed = 2222;		  //random seed for packet sampling
VF_ratio = 1;       //packet sampling ratio: 0(1/1),1(1/8),2(1/16),3(1/32),...
VF_Elephant = 100;  //threshold values for large flow filtering
VF_Stat1 = 500;     //Judgment 1
VF_Stat2 = 2000;    //Judgment 2
VF_Stat3 = 5000;    //Judgment 3
VF_Idle = 1.0;      //idle time
```

##### 2.run

- Delete the `CMakeCache.txt` file from the `build` folder, then copy the `data.cfg` file from the project root directory to the `build` folder
- Modify the `VF_File` and `VF_bigdata_csv` paths in the `data.cfg` file just copied to the `build` folder to your own path
- Compile and run your own `VideoFlows.exe` file

### FlowHttpTls.exe

This program is used to generate sample sets with labels (video flows and non-video flows). The functions are to analyze the `handshake` protocol information of the flows, extract SNI and batch label the flows.

`VideoFlows.exe` needs to be run under Windows. Below we show how to compile and run `VideoFlows.exe`.

#### Compiling Environment

Windows: `Visual Studio Code 64-bit` + `CMake 3.19.3` + `MinGW 8.1.0`

#### Build

```shell
$ make FlowHttpTls
```

#### Run

##### 1.Modify the settings in `data.cfg`

```C++
//==============================FlowHttpTls.exe==========
// the path to the original pcap folder to be tagged, with all pcap files under the bulk folder
FHT_path = "E:/Code Repository/video_traffic/data/";

FHT_bit = 25;       //the number of bits of hash value
FHT_filter = 100;  //threshold values for large flow filtering
```

##### 2.run

- Delete the `CMakeCache.txt` file from the `build` folder, then copy the `data.cfg` file from the project root directory to the `build` folder
- Modify the `VF_File` and `VF_bigdata_csv` paths in the `data.cfg` file you just copied to the `build` folder to your own path
- Compile and generate your own `FlowHttpTls.exe` file

### Machine Learning

Apply the `video_traffic_project` project to video traffic identification.

The video_traffic_project outputs labeled feature vectors to `*.csv` files, and each feature vector corresponds to a unique `HashValue`.

As shown in [Figure 1](#Figure 1. System Architecture), the whole video traffic identification project is divided into an offline model training phase and an online model deployment phase.

#### 1.Model training phase

- First, we collect video traffic from multiple platforms as a dataset. 
- Second, we label the video flows based on their handshake or request information and extract features to obtain a labeled sample set. 
- Finally, we use a supervised machine learning algorithm to train the labeled sample set to obtain a classification model.

#### 2.Model Deployment Phase. 

- First, we perform sequential sampling of packets arriving in the high-speed network. 
- Second, we reassemble the obtained packets into flows and extract features from the flows. 
- Finally, we apply the classification model obtained during the model training phase to identify the new flow.

###### Figure 1. System Architecture

![System Architecture](https://github.com/lexin2022/VideoTraffic/blob/main/images/System%20Architecture.png)

## Others

The project also contains some other tools that are coded in python and stored in the `tools` folder.

### Requirements

Installing Requirements

```shell
pip install -r requirements.txt
```

### Using

#### Data_Visualization

Includes some drawing-related programs

- `column_chart.py`: Plotting bar graphs
- `pck_count.py`: Count the distribution of the length of each packet in a flow
- `PLD_Drawn.py`: Plot the probability density distribution based on the packet length in a flow
- `plt_zexiantu.py`: Drawing line graphs
- `tmGap.py`: Statistical packet transmission time interval

#### Randomfrorest

Contains some procedures related to model training

- `quick_model_train.py`: Model Training
- `quick_model_predict.py`: Apply models to the dataset for prediction
- `select_features.py`: Select a small number of critical features based on feature importance to construct feature space
- `ITP-KNN.py`: Reproduction of the method in paper 1
- `silhouette.py`: Reproduction of the method in paper 2

## REFERENCES

> [1] Y. Liu, S. Li, C. Zhang, C. Zheng, Y. Sun, and Q. Liu,“Itp-knn: Encrypted video flow identification based on the intermittent traffic pattern of video and k-nearest neighbors classification,” in International Conference on Computational Science. Springer, 2020, pp. 279–293.
>
> [2] F. Li, J. W. Chung, and M. Claypool, “Silhouette: Identifying youtube video flows from encrypted traffic,” in Proceedings of the 28th ACM SIGMM Workshop on Network and Operating Systems Support for Digital Audio and Video, 2018, pp. 19–24.

## Contact Us

If you have any problems, please contact [xle@seu.edu.cn](mailto:xle@seu.edu.cn).
