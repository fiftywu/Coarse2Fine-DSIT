# Dynamic2static-Image-Transformation
* Dynamic Region Detection and Static Scene Recovery in Visual SLAM System.
* Acknowledge: **Postgraduate Research＆Practice Innovation Program of Jiangsu Province SJCX20_0035**
* Time: 2020.06--2021.06



# 1 Patent Experiment Setup

- Train Model-RGB-G1 under Emptycites_Carla_SLAM train dataset **RGB**
- Test Model-RGB-G1 in THREE test dataset **RGB**

- FIx Model-RGB-G1 train Model-RGB-G2 under Emptycities_Carla_SLAM train dataset **RGB**



# 2 Paper Experiment Setup

## 2.1 Dataset

### 2.1.1 Generation Aspect

- train Based: Emptycites_Carla_SLAM train dataset
- test Based: Emptycites_Carla_SLAM test dataset, and build another TWO (different dynamic rate)

### 2.1.2 Re-localization Aspect

- ref: https://github.com/FiftyWu/Carla-Visual-Relocalization

- The same scene images under different dynamic rate set-up

- Compare ORB based and deep feature based re-localization methods

## 2.2 Experiment

- Test Emptycities Dataset performance in THREE test dataset **GRAY**
- Train & Test Emptycities_SLAM model performance in THREE test dataset under 256X256 **GRAY**
- Train Model-GRAY-G1 under Empties_Carla_SLAM train dataset **GRAY**
- Test Model-GRAY-G1 in THREE test dataset **GRAY**
- FIx Model-GRAY-G1 train Model-RGB-G2 under Empties_Carla_SLAM train dataset **GRAY** 
- Train Model-GRAY-G under Empties_Carla_SLAM train dataset **GRAY**

