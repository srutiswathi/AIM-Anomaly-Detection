# AIM Spring 2024

## Zone Defend :shield:

In a modern world teeming with threats, the number of anomalies in college campuses is rapidly increasing. From stealing and vandalism of students' property to road accidents and shootings in parking lots, traditional security measures fall short in detecting and deterring these evolving threats, leaving campuses vulnerable and communities at risk. 

That is until **Zone Defend**. Zone Defend is an AI model designed to detect common anomalies on college campuses using real-time surveillance videos. Our product is designed for medium to large college campuses which is over 5000 campuses in the U.S and is targeted towards the school's safety department.

## Our Team :sunglasses:

| Name                  | Role        |
| -------------------   | ----------- |
| Sruti Karthikeyan     | _Mentor_    |
| Anthika Gunaselan     | _Developer_ |
| Laksha Arora          | _Developer_ |
| Love Bhusal           | _Developer_ |
| Zubiya Syeda          | _Developer_ |
| Daniel Nguyen         | _Developer_ |

## Contributions :wave:

Description

## Try it out! :camera:

1. In [demo.ipynb](https://github.com/srutiswathi/AIM-Anomaly-Detection/blob/main/demo.ipynb), resize your video
```python
resize_video(old_video_path, new_video_name, new_video_root = "./demo_videos")
```

2. Extract RGB and flow frames
```python
video_to_rgb(video_path, video_name, new_rgb_root = "./demo_rgb")
video_to_flow(video_path, video_name, new_flow_root = "./demo_flow")
```
3. Finally, run the **Demo** section and see the predictions!
```python
Campfire
+---------------+-------------+
|     Class     | Probability |
+---------------+-------------+
|     Arson     |   0.93741   |
|   Vandalism   |   0.05569   |
| RoadAccidents |   0.00514   |
|    Fighting   |   0.00074   |
|    Assault    |   0.00060   |
|    Shooting   |   0.00032   |
|    Stealing   |   0.00007   |
|     Normal    |   0.00004   |
+---------------+-------------+
```

## Dataset :floppy_disk:

Our dataset consists of videos from [UCF-Crime](https://www.crcv.ucf.edu/projects/real-world/). Videos are from relevant categories: _Arson_, _Assault_, _Fighting_, _Road Accidents_, _Shooting_, _Stealing_, _Vandalism_, and _Normal_. We clip each video to capture the timestamp of the corresponding action and label each video with an action.

### Training Size

| Category | Count |
| ------------------- | ----------- |
| Arson | 9 |
| Assault | 3 |
| Fighting | 5 |
| Road Accidents | 23 |
| Shooting | 9 |
| Stealing | 5 |
| Vandalism | 5 |
| Normal | 41 |

### Training/Validation Split

| Training | Validation |
| ------------------- | ----------- |
| 69 | 31 |

## Training :robot:

### Training 3

#### Training 3 Details

| Stat | Value |
| ------------------- | ----------- |
| Initial Learning Rate | 0.1 |
| Max Steps | 5000 |
| Batch Size | 10 |
| Optimizer | SGD with momentum 0.9 |

#### Training 3 Data Augmentation

Video frames are resized to 240 pixels on the smaller side. During training phase, a 224 x 224 random crop and random horizontal flip is performed.

#### Training 3 Loss Graph RGB
![Training 3 Validation Loss](https://github.com/srutiswathi/AIM-Anomaly-Detection/blob/main/newmodels_info/Training1/validationlossrgb.png?raw=true)
![Training 3 Training Loss](https://github.com/srutiswathi/AIM-Anomaly-Detection/blob/main/newmodels_info/Training1/traininglossrgb.png?raw=true)

#### Training 3 Loss Graph Flow
![Training 3 Validation Loss](https://github.com/srutiswathi/AIM-Anomaly-Detection/blob/main/newmodels_info/Training1/validationlossflow.png?raw=true)
![Training 3 Training Loss](https://github.com/srutiswathi/AIM-Anomaly-Detection/blob/main/newmodels_info/Training1/traininglossflow.png?raw=true)

## Testing :mag:
Testing consisted of selecting 5 videos from each category from our dataset (that was not used in training) for a total of 40 videos. Training 3 shows the most promising results with an average ROC AUC Score of 78%. As observed in [Quo Vadis](https://arxiv.org/abs/1705.07750), the I3D model greatly benefitted from starting with a model pretrained on Kinetics. 
### Training 3 (BEST) ROCAUC
Training 3 started training on a Kinetics-pretrained I3D model.\
```ROC AUC SCORE: 78%```\
![Training 3 (BEST) ROCAUC](https://github.com/srutiswathi/AIM-Anomaly-Detection/blob/main/newmodels_info/Training3/rocauc3.png?raw=true)
### Training 4 (WORST) ROCAUC
Training 4 started training on a blank I3D model.\
```ROC AUC SCORE: 58%```\
![Training 4 (WORST) ROCAUC](https://github.com/srutiswathi/AIM-Anomaly-Detection/blob/main/newmodels_info/Training4/rocauc4.png?raw=true)

## Future Plans :chart_with_upwards_trend:

- Increase dataset size
- Real-world implementations
- Partner with schools and universities

## Credits :books:

- Our model is based on Two-Stream Inflated 3D from [Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset](https://github.com/google-deepmind/kinetics-i3d)
- Our dataset is retrieved from [UCF-Crime](https://www.crcv.ucf.edu/projects/real-world/)
- [I3D PyTorch](https://github.com/piergiaj/pytorch-i3d/tree/master)
- [Denseflow code](https://github.com/qijiezhao/py-denseflow/tree/master)
