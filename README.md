# Social Distancing Violation Detection
Includes Object detection (efficientdet d4), Object tracking and Social Distance Logic.

# Folder structure
```
├── app.py
├── data
│   ├── input
│   │   └── input_video.mp4
│   └── output
│       └── output.mp4
├── __init__.py
├── object_detection
│   ├── efficientdet.py
│   ├── __init__.py
│   └── model
│       ├── config.json
│       ├── efficientdet_d4.pb
│       └── label_map.json

├── object_tracking
│   ├── deep_sort
│   │   ├── detection.py
│   │   ├── __init__.py
│   │   ├── iou_matching.py
│   │   ├── kalman_filter.py
│   │   ├── linear_assignment.py
│   │   ├── model
│   │   │   └── mars-small128.pb
│   │   ├── nn_matching.py
│   │   ├── preprocessing.py
│   │   ├── tracker.py
│   │   └── track.py
│   └── tools
│       ├── freeze_model.py
│       ├── generate_detections.py
│       └── __init__.py
├── utils
│   ├── get_params.py
│   ├── gui.py
│   ├── __init__.py
│   ├── projection.py
│   ├── violation_check.py
│   └── visualize.py



```

# Requirements
opencv-python          4.2.0.34  
pandas                 1.0.3  
scikit-learn           0.22  
tensorflow-gpu         1.15.0




# How to run the End to End pipeline

* Insert the video you want to run pipeline in "Data" folder.  

* Get the "efficientdet-d4_frozen.pb" and "mars-small128.pb" from google drive and paste in respective folder mentioned above.

* To run the main script which uses the Object detection (Efficientdet d4), DeepSort Object Tracker and Social distance logic, run the following command.


```
python3 app.py
```

# Output
Generated Output files are saved as output/output_video.mp4 , violation_id_counts.csv,  report.txt

# TODO

1. Introduce argument parser.
2. Increase number of points in gui to 12(4 for homography, 4 for dimension of birds eye view, 2 for upper distance threshold, 2 for lower distance threshold)
3. Write print statements to “report.txt“ file
4. Add comments wherever necessary.
5. Remove duplicate or redundant and unnecessary files.

