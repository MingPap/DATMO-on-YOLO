# KITTI-VOL-YOLO-format
Some scripts for converting kitti data format to YOLO

## STEP1.  modify_annotations_txt.py
  This script is used for **merging and modifying** labels of origin kitti labels.
 
 **Note:** backup the origin kitti labels before you use it
 
## STEP2.  txt_to_xml.py
  This script is used for **generating VOL format labels** like *xxx.xml*.
  You should mkdir 'Annotations' folder in current dir before you use it
 
## STEP3.  kitti_label.py
  This script is modified by 'voc_label.py' provided by ***pjreddie/darknet*** to generate labels for YOLO. Generated labels has the same name with the pictures and also the origin labels.
 
  It also generates *train.txt* which contains absolutely paths of the pictures, which is important for training on YOLO.
 
  **Note:** you should put the labels in the same folder of pictures before you train
 
## STEP4. create_train_test_txt.py   ***(optional)***
  Randomly divide kitti dataset into proportional parts like *train.txt* *test.txt* *val.txt* *trainval.txt* and so on. This *.txt* files contains absolutely paths of the pictures like **STEP3**, and can be directly used for training. 
  
  Also you should create or modify relevant folders which you can see in the code before you use it



