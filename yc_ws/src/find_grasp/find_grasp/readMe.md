# Graspnet

This is a model to predict the grasps based on the pointclouds. The input is the egbd image, the output is the predicted grasps

### Dataset
To train the model, make sure you install the correct datasets first. 
- Grapsnet1b: https://graspnet.net/datasets.html
- Metagraspnet: https://github.com/maximiliangilles/MetaGraspNet

To convert the Metagraspnet labels into Graspnet1b:
```
cd dataset
python meta_data_handler.py
```

To view the Metagraspnet Ground Truth:
```
cd MetaGraspNet
# the scene id can be changed, the data_root id also can be changed
python ./Scripts/colored_pcl.py --data_root /media/zhy/Data2TB1/MetaGraspNet/mnt/data1/data_ifl --scene 2153 --visualize_parallel_gripper --colorize_per_score analytical --viewpt 0
```

To generate the ground truth Graspnet1b:
```
cd dataset
python generate_graspness.py
# after this process finished, run the next command line:
python simplify_dataset.py
```


First install the poitnet2:
```
cd pointnet2
python setup.py install
cd ..
cd pointnet2_
python setup.py install
```
## Training
Then, run the training file from the root directory of the project with the following commands.
If you want to train graspnet1B, then you need to run:
```
python train_GPU.py --dataset_root /media/zhy/Data2TB1/GraspNet1B --log_dir logs/log
```
If you want to train MetaGraspnet, then you need to run:
```
# first generat the folder: logs/log/meta_new
sudo python train_GPU.py --log_dir logs/log/meta_new --batch_size 1
```
If you want to train the combined dataset, then you need to run:
```
# first generat the folder: logs/log/combined
sudo python train_GPU.py --log_dir logs/log/combined --batch_size 1
```

### Using the onnx model
- convert the model into onnx model:
```
cd models
python3 convert_model_to_onnx.py
```

- run the onnx model:
one examle of scene id is 168
```
cd evaluation
python3 inference.py --scene_id 168
```

- run the original model:
under the root folder

one examle of scene id is 168
```
python3 check_dataset.py --scene_id 168
```

### Structure
The file structure of the model is as follows:
- dataset: Functions for generate dataset for training
  - graspnet_dataset.py: this is the data structure for graspnet-1b
  - meta_data_handler.py: this is the meta conversion python file
  - metagraspnet_dataset.py: this is the data structure for metagraspnet
  - combined_dataset.py: this is teh data structure for the combined dataset
- model: contains the files for 
  - convert_model_to_onnx.py: convert the model into onnx model
  - graspPcPnnx.py: run the onnx model inference
  - graspnet.py: Class for the structure of the model structure
- visuals: Fucntions for visualizing the grasps predicted
- utils: Directory for all helping functions used in various processes
 trained and tested without needing extensive installation of different packages
    - train_CPU.py: File to run the training process for the model
    - test.py: File to inference using the original model
    - compare_grasps: File to evaluate the grasps predicted by the model and the graspnet-1b