### Project overview
Object detection is one of the most important tasks of self-driving cars which are required to understand sorrounding environments and properly navigate with safety.

In this project, I try to train a model which detects typical traffic participants, namely vehicles, pdestrians and cyclists, from images taken in a variety of environments. Starting with a pre-trained model, I experiment with some hyper parameters and image augmentations to examine how they affect the model performance.

It is a critical step of training a ML model to sufficiently understand data to be used. Therefore I perform a thorough EDA in advance of the training.

### Set up
<!-- This section should contain a brief description of the steps to follow to run the code for this repository. -->
Here is a set up instruction to run the code in this repository.

1. Build the docker image with:
```
docker build -t <IMAGE NAME> -f Docker file .
```
2. Create a container with:
```
 docker run --shm-size=<ALLOCATED MEMORY SIZE> --gpus all -v <PATH TO LOCAL PROJECT FOLDER>:/app/project/ --network=host -it <IMAGE NAME> bash
```
(Hereafter, you are in the container)

3. Download and process the data:
```
python download_process.py --data_dir /app/project/data/processed/ --temp_dir /app/project/data/raw/
```
4. Create splits:
```
python create_splits.py --data_dir app/project/data/
```
5. Training:
```
python model_main_tf2.py --model_dir=/app/project/training/reference/ --pipeline_config_path=/app/project/training/reference/pipeline_new.config
```
6. Evaluation:
```
python model_main_tf2.py --model_dir=/app/project/training/reference/ --pipeline_config_path=/app/project/training/reference/pipeline_new.config --checkpoint_dir=/app/project/training/reference/
```
7. Monitor the processses with TensorBoard:
```
tensorboard --logdir=training
```

### Dataset
#### Dataset analysis
This section should contain a quantitative and qualitative description of the dataset. It should include images, charts and other visualizations.
#### Cross validation
This section should detail the cross validation strategy and justify your approach.

### Training
#### Reference experiment
This section should detail the results of the reference experiment. It should includes training metrics and a detailed explanation of the algorithm's performances.

#### Improve on the reference