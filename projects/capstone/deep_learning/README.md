## Synopsis

Submission report and code for [Capstone Project] (https://docs.google.com/document/d/1L11EjK0uObqjaBhHNcVPxeyIripGHSUaoEWGypuuVtk/pub).
Under Udacity's Machine Learning Engineer Nanodegree.

Multi-digit recognition with Convolutional Neural Networks.
A deep feedforward neural network is designed, developed and evaluated to classify digits from the Street View House Numbers (SVHN) dataset.

## Libraries / Packages

The code has the following prerequisites:
* Python 2.7
* TensorFlow
* numpy
* matplotlib
* seaborn
* h5py

## Dataset

This project uses the [Street View House Numbers(SVHN)](http://ufldl.stanford.edu/housenumbers/) dataset which can be downloaded.

>SVHN is a real-world image dataset for developing machine learning and object recognition algorithms with minimal requirement on data preprocessing and formatting. It can be seen as similar in flavor to MNIST (e.g., the images are of small cropped digits), but incorporates an order of magnitude more labeled data (over 600,000 digit images) and comes from a significantly harder, unsolved, real world problem (recognizing digits and numbers in natural scene images). SVHN is obtained from house numbers in Google Street View images. 

## Code
The following files are included for the project:
* '1_preprocess_single.ipynb' - Code for preprocessing SVHN Format 2 (single digits). Not used in the report.
* '2_model_single.ipynb' - Code for single digit CNN model. Not used in the report.
* '3_preprocess_multi.ipynb' - Code for preprocessing SVHN Format 1 (multiple digits).
* '4_model_multi.ipynb' - Code for multi-digit CNN model. Final model for the report.

* '5_figplot_helper.ipynb' - Helper code to plot learning curves and loss curves against multiple model runs.
* '6_model_eval_valid.ipynb'. - Helper code to run sensitivity analysis on model.
* '7_data_exploration_helper.ipynb' - Helper code to run visualizations for data exploration.

## Installation / Running

This project is run under Amazon Web Services (AWS) platform for computational machine learning.
This [Amazon Machine Image (AMI)](https://aws.amazon.com/marketplace/pp/B01EYKBEQ0) is used for developing the project code in Jupyter notebook.
The AMI comes with TensorFlow and other relevant packages pre-installed.

For instructions on setting up an Amazon EC2 Instance to run, you may find it [here] (https://aws.amazon.com/getting-started/tutorials/launch-a-virtual-machine/).
Instructions on connecting to the EC2 Instance are found [here] (http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AccessingInstances.html)
Then connect to the EC2 Instance and in the terminal window, launch Jupyter notebook:

'jupyter notebook'

And open your internet browser and navigate to the Public IP of the Instance, followed by the port Jupyter notebook uses (typically 8888):

'{Public IP}:8888'

Else, if running in a local machine, then open your command window or terminal and execute the code files below:

'jupyter notebook 3_preprocess_multi.ipynb'
'jupyter notebook 4_model_multi.ipynb'

To load TensorBoard, in your command window or terminal:
The general command line format is as follows:
'tensorboard --logdir=\path_to_event_files'

For example, to run TensorBoard, type 
'tensorboard --logdir=\home\ubuntu\model'
Then, launch TensorBoard by navigating to your internet browser and type 
'{Private IP Address}:6006'




