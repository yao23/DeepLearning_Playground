# Bike Sharing Pattern Prediction
In this project, I build a neural network from scratch to carry out a prediction problem on a real dataset! By building a neural network from the ground up, I have a much better understanding of gradient descent, backpropagation, and other concepts that are important to know before I move to higher level tools such as PyTorch. I also get to see how to apply these networks to solve real prediction problems!

The data comes from the [UCI Machine Learning Database](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset).

## Instructions
1. Download the project materials from my [GitHub repository](https://github.com/yao23/DeepLearning_Playground). You can get download the repository with 
	```
	git clone https://github.com/yao23/DeepLearning_Playground.git.
	```
1. cd into the project-bikesharing directory.
1. Download anaconda or miniconda based on the instructions in the [Anaconda lesson](https://classroom.udacity.com/nanodegrees/nd101/parts/2a9dba0b-28eb-4b0e-acfa-bdcf35680d90/modules/aba54606-cf35-4a77-b643-efec6a90bfa1/lessons/9e9ed61d-20c3-4431-95aa-a1099f28d601/concepts/4cdc5a26-1e54-4a69-8eb4-f15e37aaab7b). These are also outlined in the repository README.
1. Create a new conda environment:
	```
	conda create --name deep-learning python=3
	```
1. Enter your new environment:
	* Mac/Linux: 
		```
		>> source activate deep-learning
		```
	* Windows: 
		```
		>> activate deep-learning
		```
1. Ensure you have numpy, matplotlib, pandas, and jupyter notebook installed by doing the following:
	```
	conda install numpy matplotlib pandas jupyter notebook
	```
1. Run the following to open up the notebook server:
	```
	jupyter notebook
	```
1. In your browser, open Predicting_bike_sharing_data.ipynb. Note that in the previous workspace this was called Your_first_neural_network.ipynb but the contents are the same, this is just a descriptive difference.
1. Follow the instructions in the notebook; they will lead you through the project. You'll ultimately be editing the my_answers.py python file, whose components are imported into the notebook at various places.
1. Ensure you've passed the unit tests in the notebook.