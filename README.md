The directory is organized as follows:

The task descriptions are within text files. Each contains the prompt for the experimental configurations described in my paper. They are numbered consistently as those in the paper. 

preprocess.py contains code for processing the data into the format needed for use with chatgpt. 

model.py contains code for loading the data set. For task 3, the parameter, training_mode should be set to true and n_training_instances should be a positive integer. 
For tasks 1 and 2, training_mode should be set to false for zero-shot classification.

Dependencies are listed in requirements.txt

The code can be run using python model.py. 

Note: this does require an openai api key to be stored in txt file in the same directory as the model.py script. 
