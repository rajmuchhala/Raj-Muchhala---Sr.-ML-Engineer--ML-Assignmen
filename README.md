# License plate OCR solution

# Dataset

The image HDR dataset contains 652 normal and HDR readings of cropped license plates with a csv containing
annotation as well. You can download the dataset from here : https://medusa.fit.vutbr.cz/traffic/research-topics/general-traffic-analysis/holistic-recognition-of-low-quality-license-plates-by-cnn-using-track-annotated-data-iwt4s-avss-2017/

# File : trainVal.csv

1. track_id - ID of specific track based on tracker
2. image_path - path to image in archive structure
3. lp - ground truth text for license plate
4. train - Train/test split. 0 - test, 1 - train

# Preprocessing the Dataset

1. After reading the csv file as a pandas dataframe, shuffle the dataframe before splitting the data into Train and Test.
2. Preprocess the images to grayscale, resize every image to size(32,128) and then normalize images by dividing it by 255.
3. To preprocess the labels, read the 'lp' column values from the dataframe and compute some numerical value for each character/number in the labels by using an encoding function.

# Network Architecture

The network architecture is inspired by https://arxiv.org/pdf/1507.05717.pdf. 
Source model consists of three main components:

1. CNN for extraction of spatial feayures from the image
2. RNN to predict sequential output per time-step
3. CTC loss function which is transcription layer used to predict output for each time step.

# Loss Function

We use the CTC(Connectionist Temporal Classification) loss function which is widely used for text recognition problems. 
CTC loss function requires four arguments to compute the loss : predicted labels, ground truth labels, input sequence length to the LSTM and ground truth label length. Since keras does not have a CTC implementation, a custom loss function is created and passed  into the model. This model is used for training and for testing we will use the model that we have created earlier “act_model”. 

# Training

The Dataset has a total of only 652 cropped license plate images which is not enough to train the entrire network from scratch with random weight initializations. Therefore, we are going to apply Transfer Learning and use a pre-trained model trained on a large dataset with pre-trained weights. 

The Pre-trained model and weights used here is available at https://github.com/TheAILearner/A-CRNN-model-for-Text-Recognition-in-Keras with full source code and documentation.

# Approach

Since our HDR dataset is small and very similar to the dataset on which the Pre-trained model was trained on, we will initialize our model weights with the pre-trained model's weights and then fine-tune the entire network by training on 80% of the images and testing on the remaining 20%. Overfitting concerns due to fine-tuning were handled by using Dropout layers in the Network and also by using early stopping if necessary. 

After fine-tuning the network for 10 epochs with a batch size of 16, we save the final weights as 'best_model.hdf5' based on the value of Validation loss. 

# Evaluation on the Test set

Our model is now trained and fine-tuned on around 520 HRD license plate images and we can now evaluate the model to predict text from the 130 Validation/Test images using the model.predict() function and a CTC decoder. Some of the validation text results are displayed alongside the original text. We actually get decent results on our test set.

# Evaluation Metric: Accuracy

To test how well our model performs on the test set , we use the "Accuracy" metric and calculate the accuracy on the test set. Test accuracy is somewhere between 95-97% depending on our shuffled data split. 

# Test the OCR model on your own custom image

We now have our OCR model ready that can read the text on the license plates with decent accuracy. To test the OCR model on your own custom image, use the 'test_ocr' method which takes a cropped image as an input and outputs the text on the license plate.
