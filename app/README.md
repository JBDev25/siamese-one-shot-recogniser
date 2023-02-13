# Face Recogniser

# How to use it
## Setup
Ensure the model is loaded into `application_data/model/siamese_model.h5`
##Run the App
1) Enable the virtual env
`source venv/bin/activate`
2) Install Requirements
`pip install -r requirements.txt`
3) Run the App
`python main.py`
## Use the App
1)Use the "Caturue anchor" button to gather the images against wich the input image will be validated. You can gather as many as 10.
2) "Validate Image" will validate the present frame against the validation images.
Note: Depending on the speed of your machine this may take up to 10 seconds.

#How it works
The siameas model is loaded into memeory and performs a validation on an input image against a set number of verification images, returning a float from 0-1.
The theshold of postive ID is 0.5 and the verification threshold (the number of postives/total varification images) is 0.5.
