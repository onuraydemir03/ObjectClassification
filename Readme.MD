# Training a ResNet18 on Custom Dataset

## Steps
***
This project made by steps
* Step 1 - Downloading images
~~~
- download_data.py 
This script downloads the crops & saves them in usable format
~~~
* Step 2 - Crop Detections
~~~
- crop_images.py
This script crops the images by detection bboxes
~~~
* Step 3 - Split Dataset (Train & Val)
~~~
- split_dataset.py
This script splits the dataset into multiple sets
~~~
* Step 4 - Train the Network
~~~
- train.py
Trains the ResNet18 network
~~~
* Step 5 - Extract Features
~~~
- feature_extractor.py
Extracts features from all the image crops and saves them as a torch dict
~~~
* Step 5 - Get the most similar X samples
~~~
- find_similars.py
Finds the most similar X crops by given one test sample, creates a result image
~~~
***