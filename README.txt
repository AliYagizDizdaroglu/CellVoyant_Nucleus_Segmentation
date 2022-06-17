This project consists of 2 different solutions to the expected results:

* statistics.csv 
* layered/masked images


Solution 1, Manual Trained Model:

* Chose a pretty common and strong deep learning algorithm/model named "U-net" which is a CNN type.
* Had to resize the images to 768x768 because of RAM issues. Then converted to original size forming pieces of small images.
* Size re-conversion is a bit scuffed.
* For details please see the code.


Solution 2, Pre-Trained Model:

* Used pre-trained nucleus segmentation module.
* After that process, necessary steps have been conducted as follows: Preprocessing, cleaning, labelling, formatting etc.
* As a conclusion, this model seemed to have better performance overall.