# Fetal brain segmentation in T2-weighted MRI

This is a deep learning model for segmentation of brain in fetal T2  MR images.

Developed at IMAGINE laboratory (https://imagine.med.harvard.edu/) of Boston Children's Hospital.

Training code can be provided to the interested upon request (davood.karimi@childrens.harvard.edu).

## Docker model

A trained model can downloaded and run on your images using the following two lines, where in the second line IMG_DIR should be replaced with your local directory where your images have been saved. The code automatically creates a "segmentation" subdirectory under IMG_DIR, where the segmentations are saved.

docker pull davoodk/brain_extraction

docker run   --mount src=/IMG_DIR/,target=/src/test_images/,type=bind  brain_extraction:1.0


## Example results:

### Older fetus

![older](sample_results/older.png)  


### Younger fetus

![older](sample_results/younger.png)  

