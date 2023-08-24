# Transfer learning induced DeIT for alcohol consumption detection

A Pytorch C++ based API for alcohol consumption detection using periocular NIR images on mobile devices. Here, we employ transfer learning induced [Data-efficient Image Transformer(DeIT)](https://arxiv.org/pdf/2012.12877.pdf) model for alcohol consumption detection using periocular NIR iris images [dataset](https://ieee-dataport.org/documents/nir-iris-images-under-alcohol-effect). 

## Deployment Code:

Follow these steps to utilize the API for your mobile devices:

1. git clone https://github.com/system-reboot/DeiT-for-alcohol-consumption-Detection.git
2. cd DeiT-for-alcohol-consumption-Detection
3. mkdir build/ && cd build/
4. cmake -DCMAKE_PREFIX_PATH=<absolute_path_to_libtorch>
5. make
6. ./Alcohol-Consumption-Detector path-to-your-image-file

## Files:

1. model/binary_class_model.ipynb - Model trained to detect if the subject is under alcohol consumption or not.
2. model/multi_class_model.ipynb - Model trained to study the temporal impact of alcohol on iris central nervous system(CNS).
3. src/main.cpp - Sample main file to load and preprocess sample images which is then classified using the pre-trained model.

## Note:

The weights utilized for training the above model has not published due to privacy concerns of the subjects in the dataset. One can contact [Juan Tapia Farias](https://scholar.google.es/citations?user=DtZ-bo8AAAAJ&hl=es) regarding the datset.
