@echo off
"../bin/classification_multilabel" deploy.prototxt trainedmodels/AlexNet_iter_1000.caffemodel labels/labels.txt ZnCar/Train/1.jpg
pause