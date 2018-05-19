@echo off
::"build/tools/caffe.exe" device_query --gpu=0
::set modelname=googlenet
set modelname=alexnet
::set modelname=reference_caffenet

"build/tools/caffe.exe" time --model=models/bvlc_%modelname%/deploy.prototxt --gpu all
pause