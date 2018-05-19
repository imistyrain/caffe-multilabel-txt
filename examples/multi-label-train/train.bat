@echo off
"../../build/tools/caffe" train -solver  solver.prototxt -gpu 0
pause