#!/bin/bash

#
cc treecnv.c -o treecnv.so -I ~/torch/install/include -O3 -fopenmp -lm -fpic -shared
