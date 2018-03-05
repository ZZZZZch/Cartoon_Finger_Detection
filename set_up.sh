#! /bin/bash

# Shell script set_up.sh
# Set up opencv-contrib

yes|yum install python34 python34-setuptools wget python34-devel python-qt4
yes|easy_install-3.4 pip

# Set up darkflow
pip3 install numpy scipy tensorflow cython opencv-contrib-python

python3 setup.py build_ext --inplace
pip3 install -e .
pip3 install .

# Run
python3 Search_Page.py --imgdir ./cartoon/all_imgs --model cfg/tiny-yolo-4c.cfg --load -1 --gpu 0
