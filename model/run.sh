#!/bin/bash
set -eu

usage()
{
  echo "usage: IMAGE_ZIP IMAGE_DIR"
  echo "IMAGE_ZIP: the original tiny-imagenet zip file"
  echo "IMAGE_DIR: a working directory for this run,"
  echo "           which will be removed and re-created"
}

if (( ${#} != 2 ))
then
  usage
  exit 1
fi

IMAGE_ZIP=$1
IMAGE_DIR=$2

# IMAGE_ZIP=$HOME/work/ai/tiny-imagenet-200.zip

# IMAGE_DIR=/dev/shm/resnet-input

# rm -rf $IMAGE_DIR; mkdir -p $IMAGE_DIR
echo Unzipping images...
# time unzip $IMAGE_ZIP -d $IMAGE_DIR &> /dev/null
echo Running ResNet50...
# --exclude-range string is of the form 1.1.2.1, which means the dataset into 8 pieces, keep the 7th and delete everyting else
which python
python3 keras_resnet50.py \
        --zip $IMAGE_ZIP  \
        --data-dir $IMAGE_DIR \
        --epochs 1 \
        --exclude-range 1.2
