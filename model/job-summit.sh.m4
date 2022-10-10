#!/bin/bash

# JOB SUMMIT SH M4
# Requires environment variables THIS, OUTPUT

# Use dnl as comment:
m4_changecom(`dnl')
# This does environment variable substitution in this file:
m4_define(`getenv', `m4_esyscmd(printf -- "$`$1'")')

#BSUB -P MED106
#BSUB -J ResNet-50
#BSUB -nnodes 1
#BSUB -W 30
#BSUB -cwd getenv(THIS)
#BSUB -o getenv(OUTPUT)
#BSUB -e getenv(OUTPUT)
#BSUB -alloc_flags NVME

module load ibm-wml-ce/1.7.0-3

set -eu
set -x

which python

THIS=getenv(THIS)
OUTPUT=getenv(OUTPUT)

IMAGE_ZIP=$THIS/tiny-imagenet-200.zip
IMAGE_DIR=$THIS

hostname
pwd

cd $THIS

pwd

JSRUN_FLAGS="-n1 -r1 -a1 -g1"

NVM=/mnt/bb/$USER

time jsrun -n 1 echo OK
time jsrun -n 1 printenv PATH
# time jsrun -n 1 ls /usr/bin
# cp -uv /usr/bin/unzip .
# time jsrun -n 1 ./unzip -q $IMAGE_ZIP -d $NVM

time jsrun -n 1 ls -ld $NVM
time jsrun -n 1 ls -l  $NVM

echo Running ResNet50...
time jsrun $JSRUN_FLAGS                      \
  python3 $THIS/keras_resnet50.py            \
    --zip $IMAGE_ZIP \
    --data-dir $NVM \
    --epochs 10                               \
    --exclude-range 1.2
echo EXIT CODE: $?

echo Complete.

# Local Variables:
# mode: m4;
# End:
