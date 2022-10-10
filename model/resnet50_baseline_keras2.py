
# RESNET50 BASELINE KERAS
# CANDLE-compliant version

import pprint

def setup(params):
    print("resnet50: setup(): params:")
    pprint.pprint(params)

def initialize_parameters(default_model='resnet50_default_model.txt'):
    print("resnet50: initialize_parameters... ")
    return {}

class ResNet50_Args:
    def __init__(self, data_dir, epochs, exclude_range):
        self.data_dir      = data_dir
        self.epochs        = epochs
        self.exclude_range = exclude_range
        # Defaults from keras_resnet50
        self.batch_size = 32
        self.val_batch_size = 32
        self.base_lr = 0.0125
        self.warmup_epochs = 5
        self.momentum = 0.9
        self.wd = 0.00005
        self.model_type = 50

def run(params):
    print("resnet50: run(): params")
    pprint.pprint(params)
    data_dir = getenv("RESNET50_DATA_DIR")
    args = ResNet50_Args(data_dir,
                         epochs=params["epochs"],
                         exclude_range=params["node"])
    import keras_resnet50
    keras_resnet50.run(args)
    # Example from run.sh:
    # --train-dir $IMAGE_DIR/tiny-imagenet-200/train
    # --val-dir $IMAGE_DIR/tiny-imagenet-200/val
    # --epochs 1
    # --exclude-range 1.1.2.1

def getenv(name):
    import os
    if name not in os.environ:
        print("put '%s' in the environment!" % name)
        raise Exception("put '%s' in the environment!" % name)
    return os.getenv(name)
