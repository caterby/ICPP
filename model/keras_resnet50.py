# This model is an example of a computation-intensive model that achieves good accuracy on an image
# classification task.  It brings together distributed training concepts such as learning rate
# schedule adjustments with a warmup, randomized data reading, and checkpointing on the first worker
# only.
#
# Note: This model uses Keras native ImageDataGenerator and not the sophisticated preprocessing
# pipeline that is typically used to train state-of-the-art ResNet-50 model.  This results in ~0.5%
# increase in the top-1 validation error compared to the single-crop top-1 validation error from
# https://github.com/KaimingHe/deep-residual-networks.
#

import tensorflow.keras
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import os, time, timeit

import preprocess

def set_parallelism_threads():
    """ Set the number of parallel threads according to the number available on the hardware
    """

    if 'NUM_INTRA_THREADS' in os.environ and 'NUM_INTER_THREADS' in os.environ:
        print('Using Thread Parallelism: {} NUM_INTRA_THREADS, {} NUM_INTER_THREADS'.format(os.environ['NUM_INTRA_THREADS'], os.environ['NUM_INTER_THREADS']))
        tf.config.threading.set_inter_op_parallelism_threads(int(os.environ['NUM_INTER_THREADS']))
        tf.config.threading.set_intra_op_parallelism_threads(int(os.environ['NUM_INTRA_THREADS']))

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Keras ImageNet Example',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--zip', default=None,
                        help='path to training zip (optional)')
    parser.add_argument('--data-dir', default=None,
                        help='path to data')
    # Default settings from https://arxiv.org/abs/1706.02677.
    parser.add_argument('--batch-size', type=int, default=32,
                        help='input batch size for training')
    parser.add_argument('--val-batch-size', type=int, default=32,
                        help='input batch size for validation')
    parser.add_argument('--epochs', type=int, default=90,
                        help='number of epochs to train')
    parser.add_argument('--base-lr', type=float, default=0.0125,
                        help='learning rate for a single GPU')
    parser.add_argument('--warmup-epochs', type=float, default=5,
                        help='number of warmup epochs')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--wd', type=float, default=0.00005,
                        help='weight decay')
    parser.add_argument('--exclude-range', default='1', help='exclusion range in 1.x.y.z format')
    parser.add_argument('--model-type', type=int, default=50, help='model type: 50 - ResNet50, 121 - DenseNet121, 152 - ResNet152')

    args = parser.parse_args()

    if args.data_dir == None:
        raise ValueError("You must provide --data-dir !")
    return args

def check_exists(label, fn):
    if not os.path.exists(fn):
        raise Exception("does not exist: (%s) '%s'" % (label, fn))

def run(args):

    print("TF version: " + tf.__version__)

    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    set_parallelism_threads()
    verbose = 1

    words_txt = args.data_dir + "/tiny-imagenet-200/words.txt"
    print(words_txt)
    if os.path.exists(words_txt):
        preprocess.restore_dir(args.data_dir + "/tiny-imagenet-200")
    elif args.zip != None:
        print("extracting zip: '%s' to '%s'" % (args.zip, args.data_dir))
        import zipfile
        start = time.time()
        with zipfile.ZipFile(args.zip, "r") as zf:
            zf.extractall(path=args.data_dir)
        stop = time.time()
        print("time unzip: %2.3f" % (stop-start))
    else:
        print("using existing data_dir: '%s'" % args.data_dir)

    # Training data iterator.
    train_dir = args.data_dir + "/tiny-imagenet-200/train"
    val_dir   = args.data_dir + "/tiny-imagenet-200/val"
    check_exists("train_dir", train_dir)
    check_exists("val_dir",   val_dir)

    preprocess.delete_region(train_dir, args.exclude_range)

    train_gen = image.ImageDataGenerator(
        width_shift_range=0.33, height_shift_range=0.33, zoom_range=0.5, horizontal_flip=True,
        preprocessing_function=tf.keras.applications.resnet50.preprocess_input)
    train_iter = train_gen.flow_from_directory(train_dir,
                                               batch_size=args.batch_size,
                                               target_size=(224, 224))

    # Validation data iterator.
    test_gen = image.ImageDataGenerator(
        zoom_range=(0.875, 0.875), preprocessing_function=tf.keras.applications.resnet50.preprocess_input)
    test_iter = test_gen.flow_from_directory(val_dir,
                                             batch_size=args.val_batch_size,
                                             target_size=(224, 224))

    # Set up standard ResNet-50 model.
    if args.model_type == 50:
        print("Running with ResNet50")
        model = tf.keras.applications.resnet50.ResNet50(include_top=True, weights=None, classes=200)
    elif args.model_type == 121:
        print("Running with DenseNet121")
        model = tf.keras.applications.DenseNet121(include_top=True, weights=None, classes=200)
    else:
        print("Running with ResNet152")
        model = tf.keras.applications.ResNet152(include_top=True, weights=None, classes=200)

    # ResNet-50 model that is included with Keras is optimized for inference.
    # Add L2 weight decay & adjust BN settings.
    model_config = model.get_config()
    for layer, layer_config in zip(model.layers, model_config['layers']):
        if hasattr(layer, 'kernel_regularizer'):
            regularizer = tf.keras.regularizers.l2(args.wd)
            layer_config['config']['kernel_regularizer'] = \
                {'class_name': regularizer.__class__.__name__,
                 'config': regularizer.get_config()}
        if type(layer) == tf.keras.layers.BatchNormalization:
            layer_config['config']['momentum'] = 0.9
            layer_config['config']['epsilon'] = 1e-5

    model = tf.keras.models.Model.from_config(model_config)
    opt = tf.keras.optimizers.SGD(lr=args.base_lr, momentum=args.momentum)

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=opt,
                  metrics=['accuracy'])
    # , 'top_k_categorical_accuracy'

    # ,              experimental_run_tf_function=False

    start_ts = timeit.default_timer()
    print("train_iter: ", len(train_iter))
    model.fit(train_iter,
              steps_per_epoch=len(train_iter),
              epochs=args.epochs,
              verbose=verbose,
              initial_epoch=0,
              validation_data=test_iter,
              validation_steps=len(test_iter)) # // hvd.size()

if __name__ == "__main__":
    args = parse_args()
    run(args)
