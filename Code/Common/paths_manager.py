import os

_CWD = os.getcwd()
_CODE_DIR = os.path.abspath(os.path.join(_CWD, os.pardir))
_ROOT_DIR = os.path.abspath(os.path.join(_CODE_DIR, os.pardir))
_PATH_TFRECORDS = os.path.join(_CODE_DIR, 'TFRecords')
_PATH_TFRECORDS_TRAIN = os.path.join(_PATH_TFRECORDS, 'Training')
_PATH_TFRECORDS_TEST = os.path.join(_PATH_TFRECORDS, 'Testing')
_DATA_DIR = os.path.join(_ROOT_DIR, 'Data')
_KITTY_DIR = os.path.join(_DATA_DIR, 'Kitti')
_PATH_IMAGES = os.path.join(_KITTY_DIR, 'Images')
_PATH_LABELS = os.path.join(_KITTY_DIR, 'Labels')


class Path:
    CWD = _CWD,
    CODE_DIR = _CODE_DIR,
    ROOT_DIR = _ROOT_DIR,
    PATH_TFRECORDS = _PATH_TFRECORDS,
    PATH_TFRECORDS_TRAIN = _PATH_TFRECORDS_TRAIN,
    PATH_TFRECORDS_TEST = _PATH_TFRECORDS_TEST,
    DATA_DIR = _DATA_DIR,
    KITTY_DIR = _KITTY_DIR,
    PATH_IMAGES = _PATH_IMAGES,
    PATH_LABELS = _PATH_LABELS
