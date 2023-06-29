import os


def resource_dir_path():
    return os.path.abspath(os.path.dirname(__file__))


def img_dir_path():
    return os.path.join(os.path.dirname(resource_dir_path()), 'img')
