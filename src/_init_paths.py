import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add lib to PYTHONPATH
lib_path1 = osp.join(this_dir, 'lib')
lib_path2 = osp.join(this_dir, 'tools')
add_path(lib_path1)
add_path(lib_path2)
