import os


def list_dir(dirpath: str) -> (list, list):
    files, dirs = [], []
    for dpath, dnames, fnames in os.walk(dirpath, ):
        files.extend(fnames)
        dirs.extend(dnames)
        break
    return sorted(dirs), sorted(files)
