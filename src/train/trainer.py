import sys
import logging
import pandas as pd

from src.train.model import Model
from src.utils.load_dataset import LoadSSJ500k, LoadBSNLP, LoadCombined
from src.utils.utils import list_dir


logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s'
)
logger = logging.getLogger('TrainLOOStrategy')



def main():
    pass

if __name__ == '__main__':
    main()
