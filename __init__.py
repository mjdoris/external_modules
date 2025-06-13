from configparser import ConfigParser
from pathlib import Path
from soldet.utilities import config, change_exp, change_path, soldet_to_h5
from soldet.io import download_ds
from soldet.SolitonDetector import SolitonDetector
from soldet.object_model import ObjectDetector
from soldet.classifier_nn import CNN_MLST2021_modern as Classifier

soldet_path = Path(__file__).parent
home_dir = Path.home()

if not soldet_path.joinpath('CONFIG.ini').is_file():
    configfile = ConfigParser()
    configfile['PATHS'] = {'data_path': str(home_dir.joinpath('soldet')), 'def_exp_name': 'soldet_ds'}
    configfile.write(open(soldet_path.joinpath('CONFIG.ini'), 'w'))