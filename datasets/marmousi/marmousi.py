import numpy as np

from examples.seismic import Model
from seismic_wrapper.model_creation import gardners_relation
from datasets.dataset_base import DatasetBase

import os
import array
import re


_MARMOUSI_ROOT = '/home/filipe.nolasco/projects/devito-examples/sandbox/datasets/marmousi'


class MarmousiDataset(DatasetBase):
    """
    # MarmousiDataset

    Holds the Marmousi model data.

    Instances if this class should be created through the `load_marmousi_dataset` function.
    """
    def __init__(self, input_params: dict, vp: np.ndarray):
        """
        ### Parameters
        - `input_params`: parameter dictionary loaded from the input.dat file;
        - `vp`: flat velocity profile array;
        """
        self._input_params = input_params
        self._init_params()
        self.vp = np.reshape(vp, (self.nx, self.nz))
        self.rho = gardners_relation(self.vp)


    def _init_params(self):
        for k, v in self._input_params.items():
            isint = re.fullmatch('[0-9]+', v)
            isfloat = re.fullmatch('[0-9]+(\\.[0-9]*)+', v)
            isstr = not (isint or isfloat)

            setattr(self, k, str(v) if isstr else 
                             int(v) if isint else 
                             float(v))


def load_marmousi_dataset() -> MarmousiDataset:
    """
    ### Description

    - Loads Marmousi data and fills it into a `MarmousiDataset` object.

    ### Returns

    - A `MarmousiDataset` object containing the loaded data.
    """
    input_params = load_input(os.path.join(_MARMOUSI_ROOT, 'input.dat'))
    vp = load_velocity_profile(os.path.join(_MARMOUSI_ROOT, input_params.get('velfile')))

    return MarmousiDataset(input_params, vp)


def load_input(path: str) -> dict:
    """
    ### Description

    - Loads the model header (input.dat) and fills it into a `dict`.

    ### Parameters

    - path: the path to the input.dat file.

    ### Returns

    - A `dict` object containing the header data.
    """
    input_params = {}
    with open(path, 'r') as fin:
        lines = fin.readlines()
        for line in lines:
            tokens = line.split('=')
            key = tokens[0].strip()
            val = tokens[-1].strip()

            input_params[key] = val

    return input_params


def load_velocity_profile(path: str) -> np.ndarray:
    """
    ### Description

    - Loads the model velocity profile and fills it into a numpy `ndarray`
        with dtype = float32.

    ### Parameters

    - path: the path to the velocity profile file.

    ### Returns

    - A 1-dimensional `ndarray` object containing the velocity data.
    """
    # with open(path, 'rb') as fin:
    #     raw_bytes = fin.read()
    #     vel_data = array.array('f', raw_bytes)
    #     vel_profile = np.array(vel_data, dtype=np.float32)
    #     vel_profile /= 1000  # Convert to km/s

    #     return vel_profile

    return np.fromfile(path, dtype=np.float32)/1000
