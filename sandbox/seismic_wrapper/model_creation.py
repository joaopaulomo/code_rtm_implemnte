from devito import *
from examples.seismic import Model

import numpy as np

import json
import os


def clone_model(model: Model):
    kwargs = extract_model_args(model)
    return Model(**kwargs)


def smooth_model(model: Model, sigma, smooth_b: bool = True):
    model0 = clone_model(model)
    gaussian_smooth(model0.vp.data, sigma)
    if getattr(model, 'b', None) != None and smooth_b:
        gaussian_smooth(model0.b.data, sigma)

    return model0


def extract_model_args(model: Model) -> dict:
    """
    Extract the arguments passed to the `Model` constructor from the corresponding object.
    """

    kwargs = {
        'vp': model.vp.data[model.nbl:-model.nbl, model.nbl:-model.nbl],
        'origin': model.origin, 
        'shape': model.shape, 
        'spacing': model.spacing, 
        'space_order': model.space_order,
        'grid': model.grid,
        'nbl': model.nbl,
        'bcs': 'damp'
    }

    if getattr(model, 'b', None) != None:
        kwargs['b'] = model.b.data[model.nbl:-model.nbl, model.nbl:-model.nbl]

    return kwargs
    

def gardners_relation(vp: np.ndarray) -> np.ndarray:
    return 0.31*(vp*1000)**0.25


def load_from_file(header_filename: str) -> Model:
    with open(header_filename, 'r') as fin:
        header = json.load(fin)
        
        nx = header.get('nx')
        nz = header.get('nz')
        
        dx = header.get('dx')
        dz = header.get('dz')

        space_order = header.get('space_order')
        nbl = header.get('nbl')

        vp = None

        if os.path.isabs(header.get('vp')):
            vp = np.fromfile(header.get('vp'), dtype=np.float32)
        else:
            header_dir = os.path.dirname(header_filename)
            vp_filename = os.path.join(header_dir, header.get('vp'))

            vp = np.fromfile(vp_filename, dtype=np.float32).reshape(nx, nz)
        
        if header.get('vp_unit', 'km/s') == 'm/s':
            vp = vp/1000


        kwargs = {
            'vp': vp,
            'origin': (0., 0.), 
            'shape': (nx, nz), 
            'spacing': (dx, dz), 
            'space_order': space_order,
            'nbl': nbl,
            'bcs': 'damp'
        }

        return Model(**kwargs)