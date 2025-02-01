import numpy as np

from examples.seismic import *

class LSRTMTutorialDataset():
    def __init__(self, layers_vp = np.array([1.0, 1, 1, 1]), layers_rho = np.array([1.0, 1, 1, 1])):
        self.layers_vp = layers_vp
        self.layers_rho = layers_rho

    
    @property
    def model(self):
        return self.get_model()
    
    def get_model(self, use_density = True):
        shape = (101, 101)
        spacing = (10., 10.)
        origin = (0., 0.)
        bcs = 'damp'
        nbl = 50
        space_order = 8


        vp = np.empty(shape, dtype=np.float32)
        vp[:] = self.layers_vp[0]
        vp[:, 30:65] = self.layers_vp[1]
        vp[:, 65:101] = self.layers_vp[2]
        vp[40:60, 35:55] = self.layers_vp[3]
        
        rho = np.empty(shape, dtype=np.float32)
        rho[:] = self.layers_rho[0]
        rho[:, 30:65] = self.layers_rho[1]
        rho[:, 65:101] = self.layers_rho[2]
        rho[40:60, 35:55] = self.layers_rho[3]

        kwargs = {
            'vp': vp,
            'origin': origin, 
            'shape': shape, 
            'spacing': spacing, 
            'space_order': space_order,
            'nbl': nbl,
            'bcs': bcs
        }

        if use_density:
            kwargs['b'] = 1/rho

        return Model(**kwargs)
