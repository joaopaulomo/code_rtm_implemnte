from examples.seismic import Model

import seismic_wrapper.model_creation as mc
from datasets.dataset_base import DatasetBase

import numpy as np

class LayerCakeDataset(DatasetBase):
    def __init__(self, layer_vp: list, layer_h: list, nx = 101, dx: float = 10, dz: float = 10):
        super().__init__()
        self.nx = nx
        self.nz = int(np.sum(np.array(layer_h)))
        self.dx = dx
        self.dz = dz

        self.layer_vp = layer_vp

        self.vp = np.empty((self.nx, self.nz), dtype=np.float32)
        total_h = 0
        for i in range(len(layer_h)):
            self.vp[:, total_h:total_h + layer_h[i]] = self.layer_vp[i]
            total_h += layer_h[i]
        
        self.rho = mc.gardners_relation(self.vp)
