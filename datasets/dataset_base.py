from examples.seismic import Model

import numpy as np
from abc import ABC, abstractmethod

class DatasetBase(ABC):
    # def __init__(self):
    #     self.nx = 0
    #     self.nz = 0
    #     self.vp = np.empty(0)
    #     self.rho = np.empty(0)

    @property
    def model(self):
        return self.get_model()


    def get_model(self, use_density = True):
        origin = (0., 0.)
        spacing = (self.dx, self.dz)
        shape = (self.nx, self.nz)
        space_order = 8
        nbl = 40
        vp = self.vp
        bcs = 'damp'
        b = 1/self.rho

        if use_density:
            return Model(
                origin      = origin,
                spacing     = spacing,
                shape       = shape,
                space_order = space_order,
                nbl         = nbl,
                vp          = vp,
                b           = b,
                bcs         = bcs
            )
        else:
            return Model(
                origin      = origin,
                spacing     = spacing,
                shape       = shape,
                space_order = space_order,
                nbl         = nbl,
                vp          = vp,
                bcs         = bcs
            )
