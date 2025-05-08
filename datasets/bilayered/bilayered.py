from examples.seismic import Model

import numpy as np


class BilayeredDataset():
    """
    # BilayeredDataset

    Synthetic dataset with two equal sized, inclined layers with constant velocity and density.
    """

    def __init__(self, vp1: float = 1, vp2: float = 1, rho1: float = 1, rho2: float = 1, theta: float = 0):
        """
        ### Parameters
        - `vp1`: top layer's compressional wave velocity in km/s;
        - `vp2`: bottom layer's compressional wave velocity in km/s;
        - `rho1`: top layer's density in g/cm³;
        - `rho2`: bottom layer's density in g/cm³;
        - `theta`: angle in radians between the interface and the horizontal surface;
        """

        self.vp1 = vp1
        self.vp2 = vp2
        self.rho1 = rho1
        self.rho2 = rho2
        self.theta = theta


    @property
    def model(self):
        """
        ### Returns

        - A `SeismicModel` instance with the synthesized data.
        """
        return self.get_model()


    def get_model(self, use_density = True):
        origin = (0., 0.)
        spacing = (10., 10.)
        shape = (101, 101)
        space_order = 4
        nbl = 10
        bcs = 'damp'

        intf = lambda x: np.tan(self.theta) * (x - shape[0]/2) + shape[1]/2

        vp = np.empty(shape, dtype=np.float32)

        rho = np.empty(shape, dtype=np.float32)
        
        for i in range(shape[0]):
            for j in range(shape[1]):
                vp[i, j]  = self.vp1  if j < intf(i) else self.vp2
                rho[i, j] = self.rho1 if j < intf(i) else self.rho2


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
        