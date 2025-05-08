# from devito import *
from examples.seismic import Model, AcquisitionGeometry
import numpy as np

import json
import os

from seismic_wrapper.multishot_geometry_manager import MultishotGeometryManager
import seismic_wrapper.model_creation as mc

class GeometryHeader:
    pass

def create_geometry(model: Model, tn: int, src_depth: float, src_type: str, f_peak: float):
    nreceivers = model.shape[0]
    
    t0 = 0
    tn = tn
    src_coords = np.empty((1, 2))
    src_coords[0, :] = np.array(model.domain_size)/2
    src_coords[0, 1] = src_depth

    rec_coords = np.empty((nreceivers, 2))
    rec_coords[:, 0] = np.linspace(0, model.domain_size[0], num = nreceivers)
    rec_coords[:, 1] = src_depth

    return AcquisitionGeometry(
        model, rec_coords, src_coords, t0, tn, f0=f_peak, src_type=src_type
    )


def create_multishot_geometry(model: Model, duration: int, src_depth: float, src_type: str, f_peak: float, nshots: int):
    geometry = create_geometry(model, duration, src_depth, src_type, f_peak)
    src_positions = np.empty((nshots, 2), dtype=np.float32)
    src_positions[:, 0] = np.linspace(0, model.domain_size[0], num=nshots)
    src_positions[:, 1] = src_depth

    return MultishotGeometryManager(geometry, src_positions, model)


def load_multishot_from_file(filename: str) -> MultishotGeometryManager:
    with open(filename, 'r') as fin:
        geometry_desc = json.load(fin)

        tn = geometry_desc.get('tn')
        src_depth = geometry_desc.get('src_depth')
        src_type = geometry_desc.get('src_type')
        f_peak = geometry_desc.get('f_preak')
        nshots = geometry_desc.get('nshots')

        model = None

        if os.path.isabs(geometry_desc.get('model')):
            model = mc.load_from_file(geometry_desc.get('model'))
        else:
            file_dir = os.path.dirname(filename)
            model_filename = os.path.join(file_dir, geometry_desc.get('model'))
            model = mc.load_from_file(model_filename)


        ms_geometry = create_multishot_geometry(
            model, tn, src_depth, src_type, f_peak, nshots
        )

        return ms_geometry
