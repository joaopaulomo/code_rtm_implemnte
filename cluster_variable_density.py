from devito import *
from examples.seismic import (
    Model, AcquisitionGeometry, PointSource, RickerSource, TimeAxis
)
from examples.seismic.acoustic import AcousticWaveSolver
from variable_density import VariableDensityAcousticWaveSolver
from extra_plotting import *
from datasets.marmousi import load_marmousi_dataset
from examples.seismic.plot_model_mod import (plot_image2,plot_shotrecord)
from IPython.display import clear_output
import matplotlib.pyplot as plt


import numpy as np
import gc

nx, nz = 738, 240
nbl = 150 
nshots = 738
space_order = 8
dtype = np.float32
shape = (nx, nz)
spacing = (12.5, 12.5)
origin = (0., 0.)

v = np.fromfile("/home/joao.santana/Cluster_code_variable_density/code_rtm_implemnte/marmousi-resample-738x240.bin", dtype=dtype).reshape([nx, nz]) / 1000
rho = 0.31 * (v * 1000)**0.25

model = Model(
    vp=v, origin=origin, shape=shape, spacing=spacing, space_order=space_order,
    nbl=nbl, b=1 / rho, bcs="damp"
)

model0 = Model(
    vp=v.copy(), origin=origin, shape=shape, spacing=spacing, space_order=space_order,
    nbl=nbl, b=1 / rho
)
gaussian_smooth(model0.vp, sigma=(6, 6))

t0, tn, dt = 0., 5000., 0.0008
time_range = TimeAxis(start=t0, stop=tn, step=dt)

f0 = 0.030
src = RickerSource(name='src', grid=model.grid, f0=f0, npoint=1, time_range=time_range)
src.coordinates.data[0, :] = np.array(model.domain_size) * 0.5
src.coordinates.data[0, -1] = 1.

rec_coordinates = np.empty((model.shape[0], 2))
rec_coordinates[:, 0] = np.linspace(0, model.domain_size[0], num=model.shape[0])
rec_coordinates[:, 1] = 1.

geometry = AcquisitionGeometry(
    model, rec_coordinates, src.coordinates.data, t0, tn, f0=f0, src_type='Ricker'
)

solver = VariableDensityAcousticWaveSolver(model, geometry, space_order=space_order)
true_d , _, _ = solver.forward(vp=model.vp)
smooth_d, _, _ = solver.forward(vp=model0.vp)

sismo_real = true_d.data
sismo_smooth = smooth_d.data
sismo_residual = smooth_d.data - true_d .data

sismo_images = [sismo_real, sismo_smooth, sismo_residual]
sismo_names = ["sismo_real.png", "sismo_smooth.png", "sismo_residual.png"]

for imge, nam in zip(sismo_images, sismo_names):
    plot_shotrecord(imge, model, t0, tn, clip_percent=99, clip_low=1)
    plt.savefig(nam, dpi=300, bbox_inches='tight')
    plt.close()

def ImagingOperator(model: Model, image: Function, geometry: AcquisitionGeometry):
    v = TimeFunction(name='v', grid=model.grid, time_order=2, space_order=model.space_order)
    u = TimeFunction(name='u', grid=model.grid, time_order=2, space_order=model.space_order, save=geometry.nt)

    kappa = 1 / (model.m * model.b)
    eqn = v.dt2 - kappa * div(model.b * grad(v, shift=0.5), shift=-0.5) + model.damp * v.dt.T
    stencil = Eq(v.backward, solve(eqn, v.backward))

    dt = model.critical_dt

    residual = PointSource(name='residual', grid=model.grid, time_range=geometry.time_axis,
                           coordinates=geometry.rec_positions)
    res_term = residual.inject(field=v.backward, expr=residual * dt**2 / model.m)

    image_update = Eq(image, image + u * v)

    return Operator([stencil] + res_term + [image_update], subs=model.spacing_map)

image = Function(name='image', grid=model.grid)
op = ImagingOperator(model, image, geometry)

source_locations = np.empty((nshots, 2), dtype=np.float32)
source_locations[:, 0] = np.linspace(0., model.domain_size[0], num=nshots)
source_locations[:, 1] = 1.

for i in range(nshots):
    if i % 5 == 0:
        print(f'Imaging source {i + 1} out of {nshots}')

    geometry.src_positions[0, :] = source_locations[i, :]

    true_d, _, _ = solver.forward(vp=model.vp)
    smooth_d, u0, _ = solver.forward(vp=model0.vp, save=True)

    v = TimeFunction(name='v', grid=model.grid, time_order=2, space_order=model.space_order)
    residual = smooth_d.data - true_d.data

    op(u=u0, v=v, vp=model0.vp, b=model0.b, dt=model0.critical_dt, residual=residual)

    del true_d, smooth_d, u0, v, residual
    gc.collect()

sliced_image = image.data[model.nbl:-model.nbl, model.nbl:-model.nbl]
plot_image2(np.diff(sliced_image, axis=1),model,clip_percent=98,clip_low=2)
plt.savefig('normal_variable.png', dpi=300, bbox_inches='tight')

laplace_result = Function(name='lap',grid=model.grid, space_order=space_order)
stencil = Eq(laplace_result, div(grad(image)))
op = Operator([stencil])
op.apply()

sliced_laplace = laplace_result.data[model.nbl:-model.nbl, model.nbl:-model.nbl]
plot_image2(sliced_laplace,model,clip_percent=98,clip_low=2)
plt.savefig('lapla_variable.png', dpi=300, bbox_inches='tight')

