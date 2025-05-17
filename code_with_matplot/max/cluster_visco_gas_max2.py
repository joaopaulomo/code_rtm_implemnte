from devito import *
from devito import Function, TimeFunction, Operator, Eq, solve
from examples.seismic import (
    Model, ModelViscoacoustic, TimeAxis, RickerSource, Receiver, AcquisitionGeometry, PointSource
)
from examples.seismic.plot_model_mod import (plot_image2,plot_shotrecord)
from examples.seismic.viscoacoustic import ViscoacousticWaveSolver
from examples.seismic.viscoacoustic.operators import sls_2nd_order, kv_2nd_order, maxwell_2nd_order
from utils import acoustic_2nd_order, modelling
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import gc
from IPython.display import clear_output

nx, nz = 400, 200
shape = (nx, nz)
dtype = np.float32
spacing = (24.8125, 20.1)
origin = (0, 0)
space_order = 16
nbl = 200
pad = 40

v = np.empty(shape, dtype=dtype)
rho = np.empty(shape, dtype=dtype)
qp = np.empty(shape, dtype=dtype)


v = np.empty(shape, dtype=dtype)
path = "/home/joao.santana/cluster_gas/code_rtm_implemnte/code_with_matplot/max/V_400x200_chamine.bin"
a = open(path)
v = np.fromfile(a, dtype=dtype).reshape([nx, nz])
v = v / 1000
a.close()

path_2 = "/home/joao.santana/cluster_gas/code_rtm_implemnte/code_with_matplot/max/Q_400x200_chamine.bin"
c = open(path)
qp = np.fromfile(c, dtype=dtype, count=nx*nz).reshape([nx, nz])
c.close()

rho[:] = 0.31*(v[:]*1000.)**0.25  


model = ModelViscoacoustic(vp=v, origin=origin, qp=qp, b=1/rho, shape=shape, spacing=spacing,
                          space_order=space_order, nbl=nbl)


model = ModelViscoacoustic(
    space_order=space_order, vp=v, qp=qp, b=1 / rho,
    origin=origin, shape=shape, spacing=spacing, nbl=nbl
)

model0 = ModelViscoacoustic(
    space_order=space_order, vp=v, qp=qp, b=1 / rho,
    origin=origin, shape=shape, spacing=spacing, nbl=nbl
)

filter_sigma = (6, 6)
gaussian_smooth(model0.vp, sigma=filter_sigma)
gaussian_smooth(model0.qp, sigma=filter_sigma)
gaussian_smooth(model0.b, sigma=filter_sigma)

t0, tn = 0., 4000.
dt = model.critical_dt*0.7
dt=dt.astype(np.float32)
time_range = TimeAxis(start=t0, stop=tn, step=dt)

f0 = 0.020
src = RickerSource(name='src', grid=model.grid, f0=f0, npoint=1, time_range=time_range)
src.coordinates.data[0, :] = np.array(model.domain_size) * .5
src.coordinates.data[0, -1] = 20.1  

rec_coordinates = np.empty((model.shape[0], 2))
rec_coordinates[:, 0] = np.linspace(0, model.domain_size[0], num=model.shape[0])
rec_coordinates[:, 1] = 20.1

geometry = AcquisitionGeometry(model, rec_coordinates, src.coordinates.data, t0, tn, f0=f0, src_type='Ricker')
geometry.resample(dt)

solver = ViscoacousticWaveSolver(model, geometry, space_order=space_order, kernel='kv')

d0 = solver.forward(vp=model.vp, qp=model.qp, b=model.b, dt=dt)[0]
d1 = solver.forward(vp=model0.vp, qp=model0.qp, b=model.b, dt=dt)[0]

sismo_real = d0.data
sismo_smooth = d1.data
sismo_residual = d1.data - d0.data

sismo_images = [sismo_real, sismo_smooth, sismo_residual]
sismo_names = ["sismo_real.png", "sismo_smooth.png", "sismo_residual.png"]

for imge, nam in zip(sismo_images, sismo_names):
    plot_shotrecord(imge, model, t0, tn, clip_percent=99, clip_low=1)
    plt.savefig(nam, dpi=300, bbox_inches='tight')
    plt.close()




nshots = 300
source_locations = np.empty((nshots, 2), dtype=np.float32)
source_locations[:, 0] = np.linspace(10., model.domain_size[0], num=nshots)
source_locations[:, 1] = 5.

d0 = np.empty((nshots, int(time_range.num), shape[0]), dtype=np.float32)

for i in range(nshots):
    print('Shot source %d out of %d' % (i + 1, nshots))
    geometry.src_positions[0, :] = source_locations[i, :]

    d = solver.forward(vp=model.vp, qp=model.qp, b=model.b, dt=dt)[0]
    d0[i] = d.data[:]

    clear_output(wait=True)
    gc.collect()



kernels = {
    'sls2': sls_2nd_order,
    'kv2': kv_2nd_order,
    'max2': maxwell_2nd_order,
    'acoustic2': acoustic_2nd_order
}

def ImagingOperator(model, model0, image, dt, kernel="max2"):
    v = TimeFunction(name='v', grid=model.grid, time_order=2, space_order=space_order, staggered=NODE)
    u = TimeFunction(name='u', grid=model.grid, time_order=2, space_order=space_order, save=time_range.num, staggered=NODE)
    eq_kernel = kernels[kernel]
    eqn = eq_kernel(model, geometry, v, forward=False)

    d0 = PointSource(name='d0', grid=model.grid, time_range=geometry.time_axis, coordinates=geometry.rec_positions)
    res_term = d0.inject(field=v.backward, expr=d0 * dt ** 2 / model.m)

    image_update = Eq(image, image - u * v)
    return Operator([eqn] + res_term + [image_update], subs=model.spacing_map)

source_locations = np.empty((nshots, 2), dtype=np.float32)
source_locations[:, 0] = np.linspace(0., model.domain_size[0], num=nshots)
source_locations[:, 1] = 1

def rtm(kernel, dt):
    image = Function(name='image', grid=model.grid, space_order=space_order)
    op_imaging = ImagingOperator(model, model0, image, dt, kernel=kernel)

    for i in range(nshots):
        print('Imaging %s for source %d out of %d' % (kernel, i + 1, nshots))
        _, u0 = modelling(model0, time_range, f0, dt, sl=source_locations[i, :], kernel=kernel, time_order=2)
        v = TimeFunction(name='v', grid=model.grid, time_order=2, space_order=space_order)

        if kernel != "acoustic2":
            op_imaging(u=u0, v=v, vp=model0.vp, b=model0.b, qp=model0.qp, dt=dt, d0=d0[i])
        else:
            op_imaging(u=u0, v=v, vp=model0.vp, b=model0.b, dt=dt, d0=d0[i])

        clear_output(wait=True)
    return image

kernel = 'acoustic2'
data1 = rtm(kernel, dt)

kernel = 'max2'
data2 = rtm(kernel, dt)

kernel = 'kv2'
data3 = rtm(kernel, dt)

kernel = 'sls2'
data4 = rtm(kernel, dt)

sliced_image_1 = data1.data[model.nbl:-model.nbl, model.nbl:-model.nbl]
sliced_image_2 = data2.data[model.nbl:-model.nbl, model.nbl:-model.nbl]
sliced_image_3 = data3.data[model.nbl:-model.nbl, model.nbl:-model.nbl]
sliced_image_4 = data4.data[model.nbl:-model.nbl, model.nbl:-model.nbl]


imagens = [sliced_image_1, sliced_image_2, sliced_image_3, sliced_image_4]
nomes = ["acoustic_normal.png", "max2_normal.png", "kv2_normal.png", "sls_normal.png"]

for img, nome in zip(imagens, nomes):
    plot_image2(np.diff(img, axis=1), model, clip_percent=98, clip_low=2)
    plt.savefig(nome, dpi=300, bbox_inches='tight')
    plt.close()

laplace_result_1 = Function(name='lap', grid=model.grid, space_order=space_order)
laplace_result_2 = Function(name='lap', grid=model.grid, space_order=space_order)
laplace_result_3 = Function(name='lap', grid=model.grid, space_order=space_order)
laplace_result_4 = Function(name='lap', grid=model.grid, space_order=space_order)

stencil = Eq(laplace_result_1, data1.laplace)
op = Operator([stencil])
op.apply()

stencil = Eq(laplace_result_2, data2.laplace)
op = Operator([stencil])
op.apply()

stencil = Eq(laplace_result_3, data3.laplace)
op = Operator([stencil])
op.apply()

stencil = Eq(laplace_result_4, data4.laplace)
op = Operator([stencil])
op.apply()

sliced_laplace_1 = laplace_result_1.data[model.nbl:-model.nbl, model.nbl:-model.nbl]
sliced_laplace_2 = laplace_result_2.data[model.nbl:-model.nbl, model.nbl:-model.nbl]
sliced_laplace_3 = laplace_result_3.data[model.nbl:-model.nbl, model.nbl:-model.nbl]
sliced_laplace_4 = laplace_result_4.data[model.nbl:-model.nbl, model.nbl:-model.nbl]

imagens = [sliced_laplace_1, sliced_laplace_2, sliced_laplace_3, sliced_laplace_4]
nomes = ["laplace_acoust.png", "laplace_max.png", "laplace_kv.png", "laplace_sls.png"]

for img, nome in zip(imagens, nomes):
    plot_image2(np.diff(img, axis=1), model, clip_percent=98, clip_low=2)
    plt.savefig(nome, dpi=300, bbox_inches='tight')
    plt.close()

