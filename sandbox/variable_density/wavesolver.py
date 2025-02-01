from devito import *
from devito.tools import memoized_meth
from examples.seismic import Model, AcquisitionGeometry

from variable_density.operators import *


class VariableDensityAcousticWaveSolver():
    def __init__(self, model: Model, geometry: AcquisitionGeometry, kernel='OT2', space_order=4, **kwargs):
        self.model = model
        self.model._initialize_bcs(bcs="damp")
        self.geometry: AcquisitionGeometry = geometry

        assert self.model.grid == geometry.grid

        self.space_order = space_order
        self.kernel = kernel

        # Cache compiler options
        self._kwargs = kwargs


    @property
    def dt(self):
        # Time step can be \sqrt{3}=1.73 bigger with 4th order
        if self.kernel == 'OT4':
            return self.model.dtype(1.73 * self.model.critical_dt)
        return self.model.critical_dt


    @memoized_meth
    def op_fwd(self, save=None):
        """Cached operator for forward runs with buffered wavefield"""
        return ForwardOperator(self.model, save=save, geometry=self.geometry,
                               kernel=self.kernel, space_order=self.space_order,
                               **self._kwargs)


    @memoized_meth
    def op_born(self):
        return BornOperator(self.model, save=None, geometry=self.geometry,
                               kernel=self.kernel, space_order=self.space_order,
                               **self._kwargs)

    @memoized_meth
    def op_grad(self, save=True):
        return GradientOperator(self.model, save=save, geometry=self.geometry,
                               kernel=self.kernel, space_order=self.space_order,
                               **self._kwargs)


    def forward(self, src=None, rec=None, u=None, model: Model =None, save=None, **kwargs):
        # Source term is read-only, so re-use the default
        src = src or self.geometry.src
        # Create a new receiver object to store the result
        rec = rec or self.geometry.rec

        # Create the forward wavefield if not provided
        u = u or TimeFunction(name='u', grid=self.model.grid,
                              save=self.geometry.nt if save else None,
                              time_order=2, space_order=self.space_order)

        model = model or self.model
        # Pick vp from model unless explicitly provided
        kwargs.update(model.physical_params(**kwargs))

        summary = self.op_fwd(save).apply(
            src = src, rec = rec, u = u, dt = kwargs.pop('dt', self.dt), **kwargs
        )

        return rec, u, summary


    def jacobian(self, dmin, src=None, rec=None, u=None, du=None, model=None, **kwargs):
        """
        Linearized Born modelling function that creates the necessary
        data objects for running an adjoint modelling operator.

        Parameters
        ----------
        src : SparseTimeFunction or array_like, optional
            Time series data for the injected source term.
        rec : SparseTimeFunction or array_like, optional
            The interpolated receiver data.
        u : TimeFunction, optional
            The forward wavefield.
        du : TimeFunction, optional
            The scattered wavefield.
        model : Model, optional
            Object containing the physical parameters.
        vp : Function or float, optional
            The time-constant velocity.
        """
        src = src or self.geometry.src
        rec = rec or self.geometry.rec

        u = u or TimeFunction(name='u', grid=self.model.grid, time_order=2, space_order = self.space_order)
        du = du or TimeFunction(name='du', grid=self.model.grid, time_order=2, space_order = self.space_order)
        
        model = model or self.model
        kwargs.update(model.physical_params(**kwargs))

        summary = self.op_born().apply(rec=rec, src=src, u=u, du=du, dm=dmin, 
                                       dt=kwargs.pop('dt', self.dt), **kwargs)

        return rec, u, du, summary


    def jacobian_adjoint(self, rec, u, src=None, v=None, gradient=None, model=None,
                         checkpointing=False, **kwargs):
        """
        Gradient modelling function for computing the adjoint of the
        Linearized Born modelling function, ie. the action of the
        Jacobian adjoint on an input data.

        Parameters
        ----------
        rec : SparseTimeFunction
            Receiver data.
        u : TimeFunction
            Full wavefield `u` (created with save=True).
        v : TimeFunction, optional
            Stores the computed wavefield.
        grad : Function, optional
            Stores the gradient field.
        model : Model, optional
            Object containing the physical parameters.
        vp : Function or float, optional
            The time-constant velocity.

        Returns
        -------
        Gradient field and performance summary.
        """
        dt = kwargs.pop('dt', self.dt)
        
        gradient = gradient or Function(name='grad', grid=self.model.grid)

        v = v or TimeFunction(name='v', grid=self.model.grid,
                              time_order=2, space_order=self.space_order)
        
        model = model or self.model

        kwargs.update(model.physical_params(**kwargs))

        summary = self.op_grad().apply(rec=rec, grad=gradient, v=v, u=u, dt=dt, **kwargs)

        return gradient, summary
    

    born = jacobian
    gradient = jacobian_adjoint