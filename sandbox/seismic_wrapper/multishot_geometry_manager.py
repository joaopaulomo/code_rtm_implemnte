from examples.seismic import AcquisitionGeometry, Model

import numpy as np

class MultishotGeometryManager:
    """
    Manages an AquisitionGeometry for multishots.
    """
    def __init__(self, geometry: AcquisitionGeometry, src_positions: np.ndarray, model: Model):
        self.geometry = geometry
        self.src_positions = src_positions
        self._model = model

        self._current_index = self.set_source(0)


    def _check_in_bounds(self, i: int) -> bool:
        return i >= 0 and i < self.nshots


    @property
    def nshots(self) -> int:
        return self.src_positions.shape[0]


    @property
    def model(self) -> Model:
        return self._model


    def set_source(self, i: int):
        """
        Sets the current geometry source to the `i`-th source.

        Parameters
        ----------

        - `i`: Source index;

        Returns
        -------

        `i` if the source was successfully set. `None` if the provided index was invalid.
        """
        if not self._check_in_bounds(i):
            return None
        
        self.geometry.src_positions[0, :] = self.src_positions[i, :]

        return i


    def next_src(self):
        self._current_index = self.set_source((self._current_index + 1) % self.nshots)

    
    def prev_src(self):
        self._current_index = self.set_source((self._current_index - 1) % self.nshots)