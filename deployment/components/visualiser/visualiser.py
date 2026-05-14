import numpy as np
import cv2 as cv

from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import InPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol

class ForceVisualiser(AbstractProcess):
    """A process that visualises the forces applied to the robot in real-time.
    """
    def __init__(self, **kwargs):
        self.a_in = InPort(shape=(3,))  # Assuming 3D force vector


        super().__init__(**kwargs)


@implements(proc=ForceVisualiser, protocol=LoihiProtocol)
@requires(CPU)
class PySparseForceVisualiserModel(PyLoihiProcessModel):
    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float64)

    def __init__(self, proc_params):    
        super().__init__(proc_params)
        self.window_name = "Force Visualiser"
        cv.namedWindow(self.window_name, cv.WINDOW_NORMAL)

    def plot_force(self, force_values) -> None:
        # Plot a simple line plot indicating the force values over time
        # TODO: Implement
        pass
        
    def run_spk(self) -> None:
        data = self.a_in.recv()

        if data is not None:
            self.plot_force(data)
