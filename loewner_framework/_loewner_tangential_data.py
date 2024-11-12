import numpy as _np

from . import generalized_sylvester as _gs
from . import linear_daes as _ld

from ._right_and_left_tangential_data import RightTangentialData, LeftTangentialData



class LoewnerTangentialData:
    def __init__(self, rtd: RightTangentialData, ltd: LeftTangentialData):
        """An object used to derive and encapsulate the Loewner matrix and shifted Loewner matrix associated to the right and left tangential interpolation data used for initialization.

        Args:
            rtd: a RightTangentialData object representing the right tangential interpolation data. Should have the isComplete property True.
            ltd: a LeftTangentialData object representing the left tangential interpolation data. Should have the isComplete property True.

        Returns:
            A LoewnerTangentialData object containing the right and left tangential data, along with the Loewner matrices derived from the data, and that can be provided to the InterpolantFactory constructor for the purposes of building interpolants.

        Raises:
            ValueError: The argument rtd is not an instance of RightTangentialData or the argument ltd is not an instance of LeftTangentialData.
            ValueError: The provided RightTangentialData or LeftTangentialData is incomplete.
            ValueError: The dimensions of the provided RightTangentialData and LeftTangentialData are inconsistent.
        """
        if not isinstance(rtd, RightTangentialData) or not isinstance(ltd, LeftTangentialData):
            raise ValueError("The argument rtd is not an instance of RightTangentialData or the argument ltd is not an instance of LeftTangentialData.")
        if not rtd.isComplete or not ltd.isComplete:
            raise ValueError("The provided RightTangentialData or LeftTangentialData is incomplete.")
        if not rtd.m == ltd.m or not rtd.p == ltd.p:
            raise ValueError("The dimensions of the provided RightTangentialData and LeftTangentialData are inconsistent.")
        
        self._rtd = rtd
        self._ltd = ltd
        self._isComplete = False
        self._isFromSystem = rtd.isFromSystem and ltd.isFromSystem

        try:
            self._Loewner = _gs.solve(_np.eye(ltd.nu,ltd.nu), rtd.Lambda, -ltd.M, _np.eye(rtd.rho,rtd.rho), ltd.L @ rtd.W - ltd.V @ rtd.R)
            self._shiftedLoewner = _gs.solve(_np.eye(ltd.nu,ltd.nu), rtd.Lambda, -ltd.M, _np.eye(rtd.rho,rtd.rho), ltd.L @ rtd.W @ rtd.Lambda - ltd.M @ ltd.V @ rtd.R)
        except:
            raise ValueError("Failed to solve generalized Sylvester equations defining the Loewner matrix and the shifted Loewner matrix from provided tangential data.")
        self._isComplete = True

    
        
    @property
    def rtd(self) -> RightTangentialData | None:
        return self._rtd
    
    @property
    def rho(self) -> int:
        if isinstance(self.rtd, RightTangentialData):
            return self.rtd.rho
        else:
            return -1
    
    @property
    def m(self) -> int:
        if isinstance(self.rtd, RightTangentialData):
            return self.rtd.m
        else:
            return -1
    
    @property
    def Lambda(self) -> _np.ndarray | None:
        if isinstance(self.rtd, RightTangentialData):
            return self.rtd.Lambda
        else:
            return None
    
    @property
    def R(self) -> _np.ndarray | None:
        if isinstance(self.rtd, RightTangentialData):
            return self.rtd.R
        else:
            return None
    
    @property
    def W(self) -> _np.ndarray | None:
        if isinstance(self.rtd, RightTangentialData):
            return self.rtd.W
        else:
            return None
    
    @property
    def ltd(self) -> LeftTangentialData | None:
        return self._ltd
    
    @property
    def nu(self) -> int:
        if isinstance(self.ltd, LeftTangentialData):
            return self.ltd.nu
        else:
            return -1
    
    @property
    def p(self) -> int:
        if isinstance(self.ltd, LeftTangentialData):
            return self.ltd.p
        else:
            return -1
    
    @property
    def M(self) -> _np.ndarray | None:
        if isinstance(self.ltd, LeftTangentialData):
            return self.ltd.M
        else:
            return None
    
    @property
    def L(self) -> _np.ndarray | None:
        if isinstance(self.ltd, LeftTangentialData):
            return self.ltd.L
        else:
            return None
    
    @property
    def V(self) -> _np.ndarray | None:
        if isinstance(self.ltd, LeftTangentialData):
            return self.ltd.V
        else:
            return None
    
    @property
    def isComplete(self) -> bool:
        return self._isComplete
    
    @property
    def isFromSystem(self) -> bool:
        return self._isFromSystem
    
    @property
    def Loewner(self) -> _np.ndarray | None:
        if self.isComplete:
            return self._Loewner
        else:
            return None
        
    @property
    def shiftedLoewner(self) -> _np.ndarray | None:
        if self.isComplete:
            return self._shiftedLoewner
        else:
            return None
        
    def __repr__(self) -> str:
        return f"LoewnerTangentialData\nrtd.isComplete = {self.rtd.isComplete}, ltd.isComplete = {self.ltd.isComplete}, isComplete = {self.isComplete}, isFromSystem = {self.isFromSystem},\n\nLoewner =\n{self.Loewner},\n\nshiftedLoewner =\n{self.shiftedLoewner}\n"
        
    def __bool__(self) -> bool:
        return self.isComplete