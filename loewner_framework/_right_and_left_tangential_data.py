import numpy as _np

from . import generalized_sylvester as _gs
from . import linear_daes as _ld



class RightTangentialData:
    def __init__(self, Lambda: _np.ndarray, R: _np.ndarray, W: _np.ndarray | _ld.LinearDAE | None = None):
        """An object used to encapsulate the right tangential interpolation data used for initialization of the LoewnerTangentialData object. The object can also derive the data from a provided system.

        The dimensions must be consistent. Lambda must be square, R must have number of columns equal to the number of rows/columns in Lambda, and if W is a matrix it must have number of columns equal to the number of rows/columns in Lambda. If W is a system in the form of a LinearDAE object, then R must have number of rows equal to the system's number of inputs.
        
        Args:
            Lambda: a square (rho times rho) 2darray with eigenvalues representing the right tangential data set interpolation frequencies.
            R: a 2darray (m times rho) representing the right tangential data set interpolation directions.
            W: either a 2darray (p times rho) representing the values of the function evaluated at the corresponding interpolation frequencies and along the corresponding interpolation directions, or a linear_daes.LinearDAE whose transfer function will be used to generate the 2darray.

        Returns:
            A RightTangentialData object containing the right tangential data matrices.

        Raises:
            ValueError: Inconsistent shape.
            ValueError: Can't calculate W from provided system.
        """
        # make sure that everything is 2d array for now
        if not Lambda.ndim == 2 or not R.ndim == 2:
            raise ValueError("Inconsistent shape.")

        self._rho, rho_2 = Lambda.shape
        self._m, rho_3 = R.shape

        # check dimensions are consistent
        if not self._rho == rho_2:
            raise ValueError("Inconsistent shape.")
        if not self._rho == rho_3:
            raise ValueError("Inconsistent shape.")

        self._Lambda = Lambda
        self._R = R

        # whether these flags are true depends on if and how self._W is determined below
        self._isComplete = False
        self._isFromSystem = False
        
        if isinstance(W, _np.ndarray):
            # data given as a matrix
            # check that sizes are consistent
            if not W.ndim == 2:
                raise ValueError("Inconsistent shape.")
            if not W.shape[1] == self._rho:
                raise ValueError("Inconsistent shape.")
            # all is good, set flags
            self._p = W.shape[0]
            self._W = W
            self._isComplete = True
        elif isinstance(W, _ld.LinearDAE):
            #calculate self._W from system held by W
            if W.isRegular:
                # provided system is regular
                # try to get the tangential generalized controllability matrix from its generalized sylvester equation
                try:
                    self._X = _gs.solve(W.E, self._Lambda, -W.A, _np.eye(self._rho,self._rho), W.B @ self._R)
                except:
                    self._p = None
                    self._isComplete = False
                    self._W = None
                    raise ValueError("Can't calculate W from provided system.")
                if not self._m == W.m:
                    # this shouldn't happen
                    self._p = None
                    self._isComplete = False
                    self._W = None
                    raise ValueError("Can't calculate W from provided system.")
                # all is good, set flags indicating that this object is complete, and X exists as a variable
                self._W = W.C @ self._X + W.D @ self._R
                self._p = W.p
                self._isComplete = True
                self._isFromSystem = True
            else:
                # can't get data from irregular system
                self._p = None
                self._isComplete = False
                self._W = None
                raise ValueError("Can't calculate W from provided system.")
        else:
            # W empty or bad
            self._isComplete = False
            self._p = None
            self._W = None

    @property
    def rho(self) -> int:
        return self._rho
        
    @property
    def m(self) -> int:
        return self._m
        
    @property
    def p(self) -> int | None:
        return self._p
        
    # indicates if the object contains all of self.Lambda, self.R, and self.W with consistent sizes for the Loewner framework
    @property
    def isComplete(self) -> bool:
        return self._isComplete
    
    # indicates whether or not self.X exists or is None
    @property
    def isFromSystem(self) -> bool:
        return self._isFromSystem

    @property
    def Lambda(self) -> _np.ndarray:
        return self._Lambda
        
    @property
    def R(self) -> _np.ndarray:
        return self._R
        
    @property
    def W(self) -> _np.ndarray | None:
        return self._W
    
    @property
    def X(self) -> _np.ndarray | None:
        if self._isComplete and self._isFromSystem:
            return self._X
        else:
            return None
        
    def __repr__(self) -> str:
        return f"RightTangentialData\nisComplete = {self.isComplete}, isFromSystem = {self.isFromSystem},\nrho = {self.rho}, m = {self.m}, p = {self.p},\n\nLambda =\n{self.Lambda},\n\nR =\n{self.R},\n\nW =\n{self.W},\n\nX =\n{self.X}\n"
        
    def __bool__(self) -> bool:
        return self._isComplete









class LeftTangentialData:
    def __init__(self, M: _np.ndarray, L: _np.ndarray, V: _np.ndarray | _ld.LinearDAE | None = None):
        """An object used to encapsulate the left tangential interpolation data used for initialization of the LoewnerTangentialData object. The object can also derive the data from a provided system.

        The dimensions must be consistent. M must be square, L must have number of rows equal to the number of rows/columns in M, and if V is a matrix it must have number of rows equal to the number of rows/columns in M. If V is a system in the form of a LinearDAE object, then L must have number of columns equal to the system's number of outputs.
        
        Args:
            M: a square (nu times nu) 2darray with eigenvalues representing the left tangential data set interpolation frequencies.
            L: a 2darray (nu times p) representing the left tangential data set interpolation directions.
            V: either a 2darray (nu times m) representing the values of the function evaluated at the corresponding interpolation frequencies and along the corresponding interpolation directions, or a linear_daes.LinearDAE whose transfer function will be used to generate the 2darray.

        Returns:
            A LeftTangentialData object containing the left tangential data matrices.

        Raises:
            ValueError: Inconsistent shape.
            ValueError: Can't calculate V from provided system.
        """
        # make sure that everything is 2d array for now
        if not M.ndim == 2 or not L.ndim == 2:
            raise ValueError("Inconsistent shape.")

        self._nu, nu_2 = M.shape
        nu_3, self._p = L.shape

        # check dimensions are consistent
        if not self._nu == nu_2:
            raise ValueError("Inconsistent shape.")
        if not self._nu == nu_3:
            raise ValueError("Inconsistent shape.")

        self._M = M
        self._L = L

        # whether these flags are true depends on if and how self._W is determined below
        self._isComplete = False
        self._isFromSystem = False
        
        if isinstance(V, _np.ndarray):
            # data given as a matrix
            # check that sizes are consistent
            if not V.ndim == 2:
                raise ValueError("Inconsistent shape.")
            if not V.shape[0] == self._nu:
                raise ValueError("Inconsistent shape.")
            # all is good, set flags
            self._m = V.shape[1]
            self._V = V
            self._isComplete = True
        elif isinstance(V, _ld.LinearDAE):
            #calculate self._V from system held by V
            if V.isRegular:
                # provided system is regular
                # try to get the tangential generalized controllability matrix from its generalized sylvester equation
                try:
                    #self._Y = _gs.solve(W.E, self._Lambda, -W.A, _np.eye(self._rho,self._rho), W.B @ self._R)
                    self._Y = _gs.solve(_np.eye(self._nu,self._nu), V.A, -self._M, V.E, -self._L @ V.C)
                except:
                    self._m = None
                    self._isComplete = False
                    self._V = None
                    raise ValueError("Can't calculate V from provided system.")
                if not self._p == V.p:
                    # this shouldn't happen
                    self._m = None
                    self._isComplete = False
                    self._V = None
                    raise ValueError("Can't calculate V from provided system.")
                # all is good, set flags indicating that this object is complete, and X exists as a variable
                self._V = self._Y @ V.B + self._L @ V.D
                self._m = V.m
                self._isComplete = True
                self._isFromSystem = True
            else:
                # can't get data from irregular system
                self._m = None
                self._isComplete = False
                self._V = None
                raise ValueError("Can't calculate V from provided system.")
        else:
            # V empty or bad
            self._isComplete = False
            self._m = None
            self._V = None

    @property
    def nu(self) -> int:
        return self._nu
        
    @property
    def p(self) -> int:
        return self._p
        
    @property
    def m(self) -> int | None:
        return self._m
        
    # indicates if the object contains all of self.Lambda, self.R, and self.W with consistent sizes for the Loewner framework
    @property
    def isComplete(self) -> bool:
        return self._isComplete
    
    # indicates whether or not self.X exists or is None
    @property
    def isFromSystem(self) -> bool:
        return self._isFromSystem

    @property
    def M(self) -> _np.ndarray:
        return self._M
        
    @property
    def L(self) -> _np.ndarray:
        return self._L
        
    @property
    def V(self) -> _np.ndarray | None:
        return self._V
    
    @property
    def Y(self) -> _np.ndarray | None:
        if self._isComplete and self._isFromSystem:
            return self._Y
        else:
            return None
        
    def __repr__(self) -> str:
        return f"LeftTangentialData\nisComplete = {self.isComplete}, isFromSystem = {self.isFromSystem},\nnu = {self.nu}, p = {self.p}, m = {self.m},\n\nM =\n{self.M},\n\nL =\n{self.L},\n\nV =\n{self.V},\n\nY =\n{self.Y}\n"
        
    def __bool__(self) -> bool:
        return self._isComplete


