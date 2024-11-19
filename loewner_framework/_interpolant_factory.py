import numpy as _np
import scipy.signal as _spsg

from . import generalized_sylvester as _gs
from . import linear_daes as _ld

from ._right_and_left_tangential_data import RightTangentialData, LeftTangentialData
from ._loewner_tangential_data import LoewnerTangentialData



class InterpolantFactory:
    def __init__(self, loewnertd : LoewnerTangentialData):
        """An object used to construct interpolants for the LoewnerTangentialData it is provided with, along with some helper methods for determining free parameter shapes and checking free parameter consistency.

        Args:
            loewnertd: a LoewnerTangentialData object representing the tangential interpolation data, with the isComplete property being True.

        Returns:
            An InterpolantFactory object that can be used for building interpolants in the form of linear_daes.LinearDAE objects.

        Raises:
            ValueError: Provided data is not an instance of LoewnerTangentialData.
            ValueError: Provided LoewnerTangentialData, or its contained Right/LeftTangentialData, is not complete.
        """
        if not isinstance(loewnertd, LoewnerTangentialData):
            raise ValueError("Provided data is not an instance of LoewnerTangentialData.")
        if not loewnertd.isComplete or not loewnertd.rtd.isComplete or not loewnertd.ltd.isComplete:
            raise ValueError("Provided LoewnerTangentialData, or its contained Right/LeftTangentialData, is not complete.")
        
        self._loewnertd = loewnertd
        self._isComplete = True

    @property
    def isComplete(self) -> bool:
        return self._isComplete
    
    @property
    def loewnertd(self) -> LoewnerTangentialData | None:
        if self.isComplete:
            return self._loewnertd
        else:
            return None
        
    @property
    def rho(self):
        return self.loewnertd.rtd.rho
    
    @property
    def nu(self):
        return self.loewnertd.ltd.nu
    
    @property
    def m(self):
        return self.loewnertd.rtd.m
    
    @property
    def p(self):
        return self.loewnertd.ltd.p
    
    @property
    def Lambda(self):
        return self.loewnertd.rtd.Lambda
    
    @property
    def M(self):
        return self.loewnertd.ltd.M
    
    @property
    def R(self):
        return self.loewnertd.rtd.R
    
    @property
    def L(self):
        return self.loewnertd.ltd.L
    
    @property
    def W(self):
        return self.loewnertd.rtd.W
    
    @property
    def V(self):
        return self.loewnertd.ltd.V
    
    @property
    def Loewner(self):
        return self.loewnertd.Loewner
    
    @property
    def shiftedLoewner(self):
        return self.loewnertd.shiftedLoewner
        
    def minimal_order(self, D: _np.ndarray | None = None, label: str = "") -> _ld.LinearDAE | None:
        """Provided that the Loewner matrices are square (so nu == rho), construct a minimal dimension interpolant of the tangential data using the provided free parameter D.

        Given desired interpolant free parameter D, construct a LinearDAE that interpolates the data used to initialized the InterpolantFactory object. The provided free parameter shape should be consistent with the dictionary returned by the parameter_dimensions method when its argument is rho or nu, and therefore should return a tuple from the check_consistent_shapes method with the first entry being True.

        Args:
            D: a 2darray representing the D free parameter, or None if D will be zero.
            label: the label to be assigned to the returned linear_daes.LinearDAE object.

        Returns:
            A linear_daes.LinearDAE object constructed from the interpolation data used to initialize the InterpolantFactory object, along with the interpolant free parameter D provided to this method. If an exception is not raised, the returned system will belong to the parameterization of all Loewner framework interpolants and will therefore be an interpolant of the right and left tangential data sets by construction.

        Raises:
            ValueError: If the matrix D is provided, it must be a 2darray.
            ValueError: Provided D matrix must be a 2darray.
            ValueError: Provided D matrix dimensions are not consistent with the tangential data.
        """
        if self.isComplete:
            if not isinstance(D, _np.ndarray):
                if D == None:
                    return _ld.LinearDAE(self.loewnertd.shiftedLoewner, # A
                                         -self.loewnertd.ltd.V, # B
                                         self.loewnertd.rtd.W, # C
                                         _np.zeros((self.loewnertd.rtd.p, self.loewnertd.ltd.m)), # D
                                         self.loewnertd.Loewner, # E
                                         str(label)) # label
                else:
                    raise ValueError("If the matrix D is provided, it must be a 2darray.")
            elif not isinstance(D, _np.ndarray) or not D.ndim == 2:
                raise ValueError("Provided D matrix must be a 2darray.")
            elif not D.shape == (self.loewnertd.rtd.p, self.loewnertd.ltd.m):
                raise ValueError("Provided D matrix dimensions are not consistent with the tangential data.")
            else:
                return _ld.LinearDAE(self.loewnertd.shiftedLoewner - self.loewnertd.ltd.L @ D @ self.loewnertd.rtd.R, # A
                                     -(self.loewnertd.ltd.V - self.loewnertd.ltd.L @ D), # B
                                     self.loewnertd.rtd.W - D @ self.loewnertd.rtd.R, # C
                                     D, # D
                                     self.loewnertd.Loewner, # E
                                     str(label)) # label
        else:
            return None
    
    def parameter_dimensions(self, total_dimension: int) -> tuple[bool,dict]:
        """Get the shape required for each free parameter to achieve an interpolant with a desired total dimension.

        Given a desired interpolant dimension, total_dimension, return a dictionary with keys D, P, Q, G, T, H, and F (associated to the interpolant free parameter names used in the parameterization(...) method), where for each key the corresponding value is the ndarray shape required to achieve a consistent interpolant with that total dimension. If the value associated to the key is None, then that parameter can not be used in an interpolant of that dimension (i.e. in the minimal order family of interpolants, where total_dimension = rho = nu, the only free parameter is D, so the keys P, Q, G, T, H, F will have value None, and the key D will have value (p,m)).

        Args:
            total_dimension: the total interpolant dimension for which the required shapes of D, P, Q, G, T, H, and F will be determined.

        Returns:
            A tuple where the first entry is a boolean indicating if the desired total_dimension is possible, and where the second entry is a dictionary with keys associated to the interpolant free parameters D, P, Q, G, T, H, F, for which the associated values are the required shapes of the parameters, or None if the parameter cannot be used for the given total_dimension.
        """
        # return the shape required for each of the parameters to achieve a consistent system dimension of total_dimension
        shape_dict = {"D": None, "P": None, "Q": None, "G": None, "T": None, "H": None, "F": None}
        if total_dimension < self.rho or total_dimension < self.nu:
            # generally, can't achieve a smaller dimension than max(rho,nu)
            return (False, shape_dict)
        elif total_dimension == self.rho and total_dimension == self.nu:
            # this is exactly the minimal order family, D is the only free parameter
            shape_dict["D"] = (self.p, self.m)
            return (True, shape_dict)
        elif total_dimension == self.rho:
            # nu < total_dimension, this is the tall case where rows are added, but not columns; free parameters are P, T, D
            shape_dict["P"] = (total_dimension - self.nu, self.rho)
            shape_dict["T"] = (total_dimension - self.nu, self.m)
            shape_dict["D"] = (self.p, self.m)
            return (True, shape_dict)
        elif total_dimension == self.nu:
            # rho < total_dimension, this is the wide case where columns are added, but not rows; free parameters are Q, H, D
            shape_dict["Q"] = (self.nu, total_dimension - self.rho)
            shape_dict["H"] = (self.p, total_dimension - self.rho)
            shape_dict["D"] = (self.p, self.m)
            return (True, shape_dict)
        else:
            # rho, nu < total_dimension, this is the case where both rows and columns are added; free parameters are D, P, Q, G, T, H, F
            shape_dict["D"] = (self.p, self.m)
            shape_dict["P"] = (total_dimension - self.nu, self.rho)
            shape_dict["Q"] = (self.nu, total_dimension - self.rho)
            shape_dict["G"] = (total_dimension - self.nu, total_dimension - self.rho)
            shape_dict["T"] = (total_dimension - self.nu, self.m)
            shape_dict["H"] = (self.p, total_dimension - self.rho)
            shape_dict["F"] = (total_dimension - self.nu, total_dimension - self.rho)
            return (True, shape_dict)

    def check_consistent_shapes(self,   D: _np.ndarray | None = None, P: _np.ndarray | None = None,
                                        Q: _np.ndarray | None = None, G: _np.ndarray | None = None,
                                        T: _np.ndarray | None = None, H: _np.ndarray | None = None,
                                        F: _np.ndarray | None = None) -> tuple[bool,int]:
        """Check if the provided interpolant free parameters are consistent with any valid total interpolant dimension.

        Given desired interpolant free parameters D, P, Q, G, T, H, and F, check that the parameters are consistent such that they form an interpolant that is a "square" system. The parameter sizes should be compatible with the dictionary returned by the parameter_dimensions method for a given interpolant dimension.

        Args:
            D: a 2darray representing the D free parameter, or None if D will be zero.
            P: a 2darray representing the P free parameter, or None if the parameter will not be used.
            Q: a 2darray representing the Q free parameter, or None if the parameter will not be used.
            G: a 2darray representing the G free parameter, or None if the parameter will not be used.
            T: a 2darray representing the T free parameter, or None if the parameter will not be used.
            H: a 2darray representing the H free parameter, or None if the parameter will not be used.
            F: a 2darray representing the F free parameter, or None if the parameter will not be used.

        Returns:
            A tuple where the first entry is a boolean indicating if the provided matrices are consistent with a valid total interpolant dimension, and an integer equal to the total interpolant dimension if the first tuple element is True or -1 if the first tuple element is False.
        """
        # We allow D == None to be consistent (defaults to a matrix of zeros); D always shows up as a free parameter, so we check this first and independently of the other parameters
        if isinstance(D, _np.ndarray):
            if not D.shape == (self.p, self.m):
                # inconsistent shape
                return (False, -1)
        elif not D is None:
            return (False, -1)
            
        # now D is consistent, i.e. D.shape = (p,m)

        # there are now 4 possibly consistent cases to check for the parameters P, Q, G, T, H, and F
        # (1) both rows and columns are added, so none of P, Q, G, T, H, F can be None, and must all have the correct dimension
        if isinstance(G, _np.ndarray) and isinstance(F, _np.ndarray) and isinstance(P, _np.ndarray) and isinstance(Q, _np.ndarray) and isinstance(T, _np.ndarray) and isinstance(H, _np.ndarray):
            assumed_total_order = self.rho + Q.shape[1] # the number of columns in the E matrix must be equal to the total dimension. We will now check against this assumed total order.
            _, shape_dict = self.parameter_dimensions(assumed_total_order)
            if not shape_dict["P"] == P.shape: return (False, -1)
            elif not shape_dict["Q"] == Q.shape: return (False, -1)
            elif not shape_dict["G"] == G.shape: return (False, -1)
            elif not shape_dict["T"] == T.shape: return (False, -1)
            elif not shape_dict["H"] == H.shape: return (False, -1)
            elif not shape_dict["F"] == F.shape: return (False, -1)
            else: return (True, assumed_total_order)
        # (2) minimal-order for wide Loewner matrix scenario (minimal order with rho > nu); only rows are added, so P and T can't be None and must have consistent dimensions, and Q, G, H, and F must be None
        elif isinstance(P, _np.ndarray) and isinstance(T, _np.ndarray) and Q is None and G is None and H is None and F is None:
            assumed_total_order = self.rho # in the tall scenario the number of columns in the E matrix must be the total dimension (rho)
            _, shape_dict = self.parameter_dimensions(assumed_total_order)
            if not shape_dict["P"] == P.shape: return (False, -1)
            elif not shape_dict["Q"] == None: return (False, -1)
            elif not shape_dict["G"] == None: return (False, -1)
            elif not shape_dict["T"] == T.shape: return (False, -1)
            elif not shape_dict["H"] == None: return (False, -1)
            elif not shape_dict["F"] == None: return (False, -1)
            else: return (True, assumed_total_order)
        # (3) minimal-order for tall Loewner matrix scenario (minimal order with nu > rho); only columns are added, so Q and H can't be None and must have consistent dimensions, and P, G, F, and T must be None
        elif isinstance(Q, _np.ndarray) and isinstance(H, _np.ndarray) and P is None and G is None and F is None and T is None:
            assumed_total_order = self.nu # in the wide scenario the number of rows in the E matrix must be the total dimension (nu)
            _, shape_dict = self.parameter_dimensions(assumed_total_order)
            if not shape_dict["P"] == None: return (False, -1)
            elif not shape_dict["Q"] == Q.shape: return (False, -1)
            elif not shape_dict["G"] == None: return (False, -1)
            elif not shape_dict["T"] == None: return (False, -1)
            elif not shape_dict["H"] == H.shape: return (False, -1)
            elif not shape_dict["F"] == None: return (False, -1)
            else: return (True, assumed_total_order)
        # (4) minimal-order for square Loewner matrix scenario (minimal order with rho == nu); neither rows nor columns are added and the Loewner matrices are square, so all of P, Q, G, T, H, and F must be None
        elif P is None and Q is None and G is None and T is None and H is None and F is None:
            assumed_total_order = self.rho # we could also take nu; to be consistent they must be equal, and if they are not equal they certainly won't pass the following checks on the shape dictionary
            _, shape_dict = self.parameter_dimensions(assumed_total_order)
            if not self.rho == self.nu: return (False, -1)
            elif not shape_dict["P"] == None: return (False, -1)
            elif not shape_dict["Q"] == None: return (False, -1)
            elif not shape_dict["G"] == None: return (False, -1)
            elif not shape_dict["T"] == None: return (False, -1)
            elif not shape_dict["H"] == None: return (False, -1)
            elif not shape_dict["F"] == None: return (False, -1)
            else: return (True, assumed_total_order)
        # if we make it here, the scenario is not one that is acceptable
        else:
            return (False, -1)

    def parameterization(self,  D: _np.ndarray | None = None, P: _np.ndarray | None = None,
                                Q: _np.ndarray | None = None, G: _np.ndarray | None = None,
                                T: _np.ndarray | None = None, H: _np.ndarray | None = None,
                                F: _np.ndarray | None = None, label: str = "") -> _ld.LinearDAE:
        """Construct an interpolant of the tangential data using the provided free parameters D, P, Q, G, T, H, and F, with the name provided by the label argument.

        Given desired interpolant free parameters D, P, Q, G, T, H, and F, construct a LinearDAE that interpolates the data used to initialized the InterpolantFactory object. The provided free parameter shapes should be consistent with the dictionary returned by the parameter_dimensions method, and therefore should return a tuple from the check_consistent_shapes method with the first entry being True.

        Args:
            D: a 2darray representing the D free parameter, or None if D will be zero.
            P: a 2darray representing the P free parameter, or None if the parameter will not be used.
            Q: a 2darray representing the Q free parameter, or None if the parameter will not be used.
            G: a 2darray representing the G free parameter, or None if the parameter will not be used.
            T: a 2darray representing the T free parameter, or None if the parameter will not be used.
            H: a 2darray representing the H free parameter, or None if the parameter will not be used.
            F: a 2darray representing the F free parameter, or None if the parameter will not be used.
            label: the label to be assigned to the returned linear_daes.LinearDAE object.

        Returns:
            A linear_daes.LinearDAE object constructed from the interpolation data used to initialize the InterpolantFactory object, along with the interpolant free parameters provided to this method. If an exception is not raised, the returned system will belong to the parameterization of all Loewner framework interpolants and will therefore be an interpolant of the right and left tangential data sets by construction.

        Raises:
            ValueError: Inconsistent shape.
        """
        # check shapes are consistent
        consistent, total_dimension = self.check_consistent_shapes(D=D, P=P, Q=Q, G=G, T=T, H=H, F=F)
        if consistent:
            if total_dimension == self.nu and total_dimension == self.rho:
                # square min order case
                if not P is None or not Q is None or not G is None or not T is None or not H is None or not F is None:
                    raise ValueError("Inconsistent shape.")
                else:
                    return self.minimal_order(D, label=label)
            elif total_dimension == self.nu and total_dimension > self.rho:
                # tall Loewner matrix case; only columns added
                if not P is None or not G is None or not F is None or not T is None:
                    raise ValueError("Inconsistent shape.")
                else:
                    if D is None:
                        D = _np.zeros((self.p, self.m))
                    E = _np.concatenate((self.Loewner, Q), axis=1)
                    A = _np.concatenate((self.shiftedLoewner - self.L @ D @ self.R, self.M @ Q + self.L @ H), axis=1)
                    B = -(self.V - self.L @ D)
                    C = _np.concatenate((self.W - D @ self.R, H), axis=1)
                    return _ld.LinearDAE(A=A, B=B, C=C, D=D, E=E, label=label)
            elif total_dimension > self.nu and total_dimension == self.rho:
                # wide Loewner matrix case; only rows added
                if not Q is None or not G is None or not H is None or not F is None:
                    raise ValueError("Inconsistent shape.")
                else:
                    if D is None:
                        D = _np.zeros((self.p, self.m))
                    E = _np.concatenate((self.Loewner, P), axis=0)
                    A = _np.concatenate((self.shiftedLoewner - self.L @ D @ self.R, P @ self.Lambda + T @ self.R), axis=0)
                    B = -_np.concatenate((self.V - self.L @ D, T), axis=0)
                    C = self.W - D @ self.R
                    return _ld.LinearDAE(A=A, B=B, C=C, D=D, E=E, label=label)
            elif total_dimension > self.nu and total_dimension > self.rho:
                # both rows and columns added
                if D is None:
                    D = _np.zeros((self.p, self.m))
                E = _np.concatenate((_np.concatenate((self.Loewner, Q), axis=1), _np.concatenate((P, G), axis=1)), axis=0)
                A = _np.concatenate((_np.concatenate((self.shiftedLoewner - self.L @ D @ self.R, self.M @ Q + self.L @ H), axis=1), _np.concatenate((P @ self.Lambda + T @ self.R, F), axis=1)), axis=0)
                B = -_np.concatenate((self.V - self.L @ D, T), axis=0)
                C = _np.concatenate((self.W - D @ self.R, H), axis=1)
                return _ld.LinearDAE(A=A, B=B, C=C, D=D, E=E, label=label)
            else:
                raise ValueError("Inconsistent shape.")
        else:
            raise ValueError("Inconsistent shape.")
        
    def double_order_pole_placed(self,  desired_poles: _np.ndarray, D: _np.ndarray | None = None, label: str = "") -> _ld.LinearDAE:
        """Given a square and invertible Loewner matrix, construct an interpolant of the tangential data of twice the minimal order, but having prescribed poles.

        Given desired interpolant free parameter D, and a desired set of 2*rho system poles, construct a LinearDAE that interpolates the data used to initialized the InterpolantFactory object. The provided free parameter shape should be consistent with the dictionary returned by the parameter_dimensions method, and therefore should return a tuple from the check_consistent_shapes method with the first entry being True. The Loewner matrix should be square and invertible.

        Args:
            D: a 2darray representing the D free parameter, or None if D will be zero.
            desired_poles: a 1darray representing the 2*rho desired interpolant poles; desired_poles should not contain any eigenvalues in the RightTangentialData and LeftTangentialData matrices Lambda and M, respectively.
            label: the label to be assigned to the returned linear_daes.LinearDAE object.

        Returns:
            A linear_daes.LinearDAE object constructed from the interpolation data used to initialize the InterpolantFactory object, along with the interpolant free parameter provided to this method. If an exception is not raised, the returned system will belong to the parameterization of all Loewner framework interpolants and will therefore be an interpolant of the right and left tangential data sets by construction. The system will have dimension 2*rho.

        Raises:
            ValueError: Inconsistent shape.
            ValueError: Loewner matrix is singular.
            ValueError: Inconsistent number of poles to place.
        """
        # first, check that the Loewner matrix is square and invertible
        if not self.rho == self.nu:
            raise ValueError("Inconsistent shape.")
        if _np.linalg.det(self.Loewner) == 0:
            raise ValueError("Loewner matrix is singular.")
        
        # check that D is the right shape, or None (in which case, set to 0 matrix of correct shape)
        if D is None:
            D = _np.zeros((self.p, self.m))
        elif not D.shape == (self.p, self.m):
            raise ValueError("Inconsistent shape.")
        
        # check that the number of poles is 2*rho
        if not desired_poles.size == 2*self.rho:
            raise ValueError("Inconsistent number of poles to place.")
        poles = desired_poles if desired_poles.ndim == 1 else desired_poles.flatten()
        
        # try to calculate the required T and H parameters via pole placement
        A1, A2, B, C = _np.linalg.inv(self.Loewner) @ self.shiftedLoewner, self.shiftedLoewner @ _np.linalg.inv(self.Loewner), _np.linalg.inv(self.Loewner) @ self.L, -self.R @ _np.linalg.inv(self.Loewner)
        Hbar = _spsg.place_poles(A1, -B, poles[0:self.rho])
        Tbar = _spsg.place_poles(A2.T, C.T, poles[self.rho:2*self.rho])
        H = Hbar.gain_matrix + D @ self.R
        T = -Tbar.gain_matrix.T - self.L @ D

        # assign the remaining required free parameters for construction via the parameterization method
        P = _np.zeros((self.rho, self.rho))
        Q = _np.zeros((self.rho, self.rho))
        G = self.Loewner
        F = self.shiftedLoewner - self.L @ D @ self.R - T @ self.R + self.L @ H

        # check shapes are consistent
        consistent, total_dimension = self.check_consistent_shapes(D=D, P=P, Q=Q, G=G, T=T, H=H, F=F)
        if not consistent or not total_dimension == 2*self.rho:
            raise ValueError("Inconsistent shape.")
        
        # build the interpolant
        return self.parameterization(D, P, Q, G, T, H, F, label)

    def __repr__(self) -> str:
        return f"InterpolantFactory\nisComplete = {self.isComplete}\n"
    
    def __bool__(self) -> bool:
        return self.isComplete