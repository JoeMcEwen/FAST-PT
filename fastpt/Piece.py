"""
Piece

Defines the Piece class and some subclasses like P13likePiece, P22likePiece,
ArrayPiece or CorrelationAtom. The idea being that each correlation is basically
TODO: Complete description...
Currently only used by LVDM

"""

from abc import ABC, abstractmethod
import numbers
from copy import deepcopy

import sys
sys.path.append("../")

import numpy as np
from numpy import exp, log
from numpy import logical_not as lnot, logical_and as land, logical_or as lor, logical_xor as lxor
from functools import reduce
from scipy.signal import fftconvolve
import scipy.interpolate as interpolate

# import FASTPT



# Piece ########################################################################

class Piece(ABC):
    def __init__(self, name="Unnamed_Piece"):
        self.name = name
        self.k = np.array([])
        self.P = np.array([])

    @abstractmethod
    def integrate(self, k, P, remember=True):
        return self.k, self.P # integrate is expected to return a k and P

    def forget(self):
        self.k = np.array([])
        self.P = np.array([])



# ArrayPiece ###################################################################

class ArrayPiece(Piece):
    def __init__(self, name="Unnamed_ArrayPiece", k=np.array([]), P=np.array([])):
        Piece.__init__(name)
        self.k = k
        self.P = P

    def integrate(self, k, P, remember=True):
        if remember:
            self.k, self.P = k, P
        return k, P


# P13likePiece #################################################################

class P13likePiece(Piece):

    def __init__(self,
                 name: str = "Unnamed_P13likePiece",
                 k_correction=(lambda k: 1),
                 **Zkwargs):
        """[summary]

        :param name: [description], defaults to "Unnamed_P13likePiece"
        :type name: str, optional
        :param k_correction: [description], defaults to (lambda k: 1)
        :type k_correction: tuple, optional
        :type Zargs:
        """        
        Piece.__init__(self, name)
        self.Z = lambda q: self.__init_Z(s=q, **Zkwargs)
        self.k_correction = k_correction

    def __init_Z(self, s, Zmid, Zlow=None, Zhig=None, cut=(-7,7)):
        """Returns an array representing Z for any values. This is achieved by
        piecing together 

        :param s: [description]
        :type s: [type]
        :param Zmid: [description]
        :type Zmid: [type]
        :param Zlow: [description], defaults to None
        :type Zlow: [type], optional
        :param Zhig: [description], defaults to None
        :type Zhig: [type], optional
        :param cut: [description], defaults to (-7,7)
        :type cut: tuple, optional
        :return: [description]
        :rtype: [type]
        """        
        #TODO: Clean a bit the cut and Zhig/Zlow things...

        if isinstance(cut, numbers.Number):
            cut = (-cut, cut)
        elif cut == None:
            cut = (-np.inf, np.inf)

        if Zhig == None:
            Zhig = Zmid
        if Zlow == None:
            Zlow = Zmid

        higi = s < exp(cut[0])
        lowi = s > exp(cut[1])

        s0or1 = lor(s==0, s==1)

        midi = lnot(reduce(lor, (higi, lowi, s0or1)))

        Z = np.empty(s.shape)

        Z[higi] = Zhig(s[higi])
        Z[midi] = Zmid(s[midi])
        Z[lowi] = Zlow(s[lowi])

        Z = self.__extrapolate_missing_values(s, Z, s0or1)

        return Z

    @staticmethod
    def __extrapolate_missing_values(in_val, out_val, missing, **kwargs):
        """Receiving the `in_val` values at the `out_val` positions,
        interpolates the `missing` components of `in_val` with
        `scipy.interpolate.interp1d`. The `**kwargs` are fowarded to the latter
        function. This function thus returns a completed version of `out_val`"""
        out_val[missing] = interpolate.interp1d(in_val[lnot(missing)],
                                                out_val[lnot(missing)],
                                                fill_value="extrapolate",
                                                bounds_error=False,
                                                **kwargs
                                                )(in_val[missing])
        return out_val

    def integrate(self, k, P, method="fftconvolve", use_cache=False, remember=True):
        result = self.__integrate_with_convolution(k, P, method=method)
        if remember:
            self.k = k
            self.P = result
        return self

    def __integrate_with_convolution(self, k, P, use_cache=False, method="fftconvolve"):
        N = k.size
        m = np.arange(-N + 1, N)
        delta = log(k[-1] / k[0]) / N
        sm = m * delta #+ log(k[0])

        GD = self.Z(exp(-sm))
        GD *= exp(-3*sm)
        FD = P
        #FD = P_interp(exp(mdelta))

        #trim = lambda x: x[N-1:2*N-1]

        if method == "fftconvolve":
            fftc = fftconvolve(FD, GD, mode="valid")
        elif method == "np.convolve":
            fftc = np.convolve(FD, GD, mode="valid")
        else:
            raise ValueError(f"Integration method unknown or unsupported: {method}")

        # C = P * k**3 * self.k_correction(k)
        # DONE Directly incorporate the 1/(2pi)^3 here
        C = P * k**3 * self.k_correction(k) / (2 * np.pi) ** 3

        return C * delta * fftc




# P22likePiece #################################################################

class P22likePiece(Piece):
    def __init__(self, name="Unnamed_P22likePiece", kernel=None, fastpt_key=None):
        """P22likePiece built to represent a convolution piece of the same form
        as the P22 power spectrum contribution.

        :param name: descriptive name of the piece, defaults to "Unnamed_P22likePiece"
        :type name: str, optional
        :param kernel: in order [denominator, numerator, function at mu=1 and q=k], defaults to None
        :type kernel: iterable of three functions, optional
        :param fastpt_key: string designating which Piece this corresponds to in FAST-PT, defaults to None
        :type fastpt_key: str, optional
        """        

        self.name = name
        self.kernel = kernel
        self.integral = [np.array([]), np.array([])]
        self.fastpt_key = fastpt_key
        self.k = np.array([])
        self.P = np.array([])

    def integrate_with_FASTPT(self, k, P):
        if self.fastpt_key == None:
            raise ValueError("The fastpt_key is undefined.")
            #TODO: Implement the FAST-PT interface

        raise NotImplementedError("Not ready... yet.")

    def __init_kernel(self, k, q, mu):
        """Usually, the kernel can be expressed as a fraction: numerator over
        denominator. However, when k=q, the denominator may go to 0, while the
        analytical expression can in fact be simplified. Hence this function
        and its arguments.

        This functions assumes k and q are of the same size."""
        #  TODO  Tidy up a bit below

        mu1 = 1.

        numerator = self.kernel[0](k, q, mu)
        denominator = self.kernel[1](k, q, mu)
        diagonal_val = self.kernel[2](k, q, mu1)

        denom_mu1 = denominator[:,:,-1].view()
        rows, cols = np.indices(denom_mu1.shape)
        denom_mu1[np.diagonal(rows), np.diagonal(cols)] = 1

        numer_mu1 = numerator[:,:,-1].view()
        rows, cols = np.indices(numer_mu1.shape)
        numer_mu1[np.diagonal(rows), np.diagonal(cols)] = diagonal_val

        return numerator / denominator

    def integrate_with_trapz(self, k, P, mu_param=[-1, 1, 200], q_param=[1e-5, 1e2, 796]):
        if self.kernel == None:
            raise ValueError("The kernel is undefined")

        P_interp = lambda k_: interpolate.interp1d(k, P,
            fill_value=0, bounds_error=False, assume_sorted=True)(np.abs(k_))

        # Integration variables
        delta = np.log(q_param[1]/q_param[0])/q_param[2]

        k = q_param[0] * np.exp(np.arange(q_param[2]) * delta)

        q = deepcopy(k)
        mu = np.sin(np.pi/2*np.linspace(mu_param[0], mu_param[1], num=mu_param[2], endpoint=True))
        #mu = np.linspace(mu_param[0], mu_param[1], num=mu_param[2], endpoint=True)

        kernel = self.__init_kernel(k, q, mu, *self.kernel)
        kx, qx, mux = np.ix_(k, q, mu)

        # Takes a kernel (function) and incorporate it to create the matrix to integrate
        I = qx**2 * np.apply_along_axis(P_interp, 1, qx) \
            * np.apply_along_axis(P_interp, 1, np.sqrt(kx**2 + qx**2 - 2*kx*qx*mux)) \
            * kernel(kx, qx, mux)

        # Integrate the matrix provided
        return k, 2*np.pi * np.trapz(np.trapz(I + np.flip(I, 2), x=mu, axis=2)/2, x=q, axis=1) / (2*np.pi)**3

    def integrate(self, k, P, method="FASTPT", use_cache=False, remember=True):
        methods = {"fastpt": self.integrate_with_FASTPT,
                   "trapz": self.integrate_with_trapz}

        if not method in methods:
            raise ValueError("Integration method unknown or unsupported.")

        k, result = methods[method.strip().lower()](k, P)
        if remember:
            self.k = k
            self.P = P

        return self

    def integrate_with_params(self, k, P):
        raise NotImplementedError("Not ready... yet.")



# CorrelationAtom ##############################################################

class CorrelationAtom(Piece):
    def __init__(self, name, pieces):
        self.name = name
        self.pieces = []
        self.extend(pieces)

    def __iter__(self):
        for p in self.__pieces:
            yield p

    def __getitem__(self, item):
        return self.__pieces[item]

    def __len__(self):
        return len(self.__pieces)

    def append(self, pieces):
        if isinstance(pieces, Piece):
            self.__pieces.append(pieces)
        else:
            raise TypeError("The pieces must be a Piece")

    def extend(self, pieces):
        if hasattr(pieces, "__iter__"):
            for p in pieces:
                self.append(p)
        else:
            raise AttributeError("The pieces must be iterable")

    @property
    def pieces(self):
        return self.__pieces

    @pieces.setter
    def pieces(self, new_p):
        if np.all([isinstance(p, Piece) for p in new_p]):
            self.__pieces = list(new_p)
        else:
            raise TypeError("The pieces attribute must be a list of Piece instances")

    @property
    def k(self):
        if len(self.__pieces) == 1 or self.__pieces and reduce(np.allclose, [p.k for p in self.__pieces]):
            return self.__pieces[0].k.copy()
        else:
            return np.array([])

    @k.setter
    def k(self, new_k):
        raise TypeError("k cannot be directly set on a CorrelationAtom")

    @property
    def P(self):
        if self.__pieces:
            return np.sum([p.P for p in self.pieces], axis=0)
        else:
            return np.array([])

    @P.setter
    def P(self, new_P):
        raise TypeError("P cannot be directly set on a CorrelationAtom")

    @property
    def P13likePieces(self):
        return [p for p in self.__pieces if isinstance(p, P13likePiece)]

    @property
    def P22likePieces(self):
        return [p for p in self.__pieces if isinstance(p, P22likePiece)]

    def integrate(self, k, P, remember=True):
        for p in self.__pieces:
            p.integrate(k, P, remember=remember)
        return self

    def forget(self):
        for p in self.__pieces:
            p.forget()
