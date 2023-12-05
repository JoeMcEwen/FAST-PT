"""
Defines the tidal terms proportional to b1⋅C1 and b1⋅C1δ and returns the
corresponding coefficients through the IA_tidal() function.

More specifically, the coefficients hereafter accounts for the kernels
intervening in the following terms (using the notations used in 1603.04826v2):

    bs C1 /2 ⟨ s²(k) | fE(k') δ(k') ⟩
                == bs C1 fE(k) D(k+k') ∫ S2(q,k-q)  F2(q,k-q) P(q) P(k-q) dq

    bs C1δ/2 ⟨ s²(k) | ∫ 1/(2π)³ fE(q) δ(q)δ(k-q) dq ⟩
                == bs C1delta  D(k+k') ∫ S2(q,k-q) fE(q)      P(q) P(k-q) dq

where D denotes the Dirac delta. Switch the E indices to B to get the B-mode
parts.

See also
--------
IA_tidal.IA_tidal
"""

from copy import deepcopy

import numpy as np

from .J_table import J_table
from .Piece import P13likePiece


# (i) Ordering is α, β, l₁, l₂, l, A coefficient

_l_mat_null = np.array([[0, 0, 0, 0, 0, 0]], dtype=float)

# Terms proportional to bs·C1:
_l_mat_bsC1 = np.array(
    [[ 0,  0, 0, 0, 0,   8./315],
    [  1, -1, 0, 0, 1,   4./15 ],
    # [  1, -1, 0, 0, 1,   2./15 ],
    # [ -1,  1, 0, 0, 1,   2./15 ],
    [  0,  0, 0, 0, 2, 254./441],
    [  1, -1, 0, 0, 3,   2./5  ], # Equivalent to the two lines commented
                                  # because of symmetry
    # [  1, -1, 0, 0, 3,  1./5  ],
    # [ -1,  1, 0, 0, 3,  1./5  ],
    [  0,  0, 0, 0, 4,  16./245]], dtype=float)
# The two matrices are the same in this case. Only the fEk and fBk will affect
# the result, hence:
# l_mat_B = deepcopy(l_mat_E)

# Terms proportional to bs·C1δ:
_l_mat_bsC1δ = np.array([[0, 0, 0, 2, 2, 2./3]], dtype=float)
# The B component is null, so:
# l_mat_Bδ = _l_mat_null

_l_mat_bsC2 = np.array(
    [[0, 0, 0, 0, 0, - 2./45],
    [ 0, 0, 1, 1, 1,   2./ 5],
    [ 0, 0, 0, 0, 2, -11./63],
    # [ 0, 0, 0, 2, 2, - 4./ 9],
    [ 0, 0, 0, 2, 2, - 2./ 9],
    [ 0, 0, 2, 0, 2, - 2./ 9],
    [ 0, 0, 1, 1, 3,   3./ 5],
    [ 0, 0, 0, 0, 4, - 4./35]], dtype=float)

_l_mat_bsCt = np.array(
    [[0, 0, 0, 0, 0,   8./315],
    [ 0, 0, 0, 0, 2, -40./441],
    [ 0, 0, 0, 0, 4,  16./245]], dtype=float)

_l_mat_b1Ct = np.array(
    [[0,  0, 0, 0, 0, -36./ 245],
    [ 1, -1, 0, 0, 1, - 4./  35],
    [ 0,  0, 0, 0, 2,  44./ 343],
    [ 1, -1, 0, 0, 3,   4./  35],
    [ 0,  0, 0, 0, 4,  32./1715]], dtype=float)

# l_mat_A00E = np.array([[0,0,0,2,0,17./21],\
#       [0,0,0,2,2,4./21],\
#       [1,-1,0,2,1,1./2],\
#       [-1,1,0,2,1,1./2]], dtype=float)

_l_mat_b2C1 = np.array(
    [[ 0,  0, 0, 0, 0, 17./21],
    [  1, -1, 0, 0, 1,  1./ 2],
    [ -1,  1, 0, 0, 1,  1./ 2],
    [  0,  0, 0, 0, 2,  4./21]], dtype=float)

_l_mat_b2C1δ = np.array(
    [[ 0, 0, 0, 2, 0, 1.]], dtype=float) # _l_mat_null # ???

_l_mat_b2C2 = np.array(
    [[0, 0, 0, 0, 0, -1./6],
    [ 0, 0, 1, 1, 1,  3./2],
    [ 0, 0, 2, 0, 0, -1./3],
    [ 0, 0, 0, 2, 0, -1./3],
    [ 0, 0, 0, 0, 2, -1./3]], dtype=float)

_l_mat_b2Ct = np.array(
    [[0, 0, 0, 0, 0, -4./21],
    [ 0, 0, 0, 0, 2,  4./21]], dtype=float)


def _to_J(matrix):
    """
    Independently applies J_table to each row of the matrix `matrix` provided.

    Parameters
    ----------
    matrix : 2D-array
        A 2D-array whose rows are the coefficients α, β, l₁, l₂, l and A, in
        this order.

    Returns
    -------
    2D-array
        A 2D-array where each row is the result of `J_table` with the
        coefficient α, β, l₁, l₂, l and A as arguments.

    See also
    --------
    J_table.J_table : Computes (α, β, l1, l2, l, J1, J2, Jk, A, B).
    """
    
    return np.vstack(tuple(J_table(row) for row in matrix))


def _tables_to_J(*table_list):
    """
    Independently applies the `_to_J` function to each list provided via a
    simple Python list comprehension.

    Returns
    -------
    list of 2D-arrays
        List of the matrices resulting from `_to_J` being applied to each
        input matrices.

    See also
    --------
    _to_J : Applies J_table to each row of a matrix.
    J_table.J_table : Computes (α, β, l1, l2, l, J1, J2, Jk, A, B).

    Examples
    --------
    >>> _tables_to_J(table_1, table_2, table_3):
    """    
    
    return [_to_J(table) for table in table_list]


def IA_tidal():
    """
    Outputs four 2D arrays whose rows correspond to the `J_table`
    function applied on each set of coefficients (α, β, l₁, l₂, l, A)
    characterising each kernel.

    Schematically, the ordering is:
    1. bs C1 term
    2. bs C1δ term
    3. bs C2 term
    4. bs bt term
    5. b1 bt term
    All cover the E component: the B counterpart is null in all these cases.
    
    Returns
    -------
    list of 2D-arrays
        Returns, ordered in a list, the the coefficients from `J_table` for each
        of the pieces listed above.
    
    See also
    --------
    J_table.J_table : Computes (α, β, l1, l2, l, J1, J2, Jk, A, B).
    """

    return _tables_to_J(_l_mat_bsC1, _l_mat_bsC1δ, _l_mat_bsC2,
                        _l_mat_bsCt, _l_mat_b1Ct)


def IA_bs():
    """
    Outputs four 2D arrays whose rows correspond to the J_table
    function applied on each set of coefficients (α, β, l₁, l₂, l, A)
    characterising each kernel.

    Schematically, the ordering is:
    1. bs C1 term
    2. bs C1δ term
    3. bs C2 term

    All cover the E component: the B counterpart is null in all these cases.

    Returns
    -------
    list of 2D-arrays
        Returns, ordered in a list, the the coefficients from `J_table` for each
        of the pieces listed above.

    See also
    --------
    J_table.J_table : Computes (α, β, l1, l2, l, J1, J2, Jk, A, B).

    """

    return _tables_to_J(_l_mat_bsC1, _l_mat_bsC1δ, _l_mat_bsC2)


def IA_b2():
    """
    Outputs four 2D arrays whose rows correspond to the J_table
    function applied on each set of coefficients (α, β, l₁, l₂, l, A)
    characterising each kernel.

    Schematically, the ordering is:
    1. b2 C1 term
    2. b2 C1δ term
    3. b2 C2 term
    All cover the E component: the B counterpart is null in all these cases.

    Returns
    -------
    list of 2D-arrays
        Returns, ordered in a list, the the coefficients from `J_table` for each
        of the pieces listed above.

    See also
    --------
    J_table.J_table : Computes (α, β, l1, l2, l, J1, J2, Jk, A, B).
    """
    
    # b2.C1
    # b2.C2
    # b2.C1δ
    return _tables_to_J(_l_mat_b2C1, _l_mat_b2C1δ, _l_mat_b2C2)


def IA_b2bs():
    """
    Outputs four 2D arrays whose rows correspond to the J_table
    function applied on each set of coefficients (α, β, l₁, l₂, l, A)
    characterising each kernel.

    Schematically, the ordering is:
    
    1. b2 C1 term
    2. b2 C1δ term
    3. b2 C2 term
    
    4. bs C1 term
    5. bs C1δ term
    6. bs C2 term
    
    All cover the E component: the B counterpart is null in all these cases.

    Returns
    -------
    list of 2D-arrays
        Returns, ordered in a list, the the coefficients from `J_table` for each
        of the pieces listed above.

    See also
    --------
    J_table.J_table : Computes (α, β, l1, l2, l, J1, J2, Jk, A, B).
    """
    
    return IA_b2() + IA_bs()


def IA_Ct():
    """
    Outputs four 2D arrays whose rows correspond to the J_table
    function applied on each set of coefficients (α, β, l₁, l₂, l, A)
    characterising each kernel.

    Schematically, the ordering is:
    1. bs Ct term
    2. b1 Ct term
    3. b2 Ct term
    All cover the E component: the B counterpart is null in all these cases.

    Returns
    -------
    list of 2D-arrays
        Returns, ordered in a list, the the coefficients from `J_table` for each
        of the pieces listed above.

    See also
    --------
    J_table.J_table : Computes (α, β, l1, l2, l, J1, J2, Jk, A, B).
    """    

    return _tables_to_J(_l_mat_bsCt, _l_mat_b1Ct, _l_mat_b2Ct)

def IA_bsC1_P13like(k, P):

    return P13likePiece("P13_bsC1",
            Zmid = lambda α : ((1/84) * (np.pi * ((α**-5) * ((4 * (α * (15+
                ((-55 * (α**2))+((73 * (α**4))+(15 * (α**6)))))))+(-15 *
                (((-1+(α**2))**4) * np.log((((-1+α)**-2) * ((1+α)**2))))))))),
            Zlow = lambda α : ((128/4851) * (np.pi * ((α**-8) * (-1+((-11 *
                (α**2))+((99 * (α**4))+((-231 * (α**6))+(231 * (α**8))))))))),
            Zhig = lambda α : ((-128/63063) * (np.pi * ((α**2) * (-1287+
                ((143 * (α**2))+((13 * (α**4))+(3 * (α**6))))))))
        ).integrate(k, P, method="fftconvolve").P