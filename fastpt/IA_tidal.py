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

See IA_tidal.IA_tidal()
"""

from copy import deepcopy

import numpy as np

from .J_table import J_table

def IA_tidal():
    """
    Outputs four 2D arrays whose rows correspond to the J_table
    function applied on each set of coefficients (α, β, l₁, l₂, l, A)
    characterising each kernel.

    Schematically, the ordering is:
    1. bs C1 term
    2. bs C1δ term
    3. bs C2 term
    4. bs bt term
    5. b1 bt term
    All cover the E component: the B counterpart is null in all these cases.

    See J_table.J_table(...)
    """

    # Ordering is α, β, l₁, l₂, l, A coefficient

    # Terms proportional to bs·C1:
    l_mat_C1 = np.array(
        [[ 0,  0, 0, 0, 0,   8./315],
        [  1, -1, 0, 0, 1,   4./15 ],
        # [  1, -1, 1, 0, 0,   2./15 ],
        # [ -1,  1, 1, 0, 0,   2./15 ],
        [  0,  0, 0, 0, 2, 254./441],
        [  1, -1, 0, 0, 3,   2./5  ], # Equivalent to the two lines commented
                                      # because of symmetry
        # [  1, -1, 3, 0, 0,   1./5  ],
        # [ -1,  1, 3, 0, 0,   1./5  ],
        [  0,  0, 0, 0, 4,  16./245]], dtype=float)
    # The two matrices are the same in this case. Only the fEk and fBk will affect
    # the result, hence:
    # l_mat_B = deepcopy(l_mat_E)

    # Terms proportional to bs·C1δ:
    l_mat_C1δ = np.array([[0, 0, 0, 2, 2, 2./3]], dtype=float)
    # The B component is null, so:
    # l_mat_Bδ = np.array([[0, 0, 0, 0, 0, 0]], dtype=float)

    l_mat_C2 = np.array(
        [[0, 0, 0, 0, 0, - 2./45],
        [ 0, 0, 1, 1, 1,   2./ 5],
        [ 0, 0, 0, 0, 2, -11./63],
        # [ 0, 0, 0, 2, 2, - 4./ 9],
        [ 0, 0, 0, 2, 2, - 2./ 9],
        [ 0, 0, 2, 0, 2, - 2./ 9],
        [ 0, 0, 1, 1, 3,   3./ 5],
        [ 0, 0, 0, 0, 4, - 4./35]], dtype=float)

    l_mat_bt = np.array(
        [[0, 0, 0, 0, 0,   8./315],
        [ 0, 0, 0, 0, 2, -40./441],
        [ 0, 0, 0, 0, 4,  16./245]], dtype=float)

    l_mat_b1bt = np.array(
        [[0,  0, 0, 0, 0, -36./ 245],
        [ 1, -1, 0, 0, 1, - 4./  35],
        [ 0,  0, 0, 0, 2,  44./ 343],
        [ 1, -1, 0, 0, 3,   4./  35],
        [ 0,  0, 0, 0, 4,  32./1715]], dtype=float)

    # l_mat_A00E = np.array([[0,0,0,2,0,17./21],\
    #       [0,0,0,2,2,4./21],\
    #       [1,-1,0,2,1,1./2],\
    #       [-1,1,0,2,1,1./2]], dtype=float)

    to_J = lambda matrix: np.vstack(tuple(J_table(row) for row in matrix))

    tableC1 = to_J(l_mat_C1)

    tableC1δ = to_J(l_mat_C1δ)

    tableC2 = to_J(l_mat_C2)

    tablebt = to_J(l_mat_bt)

    tableb1bt = to_J(l_mat_b1bt)

    # tableA00E = to_J(l_mat_A00E)

    return tableC1, tableC1δ, tableC2, tablebt, tableb1bt
