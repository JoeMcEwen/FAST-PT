"""
LVDM kernels

The main function here is `LVDM()` which outputs 2D arrays whose rows correspond
to the `J_table` function applied on each set of coefficients
(α, β, l₁, l₂, l, A) characterising each kernel.

Schematically, the ordering is:

     0. ⟨ vvE | vvE ⟩
     1. ⟨ vvB | vvB ⟩
     2. ⟨ vuE | vuE ⟩
     3. ⟨ vuB | vuB ⟩
     4. ⟨ uuE | uuE ⟩
     5. ⟨ uuB | uuB ⟩

     6. ⟨ vvE | vuE ⟩
     7. ⟨ vvB | vuB ⟩
     8. ⟨ vvE | uuE ⟩
     9. ⟨ vvB | uuB ⟩
    10. ⟨ vuE | uuE ⟩
    11. ⟨ vuB | uuB ⟩

    12. ⟨  δ  | vvE ⟩
    13. ⟨  δ  | vuE ⟩
    14. ⟨  δ  | uuE ⟩
    15. ⟨  δ² | vvE ⟩
    16. ⟨  δ² | vuE ⟩
    17. ⟨  δ² | uuE ⟩

    18. ⟨ δ s | vvE ⟩
    19. ⟨ δ s | vvB ⟩
    20. ⟨ δ s | vuE ⟩
    21. ⟨ δ s | vuB ⟩
    22. ⟨ δ s | uuE ⟩
    23. ⟨ δ s | uuB ⟩

    24. ⟨ s s | vvE ⟩
    25. ⟨ s s | vvB ⟩
    26. ⟨ s s | vuE ⟩
    27. ⟨ s s | vuB ⟩
    28. ⟨ s s | uuE ⟩
    29. ⟨ s s | uuB ⟩

The function `LVDM()` depends on the `LVDM_mat()` function which simply returns
a list of 2D arrays whose rows are the (α, β, l₁, l₂, l, A) coefficients for
each piece. The ordering remains the same.

"""

# Scipy modules
import numpy as np
# Local modules
from .J_table import J_table
from .Piece import P13likePiece


def LVDM_mat():
    # Ordering is α, β, l₁, l₂, l, A coefficient

    l_mat_vvE_vvE = np.array(
        [[-2, -2, 0, 0, 0,   7./18],
         [-2, -2, 1, 1, 1, - 7./ 4],
         [-2, -2, 0, 2, 0,  19./36],
         [-2, -2, 2, 0, 0,  19./36],
         [-2, -2, 2, 2, 0,  19./18],
         [-2, -2, 0, 0, 2,   1./ 4]], dtype=float)

    l_mat_vvB_vvB = np.array(
        [[-2, -2, 0, 0, 0,  1./9],
         [-2, -2, 1, 1, 1,  1.  ],
         [-2, -2, 0, 2, 0, -1./9],
         [-2, -2, 2, 0, 0, -1./9],
         [-2, -2, 2, 2, 0, -8./9]], dtype=float)

    l_mat_vuE_vuE = np.array(
        [[-2, -6, 0, 0, 0,   7/ 72],
         [-4, -4, 0, 0, 0,   7/ 36],
         [-6, -2, 0, 0, 0,   7/ 72],
         [-2, -6, 1, 1, 1, - 7/ 16],
         [-4, -4, 1, 1, 1, - 7/  8],
         [-6, -2, 1, 1, 1, - 7/ 16],
         [-2, -6, 0, 2, 0,  19/144],
         [-4, -4, 0, 2, 0,  19/ 72],
         [-6, -2, 0, 2, 0,  19/144],
         [-2, -6, 2, 0, 0,  19/144],
         [-4, -4, 2, 0, 0,  19/ 72],
         [-6, -2, 2, 0, 0,  19/144],
         [-2, -6, 2, 2, 0,  19/ 72],
         [-4, -4, 2, 2, 0,  19/ 36],
         [-6, -2, 2, 2, 0,  19/ 72],
         [-2, -6, 0, 0, 2,   1/ 16],
         [-4, -4, 0, 0, 2,   1/  8],
         [-6, -2, 0, 0, 2,   1/ 16]], dtype=float)

    l_mat_vuB_vuB = np.array(
        [[-2, -6, 0, 0, 0,  1/36],
         [-4, -4, 0, 0, 0,  1/18],
         [-6, -2, 0, 0, 0,  1/36],
         [-2, -6, 1, 1, 1,  1/ 4],
         [-4, -4, 1, 1, 1,  1/ 2],
         [-6, -2, 1, 1, 1,  1/ 4],
         [-2, -6, 0, 2, 0, -1/36],
         [-4, -4, 0, 2, 0, -1/18],
         [-6, -2, 0, 2, 0, -1/36],
         [-2, -6, 2, 0, 0, -1/36],
         [-4, -4, 2, 0, 0, -1/18],
         [-6, -2, 2, 0, 0, -1/36],
         [-2, -6, 2, 2, 0, -2/ 9],
         [-4, -4, 2, 2, 0, -4/ 9],
         [-6, -2, 2, 2, 0, -2/ 9]], dtype=float)

    l_mat_uuE_uuE = np.array(
        [[-6, -6, 0, 0, 0,   7/18],
         [-6, -6, 1, 1, 1, - 7/ 4],
         [-6, -6, 0, 2, 0,  19/36],
         [-6, -6, 2, 0, 0,  19/36],
         [-6, -6, 2, 2, 0,  19/18],
         [-6, -6, 0, 0, 2,   1/ 4]], dtype=float)
    
    # # vvv BIASED VERSION vvv
    l_mat_uuE_uuE = np.array(
        [[-6, -6, 0, 0, 0,   1/18], # <- Modifications here
         [-6, -6, 1, 1, 1, - 7/ 4],
         [-6, -6, 0, 2, 0,  19/36],
         [-6, -6, 2, 0, 0,  19/36],
         [-6, -6, 2, 2, 0,  19/18],
         [-6, -6, 0, 0, 2, - 5/12]], dtype=float) # <- Modifications here

    # TEST #####################################################################
    l_mat_uuE_uuE = np.array(
        [[-6, -6, 0, 0, 0,   7/18],
         [-6, -6, 1, 1, 1, - 7/ 4],
         [-6, -6, 0, 2, 0,  19/36],
         [-6, -6, 2, 0, 0,  19/36],
         [-6, -6, 2, 2, 0,  19/18],
         [-6, -6, 0, 0, 2,   1/ 4]], dtype=float)
    ############################################################################

    l_mat_uuB_uuB = np.array(
        [[-6, -6, 0, 0, 0,  1/9],
         [-6, -6, 1, 1, 1,  1  ],
         [-6, -6, 0, 2, 0, -1/9],
         [-6, -6, 2, 0, 0, -1/9],
         [-6, -6, 2, 2, 0, -8/9]], dtype=float)

    l_mat_vvE_vuE = np.array(
        [[-2, -4, 0, 0, 0, - 7/36],
         [-4, -2, 0, 0, 0, - 7/36],
         [-2, -4, 1, 1, 1,   7/ 8],
         [-4, -2, 1, 1, 1,   7/ 8],
         [-2, -4, 0, 2, 0, -19/72],
         [-4, -2, 0, 2, 0, -19/72],
         [-2, -4, 2, 0, 0, -19/72],
         [-4, -2, 2, 0, 0, -19/72],
         [-2, -4, 2, 2, 0, -19/36],
         [-4, -2, 2, 2, 0, -19/36],
         [-2, -4, 0, 0, 2, - 1/ 8],
         [-4, -2, 0, 0, 2, - 1/ 8]], dtype=float)

    l_mat_vvB_vuB = np.array(
        [[-2, -4, 0, 0, 0,  1/18],
         [-4, -2, 0, 0, 0,  1/18],
         [-2, -4, 1, 1, 1,  1/ 2],
         [-4, -2, 1, 1, 1,  1/ 2],
         [-2, -4, 2, 0, 0, -1/18],
         [-4, -2, 2, 0, 0, -1/18],
         [-2, -4, 0, 2, 0, -1/18],
         [-4, -2, 0, 2, 0, -1/18],
         [-2, -4, 2, 2, 0, -4/ 9],
         [-4, -2, 2, 2, 0, -4/ 9]], dtype=float)

    l_mat_vvE_uuE = np.array(
        [[-4, -4, 0, 0, 0, - 7/18],
         [-4, -4, 1, 1, 1,   7/ 4],
         [-4, -4, 2, 0, 0, -19/36],
         [-4, -4, 0, 2, 0, -19/36],
         [-4, -4, 2, 2, 0, -19/18],
         [-4, -4, 0, 0, 2, - 1/ 4]], dtype=float)

    l_mat_vvB_uuB = np.array(
        [[-4, -4, 0, 0, 0,  1/9],
         [-4, -4, 1, 1, 1,  1  ],
         [-4, -4, 0, 2, 0, -1/9],
         [-4, -4, 2, 0, 0, -1/9],
         [-4, -4, 2, 2, 0, -8/9]], dtype=float)

    l_mat_vuE_uuE = np.array(
        [[-4, -6, 0, 0, 0,   7/36],
         [-6, -4, 0, 0, 0,   7/36],
         [-4, -6, 1, 1, 1, - 7/ 8],
         [-6, -4, 1, 1, 1, - 7/ 8],
         [-4, -6, 0, 2, 0,  19/72],
         [-6, -4, 0, 2, 0,  19/72],
         [-4, -6, 2, 0, 0,  19/72],
         [-6, -4, 2, 0, 0,  19/72],
         [-4, -6, 2, 2, 0,  19/36],
         [-6, -4, 2, 2, 0,  19/36],
         [-4, -6, 0, 0, 2,   1/ 8],
         [-6, -4, 0, 0, 2,   1/ 8]], dtype=float)

    l_mat_vuB_uuB = np.array(
        [[-4, -6, 0, 0, 0,  1/18],
         [-6, -4, 0, 0, 0,  1/18],
         [-4, -6, 1, 1, 1,  1/ 2],
         [-6, -4, 1, 1, 1,  1/ 2],
         [-4, -6, 0, 2, 0, -1/18],
         [-6, -4, 0, 2, 0, -1/18],
         [-4, -6, 2, 0, 0, -1/18],
         [-6, -4, 2, 0, 0, -1/18],
         [-4, -6, 2, 2, 0, -4/ 9],
         [-6, -4, 2, 2, 0, -4/ 9]], dtype=float)

    l_mat_δ_vvE = np.array( #  /!\  P13-like piece missing
        [[-2,  0, 0, 0, 0,   1./12],
         [ 0, -2, 0, 0, 0,   1./12],
         [-1, -1, 1, 1, 0, -17./14],
         [-1, -1, 0, 0, 1,  31./70],
         [-2,  0, 1, 1, 1, - 3./ 4],
         [ 0, -2, 1, 1, 1, - 3./ 4],
         [-2,  0, 0, 0, 2,   1./ 6],
         [ 0, -2, 0, 0, 2,   1./ 6],
         [-1, -1, 1, 1, 2, - 2./ 7],
         [-1, -1, 0, 0, 3,   2./35]], dtype=float)

    l_mat_δ_vuE = np.array( #  /!\  P13-like piece missing
        [[-4,  0, 0, 0, 0, - 1/12],
         [ 0, -4, 0, 0, 0, - 1/12],
         [-2, -2, 0, 0, 0, - 1/ 6],
         [-1, -3, 1, 1, 0,  17/14],
         [-3, -1, 1, 1, 0,  17/14],
         [-1, -3, 0, 0, 1, -31/70],
         [-3, -1, 0, 0, 1, -31/70],
         [-4,  0, 1, 1, 1,   3/ 4],
         [ 0, -4, 1, 1, 1,   3/ 4],
         [-2, -2, 1, 1, 1,   3/ 2],
         [-4,  0, 0, 0, 2, - 1/ 6],
         [ 0, -4, 0, 0, 2, - 1/ 6],
         [-2, -2, 0, 0, 2, - 1/ 3],
         [-1, -3, 1, 1, 2,   2/ 7],
         [-3, -1, 1, 1, 2,   2/ 7],
         [-1, -3, 0, 0, 3, - 2/35],
         [-3, -1, 0, 0, 3, - 2/35]], dtype=float)

    l_mat_δ_uuE = np.array( #  /!\  P13-like piece missing
        [[-2, -4, 0, 0, 0, - 1/12],
         [-4, -2, 0, 0, 0, - 1/12],
         [-3, -3, 1, 1, 0,  17/14],
         [-3, -3, 0, 0, 1, -31/70],
         [-2, -4, 1, 1, 1,   3/ 4],
         [-4, -2, 1, 1, 1,   3/ 4],
         [-2, -4, 0, 0, 2, - 1/ 6],
         [-4, -2, 0, 0, 2, - 1/ 6],
         [-3, -3, 1, 1, 2,   2/ 7],
         [-3, -3, 0, 0, 3, - 2/35]], dtype=float)

    l_mat_δ2_vvE = np.array(
        [[-1, -1, 1, 1, 0, -3/2],
         [-1, -1, 0, 0, 1,  1/2]], dtype=float)

    l_mat_δ2_vuE = np.array(
        [[-1, -3, 1, 1, 0,  3/4],
         [-3, -1, 1, 1, 0,  3/4],
         [-1, -3, 0, 0, 1, -1/4],
         [-3, -1, 0, 0, 1, -1/4]], dtype=float)

    l_mat_δ2_uuE = np.array(
        [[-3, -3, 1, 1, 0,  3/2],
         [-3, -3, 0, 0, 1, -1/2]], dtype=float)

    l_mat_δ_s_vvE = np.array(
        [[-1, -1, 1, 1, 0, -11/20],
         [-1, -1, 0, 0, 1, - 1/12],
         [-1, -1, 0, 2, 1,   7/12],
         [-1, -1, 1, 3, 0, -19/20]], dtype=float)

    l_mat_δ_s_vvB = np.array(
        [[-1, -1, 1, 1, 0, -1/5],
         [-1, -1, 0, 0, 1,  1/3],
         [-1, -1, 0, 2, 1,  2/3],
         [-1, -1, 1, 3, 0, -4/5]], dtype=float)

    l_mat_δ_s_vuE = np.array(
        [[-1, -3, 1, 1, 0,  11/40],
         [-3, -1, 1, 1, 0,  11/40],
         [-1, -3, 0, 0, 1,   1/24],
         [-3, -1, 0, 0, 1,   1/24],
         [-1, -3, 0, 2, 1, - 7/24],
         [-3, -1, 0, 2, 1, - 7/24],
         [-1, -3, 1, 3, 0,  19/40],
         [-3, -1, 1, 3, 0,  19/40]], dtype=float)

    l_mat_δ_s_vuB = np.array(
        [[-1, -3, 1, 1, 0, -1/10],
         [-3, -1, 1, 1, 0, -1/10],
         [-1, -3, 0, 0, 1,  1/ 6],
         [-3, -1, 0, 0, 1,  1/ 6],
         [-1, -3, 0, 2, 1,  1/ 3],
         [-3, -1, 0, 2, 1,  1/ 3],
         [-1, -3, 1, 3, 0, -2/ 5],
         [-3, -1, 1, 3, 0, -2/ 5]], dtype=float)

    l_mat_δ_s_uuE = np.array(
        [[-3, -3, 1, 1, 0,  11/20],
         [-3, -3, 0, 0, 1,   1/12],
         [-3, -3, 2, 0, 1, - 7/12],
         [-3, -3, 3, 1, 0,  19/20]], dtype=float)

    l_mat_δ_s_uuB = np.array(
        [[-3, -3, 1, 1, 0, -1/5],
         [-3, -3, 0, 0, 1,  1/3],
         [-3, -3, 0, 2, 1,  2/3],
         [-3, -3, 1, 3, 0, -4/5]], dtype=float)

    l_mat_s_s_vvE = np.array(
        [[-1, -1, 1, 1, 0,   7/10],
         [-1, -1, 0, 0, 1, - 1/ 3],
         [-1, -1, 0, 2, 1, -17/36],
         [-1, -1, 2, 0, 1, -17/36],
         [-1, -1, 2, 2, 1, -19/18],
         [-1, -1, 1, 1, 2,   2/ 3],
         [-1, -1, 1, 3, 0,  19/60],
         [-1, -1, 3, 1, 0,  19/60]], dtype=float)

    l_mat_s_s_vvB = np.array(
        [[-1, -1, 1, 1, 0,  7/15],
         [-1, -1, 0, 0, 1, -1/ 9],
         [-1, -1, 0, 2, 1, -1/ 3],
         [-1, -1, 2, 0, 1, -1/ 3],
         [-1, -1, 2, 2, 1, -8/ 9],
         [-1, -1, 1, 1, 2,  2/ 3],
         [-1, -1, 1, 3, 0,  4/15],
         [-1, -1, 3, 1, 0,  4/15]], dtype=float)

    l_mat_s_s_vuE = np.array(
        [[-1, -3, 1, 1, 0, - 7/ 20],
         [-3, -1, 1, 1, 0, - 7/ 20],
         [-1, -3, 0, 0, 1,   1/  6],
         [-3, -1, 0, 0, 1,   1/  6],
         [-1, -3, 0, 2, 1,  17/ 72],
         [-3, -1, 0, 2, 1,  17/ 72],
         [-1, -3, 2, 0, 1,  17/ 72],
         [-3, -1, 2, 0, 1,  17/ 72],
         [-1, -3, 2, 2, 1,  19/ 36],
         [-3, -1, 2, 2, 1,  19/ 36],
         [-1, -3, 1, 1, 2, - 1/  3],
         [-3, -1, 1, 1, 2, - 1/  3],
         [-1, -3, 1, 3, 0, -19/120],
         [-3, -1, 1, 3, 0, -19/120],
         [-1, -3, 3, 1, 0, -19/120],
         [-3, -1, 3, 1, 0, -19/120]], dtype=float)

    l_mat_s_s_vuB = np.array(
        [[-1, -3, 1, 1, 0,  7/30],
         [-3, -1, 1, 1, 0,  7/30],
         [-1, -3, 0, 0, 1, -1/18],
         [-3, -1, 0, 0, 1, -1/18],
         [-1, -3, 0, 2, 1, -1/ 6],
         [-3, -1, 0, 2, 1, -1/ 6],
         [-1, -3, 2, 0, 1, -1/ 6],
         [-3, -1, 2, 0, 1, -1/ 6],
         [-1, -3, 2, 2, 1, -4/ 9],
         [-3, -1, 2, 2, 1, -4/ 9],
         [-1, -3, 1, 1, 2,  1/ 3],
         [-3, -1, 1, 1, 2,  1/ 3],
         [-1, -3, 1, 3, 0,  2/15],
         [-3, -1, 1, 3, 0,  2/15],
         [-1, -3, 3, 1, 0,  2/15],
         [-3, -1, 3, 1, 0,  2/15]], dtype=float)

    l_mat_s_s_uuE = np.array(
        [[-3, -3, 1, 1, 0, - 7/10],
         [-3, -3, 0, 0, 1,   1/ 3],
         [-3, -3, 0, 2, 1,  17/36],
         [-3, -3, 2, 0, 1,  17/36],
         [-3, -3, 2, 2, 1,  19/18],
         [-3, -3, 1, 1, 2, - 2/ 3],
         [-3, -3, 1, 3, 0, -19/60],
         [-3, -3, 3, 1, 0, -19/60]], dtype=float)

    l_mat_s_s_uuB = np.array(
        [[-3, -3, 1, 1, 0,  7/15],
         [-3, -3, 0, 0, 1, -1/ 9],
         [-3, -3, 0, 2, 1, -1/ 3],
         [-3, -3, 2, 0, 1, -1/ 3],
         [-3, -3, 2, 2, 1, -8/ 9],
         [-3, -3, 1, 1, 2,  2/ 3],
         [-3, -3, 1, 3, 0,  4/15],
         [-3, -3, 3, 1, 0,  4/15]], dtype=float)

    return l_mat_vvE_vvE, l_mat_vvB_vvB, \
           l_mat_vuE_vuE, l_mat_vuB_vuB, \
           l_mat_uuE_uuE, l_mat_uuB_uuB, \
           l_mat_vvE_vuE, l_mat_vvB_vuB, \
           l_mat_vvE_uuE, l_mat_vvB_uuB, \
           l_mat_vuE_uuE, l_mat_vuB_uuB, \
           l_mat_δ2_vvE,  l_mat_δ2_vuE,  l_mat_δ2_uuE, \
           l_mat_δ_vvE,   l_mat_δ_vuE,   l_mat_δ_uuE, \
           l_mat_δ_s_vvE, l_mat_δ_s_vvB, \
           l_mat_δ_s_vuE, l_mat_δ_s_vuB, \
           l_mat_δ_s_uuE, l_mat_δ_s_uuB, \
           l_mat_s_s_vvE, l_mat_s_s_vvB, \
           l_mat_s_s_vuE, l_mat_s_s_vuB, \
           l_mat_s_s_uuE, l_mat_s_s_uuB


def LVDM():
    """
    Outputs 2D arrays whose rows correspond
    to the `J_table` function applied on each set of coefficients
    (α, β, l₁, l₂, l, A) characterising each kernel.

    Schematically, the ordering is:

        0. ⟨ vvE | vvE ⟩
        1. ⟨ vvB | vvB ⟩
        2. ⟨ vuE | vuE ⟩
        3. ⟨ vuB | vuB ⟩
        4. ⟨ uuE | uuE ⟩
        5. ⟨ uuB | uuB ⟩

        6. ⟨ vvE | vuE ⟩
        7. ⟨ vvB | vuB ⟩
        8. ⟨ vvE | uuE ⟩
        9. ⟨ vvB | uuB ⟩
        10. ⟨ vuE | uuE ⟩
        11. ⟨ vuB | uuB ⟩

        12. ⟨  δ  | vvE ⟩
        13. ⟨  δ  | vuE ⟩
        14. ⟨  δ  | uuE ⟩
        15. ⟨  δ² | vvE ⟩
        16. ⟨  δ² | vuE ⟩
        17. ⟨  δ² | uuE ⟩

        18. ⟨ δ s | vvE ⟩
        19. ⟨ δ s | vvB ⟩
        20. ⟨ δ s | vuE ⟩
        21. ⟨ δ s | vuB ⟩
        22. ⟨ δ s | uuE ⟩
        23. ⟨ δ s | uuB ⟩

        24. ⟨ s s | vvE ⟩
        25. ⟨ s s | vvB ⟩
        26. ⟨ s s | vuE ⟩
        27. ⟨ s s | vuB ⟩
        28. ⟨ s s | uuE ⟩
        29. ⟨ s s | uuB ⟩

    See `J_table.J_table(...)`
    """

    to_J = lambda matrix: np.vstack([J_table(row) for row in matrix])
    # print(to_J(LVDM_mat()[4])) # uuE uuE  TODO  Provisoire
    return (to_J(mat) for mat in LVDM_mat())

    # table_vvE_vvE = to_J(l_mat_vvE_vvE)
    # table_vvB_vvB = to_J(l_mat_vvB_vvB)
    # table_vuE_vuE = to_J(l_mat_vuE_vuE)
    # table_vuB_vuB = to_J(l_mat_vuB_vuB)
    # table_uuE_uuE = to_J(l_mat_uuE_uuE)
    # table_uuB_uuB = to_J(l_mat_uuB_uuB)

    # table_vvE_vuE = to_J(l_mat_vvE_vuE)
    # table_vvB_vuB = to_J(l_mat_vvB_vuB)
    # table_vvE_uuE = to_J(l_mat_vvE_uuE)
    # table_vvB_uuB = to_J(l_mat_vvB_uuB)
    # table_vuE_uuE = to_J(l_mat_vuE_uuE)
    # table_vuB_uuB = to_J(l_mat_vuB_uuB)

    # table_δ_vvE = to_J(l_mat_δ_vvE)
    # table_δ_vuE = to_J(l_mat_δ_vuE)
    # table_δ_uuE = to_J(l_mat_δ_uuE)
    
    # table_δ2_vvE = to_J(l_mat_δ2_vvE)
    # table_δ2_vuE = to_J(l_mat_δ2_vuE)
    # table_δ2_uuE = to_J(l_mat_δ2_uuE)

    # table_δ_s_vvE = to_J(l_mat_δ_s_vvE)
    # table_δ_s_vvB = to_J(l_mat_δ_s_vvB)
    # table_δ_s_vuE = to_J(l_mat_δ_s_vuE)
    # table_δ_s_vuB = to_J(l_mat_δ_s_vuB)
    # table_δ_s_uuE = to_J(l_mat_δ_s_uuE)
    # table_δ_s_uuB = to_J(l_mat_δ_s_uuB)
    # table_s_s_vvE = to_J(l_mat_s_s_vvE)
    # table_s_s_vvB = to_J(l_mat_s_s_vvB)
    # table_s_s_vuE = to_J(l_mat_s_s_vuE)
    # table_s_s_vuB = to_J(l_mat_s_s_vuB)
    # table_s_s_uuE = to_J(l_mat_s_s_uuE)
    # table_s_s_uuB = to_J(l_mat_s_s_uuB)

    # return table_vvE_vvE, table_vvB_vvB, \
    #        table_vuE_vuE, table_vuB_vuB, \
    #        table_uuE_uuE, table_uuB_uuB, \
    #        table_vvE_vuE, table_vvB_vuB, \
    #        table_vvE_uuE, table_vvB_uuB, \
    #        table_vuE_uuE, table_vuB_uuB, \
    #        table_δ2_vvE, table_δ2_vuE, table_δ2_uuE, \
    #        table_δ_vvE, table_δ_vuE, table_δ_uuE, \
    #        table_δ_s_vvE, table_δ_s_vvB, \
    #        table_δ_s_vuE, table_δ_s_vuB, \
    #        table_δ_s_uuE, table_δ_s_uuB, \
    #        table_s_s_vvE, table_s_s_vvB, \
    #        table_s_s_vuE, table_s_s_vuB, \
    #        table_s_s_uuE, table_s_s_uuB

def LVDM_P13like(k, P):
    P13like = []

    P13like.append( P13likePiece("P13_LVDM_d_vvE",
        Zmid = lambda α : (4*α * (-15 + 107*α**2 - 105*α**4 + 45*α**6)
                        - 15 * (-1 + α**2)**3 * (1 + 3*α**2)
                        * np.log((1 + α)**2 / (-1 + α)**2)) / (2688*α**5),
        Zlow = lambda α : 10/(4851*α**8) + 2/(147*α**6) - 2/(49*α**4) + 1/(14*α**2),
        Zhig = lambda α : 1/(6*α**2) - 2/7 + (10*α**2)/49 - (2*α**4)/63 
                        - (2*α**6)/539 - (2*α**8)/1911
    ) )

    P13like.append( P13likePiece("P13_LVDM_d_vuE",
        Zmid = lambda α : (4*α * (15 - 86*α**2 + 3*α**4 + 72*α**6) \
                        - 3 * (5 + 7*α**2 - 29*α**4 - 7*α**6 + 24*α**8) * np.log((1 + α)**2 / (-1 + α)**2)) \
                        / (2688*α**7),
        Zlow = lambda α : -71 / (48510*α**10) - 1 / (70*α**8) + 37 / (1470*α**6) - 1 / (105*α**4),
        Zhig = lambda α : 253 / 1470 - 1 / (6*α**4) + 5 / (42*α**2) - (7*α**2) / 90,
        k_correction = lambda kk: 1/(kk**4)
    ) )

    P13like.append( P13likePiece("P13_LVDM_d_uuE",
        Zmid = lambda α : -(-7 + 13*α**2) * \
                        (4*α - 12*α**3 + (-1 - 2*α**2 + 3*α**4) * np.log((1 + α)**2 / (-1 + α)**2)) \
                        / (448*α**7),
        Zlow = lambda α : 29 / (24255*α**12) - 1 / (735*α**10) - 23 / (735*α**8) + 13 / (105*α**6),
        Zhig = lambda α : -23 / 105 - 1 / (3*α**4) + 79 / (105*α**2) - 89*α**2 / 2205,
        k_correction = lambda kk: 1/(kk**6)
    ) )

    return (p13.integrate(k, P, method="fftconvolve").P for p13 in P13like)