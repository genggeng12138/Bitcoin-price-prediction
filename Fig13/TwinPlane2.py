import numpy as np

from numpy import linalg
from cvxopt import solvers, matrix



def Twin_plane_2(H, Y, C2, Epsi2, regulz2):
    r, c = H.shape

    HtH = np.dot(H.T, H)
    HtH = HtH + regulz2 * (np.identity(HtH.shape[0]))

    HtH_i_Ht = linalg.solve(HtH, H.T)

    H_HtH_i_Ht = np.dot(H, HtH_i_Ht)
    H_HtH_i_Ht = (H_HtH_i_Ht + (H_HtH_i_Ht.T)) / 2

    e = np.ones((r, 1))

    h = Y + np.dot(e, Epsi2)

    ht = h.T

    ht_H_HtH_i_Ht = np.dot(ht, H_HtH_i_Ht)

    ht_H_HtH_i_Ht_ht = ht_H_HtH_i_Ht - ht

    q = ht_H_HtH_i_Ht_ht.T

    solvers.options['show_progress'] = False
    vlb = np.zeros((r, 1))
    vub = C2 * (np.ones((r, 1)))
    # x<=vub
    # x>=vlb -> -x<=-vlb
    # cdx<=vcd
    cd = np.vstack((np.identity(r), -np.identity(r)))
    vcd = np.vstack((vub, -vlb))
    alpha = solvers.qp(matrix(H_HtH_i_Ht, tc='d'), matrix(q, tc='d'), matrix(cd, tc='d'),
                       matrix(vcd, tc='d'))  # ,matrix(0.0,(1,m1)),matrix(0.0))#,None,matrix(x0))

    alphasol = np.array(alpha['x'])

    HtH_rI = HtH + np.dot(regulz2, np.eye(c))

    u2 = np.dot(linalg.solve(HtH_rI, H.T), (h + alphasol))

    w2 = u2[:len(u2) - 1]
    b2 = u2[len(u2) - 1]
    return [w2, b2]
