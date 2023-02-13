#!/usr/bin/env python

import numpy as np

from numpy import linalg
from cvxopt import solvers, matrix


# __copyright__ = ""
# __license__ = "GPL"
# __version__ = "1.1"
# __maintainer__ = "Arnav Kansal"
# __email__ = "ee1130440@ee.iitd.ac.in"
# __status__ = "Production"


def Twin_plane_1(H, Y, C1, Epsi1, regulz1):
    r, c = H.shape

    HtH = np.dot(H.T, H)
    HtH = HtH + regulz1 * (np.identity(HtH.shape[0]))

    HtH_i_Ht = linalg.solve(HtH, H.T)

    H_HtH_i_Ht = np.dot(H, HtH_i_Ht)
    H_HtH_i_Ht = (H_HtH_i_Ht + (H_HtH_i_Ht.T)) / 2

    e = np.ones((r, 1))

    f = Y - (np.dot(e, Epsi1))

    ft = f.T

    ft_H_HtH_i_Ht = np.dot(ft, H_HtH_i_Ht)

    ft_ft_H_HtH_i_Ht = ft - ft_H_HtH_i_Ht

    q = ft_ft_H_HtH_i_Ht.T

    solvers.options['show_progress'] = False
    vlb = np.zeros((r, 1))
    vub = C1 * (np.ones((r, 1)))
    # x<=vub
    # x>=vlb -> -x<=-vlb
    # cdx<=vcd
    cd = np.vstack((np.identity(r), -np.identity(r)))
    vcd = np.vstack((vub, -vlb))
    alpha = solvers.qp(matrix(H_HtH_i_Ht, tc='d'), matrix(q, tc='d'), matrix(cd, tc='d'),
                       matrix(vcd, tc='d'))  # ,matrix(0.0,(1,m1)),matrix(0.0))#,None,matrix(x0))

    alphasol = np.array(alpha['x'])

    HtH_rI = HtH + np.dot(regulz1, np.eye(c))

    u1 = np.dot(linalg.solve(HtH_rI, H.T), (f - alphasol))

    w1 = u1[:len(u1) - 1]
    b1 = u1[len(u1) - 1]
    return [w1, b1]