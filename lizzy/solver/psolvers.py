#  Copyright 2025-2025 Simone Bancora, Paris Mulye
#
#  This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#  You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from enum import Enum, auto
from lizzy.solver.builtin.direct_solvers import *

class SolverType(Enum):
    DIRECT_DENSE = auto()
    DIRECT_SPARSE = auto()
    DIRECT_DENSE_GPU = auto()
    DIRECT_SPARSE_GPU = auto()

class PressureSolver:
    @staticmethod
    def solve(k:torch.tensor, f:torch.tensor, method:SolverType = SolverType.DIRECT_SPARSE):
        """
        Solve the system `K p = f`.

        Parameters
        ----------
        k : np.ndarray
            Stiffness matrix with Dirichlet BCs applied by row/column method. Dimension (N,N) where N is the number of Dofs (1 per node)
        f : np.ndarray
            Right-hand side vector. Zeros (no source yet) with ones in Dirichlet nodes. Dimension (N,1)
        method : SolverType
            The solver type to be used. Default is SPARSE.
        """
        match method:
            case SolverType.DIRECT_DENSE:
                p = solve_pressure_direct_dense(k, f)
            case SolverType.DIRECT_SPARSE:
                p = solve_pressure_direct_sparse(k, f)
            case _:
                raise ValueError(f"Unknown solver type: {method}")
        return p

    @staticmethod
    def apply_bcs(k, f, bcs):
        dirichlet_idx_full = torch.cat((bcs.dirichlet_idx, bcs.p0_idx), axis=0)
        dirichlet_vals_full = torch.cat((bcs.dirichlet_vals, torch.zeros( len(bcs.p0_idx))), axis=0)

        k_modified = k.copy()
        f_modified = f.copy()

        k_modified[dirichlet_idx_full, :] = 0
        k_modified[dirichlet_idx_full,dirichlet_idx_full]=1
        f_modified[dirichlet_idx_full] = dirichlet_vals_full
        # apply bcs
        # for i, node_id in enumerate(dirichlet_idx_full):
        #     k_modified[node_id, :] = 0
        #     k_modified[node_id, node_id] = 1
        #     f_modified[node_id] = dirichlet_vals_full[i]
        return k_modified, f_modified





