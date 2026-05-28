"""Custom Eryn RJ move for the (p, tau) model."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
from eryn.moves import GroupStretchMove, ReversibleJumpMove


class TauGroupStretchMove(GroupStretchMove):
    """Group stretch move using tau-coordinates from active leaves as friends."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.coords_friends = None

    def setup_friends(self, branches):
        self.coords_friends = branches["components"].coords[branches["components"].inds]

    def find_friends(self, name, s, s_inds, branch_supps=None):
        if self.coords_friends is None or self.coords_friends.shape[0] == 0:
            raise ValueError("No available friends in group proposal.")

        friends = np.zeros_like(s)
        coords_here = s[s_inds]
        n_idx = coords_here.shape[0]
        if n_idx == 0:
            return friends

        # Distances in tau-space (ndim=1).
        dist = np.abs(coords_here[:, 0, None] - self.coords_friends[:, 0][None, :])
        nfriends_eff = min(int(self.nfriends), self.coords_friends.shape[0])
        inds_choice = np.random.randint(0, nfriends_eff, size=n_idx)

        keep = np.empty(n_idx, dtype=np.int64)
        for i in range(n_idx):
            dist_row = dist[i]
            part = np.argpartition(dist_row, nfriends_eff - 1)[:nfriends_eff]
            part_sorted = part[np.argsort(dist_row[part])]
            keep[i] = part_sorted[inds_choice[i]]

        friends[s_inds] = self.coords_friends[keep]
        return friends


class TauRJMove(ReversibleJumpMove):
    """Birth/death RJ move that proposes new tau values from the prior."""

    def __init__(self, prior, *args, **kwargs):
        self.prior = prior
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_model_change_proposal(
        inds: np.ndarray,
        change: np.ndarray,
        nleaves_min: int,
        nleaves_max: int,
        random,
    ):
        ntemps, nwalkers = inds.shape[:2]
        nleaves = inds.sum(axis=-1)

        change = (
            change * ((nleaves != nleaves_min) & (nleaves != nleaves_max))
            + (+1) * (nleaves == nleaves_min)
            + (-1) * (nleaves == nleaves_max)
        )

        births = []
        deaths = []

        for t in range(ntemps):
            for w in range(nwalkers):
                c = int(change[t, w])
                inds_tw = inds[t, w]

                if c == +1:
                    inactive = np.where(~inds_tw)[0]
                    leaf = int(random.choice(inactive))
                    births.append((t, w, leaf, int(nleaves[t, w])))
                elif c == -1:
                    active = np.where(inds_tw)[0]
                    leaf = int(random.choice(active))
                    deaths.append((t, w, leaf, int(nleaves[t, w])))

        births_arr = np.asarray(births, dtype=np.int64).reshape(-1, 4)
        deaths_arr = np.asarray(deaths, dtype=np.int64).reshape(-1, 4)
        return births_arr, deaths_arr

    def get_proposal(
        self,
        all_coords,
        all_inds,
        nleaves_min_all,
        nleaves_max_all,
        random,
        **kwargs,
    ):
        inds = all_inds["components"]
        nleaves_min = nleaves_min_all["components"]
        nleaves_max = nleaves_max_all["components"]

        ntemps, nwalkers = inds.shape[:2]
        nleaves = inds.sum(axis=-1)
        q = deepcopy(all_coords)
        new_inds = deepcopy(all_inds)
        factors = np.zeros((ntemps, nwalkers), dtype=float)

        if nleaves_min == nleaves_max:
            return q, new_inds, factors

        if self.fix_change is None:
            change = random.choice(np.array([-1, +1]), size=nleaves.shape)
        else:
            change = np.full(nleaves.shape, int(self.fix_change), dtype=int)

        births_arr, deaths_arr = self.get_model_change_proposal(
            inds,
            change,
            nleaves_min,
            nleaves_max,
            random,
        )

        if deaths_arr.size > 0:
            inds_death = tuple(deaths_arr[:, :-1].T)
            new_inds["components"][inds_death] = False

            # Death proposal factor:
            # log [ r_birth_reverse / r_delete_forward ], where
            # r_delete_forward=1/p and r_birth_reverse=1/(nleaves_max-(p-1)).
            # NOTE: Eryn's ReversibleJumpMove already handles the edge-direction
            # asymmetry terms (q_birth/q_death near nleaves_min/nleaves_max).
            for row in deaths_arr:
                t, w, _, p_now = map(int, row)
                p_new = p_now - 1
                r_birth_reverse = 1.0 / (nleaves_max - p_new)
                r_delete_forward = 1.0 / p_now
                factors[t, w] += np.log(r_birth_reverse)
                factors[t, w] -= np.log(r_delete_forward)

            factors[inds_death[:2]] += self.prior.logpdf_components(
                q["components"][inds_death]
            )

        if births_arr.size > 0:
            inds_birth = tuple(births_arr[:, :-1].T)
            new_inds["components"][inds_birth] = True

            num_birth = len(inds_birth[0])
            q["components"][inds_birth] = self.prior.rvs(num_birth)

            # Birth proposal factor:
            # log [ r_delete_reverse / r_birth_forward ], where
            # r_birth_forward=1/(nleaves_max-p) and r_delete_reverse=1/(p+1).
            # NOTE: edge-direction asymmetry is added by Eryn internally.
            for row in births_arr:
                t, w, _, p_now = map(int, row)
                p_new = p_now + 1
                r_delete_reverse = 1.0 / p_new
                r_birth_forward = 1.0 / (nleaves_max - p_now)
                factors[t, w] += np.log(r_delete_reverse)
                factors[t, w] -= np.log(r_birth_forward)

            factors[inds_birth[:2]] += -self.prior.logpdf_components(
                q["components"][inds_birth]
            )

        return q, new_inds, factors
