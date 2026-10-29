"""The truth surface everywhere a phase 6 flight can go (PLAN.md phase 6, D5).

Phase 5's truth surface (truthsurface.truth: the 5 m mosaic, bilinear on a
0.05 m raster, plus a seeded rock field) covers one 150 m square around the
target. A stale map or no map can put the scan and the touchdown hundreds of
metres from there, so the ground is tiled: tile (i, j) is the 150 m square
centred i squares east and j squares north of the target, made the same way,
its rocks seeded by the draw and the tile, generated only when a flight
first touches it. Tile (0, 0) is phase 5's square with phase 5's seed
[seed, draw, 0], so its rock field is phase 5's exactly; every other tile is
seeded [seed, draw, 0, TILE_KEY + i, TILE_KEY + j]. The ground does not
depend on the gap or the landing epoch, so every flight of one draw sees the
same ground.

The tiles share one 0.05 m cell grid. A height is read from the tile whose
cell centres surround the point (truthsurface.Surface.height, so inside the
target's square heights are phase 5's); in the half-cell seam between two
tiles, bilinear between the neighbouring tiles' cells. A rock is drawn in its
own tile only: one centred near a tile's edge is cut there (as phase 5's are
at its square's edge).

The region is a square of half_tiles tiles each side of the target's tile;
a point outside it raises.
"""
import numpy as np

from . import truthsurface as TS

TILE_KEY = 1000


class Ground:
    def __init__(self, path, cx, cy, seed, draw, half_tiles, size=TS.SIZE, cell=TS.CELL, sfd=TS.SFD):
        self.path, self.cx, self.cy = path, float(cx), float(cy)
        self.seed, self.draw, self.half = int(seed), int(draw), int(half_tiles)
        self.size, self.cell, self.sfd = size, cell, sfd
        self.n = int(round(size / cell))
        self.tiles, self.rocks = {}, {}

    def key(self, i, j):
        if (i, j) == (0, 0):
            return [self.seed, self.draw, 0]
        return [self.seed, self.draw, 0, TILE_KEY + i, TILE_KEY + j]

    def tile(self, i, j):
        if max(abs(i), abs(j)) > self.half:
            raise ValueError(f"tile ({i}, {j}) is outside the ground's region of {self.half} tiles each side")
        if (i, j) not in self.tiles:
            s, rocks = TS.truth(self.path, self.cx + i * self.size, self.cy + j * self.size, self.key(i, j),
                                self.size, self.cell, self.sfd)
            self.tiles[(i, j)], self.rocks[(i, j)] = s, rocks
        return self.tiles[(i, j)]

    def tile_of(self, x, y):
        """The tile holding map point (x, y)."""
        i = int(np.floor((x - (self.cx - self.size / 2)) / self.size))
        j = int(np.floor((y - (self.cy - self.size / 2)) / self.size))
        return i, j

    def _z(self, rg, cg):
        """Heights at global cell (row from the target tile's north edge,
        column from its west edge)."""
        ti, tj = np.floor_divide(cg, self.n), -np.floor_divide(rg, self.n)
        out = np.empty(rg.shape)
        for i, j in set(zip(ti.tolist(), tj.tolist())):
            m = (ti == i) & (tj == j)
            out[m] = self.tile(i, j).z[rg[m] + j * self.n, cg[m] - i * self.n]
        return out

    def height(self, x, y):
        x = np.atleast_1d(np.asarray(x, float))
        y = np.atleast_1d(np.asarray(y, float))
        xf, yf = x.ravel(), y.ravel()
        flat = np.full(xf.shape, np.nan)
        ti = np.floor((xf - (self.cx - self.size / 2)) / self.size).astype(np.int64)
        tj = np.floor((yf - (self.cy - self.size / 2)) / self.size).astype(np.int64)
        for i, j in set(zip(ti.tolist(), tj.tolist())):
            m = (ti == i) & (tj == j)
            flat[m] = self.tile(i, j).height(xf[m], yf[m])
        seam = ~np.isfinite(flat)
        if seam.any():
            fc = (xf[seam] - (self.cx - self.size / 2)) / self.cell - 0.5
            fr = ((self.cy + self.size / 2) - yf[seam]) / self.cell - 0.5
            c0, r0 = np.floor(fc).astype(np.int64), np.floor(fr).astype(np.int64)
            tc, tr = fc - c0, fr - r0
            flat[seam] = ((1 - tr) * ((1 - tc) * self._z(r0, c0) + tc * self._z(r0, c0 + 1))
                          + tr * ((1 - tc) * self._z(r0 + 1, c0) + tc * self._z(r0 + 1, c0 + 1)))
        return flat.reshape(x.shape)

    @property
    def z_max(self):
        """The highest point of the tiles made so far."""
        return max(s.z_max for s in self.tiles.values())
