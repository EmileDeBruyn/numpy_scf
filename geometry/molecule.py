#
# Python implementation of the Crawford Lab @ Virginia Tech's c++ molecular
# programming project found at
#
# http://sirius.chem.vt.edu/wiki/doku.php?id=crawdad:programming:project1
# or
# https://github.com/CrawfordGroup/ProgrammingProjects/tree/master/Project%2301
#
# Licence originally under: http://creativecommons.org/licenses/by-nc-sa/4.0/
# Linear algebra implemented using numpy and some optional optimizations
# using the numba JIT-compiler.
#

import numpy as np
import numpy.linalg as la
from numba import jit, njit
from numba import int64, float64
import pandas as pd
import itertools as it
from my_elements import tbl as el_tbl
import scipy.constants as sc


@jit
def _tens_fo(masses, coords, ax1, ax2):
    arr = np.empty(coords.shape)
    arr[:, :2] = coords[:, [ax1, ax2]]
    arr[:, 2] = masses.ravel()
    element = np.prod(arr, axis=1).sum()
    return element


@jit
def _tens_fd(masses, coords, ax):
    axes = list(np.arange(3))
    axes.remove(ax)
    sqrs = coords**2
    element = (masses.T * (sqrs[:, axes[0]] + sqrs[:, axes[1]])).sum()
    return element

@jit
def _inertia_tensor(center_coords, tens_fo, tens_fd):
    center_coords()
    in_tens = np.frompyfunc(tens_fo, 2, 1).outer(np.arange(3), np.arange(3)).astype(np.float64)
    for i in np.arange(3):
        in_tens[i,i] = tens_fd(i)
    return in_tens


def _zvals_typecheck(zvals):
    if type(zvals[0]) == float:
        zvals = [int(z) for z in zvals]
    elif type(zvals[0]) == str:
        if any(z.isdigit() for z in zvals):
            zvals = [int(float(z)) for z in zvals]
        else:
            zvals = [el_tbl[el_tbl.symbol == sym].atomicNumber.iloc[0] for sym in zvals]
    return zvals

def _xyz_input(txt_file, comment_line=0):
    with open(txt_file, 'r') as fin:
        lines = fin.read().splitlines()
        natom = int(lines[0])
        coords = np.zeros([natom, 3], dtype="float64")
        zvals = []
        for i, x in enumerate(coords):
            line = lines[i+1+comment_line].split()
            zvals.append(line[0])
        zvals = _zvals_typecheck(zvals)
        x[:] = list(map(float, line[1:4]))
    return natom, zvals, coords


def _dist_rep(self):
    self.perms = list(it.combinations(np.arange(0,self.natom), r=2))
    atom1 = [x+1 for x, y in self.perms]
    atom2 = [y+1 for x, y in self.perms]
    dist = [self.dist_matrix[pair] for pair in self.perms]
    return pd.DataFrame({'Atom_1': atom1, 'Atom_2': atom2, 'Distance': dist})


class Molecule(object):

    """
    A simple Molecule class for computations, including calculations of bond lengths,
    bond angles, out-of-plane angles and dihedral angles. Also supports re-centering the
    coordinate system to the center-of-mass (translation). Calculates the principal moments
    inertia and rotational constants.

    Init either with normal initialisation through giving the number of atoms, the atomic
    number (zvals) as well as the coordinates of those atoms. Or, more conveniently, through
    either the .from_txt or .from_xyz methods.

    E.g.: h20 = Molecule(3, [8, 1, 1], np.array([[0, 0, -0.134503695264], [0, -1.684916670000, 1.067335684736], [1, 1.684916670000, 1.067335684736]]))

    or

    h20 = Molecule.from_txt('./h2o_geom.txt')

    Use the .rep attribute to access a dataframe containg the atoms, their z-value as well as their
    cartesian coordinates.



    Symmetry TBD.
    """

    def __init__(self, natom, zvals, coords):
        self.natom = natom
        self.charge = 0
        self.zvals = np.array(zvals)
        self.coords = coords
        self.point_group = ''
        self.rep = pd.DataFrame({'Sym': [el_tbl[el_tbl.atomicNumber == z].symbol.iloc[0] for z in zvals] , 'Z': zvals, 'x': coords[:,0], 'y': coords[:,1], 'z': coords[:,2]})
        self.dist_matrix = np.sum((coords[None,:] - coords[:, None])**2, -1)**0.5
        self.dist_rep = _dist_rep(self)
        self.bond_cond = (self.dist_matrix < 3) & (self.dist_matrix > 0)


    @classmethod
    def from_txt(class_object, txt_file):
        natom, zvals, coords = _xyz_input(txt_file, comment_line=0)
        return class_object(natom, zvals, coords)


    @classmethod
    def from_xyz(class_object, txt_file):
        natom, zvals, coords = _xyz_input(txt_file, comment_line=1)
        return class_object(natom, zvals, coords)


    def print_geom(self):
        print(self.rep)


    def translate(self, xyz):
        trans = np.array(xyz)
        self.coords = self.coords + trans


    def print_dist(self):
        print('Distances between atoms (unit):')
        print(self.dist_rep)


    def bonds(self):
        return list({tuple(sorted(pair)) for pair in zip(np.where(self.bond_cond)[0], np.where(self.bond_cond)[1])})


    def bond_counts(self):
        num, cnt = np.unique(self.bonds(), return_counts=True)
        return dict(zip(num, cnt))


    def conn_atoms(self, rng):
        num, cnt = np.unique(self.bonds(), return_counts=True)
        bonds = np.array(self.bonds())
        adjacent_bonds = dict()
        for n in np.where(np.array((num, cnt))[1] > 1)[0]:
            adj_bnds = [bond for bond in self.bonds() if n in bond]
            adjacent_bonds[n] = adj_bnds
        if rng == 2:
            return adjacent_bonds


    def unit_vector(self, bond):
        atom1, atom2 = bond
        e = np.array(-(self.coords[atom1] - self.coords[atom2])) / self.dist_matrix[atom1][atom2]
        return e


    def bond_angle(self, bond1, bond2, deg=True):
        if deg:
            return np.rad2deg(np.arccos(np.dot(self.unit_vector((bond1)), self.unit_vector((bond2)))))
        else:
            return np.arccos(np.dot(self.unit_vector((bond1)), self.unit_vector((bond2))))


    def bond_angles(self):
        if self.natom <= 2:
            raise ValueError('The number of atoms in the molecule is not sufficient to calculate any bond angles')
        bnd_angls = pd.DataFrame()
        for key, vals in self.conn_atoms(2).items():
            for pair in list(it.combinations(vals, r=2)):
                bond1, bond2 = pair
                atom1, atom2 = list(pair[0]), list(pair[1])
                atom1.remove(key)
                atom2.remove(key)
                line = {'Atom_1': atom1[0], 'Conn_Atom': key, 'Atom_2': atom2[0], 'Angle': self.bond_angle(bond1, bond2)}
                line = pd.DataFrame(line, index=[0])
                bnd_angls = bnd_angls.append(line, ignore_index=True)
        return bnd_angls


    def oop_angle(self, bond_ki, bond_kj, bond_kl, deg=True):
        eki, ekj, ekl = self.unit_vector(bond_ki), self.unit_vector(bond_kj), self.unit_vector(bond_kl)
        theta = np.arcsin(np.dot(np.cross(ekj, ekl) / np.sin(self.bond_angle(bond_kj, bond_kl, deg=False)), eki))
        theta = abs(theta)
        if deg:
            return np.rad2deg(theta)
        else:
            return theta


    def oop_angles(self):
        if self.natom <= 3:
            raise ValueError('The number of atoms in the molecule is not sufficient to calculate any out-of-plane angles')
        oop_angls = pd.DataFrame()
        for key, vals in self.conn_atoms(2).items():
            for triple in list(it.combinations(vals, r=3)):
                for i, bond_ki in enumerate(triple):
                    bond_kj, bond_kl = (x for x in triple if x != triple[i])
                    atomi, atomj, atoml = list(bond_ki), list(bond_kj), list(bond_kl)
                    atomi.remove(key)
                    atomj.remove(key)
                    atoml.remove(key)
                    line = {'Atom_i': atomi[0], 'Atom_k': key, 'Atom_j': atomj[0], 'Atom_l': atoml[0], 'OOP_angle': self.oop_angle(bond_ki, bond_kj, bond_kl)}
                    line = pd.DataFrame(line, index=[0])
                    oop_angls = oop_angls.append(line, ignore_index=True)
        return oop_angls


    def tort_angle(self, bond_ij, bond_jk, bond_kl, deg=True):
        eij, ejk, ekl = self.unit_vector(bond_ij), self.unit_vector(bond_jk), self.unit_vector(bond_kl)
        cross_ijk = np.cross(eij, ejk)
        cross_jkl = np.cross(ejk, ekl)
        phi_ijk = self.bond_angle(bond_ij, bond_jk, deg=False)
        phi_jkl = self.bond_angle(bond_jk, bond_kl, deg=False)
        tau = np.arccos(np.dot(cross_ijk, cross_jkl) / (np.sin(phi_ijk)) * np.sin(phi_jkl))
        tau = abs(tau)
        if deg:
            return np.rad2deg(tau)
        else:
            return tau


    def middle_bonds(self):
        middle_bonds = [(x,y) for x,y in self.bonds() if (x in self.conn_atoms(2).keys()) and (y in self.conn_atoms(2).keys())]
        return middle_bonds


    def possible_tortion_angles(self):
        torsion_dict = {}
        for mb in self.middle_bonds():
            conn_bonds = []
            for atom in mb:
                atom_conn_bonds = []
                for bond in self.bonds():
                    if (bond != mb) and (any(e in bond for e in mb)) and (atom in bond):
                        atom_conn_bonds.append(bond)
                conn_bonds.append(atom_conn_bonds)
            torsion_dict[mb] = conn_bonds
        return torsion_dict


    def tort_angles(self):
        tort_angls = pd.DataFrame()
        for mb, bond_list in self.possible_tortion_angles().items():
            for bond1 in bond_list[0]:
                for bond2 in bond_list[1]:
                    atom1, atom2 = list(bond1), list(bond2)
                    atom1.remove(mb[0])
                    atom2.remove(mb[1])
                    line = {'Atom_1': atom1[0], 'Atom_2': atom2[0], 'Middle_bond': f'{mb}', 'Torsion_angle': self.tort_angle(bond1, mb, bond2)}
                    line = pd.DataFrame(line, index=[0])
                    tort_angls = tort_angls.append(line, ignore_index=True)
        return tort_angls


    def masses(self):
        return np.array([el_tbl[el_tbl.atomicNumber == x].atomicMass.iloc[0] for x in self.zvals])[:, None]


    def mass_weighted_coords(self):
        mw_coords = self.coords * self.masses()
        return mw_coords


    def center_of_mass(self):
        com = self.mass_weighted_coords().sum(axis=0) / self.masses().sum()
        return com


    def center_coords(self):
        self.translate(-self.center_of_mass())


    def tens_fo(self, ax1, ax2):
        return _tens_fo(self.masses(), self.coords, ax1, ax2)


    def tens_fd(self, ax):
        return _tens_fd(self.masses(), self.coords, ax)


    def inertia_tensor(self):
        return _inertia_tensor(self.center_coords, self.tens_fo, self.tens_fd)


    def p_inertia_moments(self):
        vals = la.eigh(self.inertia_tensor())[0]
        keys = ['Ia', 'Ib', 'Ic']
        return dict(zip(keys, vals))


    def moment2const(self, I):
        br = sc.physical_constants['Bohr radius'][0]
        return sc.Planck / (8 * np.pi**2 * br**2 * sc.atomic_mass * I)

    def rot_consts(self):
        const = np.empty((3,))
        for i, I in enumerate(self.p_inertia_moments().values()):
            const[i] = self.moment2const(I)
        return const


    def rot_const_mhz(self):
        return dict(zip(['A', 'B', 'C'], self.rot_consts()*1e-6))


    def rot_const_wn(self):
        return dict(zip(['A', 'B', 'C'], self.rot_consts() / (sc.speed_of_light * 1e+2)))


    def read_hessian(self, hessian_file, skiprows=1):
        with open(hessian_file) as hf:
            natom = int(hf.read().splitlines()[0].split()[0])
        if self.natom != natom:
            raise ValueError('The number of atoms in the molecule doesn\'t match those in the hessian matrix! Try another matrix.')
        self.hessian = np.loadtxt(hessian_file, skiprows=skiprows)
        self.hessian = self.hessian.reshape((self.natom*3, self.natom*3))


    def weight_hessian(self):
        masses = self.masses().ravel()
        wts = np.repeat(masses, 3)
        wt_matrix = np.outer(wts, wts)**.5
        self.wt_hessian = self.hessian / wt_matrix

    def hessian_evals(self):
        try:
            self.wt_hessian
        except AttributeError:
            print('The weighted hessian has\'t been defined for this molecule yet!')
            return
        eig = la.eig(self.wt_hessian)[0]
        eig.sort()
        return eig[::-1]


    def harm_freq_hz(self):
        hrm_frq = self.hessian_evals()
        for x in np.nditer(hrm_frq, op_flags=['readwrite']):
            if x <= 0:
                x[...] = 0
        const = (sc.physical_constants['Hartree energy'][0] / sc.physical_constants['Bohr radius'][0]**2 / sc.atomic_mass)
        hrm_frq = hrm_frq * const
        return hrm_frq**.5


    def harmonic_freq_mhz(self):
        return self.harm_freq_hz() * 1e-6


    def harmonic_freq_wn(self):
        return self.harm_freq_hz() / (2 * np.pi * sc.speed_of_light * 1e2)
