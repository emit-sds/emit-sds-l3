"""
| ----------------------------------------------------------------------------------------------------------------------
| Date                : August 2018
| Copyright           : (C) 2018 by Ann Crabbé (KU Leuven)
| Email               : ann.crabbe@kuleuven.be
| Acknowledgements    : Translated from VIPER Tools 2.0 (UC Santa Barbara, VIPER Lab).
|                       Dar Roberts, Kerry Halligan, Philip Dennison, Kenneth Dudley, Ben Somers, Ann Crabbé
|
| This program is free software; you can redistribute it and/or modify it under the terms of the GNU
| General Public License as published by the Free Software Foundation; either version 3 of the
| License, or any later version.
|
|
| Modified for EMIT by Phil Brodrick - minor changes to constraints
| Original source: Vipertools-3.0.8
| ----------------------------------------------------------------------------------------------------------------------
"""
import numpy as np
import math


class SquareArray:
    """
    In a spectral library, each endmember is used to model all others, using metrics like RMSE, Spectral Angle, ...
    These metrics are returned as a dictionary of square images (n_endmembers x n_endmembers).

    Possible metrics (bands) are RMSE, Spectral Angle, SMA fractions, SMA shade fraction, Constraints fulfilled.

    Citation: Roberts, D.A., Gardner, M.E., Church, R., Ustin, S.L., and Green, R.O., 1997, Optimum Strategies for
    Mapping Vegetation using Multiple Endmember Spectral Mixture Models, Proceedings of the SPIE, 3118, p. 108-119
    """

    def __init__(self):
        self.fractions = None  # endmember fractions
        self.shade = None  # leftover shade
        self.rmse = None  # root mean square error
        self.angle = None  # spectral angle
        self.constraints = None  # constraints encountered?
        self.metadata = {}
        self.progress_bar = _ProgressBar()

    def _square_fractions(self, library: np.array):
        """
        Calculate fractions and store them in self.fractions.

        :param library: spectral library [spectra as columns], scaled to reflectance values, without bad bands.
        """

        n_bands, n_spectra = library.shape
        # STEP 1: get the inverse endmembers
        # 'inverse' endmembers are the result of singular value decomposition
        #       svdc says: endmember = u * v * w        with u = decomposition array, v = -1 and w = eigenvalue
        #            and:  inverse endmember = u * v / w
        #            so:   total(endmember * inverse endmember) = 1
        #
        # we can manually determine the eigenvalue of a 1D array (look it up...):
        #       w * w = lambda = total(endmember * endmember)
        #
        # substitute in the formula for the inverse endmember gives us
        #       endmember = u * v * w         together with         inverse endmember = u * v / w       gives us:
        #       endmember / w = inverse endmember * w
        #       inverse endmember = endmember / w / w = endmember / lambda = endmember / total(endmember * endmember)
        em_inverse = library / np.sum(library * library, 0)

        # STEP 2: get the fractions of modelling all endmembers with each other
        self.fractions = np.dot(em_inverse.transpose(), library)
        x = np.arange(n_spectra)
        self.fractions[x, x] = 0.0

        print("Fractions calculated")
        self.progress_bar.setValue(20)

    def _square_shade(self):
        """
        Fractions should sum to 1 so the shade fraction is 1 minus the other fractions. Store in self.shade.
        """

        self.shade = 1.0 - self.fractions
        x = np.arange(self.shade.shape[0])
        self.shade[x, x] = 0.0

        self.progress_bar.setValue(50)
        print("Shade fractions calculated")

    def _square_rmse(self, library: np.array):
        """
        Calculate the root mean square error and store in self.rmse.

        :param library: spectral library [spectra as columns], scaled to reflectance values, without bad bands.
        """

        n_bands, n_spec = library.shape

        block_size = math.ceil(250000000 / n_bands / n_spec)
        n_blocks = math.ceil(n_spec / block_size)

        rmse = np.zeros([n_spec, n_spec], dtype=np.float32)

        for b in range(n_blocks):
            start = b * block_size
            end = min((b + 1) * block_size, n_spec)

            residuals = library[:, np.newaxis, :] - \
                (library[:, start:end, np.newaxis] * self.fractions[np.newaxis, start:end, :])
            rmse[start:end, :] = np.sqrt(np.sum(residuals * residuals, axis=0) / n_bands)
            self.progress_bar.setValue(int(float(b) / n_blocks * 100))

        x = np.arange(n_spec)
        rmse[x, x] = 0.0
        self.rmse = np.float32(rmse)

        self.progress_bar.setValue(70)
        print("RMSE calculated")

    def _square_angle(self, library: np.array):
        """
        Calculate the spectral angle and store in self.angle.

        :param library: spectral library [spectra as columns], scaled to reflectance values, without bad bands
        """

        n_bands, n_spectra = library.shape

        angles_self = library / np.sqrt(np.sum(library * library, 0))
        angles_cross = np.dot(angles_self.transpose(), angles_self)
        angles_cross[angles_cross > 1] = 1  # np.across can only handle values up to 0
        angle = np.arccos(angles_cross)
        x = np.arange(n_spectra)
        angle[x, x] = 0.0
        self.angle = np.float32(angle)

        self.progress_bar.setValue(90)
        print("Spectral angle calculated")

    def _fraction_constraint(self, constraints: tuple=(-0.05, 1.05), reset: bool=False):
        """
        Indicate in self.constraints whether the fraction constraint was breached and force fractions to threshold
        values in case reset = True:

            * 0: no breach
            * 1: fraction constraint breach + fraction reset
            * 2: fraction constraint breach + no fraction reset

        :param constraints: the min and max allowable fractions
        :param reset: fractions are reset to threshold values in case of breach
        """

        # in case reset == True: cut off fractions at threshold and return 1 where constraint breached
        if reset:
            if constraints[0] != -9999:
                self.constraints[np.where(self.fractions < constraints[0])] += 1
                self.fractions[np.where(self.fractions < constraints[0])] = constraints[0]
            if constraints[1] != -9999:
                self.constraints[np.where(self.fractions > constraints[1])] += 1
                self.fractions[np.where(self.fractions > constraints[1])] = constraints[1]

        # in case reset == False: return 2 where constraint breached
        else:
            if constraints[0] != -9999:
                self.constraints[np.where(self.fractions < constraints[0])] += 2
            if constraints[1] != -9999:
                self.constraints[np.where(self.fractions > constraints[1])] += 2

        self.progress_bar.setValue(30)
        print("Fraction constraints applied")

    def _rmse_constraint(self, max_rmse):
        """
        Indicate in self.constraints whether the RMSE constraint was breached:

            * 0: no breach
            * 1: fraction constraint breach + fraction reset + no RMSE constraint breach
            * 2: fraction constraint breach + no fraction reset + no RMSE constraint breach
            * 3: no fraction constraint breach + RMSE constraint breach
            * 4: fraction constraint breach + fraction reset + RMSE constraint breach
            * 5: fraction constraint breach + no fraction reset + RMSE constraint breach

        :param max_rmse: the maximum allowable rmse
        """

        if max_rmse != -9999:
            self.constraints[np.where(self.rmse > max_rmse)] += 3

        self.progress_bar.setValue(80)
        print("RMSE constraints applied")

    def execute(self, library: np.array, constraints: tuple = (-0.05, 1.05, 0.025), reset: bool = True,
                out_rmse: bool = True, out_constraints: bool = True, out_fractions: bool = False,
                out_shade: bool = False, out_angle: bool = False, p=None) -> dict:
        """
        Execute the Square Array calculations.

        The returned result is a dictionary with numpy arrays. The dictionary has the following possible keys, depending
        on the booleans set in the method's parameters:

            'rmse', 'constraints', 'em fraction', 'shade fraction', 'spectral angle'

        The 'constraints' band has 6 values indicating which constraint was breached:

             * 0 = no breach
             * 1/2 = a fraction constraint breached (reset/no reset)
             * 3 = RMSE constraint breached
             * 4/5 = 1/2 (fractions reset/no reset) + 3 (RMSE)

        :param library: spectral library [spectra as columns], scaled to reflectance values, without bad bands
        :param constraints: min fraction, max fraction and max RMSE (use *None* for unconstrained)
        :param reset: fractions are reset to threshold values before continuing (constraints not *None*)
        :param out_rmse: set to True if the user wants this as output
        :param out_constraints: set to True if the user wants this as output
        :param out_fractions: set to True if the user wants this as output
        :param out_shade: set to True if the user wants this as output
        :param out_angle: set to True if the user wants this as output
        :param QProgressBar p: a reference to the GUI progress bar
        :return: dictionary with each output metric  as a numpy array in a key-value pair
        """
        library = np.array(library, dtype=np.float32)
        n_spectra = library.shape[1]

        if constraints is None:
            constraints = (-9999, -9999, -9999)
        if len(constraints) != 3:
            raise Exception("Constraints must be a tuple with 3 values. Set -9999 when not used.")

        if constraints[0] != -9999 and constraints[0] < -0.5:
            raise Exception("The minimum fraction constraint cannot be below -0.50.")
        if constraints[1] != -9999 and constraints[1] > 1.5:
            raise Exception("The maximum fraction constraint cannot be over 1.50.")
        if constraints[2] != -9999 and constraints[2] > 0.1:
            raise Exception("The maximum RMSE constraint cannot be over 0.10.")

        if out_constraints:# or sum(constraints) != -29997:
            self.constraints = np.zeros([n_spectra, n_spectra], dtype=np.int8)

        # run the algorithm in all possible combinations
        if p:
            self.progress_bar = p
        if out_rmse or out_constraints or out_fractions or out_shade:
            self._square_fractions(library)
            self._fraction_constraint(constraints[0:2], reset)

            if out_shade:
                self._square_shade()
            if out_rmse or constraints[2] != -9999:
                self._square_rmse(library)
                self._rmse_constraint(constraints[2])

        if out_angle:
            self._square_angle(library)

        output = {}
        if out_rmse:
            output['rmse'] = self.rmse
        if out_constraints:# and sum(constraints) != -29997:
            output['constraints'] = self.constraints
        if out_fractions:
            output['em fraction'] = self.fractions
        if out_shade:
            output['shade fraction'] = self.shade
        if out_angle:
            output['spectral angle'] = self.angle

        self.progress_bar.setValue(100)

        return output


class _ProgressBar:

    @staticmethod
    def setValue(value: int):
        """ Replacement for the GUI progress bar: do nothing """
        pass


""" MODIFICATION HISTORY:
2005-05 [IDL] Written by Kerry Halligan in IDL
2007-06 [IDL] Modified by Kerry Halligan: Added a very simple metadata file to keep track of constraints used.
2014-09 [IDL] Modified by Kenneth Dudley: Store models along rows (computation efficiency in IDL). Added band selection
              of the output bands. Bug fix in spectral angle calculation which could cause NaNs to be stored in the
              output array, these NaNs are corrected to 0 for other bands. NaNs occur when there are duplicate spectra.
2014-09 [IDL] Modified by Kenneth Dudley: Added default name for output file to speed up processing
2017-11 [IDL] Modified by Ann Crabbé: Update of the GUI + speed up spectral angle calculations
2018-07 [Python] Ported to QGIS/Python by Ann Crabbé, incl. significant re-write of code
"""
