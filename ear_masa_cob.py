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
| Original source: Vipertools-3.0.8
| ----------------------------------------------------------------------------------------------------------------------
"""
import numpy as np


class EarMasaCob:
    """  Calculate EAR, MASA, and CoB values from a square array.

     * EAR = Endmember Average RMSE
     * MASA = Minimum Average Spectral Angle
     * CoB = Count based Endmember selection (CoB)

    Citation EAR:

    Dennison, P.E., Roberts, D.A., 2003, Endmember Selection for Multiple Endmember Spectral Mixture Analysis using
    Endmember Average RMSE, Remote Sensing of Environment, 87, p. 123-135.

    Citation MASA:

    Dennison, P.E., Halligan, K. Q., Roberts, D.A., 2004, A Comparison of Error Metrics and Constraints for Multiple
    Endmember Spectral Mixture Analysis and Spectral Angle Mapper, Remote Sensing of Environment, 93, p. 359-367.

    Citations CoB:

    Roberts, D.A., Dennison, P.E., Gardner, M.E., Hetzel, Y., Ustin, S.L., Lee, C.T., 2003, Evaluation of the potential
    of Hyperion for fire danger assessment by comparison to the Airborne Visible/Infrared Imaging Spectrometer, IEEE
    Transactions on Geoscience and Remote Sensing, 41, p. 1297-1310.

    Clark, M., 2005, An assessment of Hyperspectral and LIDAR Remote Sensing for the Monitoring of Tropical Rain Forest
    Trees, PhD Dissertation, UC Santa Barbara, 319 p.
    """

    def __init__(self):
        pass

    @staticmethod
    def execute(spectral_angle_band: np.array, rmse_band: np.array, constraints_band: np.array, class_list: np.array):
        """
        Calculate EAR, MASA and COB parameters.

        Beware of the square array used as input! It must be built in 'reset' mode. Normally, the constraints band has
        6 possible values indicating which constraint was breached. Values 2 and 5 are not allowed here:

            * 0 = no breach
            * 1/2 = a fraction constraint breached (reset/no reset)
            * 3 = RMSE constraint breached
            * 4/5 = 1/2 (fractions reset/no reset) + 3 (RMSE)

        :param spectral_angle_band: Spectral Angle band from the square array
        :param rmse_band: RMSE band from the square array
        :param constraints_band: Constraints band from the square array
        :param class_list: str array with the class for each spectrum (e.g. GV or SOIL)
        :return: ear, masa, cob_in, cob_out, cob_ratio: all 1-D arrays complementing the original library's metadata.
        """

        if spectral_angle_band.shape[0] != spectral_angle_band.shape[1]:
            raise Exception("Spectral Angle band is not square")

        if rmse_band.shape[0] != rmse_band.shape[1]:
            raise Exception("RMSE band is not square")

        if constraints_band.shape[0] != constraints_band.shape[1]:
            raise Exception("Constraints band is not square")

        if 2 in constraints_band or 5 in constraints_band:
            raise Exception("Square array must be run in 'reset' mode. This one is not.")

        print('EMC calculations started')

        n_spec = class_list.shape[0]
        n_spec_range = np.arange(n_spec)

        # change the constraints band to hold '1' if no constrained was breached and '0' otherwise, in other words '1'
        # indicates a successful unmixing
        x, y = np.where(constraints_band == 0)
        constraints_band = np.zeros([n_spec, n_spec], dtype=np.bool)
        constraints_band[x, y] = 1
        constraints_band[n_spec_range, n_spec_range] = 0   # diagonal

        # get the unique groups
        class_list = np.asarray([x.lower() for x in class_list])
        groups = np.unique(class_list)
        print('Number of groups: ' + str(groups.shape[0]))

        # prep the output arrays
        ear = np.zeros(n_spec, dtype=float)             # EAR: average RMSE within the group
        masa = np.zeros(n_spec, dtype=float)            # MASA: average Spectral Angle within the group
        cob_in = np.zeros(n_spec, dtype=float)          # InCoB: number of spectra modeled by EM within the group
        cob_out = np.zeros(n_spec, dtype=int)           # OutCoB: number of spectra modeled by EM outside the group
        groups_n = np.zeros(n_spec, dtype=int)          # Helper for the CoBI calculation

        # remember in the square array: rows are the 'models' and columns are the 'pixels to be unmixed' by the models
        for group in groups:
            print('     Current group: ' + group)
            inside_indices = np.where(class_list == group)[0]
            outside_indices = np.where(class_list != group)[0]
            n = inside_indices.shape[0]
            groups_n[inside_indices] = n

            spectral_angle_inside = spectral_angle_band[inside_indices[:, None], inside_indices[None, :]]
            rmse_inside = rmse_band[inside_indices[:, None], inside_indices[None, :]]
            constraints_inside = constraints_band[inside_indices[:, None], inside_indices[None, :]]
            constraints_outside = constraints_band[inside_indices[:, None], outside_indices[None, :]]

            # number of nonzero spectral angle elements per column, to account for the effect of duplicate spectra
            n_nonzero_items = np.sum(spectral_angle_inside != 0, 1)

            # mean of the spectral angle of non-zero elements
            with np.errstate(invalid='ignore'):
                masa[inside_indices] = np.sum(spectral_angle_inside, 1)/n_nonzero_items

            # mean of the rmse of non-zero elements
            with np.errstate(invalid='ignore'):
                ear[inside_indices] = np.sum(rmse_inside, 1)/n_nonzero_items

            """
            Until all spectra have been used, do:
            1) Calculate CoB as count of all spectra (col) successfully unmixed by each model (row)
            2) Find the highest CoB (there may be ties!) and save this value
            3) Remove rows/cols of this highest CoB + all spectra modeled by it so they can no longer influence step 1
            4) Repeat steps 1-3 until there are are no available spectra [all status = 0]
            When more than 1 group exist calculate the out of group CoB and the CoBI
            """
            unused = np.ones(n, dtype=bool)
            while np.sum(unused) > 0:
                cob = np.sum(constraints_inside, 1)
                cob_max = cob.max()
                if cob_max != 0:
                    cob_index = np.where(cob == cob_max)[0]
                    cob_in[inside_indices[cob_index]] = cob_max
                else:
                    cob_index = np.where(unused)[0]
                if n < n_spec:
                    cob_out[inside_indices[cob_index]] = np.sum(constraints_outside[cob_index, :], 1)

                # set the 'pixel' containing the model, as the 'pixels' unmixed by the model to 0
                # set the same 'models' to 0
                unmixed = np.where(constraints_inside[cob_index, :])[1]
                constraints_inside[:, unmixed] = 0
                constraints_inside[unmixed, :] = 0
                constraints_inside[:, cob_index] = 0
                constraints_inside[cob_index, :] = 0
                unused[cob_index] = 0
                unused[unmixed] = 0

        # CoBI: CoB index (InCoB/OutCoB/number of spectra in the group)
        cob_ratio = np.divide(cob_in/groups_n, cob_out, out=np.zeros_like(cob_in), where=cob_out != 0)

        return ear, masa, cob_in, cob_out, cob_ratio


""" MODIFICATION HISTORY:
2005-06 [IDL] Written by Kerry Halligan
2006-12 [IDL] Modified by Kerry Halligan: allow for NaNs in square array [IDL specific issue]
2007-01 [IDL] Modified by Kerry Halligan: change IDL-specific EMC io 
2007-03 [IDL] Modified by Kerry Halligan: change IDL-specific Square Array io
2007-06 [IDL] Modified by Kerry Halligan: fix a sorting bug in the CoB calculation, free up some memory, added 'slow' 
              routine (line by line calculation of EAR and MASA).
2014-10 [IDL] Modifications by Kenneth Dudley: added support for new square array format and additional support for 
              legacy square arrays; changed method for loading in constraint array to prevent loading in a large
              floating point array when only byte data was needed; fixed "slow" mode which did not sort before finding
              unique values; changed sorting in program such that sort of input library no longer matters when appending
              data; added strupcase() to nullify case differences between classes; added code to detect duplicate 
              spectra and leave them out of EAR MASA calculations; fixed a sorting bug in "fast" mode which was applied 
              only to columns of the square array and not rows, leading to several miscalculations in the results.
2017-11 [IDL] Modified by Ann Crabbé: to show info sooner on the progress bar
2018-03 [IDL] Modified by Ann Crabbé: fixed bug where cob_out was calculated based on the wrong rows
2018-03 [Python] Ported to QGIS/Python by Ann Crabbé, incl. significant re-write of code
2018-09 [Python] Included warning when SA-constraints-band has 2 or 5, i.e. SA was run in 'non-reset' mode (not allowed)
"""
