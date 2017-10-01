// ===================================================================================
// Copyright ScalFmm 2016 INRIA, Olivier Coulaud, Bérenger Bramas,
// Matthias Messner olivier.coulaud@inria.fr, berenger.bramas@inria.fr
// This software is a computer program whose purpose is to compute the
// FMM.
//
// This software is governed by the CeCILL-C and LGPL licenses and
// abiding by the rules of distribution of free software.
// An extension to the license is given to allow static linking of scalfmm
// inside a proprietary application (no matter its license).
// See the main license file for more details.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public and CeCILL-C Licenses for more details.
// "http://www.cecill.info".
// "http://www.gnu.org/licenses".
// ===================================================================================
#ifndef FGLOBALPERIODIC_HPP
#define FGLOBALPERIODIC_HPP

///////////////////////////////////////////////////////
// Periodic condition definition
///////////////////////////////////////////////////////

/**
 * @brief The PeriodicCondition enum
 * To be able to chose the direction of the periodicity.
 */
enum PeriodicCondition {
    DirNone     = 0,

    DirPlusX    = 1 << 0,
    DirMinusX   = 1 << 1,
    DirPlusY    = 1 << 2,
    DirMinusY   = 1 << 3,
    DirPlusZ    = 1 << 4,
    DirMinusZ   = 1 << 5,

    DirX        = (DirPlusX | DirMinusX),
    DirY        = (DirPlusY | DirMinusY),
    DirZ        = (DirPlusZ | DirMinusZ),

    AllDirs     = (DirX | DirY | DirZ)
};

/**
 * @brief TestPeriodicCondition
 * @param conditions
 * @param testConditions
 * @return true if the direction is in the condition
 */
inline bool TestPeriodicCondition(const int conditions, const PeriodicCondition testConditions) {
    return (conditions & testConditions) == testConditions;
}

#endif // FGLOBALPERIODIC_HPP
