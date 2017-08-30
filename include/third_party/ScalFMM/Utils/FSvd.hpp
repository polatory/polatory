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
#ifndef FSVD_HPP
#define FSVD_HPP

#include "FGlobal.hpp"

#include "Kernels/Chebyshev/FChebTensor.hpp" 


/**
 * @class FSvd
 * 
 * @brief This class provides some functions related to SVD.
 *  
 */
namespace FSvd {


    /**
     * Computes the low-rank \f$k\f$ based on \f$\|K-U\Sigma_kV^\top\|_F \le
     * \epsilon \|K\|_F\f$, ie., the truncation rank of the singular value
     * decomposition. With the definition of the Frobenius norm \f$\|K\|_F =
     * \left(\sum_{i=1}^N \sigma_i^2\right)^{\frac{1}{2}}\f$ the determination of
     * the low-rank follows as \f$\|K-U\Sigma_kV^\top\|_F^2 = \sum_{i=k+1}^N
     * \sigma_i^2 \le \epsilon^2 \sum_{i=1}^N \sigma_i^2 = \epsilon^2
     * \|K\|_F^2\f$.
     *
     * @param[in] singular_values array of singular values ordered as \f$\sigma_1
     * \ge \sigma_2 \ge \dots \ge \sigma_N\f$
     * @param[in] eps accuracy \f$\epsilon\f$
     */ 
    template <class FReal, int ORDER>
    unsigned int getRank(const FReal singular_values[], const double eps)
    {
    	enum {nnodes = TensorTraits<ORDER>::nnodes};

    	FReal nrm2(0.);
    	for (unsigned int k=0; k<nnodes; ++k)
    		nrm2 += singular_values[k] * singular_values[k];

    	FReal nrm2k(0.);
    	for (unsigned int k=nnodes; k>0; --k) {
    		nrm2k += singular_values[k-1] * singular_values[k-1];
    		if (nrm2k > eps*eps * nrm2)	return k;
    	}
    	throw std::runtime_error("rank cannot be larger than nnodes");
    	return 0;
    }

    template <class FReal>
    unsigned int getRank(const FReal singular_values[], const unsigned int size, const double eps)
    {
    	const FReal nrm2 = FBlas::scpr(size, singular_values, singular_values);
    	FReal nrm2k(0.);
    	for (unsigned int k=size; k>0; --k) {
    		nrm2k += singular_values[k-1] * singular_values[k-1];
    		if (nrm2k > eps*eps * nrm2)	return k;
    	}
    	throw std::runtime_error("rank cannot be larger than size");
    	return 0;
    }




};

#endif /* FSVD_HPP */