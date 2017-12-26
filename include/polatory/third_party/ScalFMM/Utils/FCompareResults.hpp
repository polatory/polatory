// See LICENCE file at project root
#ifndef FCOMPARERESULTS_HPP
#define FCOMPARERESULTS_HPP

#include <cstdlib>
#include <string>
//
#include "Utils/FMath.hpp"
#include "Files/FFmaGenericLoader.hpp"

template <class FReal, class classArrayType>
FSize compareTwoArrays(const std::string &tag,  const FSize &nbParticles, const classArrayType &array1,
		const classArrayType &array2){
	//
    FMath::FAccurater<FReal> potentialDiff;
    FMath::FAccurater<FReal> fx, fy, fz;
	double energy1= 0.0, energy2= 0.0;
    FSize error = 0 ;
	for(FSize idxPart = 0 ; idxPart < nbParticles ;++idxPart){
		if(array1[idxPart].getPosition() != array2[idxPart].getPosition() ){
			std::cerr <<"Wrong positions  " <<std::endl
					<< "   P1: " <<array1[idxPart].getPosition()<<std::endl
					<< "   P2: " <<array2[idxPart].getPosition()<<std::endl
			<< "   error  " <<array1[idxPart].getPosition()-array2[idxPart].getPosition()<<std::endl;
		}
		potentialDiff.add(array1[idxPart].getPotential(),array2[idxPart].getPotential());
		fx.add(array1[idxPart].getForces()[0],array2[idxPart].getForces()[0]);
		fy.add(array1[idxPart].getForces()[1],array2[idxPart].getForces()[1]);
		fz.add(array1[idxPart].getForces()[2],array2[idxPart].getForces()[2]);
		energy1 += array1[idxPart].getPhysicalValue() *array1[idxPart].getPotential() ;
		energy2 += array2[idxPart].getPhysicalValue() *array2[idxPart].getPotential() ;
	}
	// Print for information
	std::cout << tag<< " Energy "  << FMath::Abs(energy1-energy2) << "  "
			<<  FMath::Abs(energy1-energy2) /energy1 <<std::endl;
	std::cout << tag<< " Potential " << potentialDiff << std::endl;
	std::cout << tag<< " Fx " << fx << std::endl;
	std::cout << tag<< " Fy " << fy << std::endl;
	std::cout << tag<< " Fz " << fz << std::endl;

	return error ;
}
//
template <class FReal, class classArrayType>
void computeFirstMoment( const FSize &nbParticles, const classArrayType &particles,  FPoint<FReal> &FirstMoment){

    FReal mx,my,mz ;
	//
#pragma omp parallel  for shared(nbParticles,particles) reduction(+:mx,my,mz)
	for(FSize idxPart = 0 ; idxPart < nbParticles ; ++idxPart){
		//
		mx += particles[idxPart].getPhysicalValue()*particles[idxPart].getPosition().getX() ;
		my += particles[idxPart].getPhysicalValue()*particles[idxPart].getPosition().getY() ;
		mz += particles[idxPart].getPhysicalValue()*particles[idxPart].getPosition().getZ() ;
	}
	FirstMoment.setPosition(mx,my,mz);
} ;

template < class FReal, class classArrayType>
void removeFirstMoment( const std::string& TYPE, const FSize &nbParticles, const classArrayType &particles,  FReal &volume) {
    FPoint<FReal> FirstMoment ;
    computeFirstMoment<FReal, classArrayType>( nbParticles, particles,  FirstMoment);
	std::cout << std::endl;
	std::cout << "Electric Moment          = "<< FirstMoment <<std::endl;
	std::cout << "Electric Moment norm = "<< FirstMoment.norm2()  <<std::endl;
	std::cout << "----------------------------------------------------"<<std::endl;
	std::cout << std::endl;
	//
	// Remove
	FReal coeffCorrection  = 4.0*FMath::FPi<FReal>()/volume/3.0 ;
	FReal scaleEnergy=1.0, scaleForce=1.0  ;
	//
	if (TYPE == "DLPOLY") {
		const FReal r4pie0 = FReal(138935.4835);
		scaleEnergy =  r4pie0 / 418.4 ;   // kcal mol^{-1}
		scaleForce  = -r4pie0 ;           // 10 J mol^{-1} A^{-1}
	}
	else if (TYPE == "NOSCALE") {
		scaleEnergy=1.0, scaleForce=1.0  ;
	}
	else {
		std::cerr << "In removeFirstMoment TYPE " << TYPE << " was not know. Available TYPE DLPOLY "<< std::endl;
		std::exit( EXIT_FAILURE);
	}
      //
	double tmp;
    for(FSize idx = 0 ; idx < nbParticles ; ++idx){
		tmp = particles[idx].getPosition().getX()*FirstMoment.getX()  + particles[idx].getPosition().getY()*FirstMoment.getY()
								+ particles[idx].getPosition().getZ()*FirstMoment.getZ()  ;
		FReal Q =  particles[idx].getPhysicalValue(), P = particles[idx].getPotential();
		//

		particles[idx].setPotential( P - tmp*coeffCorrection);
		//
		std::cout << "idx: "<< idx <<" old: " << particles[idx].getForces()[0] ;
		particles[idx].getForces()[0] -= Q*coeffCorrection*FirstMoment.getX() ;
		particles[idx].getForces()[1] -= Q*coeffCorrection*FirstMoment.getY() ;
		particles[idx].getForces()[2] -= Q*coeffCorrection*FirstMoment.getZ() ;
		std::cout << " new: " << particles[idx].getForces()[0] <<std::endl;

		//
		particles[idx].getForces()[0] *= scaleForce;
		particles[idx].getForces()[1] *= scaleForce;
		particles[idx].getForces()[2] *= scaleForce;
		//
		//		newEnergy += Q*particles[idx].getPotential()  ;
	}

}
#endif
