// See LICENCE file at project root
#ifndef FEXPORTDATA_HPP
#define FEXPORTDATA_HPP

// @author O. Coulaud

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>

//! \fn void exportCVS(std::ofstream& file, const FReal * particles , const FSize N, const FSize nbDataPerParticle=4)

//! \brief  Export particles in CVS Format
//!
//! Export particles in CVS Format as follow
//!      x ,  y  , z , physicalValue, P, FX,  FY,  FY
//! It is useful to plot the distribution with paraView
//!
//!  @param file stream to save the data
//!  @param  N number of particles
//!  @param  particles array of particles of type FReal (float or double) Its size is N*nbDataPerParticle
//!  @param  nbDataPerParticle number of values per particles (default value 4)
//!
template <class FReal>
void exportCVS(std::ofstream& file, const FReal * particles , const FSize N, const FSize nbDataPerParticle=4){
    FSize j = 0;
	if (nbDataPerParticle==4){
		file << " x ,  y , z, q  " <<std::endl;
	}
	else {
		file << " x ,  y , z, q P FX FY FZ" <<std::endl;
	}
    for(FSize i = 0 ; i< N; ++i, j+=nbDataPerParticle){
		file <<    particles[j]   ;
        for (FSize k = 1 ; k< nbDataPerParticle ; ++k) {
			file   << "  , "  <<    particles[j+k]     ;
		}
		file   << std::endl;
	}
}
//
//! \fn void exportCOSMOS(std::ofstream& file,  const FReal * particles, const FSize N )

//! \brief  Export particles in COSMOS Format
//!
//! Export particles in CVS Format as follow
//!      x ,  y  , z , 0.0, 0.0, 0.0, physicalValue
//!
//!  @param file stream to save the data
//!  @param  particles array of particles of type FReal (float or double) Its size is 4*N (X,Y,Z,M)
//!  @param  N number of particles
//!
template <class FReal>
void exportCOSMOS(std::ofstream& file, const FReal * particles , const FSize N){
    FSize j = 0;
	file << " x ,  y , z, q " <<std::endl;
    for(FSize i = 0 ; i< N; ++i, j+=4){
		file <<    particles[j]    << "  "    <<   particles[j+1]    << "  "   <<   particles[j+2]    << "  0.0 0.0 0.0  "   <<   particles[j+3]   <<"  " << i << std::endl;
	}
}
//
//
//! \fn void exportVTK(std::ofstream& file,  const FReal * particles, const FSize N )

//! \brief  Export particles in CVS Format
//!
//! Export particles in old polydata Format.
//!   A particle is composed of 4 fields    x ,  y  , z ,  physicalValue
//! It is useful to plot the distribution with paraView
//!
//!  @param file stream to save the data
//!  @param  particles array of particles of type FReal (float or double) Its size is 4*N (X,Y,Z,M)
//!  @param  N number of particles
template <class FReal>
void exportVTK(std::ofstream& VTKfile, const FReal * particles, const FSize N, const FSize nbDataPerParticle=4 ){
    FSize j = 0;
	//---------------------------
	// print generic information
	//---------------------------
	VTKfile << "# vtk DataFile Version 3.0" << "\n";
	VTKfile << "#  Generated bt exportVTK" << "\n";

	VTKfile << "ASCII" << "\n";
	VTKfile << "DATASET POLYDATA" << "\n";
	//
	//---------------------------------
	// print nodes ordered by their TAG
	//---------------------------------
	VTKfile << "POINTS " << N << "  float" << "\n";
	//
    for(FSize i = 0 ; i< N; ++i, j+=nbDataPerParticle){
		VTKfile <<    particles[j]    << "  "    <<   particles[j+1]    << "   "   <<   particles[j+2]      <<std::endl;
	}
	// ------------------------------------------
	VTKfile << "\n";
	VTKfile << "VERTICES  " <<  N << " " << 2*N << "\n";
    for(FSize i = 0 ; i< N; ++i){
		VTKfile <<    "  1 "    << " "    <<i<<std::endl;
	}
	VTKfile << "POINT_DATA  " <<  N << "\n";
	VTKfile << "SCALARS PhysicalValue  float 1" << "\n"
			<< "LOOKUP_TABLE default" << "\n" ;
	j = 0 ;
    for(FSize i = 0 ; i< N; ++i, j+=nbDataPerParticle){
		VTKfile <<    particles[j+3]    << " "    <<std::endl;
	}
	VTKfile << "\n";
};
//
//
//! \fn void exportVTKxml(std::ofstream& file,  const FReal * particles, const FSize N )

//! \brief  Export particles in xml polydata VTK  Format
//!
//! Export particles in the xml polydata VTK  Format.
//!   A particle is composed of 4 fields    x ,  y  , z ,  physicalValue
//! It is useful to plot the distribution with paraView
//!
//!  @param file stream to save the data
//!  @param  particles array of particles of type FReal (float or double) Its size is 4*N (X,Y,Z,M)
//!  @param  N number of particles
template <class FReal>
void exportVTKxml(std::ofstream& VTKfile, const FReal * particles, const FSize N ){
    FSize j = 0;

	VTKfile << "<?xml version=\"1.0\"?>" <<std::endl
			<< "<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\"> "<<std::endl
			<< "<PolyData>"<<std::endl
			<< "<Piece NumberOfPoints=\" " << N << " \"  NumberOfVerts=\" "<<N <<" \" NumberOfLines=\" 0\" NumberOfStrips=\"0\" NumberOfPolys=\"0\">"<<std::endl
			<< "<Points>"<<std::endl
			<< "<DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\"> "<<std::endl ;
	j = 0 ;
    for(FSize i = 0 ; i< N; ++i, j+=4){
		VTKfile <<    particles[j]    << "  "    <<   particles[j+1]    << "   "   <<   particles[j+2]      << "   "   ;
	}
	VTKfile <<std::endl<< "</DataArray> "<<std::endl
			<< "</Points> "<<std::endl
			<< "<PointData Scalars=\"PhysicalValue\" > "<<std::endl
			<< "<DataArray type=\"Float64\" Name=\"PhysicalValue\"  format=\"ascii\">"<<std::endl ;
	j = 0 ;
    for(FSize i = 0 ; i< N; ++i, j+=4){
		VTKfile <<    particles[j+3]    << " "   ;
	}
	VTKfile <<std::endl << "</DataArray>"<<std::endl
			<< "	</PointData>"<<std::endl
			<< "	<CellData>"<<" </CellData>"<<std::endl
			<< "	<Verts>"<<std::endl
			<< "	<DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">"<<std::endl ;
    for(FSize i = 0 ; i< N; ++i){
		VTKfile <<   i   << " "   ;
	}
	VTKfile<<std::endl << "</DataArray>" <<std::endl
			<< "<DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">"<<std::endl ;
    for(FSize i = 1 ; i< N+1; ++i){
		VTKfile <<   i   << " "   ;
	}
	VTKfile<<std::endl  << "</DataArray>"<<std::endl
			<< "	</Verts>"<<std::endl
			<< "<Lines></Lines>"<<std::endl
			<< "<Strips></Strips>"<<std::endl
			<< "<Polys></Polys>"<<std::endl
			<< "</Piece>"<<std::endl
			<< "</PolyData>"<<std::endl
			<< "</VTKFile>"<<std::endl;
} ;
//
//
//
//! \fn void exportVTKxml(std::ofstream& file,  const FReal * particles, const FSize N, const FSize nbDataPerParticle )

//! \brief  Export particles in CVS Format
//!
//! Export particles in the new PolyData Format.
//!   A particle is composed of 4 fields    x ,  y  , z ,  physicalValue
//! It is useful to plot the distribution with paraView
//!
//!  @param file stream to save the data
//!  @param  particles array of particles of type FReal (float or double) Its size is nbDataPerParticle*N
//!  @param  N number of particles
//!  @param  nbDataPerParticle number of values per particles (default value 4)
template <class FReal>
void exportVTKxml(std::ofstream& VTKfile, const FReal * particles, const FSize N, const FSize nbDataPerParticle ){
    FSize j = 0;

	VTKfile << "<?xml version=\"1.0\"?>" <<std::endl
			<< "<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\"> "<<std::endl
			<< "<PolyData>"<<std::endl
			<< "<Piece NumberOfPoints=\" " << N << " \"  NumberOfVerts=\" "<<N <<" \" NumberOfLines=\" 0\" NumberOfStrips=\"0\" NumberOfPolys=\"0\">"<<std::endl
			<< "<Points>"<<std::endl
			<< "<DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\"> "<<std::endl ;
	j = 0 ;
    for(FSize i = 0 ; i< N; ++i, j+=nbDataPerParticle){
		VTKfile <<    particles[j]    << "  "    <<   particles[j+1]    << "   "   <<   particles[j+2]      << "   "   ;
	}
	VTKfile <<std::endl<< "</DataArray> "<<std::endl
			<< "</Points> "<<std::endl ;
	if (nbDataPerParticle==8 ) {
		VTKfile<< "<PointData Scalars=\"PhysicalValue\" > "<<std::endl
				<< "<DataArray type=\"Float64\" Name=\"PhysicalValue\"  format=\"ascii\">"<<std::endl ;
		j = 0 ;
        for(FSize i = 0 ; i< N; ++i, j+=nbDataPerParticle){
			VTKfile <<    particles[j+3]    << " "   ;
		}
		VTKfile <<std::endl << "</DataArray>"<<std::endl ;
		VTKfile  << "<DataArray type=\"Float64\" Name=\"Potential\"  format=\"ascii\">"<<std::endl ;
		j = 0 ;
        for(FSize i = 0 ; i< N; ++i, j+=nbDataPerParticle){
			VTKfile <<    particles[j+4]    << " "   ;
		}
		VTKfile <<std::endl << "</DataArray>"<<std::endl ;
		VTKfile<< "<DataArray type=\"Float64\"  Name=\"Force\" NumberOfComponents=\"3\" format=\"ascii\"> "<<std::endl ;
		j = 0 ;
        for(FSize i = 0 ; i< N; ++i, j+=nbDataPerParticle){
			VTKfile <<    particles[j+5]    << "  "    <<   particles[j+6]    << "   "   <<   particles[j+7]      << "   "   ;
		}
		VTKfile <<std::endl<< "</DataArray> "<<std::endl;
	}
	else {
		VTKfile		<< "<PointData Scalars=\"PhysicalValue\" > "<<std::endl
				<< "<DataArray type=\"Float64\" Name=\"PhysicalValue\"  format=\"ascii\">"<<std::endl ;
		j = 0 ;
        for(FSize i = 0 ; i< N; ++i, j+=nbDataPerParticle){
			VTKfile <<    particles[j+3]    << " "   ;
		}
		VTKfile <<std::endl << "</DataArray>"<<std::endl ;
	}

	VTKfile		<< "	</PointData>"<<std::endl
			<< "	<CellData>"<<" </CellData>"<<std::endl
			<< "	<Verts>"<<std::endl
			<< "	<DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">"<<std::endl ;
    for(FSize i = 0 ; i< N; ++i){
		VTKfile <<   i   << " "   ;
	}
	VTKfile<<std::endl << "</DataArray>" <<std::endl
			<< "<DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">"<<std::endl ;
    for(FSize i = 1 ; i< N+1; ++i){
		VTKfile <<   i   << " "   ;
	}
	VTKfile<<std::endl  << "</DataArray>"<<std::endl
			<< "	</Verts>"<<std::endl
			<< "<Lines></Lines>"<<std::endl
			<< "<Strips></Strips>"<<std::endl
			<< "<Polys></Polys>"<<std::endl
			<< "</Piece>"<<std::endl
			<< "</PolyData>"<<std::endl
			<< "</VTKFile>"<<std::endl;
} ;
//
////////////////////////////////////////////////////////////////////////////////////////////////////
//                           The Driver
////////////////////////////////////////////////////////////////////////////////////////////////////
//
//! \fn void driverExportData(std::ofstream& file, const FReal * particles , const FSize N, const FSize nbDataPerParticle=4)
//! \brief  The driver to select the right format to write the data
//!
//!  The driver select the right format (CVS, COSMOS, VTK, VTP,, ...) according to the extention of the filename
//!      . cvs   CVS Format
//!      .vtk the old vtk format
//!      .vtp  the xml vtk format
//!      .cosmo  the cosmos format
//! It is useful to plot the distribution with paraView
//!
//!  @param filename the name of the file to store the data
//!  @param  N number of particles
//!  @param  particles array of particles of type FReal (float or double) Its size is N*nbDataPerParticle
//!  @param  nbDataPerParticle number of values per particles (default value 4)
template <class FReal>
void driverExportData(std::string & filename, const FReal * particles , const FSize NbPoints, const FSize nbDataPerParticle=4){
	//
	std::ofstream file( filename.c_str(), std::ofstream::out);
	// open particle file
	std::cout << "Write "<< NbPoints<<" Particles in file " << filename <<std::endl;

	if(filename.find(".vtp") != std::string::npos) {
		exportVTKxml( file, particles, NbPoints,nbDataPerParticle)  ;
	}
	else if(filename.find(".vtk")!=std::string::npos ) {
		exportVTK( file, particles, NbPoints,nbDataPerParticle)  ;
	}
	else if(filename.find(".cvs")!=std::string::npos ) {
		exportCVS( file, particles, NbPoints,nbDataPerParticle)  ;
	}
	else if(filename.find(".cosmo")!=std::string::npos ) {
		exportCOSMOS( file, particles, NbPoints)  ;
	}
	else  {
		std::cout << "Output file not allowed only .cvs, .cosmo, .vtk or .vtp extensions" <<std::endl;
		std::exit ( EXIT_FAILURE) ;
	}
}
#endif
