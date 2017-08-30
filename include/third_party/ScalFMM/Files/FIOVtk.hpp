/*
 * VtkWrite.hpp
 *
 *  Created on: 6 juin 2013
 *      Author: Arnaud Etcheverry
 // ===================================================================================
 // Logiciel: OptiDis Version 0.1
 // Propriétaires : INRIA.
 // Copyright © 2011-2012, diffusé sous les termes et conditions d’une licence propriétaire.
 // ===================================================================================
 */

#ifndef FIOVTK_HPP
#define FIOVTK_HPP
#include <ostream>
#include <fstream>
#include <string>
//
#include "../../Src/Utils/FPoint.hpp"
//
template <class FReal>
class FIOVtk {
public:

	FIOVtk() : _format("xml")
{}

	/*!
    \fn void SetFormat(const std::string& fmt)
    /brief Set format for the post-treatment (visualization) file.
	 *
      Values are available
          - vtk - old vtk format
          - xml - new vtk format
          - xmlBinary new binary vtk format

        \param fmt the format descriptor
	 */
	void SetFormat(const std::string& fmt)
	{this->_format =  fmt ; }
	/*!
    \fn void writeOctree(const std::string &vtkName,const std::string &header, class OctreeClass& tree)
    \brief Write in VTK file the Octree skeleton.

    The output file name is [VTKName].vtk

    \param VTKname filename prefix
    \param title description of the vtk file
	 */
	template<class OctreeClass>
	void writeOctree(const std::string &vtkName,const std::string &header, const OctreeClass& tree);
	//
	//---------------------------------------------------------------
	//! Print the graph in a VTK file
	//---------------------------------------------------------------
	/*!
    The output file name is [VTKName][ifile].vtk where ndigits
    specifies the number of characters to be written for [ifile].

    We save in VTK 3.0 format and in ascii mode. See for more details on this format

    http://www.cacr.caltech.edu/~slombey/asci/vtk/vtk_formats.simple.html

    The data saved are the following.
    - On Nodes: velocitie and force.
    - On segments: length, burgers vector and morton index.

    \param VTKname filename prefix
    \param ndigits the number of digits to be written
    \param ifile file number
    \param title description of the vtk file
	 */
	//===============================================================

	template<class OctreeClass>
	void writeParticules( OctreeClass& tree, const bool & ioNode,const int &  myRank,const int & numProc,
			std::string VTKname =  std::string(),std::string title =  std::string())
	{
		if(VTKname.empty()) VTKname= "particlesFile";
		if(title.empty()) title="Saving Particles  ";
		std::string VTKnameTMP(VTKname);
		if(numProc >1 ){
			std::stringstream rank ;
			rank << myRank;
			VTKnameTMP += rank.str() ;
		}
		//
		if( this->_format == "xml"){
			if(ioNode && numProc >1) this->writeXMLmasterFile(numProc,VTKname,title);
			this->writeXMLParticles(VTKnameTMP, title,tree);
		}else if( this->_format == "xmlBinary"){
			if(ioNode && numProc >1) this->writeXMLmasterFile(numProc,VTKname,title);
			//		this->writeXMLDislocationsBinary(VTKnameTMP, title,tree);
		}
		else {
			//			this->writeVTKDislocations(iterationIndice,VTKnameTMP, title);
		}
	}

private:
	//	void writeVTKDislocations(const int& iterationIndice, const std::string& VTKname ,const std::string& /*title*/) ;
	template<class OctreeClass>
	void writeXMLParticles( const std::string& VTKname,const std::string& /*title*/,  OctreeClass& tree) ;
	//	void writeXMLDislocationsBinary(const int& iterationIndice, const std::string& VTKname,const std::string& /*title*/) ;
	//
	void writeXMLmasterFile(const int & numProc,std::string& VTKname,const std::string& /*title*/);



	/*!  Define the format to write the dislocation for. The following values are available
        - vtk - old vtk format
        - xml - new vtk format
       - xmlBinary new binary vtk format
	 */
	std::string   _format ;
};

template <class FReal>
template<class OctreeClass>
void FIOVtk<FReal>::writeOctree(const std::string &vtkName,const std::string &header, const OctreeClass& tree)
{
	//
	const double boxWidth = tree.getBoxWidth();
    const        FPoint<FReal> min(tree.getBoxCenter(),-boxWidth/2);
	int          level = tree.getHeight() ;
	//
	int nx = (int)pow(2.,level-1)+1;
	//
	//------------------
	// output file name
	//------------------
	//
	std::ofstream VTKfile(vtkName.c_str(),std::ios::out);
	//	OptiDisAssert(VTKfile.is_open(),DescribeIosFailure(VTKfile).c_str());
	//---------------------------
	// print generic information
	//---------------------------
	VTKfile << "# vtk DataFile Version 3.0" << "\n";
	VTKfile << header << "\n";

	VTKfile << "ASCII" << "\n";
	VTKfile << "DATASET RECTILINEAR_GRID" << "\n";
	VTKfile << "DIMENSIONS "<<nx<<" "<<nx<<" "<<nx<<" "<<"\n";
	VTKfile << "X_COORDINATES " << nx << " float " <<"\n";
	double start = min.getX(), h = boxWidth / double(nx-1);
	for(int i =0; i < nx; ++i ){
		VTKfile << "  " << start + i*h ;
	}
	VTKfile << "\n";
	VTKfile << "Y_COORDINATES " << nx << " float " <<"\n";
	start = min.getY() ;
	for(int i =0; i < nx; ++i ){
		VTKfile << "  " << start + i*h ;
	}
	VTKfile << "\n";
	VTKfile << "Z_COORDINATES " << nx << " float " <<"\n";
	start = min.getZ() ;
	for(int i =0; i < nx; ++i ){
		VTKfile << "  " << start + i*h ;
	}
	VTKfile << "\n";
	VTKfile << "\n";
}

template <class FReal>
void FIOVtk<FReal>::writeXMLmasterFile(const int & numProc, std::string& VTKname,const std::string& /*title*/)
{

	std::string       filename;

	filename = VTKname + ".pvtp";
	std::ofstream VTKfile(filename.c_str(),std::ios::out);
	//---------------------------
	// print generic information
	//---------------------------
	VTKfile << "<?xml version=\"1.0\" ?>"<< std::endl;
	VTKfile << "<VTKFile type=\"PPolyData\" version=\"0.1\"   byte_order=\"LittleEndian\">" << std::endl;
	VTKfile << "  <PPolyData  GhostLevel=\"0\">" << std::endl;
	VTKfile << "       <PPointData>\n";
	VTKfile << "          <PDataArray type=\"Float64\" Name=\"PhysicalProperties\" NumberOfComponents=\"1\" />\n";
	VTKfile << "          <PDataArray type=\"Float64\" Name=\"Potentials\"         NumberOfComponents=\"1\" />\n";
	VTKfile << "          <PDataArray type=\"Float64\" Name=\"NodalForces\"        NumberOfComponents=\"3\" />\n";
	VTKfile << "       </PPointData>\n";
	long unsigned found = VTKname.find_last_of("/");
	//   il faut mettre les fichiers
	for(int i=0 ; i< numProc ; ++i)
	{
		std::stringstream numproc ;
		numproc <<  i ;
		filename = VTKname.substr(found+1) + numproc.str() +".vtp";
		VTKfile << "           <Piece Source=\""<<filename<<"\"/>" << std::endl;
	}
	VTKfile << "  </PPolyData>" << std::endl;
	VTKfile << "</VTKFile>" << std::endl;
	std::cout << "filename " << filename<< std::endl;
}

template <class FReal>
template<typename OctreeClass>
void FIOVtk<FReal>::writeXMLParticles( const std::string& VTKname,const std::string& /*title*/,  OctreeClass& tree)
{
	//
	// --- SAUVEGARDE
	std::string        filename;
	//------------------
	// output file name
	//------------------
	filename=VTKname +".vtp";
	//
	std::cout << "     Saving in: " << filename << "\n";

	std::ofstream VTKfile(filename.c_str(),std::ios::out);
	//---------------------------
	// print generic information
	//---------------------------
	VTKfile << "<VTKFile type=\"PolyData\" version=\"0.1\"   byte_order=\"LittleEndian\" compressor=\"vtkZLibDataCompressor\">" << std::endl;
	VTKfile << "  <PolyData>" << std::endl;
	//
	typename  OctreeClass::Iterator  octreeIterator(&tree);
    FSize nbNodes=0;
	octreeIterator.gotoBottomLeft();
	do{
		//ContainerClass* const FRestrict segments = octreeIterator.getCurrentListTargets();
		auto * const FRestrict particles = octreeIterator.getCurrentListTargets();
		nbNodes += particles->getNbParticles();
		//    Print::Out << "Index: "<<  octreeIterator.getCurrentGlobalIndex() << " nbEl: " << nbElements <<  "\n";
	} while(octreeIterator.moveRight());


	VTKfile <<"     <Piece NumberOfPoints=\""<<nbNodes<<"\" NumberOfVerts=\"0\" NumberOfLines=\""
			<< 0<<"\" NumberOfStrips=\"0\" NumberOfPolys=\"0\">"<<std::endl;
	//

	VTKfile << "       <PointData>"<<std::endl;
	VTKfile << "          <DataArray type=\"Float64\"   Name=\"PhysicalProperties\"      NumberOfComponents=\"1\" format=\"ascii\">"<< std::endl;
	octreeIterator.gotoBottomLeft();
	do{
		auto * leaf = octreeIterator.getCurrentLeaf()	;
		auto * const physicalValues = leaf->getTargets()->getPhysicalValues();
        FSize nbParticlesInLeaf       = leaf->getTargets()->getNbParticles();
		for(FSize idxPart = 0 ; idxPart < nbParticlesInLeaf ; ++idxPart){
			VTKfile << "  " << physicalValues[idxPart]  ;
		}
	} while(octreeIterator.moveRight());
	VTKfile << std::endl << "          </DataArray>"  << std::endl;
	VTKfile << "          <DataArray type=\"Float64\"   Name=\"Potentials\"      NumberOfComponents=\"1\" format=\"ascii\">"<< std::endl;
	octreeIterator.gotoBottomLeft();
	do{
		auto * leaf = octreeIterator.getCurrentLeaf()	;
		auto * const potentials = leaf->getTargets()->getPotentials();
        FSize nbParticlesInLeaf       = leaf->getTargets()->getNbParticles();
		for(FSize idxPart = 0 ; idxPart < nbParticlesInLeaf ; ++idxPart){
			VTKfile << "  " << potentials[idxPart]  ;
		}
	} while(octreeIterator.moveRight());
	VTKfile << std::endl << "          </DataArray>"  << std::endl;
	VTKfile << "          <DataArray type=\"Float64\" Name=\"NodalForces\"     NumberOfComponents=\"3\" format=\"ascii\">"<< std::endl;
	octreeIterator.gotoBottomLeft();
	do{
		auto * leaf = octreeIterator.getCurrentLeaf()	;
		FReal*const forcesX = leaf->getTargets()->getForcesX();
		FReal*const forcesY = leaf->getTargets()->getForcesY();
		FReal*const forcesZ = leaf->getTargets()->getForcesZ();
        FSize nbParticlesInLeaf       = leaf->getTargets()->getNbParticles();
		for(FSize idxPart = 0 ; idxPart < nbParticlesInLeaf ; ++idxPart){
			VTKfile << "  " << forcesX[idxPart] << "  " << forcesY[idxPart] << "  " << forcesZ[idxPart]  ;
		}
	} while(octreeIterator.moveRight());
	VTKfile << std::endl << "          </DataArray>"  << std::endl;
	VTKfile << "       </PointData>" << std::endl;
	//--------------------------------------
	VTKfile << "       <Points>" << std::endl;
	VTKfile << "          <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">"<< std::endl;
	octreeIterator.gotoBottomLeft();
	do{
		auto * leaf = octreeIterator.getCurrentLeaf()	;
		const FReal* const positionsX = leaf->getTargets()->getPositions()[0];
		const FReal* const positionsY = leaf->getTargets()->getPositions()[1];
		const FReal* const positionsZ = leaf->getTargets()->getPositions()[2];
        FSize nbParticlesInLeaf  = leaf->getTargets()->getNbParticles();
		for(FSize idxPart = 0 ; idxPart < nbParticlesInLeaf ; ++idxPart){
			VTKfile << "  " << positionsX[idxPart] << "  " << positionsY[idxPart] << "  " << positionsZ[idxPart]  ;
		}
	} while(octreeIterator.moveRight());
	// ------------------------------------------
	VTKfile << "          </DataArray>"  << std::endl;
	//--------------------------------------
	VTKfile << "       </Points>" << std::endl;
	VTKfile << "       <Verts>" << std::endl   << "       </Verts>" << std::endl;
	//

	VTKfile << "       <Lines>" << std::endl  << "       </Lines>" << std::endl;

	VTKfile << "       <Strips>" << std::endl  << "       </Strips>" << std::endl;
	VTKfile << "       <Polys>" << std::endl   << "       </Polys>" << std::endl;
	VTKfile << "     </Piece>" << std::endl;
	//

	VTKfile << "  </PolyData>" << std::endl;
	VTKfile << "</VTKFile>" << "\n";

}
#endif /* FIOVTK_HPP*/
