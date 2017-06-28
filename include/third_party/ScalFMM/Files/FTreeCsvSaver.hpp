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
#ifndef FTREECSVSAVER_HPP
#define FTREECSVSAVER_HPP

#include "../Utils/FGlobal.hpp"

#include <cstring>
#include <iostream>
#include <fstream>

/** This class is to export a tree in csv file
  *
  */
template <class FReal, class OctreeClass, class ContainerClass >
class FTreeCsvSaver {
    const bool includeHeader;   //< To include a line of header
    int nbFrames;               //< The current frame
    char basefile[512];         //< The base file name like "~/OUT/simulation%d.csv"

public:
    /** Constructor
      * @param inBasefile is the output file name, you must put %d in it
			* @param inIncludeHeader tells if header must be included
      */
    FTreeCsvSaver(const char inBasefile[], const bool inIncludeHeader = false)
        : includeHeader(inIncludeHeader), nbFrames(0) {
        strcpy(basefile, inBasefile);
    }

    /** Virtual destructor
      */
    virtual ~FTreeCsvSaver(){
    }

    /** to know how many frame has been saved
      */
    int getNbFrames() const {
        return nbFrames;
    }

    /** export a tree
      */
    void exportTree(OctreeClass*const tree){
        char currentFilename[512];
        sprintf(currentFilename, basefile, nbFrames++);

        std::ofstream file(currentFilename, std::ofstream::out );
        if(includeHeader){
            file << "x, y, z, value\n";
        }

        typename OctreeClass::Iterator octreeIterator(tree);
        octreeIterator.gotoBottomLeft();
        do{
            FReal values[4];
            {
                const ContainerClass* container = octreeIterator.getCurrentListTargets();
                for(FSize idxPart = 0 ; idxPart < container->getNbParticles() ; ++idxPart){
                    container->fillToCsv(idxPart, values);
                    file << values[0] << "," << values[1] << "," <<
                            values[2] << "," << values[3] << "\n";
                }
            }

            const bool isUsingTsm = (octreeIterator.getCurrentListTargets() != octreeIterator.getCurrentListSrc());
            if( isUsingTsm ){
                const ContainerClass* container = octreeIterator.getCurrentListSrc();
                for(FSize idxPart = 0 ; idxPart < container->getNbParticles() ; ++idxPart){
                    container->fillToCsv(idxPart, values);
                    file << values[0] << "," << values[1] << "," <<
                            values[2] << "," << values[3] << "\n";
                }
            }
        } while(octreeIterator.moveRight());

        file.close();
    }
};


#endif // FTREECSVSAVER_HPP
