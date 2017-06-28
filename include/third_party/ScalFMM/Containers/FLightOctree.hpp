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
#ifndef FLIGHTOCTREE_HPP
#define FLIGHTOCTREE_HPP

/**
* @author Berenger Bramas (berenger.bramas@inria.fr)This class is a light octree
* It is just a linked list with 8 pointers per node
* it is used to store small data in an octree way.
* @warning It can only store one level of data!
* As it is linked, the acess is always made fro the top.
*/
template <class CellClass>
class FLightOctree {
    // The node class
    struct Node {
        Node* next[8];   // Child
        CellClass* cell; // Cell data

        Node() : cell(nullptr) {
            memset(next, 0, sizeof(Node*)*8);
        }

        virtual ~Node(){
            for(int idxNext = 0 ; idxNext < 8 ; ++idxNext){
                delete next[idxNext];
            }
            delete cell;
        }
    };

    // Tree root
    Node root;

public:
    FLightOctree(){
    }

    // Insert a cell
    void insertCell(const MortonIndex& index, int level, CellClass*const inCell){
        Node* iter = &root;

        while(level){
            const int host = (index >> (3 * (level-1))) & 0x07;
            if(!iter->next[host]){
                iter->next[host] = new Node();
            }
            iter = iter->next[host];
            level -= 1;
        }

        iter->cell = inCell;
    }
    // Retreive a cell
    CellClass* getCell(const MortonIndex& index, int level) const{
        const Node* iter = &root;

        while(level){
            const int host = (index >> (3 * (level-1))) & 0x07;
            if(!iter->next[host]){
                return nullptr;
            }
            iter = iter->next[host];
            level -= 1;
        }

        return iter->cell;
    }
};

#endif // FLIGHTOCTREE_HPP
