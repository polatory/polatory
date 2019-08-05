// See LICENCE file at project root
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
