// See LICENCE file at project root
#ifndef FBASICPARTICLECONTAINER_HPP
#define FBASICPARTICLECONTAINER_HPP

#include "FAbstractParticleContainer.hpp"
#include "FAbstractSerializable.hpp"

#include "../Utils/FGlobal.hpp"
#include "../Utils/FAlignedMemory.hpp"
#include "../Utils/FMath.hpp"
#include "../Utils/FPoint.hpp"
#include "FParticleType.hpp"

#include <array>

/**
 * @author Berenger Bramas (berenger.bramas@inria.fr)
 * Please read the license
 *
 * This class defines a container which can holds one type (AttributeClass) for each particle.
 * The memory is allocated in one block and stores the positions and the request type.
 * For example if one wants to store a struct for each particle:
 * \code
 * struct AStruct{ ... };
 * FBasicParticleContainer<FReal, 1, AStruct> container;
 * \endcode
 * And then access is done using:
 * \code
 * AStruct* strucs = container.getAttributes<0>();
 * \endcode
 * For example if one wants to store 4 doubles for each particles:
 * \code
 * FBasicParticleContainer<FReal, 4, double> container;
 * \endcode
 * And then access is done using:
 * \code
 * double* v1 = container.getAttributes<0>();
 * double* v2 = container.getAttributes<1>();
 * double* v3 = container.getAttributes<2>();
 * double* v4 = container.getAttributes<3>();
 * \endcode
 * The memory is aligned to FP2PDefaultAlignement value.
 */
template <class FReal, unsigned NbAttributesPerParticle, class AttributeClass >
class FBasicParticleContainer : public FAbstractParticleContainer<FReal>, public FAbstractSerializable {
protected:
    static const FSize MemoryAlignement   = FP2PDefaultAlignement;
    static const FSize DefaultNbParticles = FSize(MemoryAlignement/sizeof(FReal));

    /** The number of particles in the container */
    FSize nbParticles;
    /** 3 pointers to 3 arrays of real to store the position */
    FReal* positions[3];
    /** The attributes requested by the user */
    AttributeClass* attributes[NbAttributesPerParticle];

    /** The allocated memory */
    FSize allocatedParticles;

    /////////////////////////////////////////////////////
    /////////////////////////////////////////////////////

    /** Ending call for pushing the attributes */
    template<int index>
    void addParticleValue(const FSize /*insertPosition*/){
    }

    /** Ending call for pushing array of attributes */
    template<int index>
    void addParticleValueS(const FSize /*insertPosition*/,const FSize /*nbParticles*/){
    }


    /** Filling call for each attributes values */
    template<int index, typename... Args>
    void addParticleValue(const FSize insertPosition, const AttributeClass value, Args... args){
        // Compile test to ensure indexing
        static_assert(index < NbAttributesPerParticle, "Index to get attributes is out of scope.");
        // insert the value
        attributes[index][insertPosition] = value;
        // Continue for reamining values
        addParticleValue<index+1>( insertPosition, args...);
    }

    /** Filling call for each attributes values
     * add multiples attributes from arrays
     */
    template<int index, typename... Args>
    void addParticleValueS(const FSize insertPosition, const FSize nbParts, const AttributeClass* value, Args... args){
        // Compile test to ensure indexing
        static_assert(index < NbAttributesPerParticle, "Index to get attributes is out of scope.");
        for(FSize idxPart = 0; idxPart<nbParts ; ++idxPart){
            // insert the value
            attributes[index][insertPosition+idxPart] = value[idxPart];
            // Continue for reamining values
        }
        addParticleValueS<index+1>( insertPosition, nbParts, args...);

    }


    /////////////////////////////////////////////////////
    /////////////////////////////////////////////////////

    void increaseSizeIfNeeded(FSize sizeInput = 1){
        if( nbParticles+(sizeInput-1) >= allocatedParticles ){
            // allocate memory
            const FSize moduloParticlesNumber = (MemoryAlignement/sizeof(FReal));
            allocatedParticles = (FMath::Max(DefaultNbParticles,FSize(FReal(nbParticles+sizeInput)*1.5)) + moduloParticlesNumber - 1) & ~(moduloParticlesNumber-1);
            // init with 0
            const size_t allocatedBytes = (sizeof(FReal)*3 + sizeof(AttributeClass)*NbAttributesPerParticle)*allocatedParticles;
            FReal* newData  = reinterpret_cast<FReal*>(FAlignedMemory::AllocateBytes<MemoryAlignement>(allocatedBytes));
            memset( newData, 0, allocatedBytes);
            // copy memory
            const char*const toDelete  = reinterpret_cast<const char*>(positions[0]);
            for(int idx = 0 ; idx < 3 ; ++idx){
                memcpy(newData + (allocatedParticles * idx), positions[idx], sizeof(FReal) * nbParticles);
                positions[idx] = newData + (allocatedParticles * idx);
                for(FSize idxEmpty = nbParticles ; idxEmpty < allocatedParticles ; ++idxEmpty){
                    positions[idx][idxEmpty] = std::numeric_limits<FReal>::max()/2;
                }
            }
            // copy attributes
            AttributeClass* startAddress = reinterpret_cast<AttributeClass*>(positions[2] + allocatedParticles);
            for(unsigned idx = 0 ; idx < NbAttributesPerParticle ; ++idx){
                memcpy(startAddress + (allocatedParticles * idx), attributes[idx], sizeof(AttributeClass) * nbParticles);
                attributes[idx] = startAddress + (idx * allocatedParticles);
            }
            // delete old
            FAlignedMemory::DeallocBytes(toDelete);
        }
    }

public:
    /////////////////////////////////////////////////////
    /////////////////////////////////////////////////////

    FBasicParticleContainer(const FBasicParticleContainer&)            = delete;
    FBasicParticleContainer& operator=(const FBasicParticleContainer&) = delete;

    /////////////////////////////////////////////////////
    /////////////////////////////////////////////////////

    /** Basic contructor */
    FBasicParticleContainer() : nbParticles(0), allocatedParticles(0){
        memset(positions, 0, sizeof(positions[0]) * 3);
        memset(attributes, 0, sizeof(attributes[0]) * NbAttributesPerParticle);
    }

    /** Simply dalloc the memory using first pointer
   */
    ~FBasicParticleContainer(){
        FAlignedMemory::DeallocBytes(positions[0]);
    }

    /**
   * @brief getNbParticles
   * @return  the number of particles
   */
    FSize getNbParticles() const{
        return nbParticles;
    }
    /**
   * @brief reset the number of particles
   * @warning Only the number of particles is set to 0, the particles are still here.
   */
    void resetNumberOfParticles()
    {
        nbParticles = 0 ;
    }
    /**
   * @brief getPositions
   * @return a FReal*[3] to get access to the positions
   */
    const FReal*const* getPositions() const {
        return positions;
    }

    /**
   * @brief getWPositions
   * @return get the position in write mode
   */
    FReal* const* getWPositions() {
        return positions;
    }

    /**
   * @brief getAttribute
   * @param index
   * @return the attribute at index index
   */
    AttributeClass* getAttribute(const int index) {
        return attributes[index];
    }

    /**
   * @brief getAttribute
   * @param index
   * @return
   */
    const AttributeClass* getAttribute(const int index) const {
        return attributes[index];
    }

    /**
   * Get the attribute with a forcing compile optimization
   */
    template <int index>
    AttributeClass* getAttribute() {
        static_assert(index < NbAttributesPerParticle, "Index to get attributes is out of scope.");
        return attributes[index];
    }

    /**
   * Get the attribute with a forcing compile optimization
   */
    template <int index>
    const AttributeClass* getAttribute() const {
        static_assert(index < NbAttributesPerParticle, "Index to get attributes is out of scope.");
        return attributes[index];
    }

    /////////////////////////////////////////////////////
    /////////////////////////////////////////////////////


    /**
   * Push multiple particles
   * Should have a particle position fallowed by attributes
   * @param Array of position, number of parts to insert, followed by array of attribute
   */
    template<typename... Args>
    void pushArray(const FPoint<FReal> * inParticlePosition, FSize numberOfParts, Args... args){
        const FSize positionToInsert = nbParticles;
        //Tests if enough space
        increaseSizeIfNeeded(numberOfParts);
        for(FSize idxPart = 0; idxPart<numberOfParts ; ++idxPart){
            // insert particle data
            positions[0][positionToInsert + idxPart] = inParticlePosition[idxPart].getX();
            positions[1][positionToInsert + idxPart] = inParticlePosition[idxPart].getY();
            positions[2][positionToInsert + idxPart] = inParticlePosition[idxPart].getZ();

        }
        // insert attribute data
        addParticleValueS<0>( nbParticles, numberOfParts ,args...);
        nbParticles += numberOfParts;

    }

    /**
     * Allocate the data to store requiereNbParticles particles
     */
    void reserve(const FSize requiereNbParticles){
        increaseSizeIfNeeded(requiereNbParticles - nbParticles);
    }


    /**
   * Push called bu FSimpleLeaf
   * Should have a particle position fallowed by attributes
   */
    template<typename... Args>
    void push(const FPoint<FReal>& inParticlePosition, Args... args){
        // enought space?
        increaseSizeIfNeeded();

        // insert particle data
        positions[0][nbParticles] = inParticlePosition.getX();
        positions[1][nbParticles] = inParticlePosition.getY();
        positions[2][nbParticles] = inParticlePosition.getZ();
        // insert attribute data
        addParticleValue<0>( nbParticles, args...);
        nbParticles += 1;
    }


    /**
   * Push called bu FSimpleLeaf
   * Should have a particle position fallowed by attributes
   */
    template<typename... Args>
    void push(const FPoint<FReal>& inParticlePosition, const std::array<AttributeClass , NbAttributesPerParticle>& values){
        // enought space?
        increaseSizeIfNeeded();

        // insert particle data
        positions[0][nbParticles] = inParticlePosition.getX();
        positions[1][nbParticles] = inParticlePosition.getY();
        positions[2][nbParticles] = inParticlePosition.getZ();
        // insert attribute data
        for(unsigned idxVal = 0 ; idxVal < NbAttributesPerParticle ; ++idxVal){
            attributes[idxVal][nbParticles] = values[idxVal];
        }
        nbParticles += 1;
    }

    /**
   * Push called by FTypedLeaf Through arranger
   * Should have a particle position fallowed by isTarget flag and attributes
   */
    template<typename... Args>
    void push(const FPoint<FReal>& inParticlePosition, const FParticleType type,
              const std::array<AttributeClass , NbAttributesPerParticle>& values){
        push(inParticlePosition,values);
    }

    /**
   * Push called usually by FTypedLeaf with the isTarget flag in addition
   */
    template<typename... Args>
    void push(const FPoint<FReal>& inParticlePosition, const FParticleType /*particleType*/, Args... args){
        push(inParticlePosition, args...);
    }

    /** set nb particles to 0 */
    void clear(){
        nbParticles = 0;
    }

    /** to enable rearranging
   * indexesToRemove must be sorted
   * it removes all the particles at position indexesToRemove
   */
    void removeParticles(const FSize indexesToRemove[], const FSize nbParticlesToRemove){
        FSize offset = 1;
        FSize idxIndexes = 1;
        FSize idxIns = indexesToRemove[0] + 1;
        for( ; idxIns < nbParticles && idxIndexes < nbParticlesToRemove ; ++idxIns){
            if( idxIns == indexesToRemove[idxIndexes] ){
                idxIndexes += 1;
                offset += 1;
            }
            else{
                for(int idx = 0 ; idx < 3 ; ++idx){
                    positions[idx][idxIns-offset] = positions[idx][idxIns];
                }
                for(unsigned idx = 0 ; idx < NbAttributesPerParticle ; ++idx){
                    attributes[idx][idxIns-offset] = attributes[idx][idxIns];
                }
            }
        }
        for( ; idxIns < nbParticles ; ++idxIns){
            for(int idx = 0 ; idx < 3 ; ++idx){
                positions[idx][idxIns-offset] = positions[idx][idxIns];
            }
            for(unsigned idx = 0 ; idx < NbAttributesPerParticle ; ++idx){
                attributes[idx][idxIns-offset] = attributes[idx][idxIns];
            }
        }
        nbParticles -= nbParticlesToRemove;
    }

    /////////////////////////////////////////////////////
    /////////////////////////////////////////////////////

    AttributeClass* getRawData(){
        return reinterpret_cast<AttributeClass*>(positions[2] + allocatedParticles);
    }

    const AttributeClass* getRawData() const {
        return reinterpret_cast<AttributeClass*>(positions[2] + allocatedParticles);
    }

    FSize getLeadingRawData() const {
        return allocatedParticles;
    }

    /////////////////////////////////////////////////////
    /////////////////////////////////////////////////////

    /** The size to send a leaf */
    FSize getSavedSize() const{
        return FSize(sizeof(nbParticles) + nbParticles * (3 * sizeof(FReal) + NbAttributesPerParticle * sizeof(AttributeClass)));
    }

    /** Save the current cell in a buffer */
    template <class BufferWriterClass>
    void save(BufferWriterClass& buffer) const{
        buffer << nbParticles;
        for(int idx = 0 ; idx < 3 ; ++idx){
            buffer.write(positions[idx], nbParticles);
        }
        for(unsigned idx = 0 ; idx < NbAttributesPerParticle ; ++idx){
            buffer.write(attributes[idx], nbParticles);
        }
    }
    /** Restore the current cell from a buffer */
    template <class BufferReaderClass>
    void restore(BufferReaderClass& buffer){
        buffer >> nbParticles;
        if( nbParticles >= allocatedParticles ){
            // allocate memory
            const FSize moduloParticlesNumber = (MemoryAlignement/sizeof(FReal));
            allocatedParticles = (nbParticles + moduloParticlesNumber - 1) & ~(moduloParticlesNumber-1);
            // init with 0
            const size_t allocatedBytes = (sizeof(FReal)*3 + sizeof(AttributeClass)*NbAttributesPerParticle)*allocatedParticles;
            FReal* newData  = reinterpret_cast<FReal*>(FAlignedMemory::AllocateBytes<MemoryAlignement>(allocatedBytes));
            memset( newData, 0, allocatedBytes);

            FAlignedMemory::DeallocBytes(positions[0]);
            for(int idx = 0 ; idx < 3 ; ++idx){
                positions[idx] = newData + (allocatedParticles * idx);
            }
            AttributeClass* startAddress = reinterpret_cast<AttributeClass*>(positions[2] + allocatedParticles);
            for(unsigned idx = 0 ; idx < NbAttributesPerParticle ; ++idx){
                attributes[idx] = startAddress + (idx * allocatedParticles);
            }
        }
        for(int idx = 0 ; idx < 3 ; ++idx){
            buffer.fillArray(positions[idx], nbParticles);
        }
        for(unsigned idx = 0 ; idx < NbAttributesPerParticle ; ++idx){
            buffer.fillArray(attributes[idx], nbParticles);
        }
    }

    /** Reset the attributes to zeros */
    void resetToInitialState(){
        for(unsigned idx = 0 ; idx < NbAttributesPerParticle ; ++idx){
            memset(attributes[idx], 0, sizeof(AttributeClass) * allocatedParticles);
        }
    }

    /** Reset the attributes to zeros */
    void resetToInitialState(const int idxAttribute){
        memset(attributes[idxAttribute], 0, sizeof(AttributeClass) * allocatedParticles);
    }
};


#endif //FBASICPARTICLECONTAINER_HPP
