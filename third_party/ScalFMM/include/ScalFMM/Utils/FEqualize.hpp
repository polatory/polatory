// See LICENCE file at project root
#ifndef FEQUALIZE_HPP
#define FEQUALIZE_HPP


#include <iostream>
#include <vector>

#include "../Utils/FAssert.hpp"
#include "../Utils/FMath.hpp"

/**
 * This class proposes a method to distribute an interval own by a worker
 * to some others.
 * It returns a vector of Package which tell what interval to sent to which process.
 * The algorithm works only if the objectives (the intervals that the workers shoud obtain
 * are exclusive and given in ascendant order).
 * Also each of the element from the current interval must belong to someone!
 * Finally the current worker is included in the Package if its objective interval
 * is related to its starting interval.
 */
class FEqualize {
    // Just return the min
    static size_t Min(const size_t v1, const size_t v2){
        return v1 < v2 ? v1 : v2;
    }

public:
    /** To represent an interval to proceed */
    struct Package{
        int idProc;
        size_t elementFrom;
        size_t elementTo;
    };


    /**
     * To know what to send to who.
     * @param myCurrentInterval current process interval
     * @param allObjectives the intervals that each process should have (in ascendant order, exclusive)
     * @return the package that the current worker should sent to others
     */
    static std::vector<Package> GetPackToSend(const std::pair<size_t, size_t> myCurrentInterval,
                                              const std::vector< std::pair<size_t,size_t> >& allObjectives){
        std::vector<Package> packToSend;

        unsigned int idxProc = 0;

        // Find the first proc to send to
        while( idxProc != allObjectives.size()
               && allObjectives[idxProc].second < myCurrentInterval.first){
            idxProc += 1;
        }

        // We will from the first element for sure
        size_t currentElement = 0;

        // Check each proc to send to
        while( idxProc != allObjectives.size()
               && allObjectives[idxProc].first < myCurrentInterval.second){
            Package pack;
            pack.idProc = idxProc;
            // The current element must start where the previous one end
            FAssertLF(allObjectives[idxProc].first < myCurrentInterval.first
                      || currentElement == allObjectives[idxProc].first - myCurrentInterval.first );
            pack.elementFrom = currentElement;
            pack.elementTo   = Min(allObjectives[idxProc].second , myCurrentInterval.second) - myCurrentInterval.first;
            // Next time give from the previous end
            currentElement   = pack.elementTo;

            if(pack.elementTo - pack.elementFrom != 0){
                packToSend.push_back(pack);
            }
            // Progress
            idxProc += 1;
        }
        FAssertLF(currentElement == myCurrentInterval.second - myCurrentInterval.first);

        return packToSend;
    }


    /**
     * To know what to recv from who.
     * @param myObjectiveInterval Interval I should have
     * @param allCurrentIntervals the intevals that each process currently contains
     * @return the package that the current worker should recv from others
     */
    static std::vector<Package> GetPackToRecv(const std::pair<size_t, size_t> myObjectiveInterval,
                                              const std::vector< std::pair<size_t,size_t> >& allCurrentIntervals){
        return GetPackToSend(myObjectiveInterval,allCurrentIntervals);
    }
};

#endif // FEQUALIZE_HPP
