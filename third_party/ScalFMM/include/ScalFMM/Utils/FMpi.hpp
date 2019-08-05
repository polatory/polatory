// See LICENCE file at project root
#ifndef FMPI_HPP
#define FMPI_HPP


#include <cstdio>
#include <stdexcept>

#include "FGlobal.hpp"
#ifndef SCALFMM_USE_MPI
#error The MPI header is included while SCALFMM_USE_MPI is turned OFF
#endif


#include "FNoCopyable.hpp"
#include "FMath.hpp"
#include "FAssert.hpp"

//Need that for converting datas
#include "FComplex.hpp"


/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////


#include <mpi.h>


/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////


/**
 * @brief MPI context management
 * @author Berenger Bramas (berenger.bramas@inria.fr)
 *
 * This class manages the initialization and destruction of the MPI context. It
 * also gives access to the main communicator via its `const FComm& global()`
 * member.
 *
 * @warning Do not create more that one instance of this class for the whole
 * program lifetime. MPI does not support multiple initializations.
 *
 * Please read the license.
 */

class FMpi {
public:
    ////////////////////////////////////////////////////////
    // MPI Flag
    ////////////////////////////////////////////////////////
    enum FMpiTag {
        // FMpiTreeBuilder
        TagExchangeIndexs = 100,
        TagSplittedLeaf = 200,
        TagExchangeNbLeafs = 300,
        TagSandSettling = 400,

        // FQuickSort
        TagQuickSort = 500,

        // FMM
        TagFmmM2M = 1000,
        TagFmmM2MSize = 1500,
        TagFmmL2L = 2000,
        TagFmmL2LSize = 2500,
        TagFmmP2P = 3000,

        // Bitonic,
        TagBitonicMin = 4000,
        TagBitonicMax = 5000,
        TagBitonicMinMess = 6000,
        TagBitonicMaxMess = 7000,

        // Last defined tag
        TagLast = 8000,
    };

    ////////////////////////////////////////////////////////
    // FComm to factorize MPI_Comm work
    ////////////////////////////////////////////////////////
    
    /**
     * \brief MPI comunicator abstraction
     *
     * This class is used to gather the usual methods related to identifying an
     * MPI communicator.
     */
    class FComm {
        int rank;   ///< rank related to the comm
        int nbProc; ///< nb proc in this group

        MPI_Comm communicator;  ///< current mpi communicator
        MPI_Group group;        ///< current mpi group


        /// Updates current process rank and process count from mpi
        void reset(){
            FMpi::Assert( MPI_Comm_rank(communicator,&rank),  __LINE__ );
            FMpi::Assert( MPI_Comm_size(communicator,&nbProc),  __LINE__ );
        }

    public:
        /// Constructor : duplicates the given communicator
        explicit FComm(MPI_Comm inCommunicator ) {
            FMpi::Assert( MPI_Comm_dup(inCommunicator, &communicator),  __LINE__ , "comm dup");
            FMpi::Assert( MPI_Comm_group(communicator, &group),  __LINE__ , "comm group");

            reset();
        }

        /// Constructor : duplicates the given communicator
        FComm(const FComm& inCommunicator ) {
            FMpi::Assert( MPI_Comm_dup(inCommunicator.communicator, &communicator),  __LINE__ , "comm dup");
            FMpi::Assert( MPI_Comm_group(communicator, &group),  __LINE__ , "comm group");

            reset();
        }

        FComm& operator=(const FComm& inCommunicator ) {
            FMpi::Assert( MPI_Comm_free(&communicator),  __LINE__ );
            FMpi::Assert( MPI_Group_free(&group),  __LINE__ );

            FMpi::Assert( MPI_Comm_dup(inCommunicator.communicator, &communicator),  __LINE__ , "comm dup");
            FMpi::Assert( MPI_Comm_group(communicator, &group),  __LINE__ , "comm group");

            reset();

            return *this;
        }

        /// Frees communicator and group
        virtual ~FComm(){
            FMpi::Assert( MPI_Comm_free(&communicator),  __LINE__ );
            FMpi::Assert( MPI_Group_free(&group),  __LINE__ );
        }

        /// Gets the underlying MPI communicator
        MPI_Comm getComm() const {
            return communicator;
        }

        /// Gets the current rank
        int processId() const {
            return rank;
        }

        /// The current number of procs in the group */
        int processCount() const {
            return nbProc;
        }

        ////////////////////////////////////////////////////////////
        // Split/Chunk functions
        ////////////////////////////////////////////////////////////

        /** Get a left index related to a size */
        template< class T >
        T getLeft(const T inSize)  const {
            const double step = (double(inSize) / double(processCount()));
            return T(FMath::Ceil(step * double(processId())));
        }

        /** Get a right index related to a size */
        template< class T >
        T getRight(const T inSize)  const {
            const double step = (double(inSize) / double(processCount()));
            const T res = T(FMath::Ceil(step * double(processId()+1)));
            if(res > inSize) return inSize;
            else return res;
        }

        /** Get a right index related to a size and another id */
        template< class T >
        T getOtherRight(const T inSize, const int other)  const {
            const double step = (double(inSize) / double(processCount()));
            const T res = T(FMath::Ceil(step * double(other+1)));
            if(res > inSize) return inSize;
            else return res;
        }

        /** Get a left index related to a size and another id */
        template< class T >
        T getOtherLeft(const T inSize, const int other) const {
            const double step = (double(inSize) / double(processCount()));
            return T(FMath::Ceil(step * double(other)));
        }

        /** Get a proc id from and index */
        template< class T >
        int getProc(const int position, const T inSize) const {
            const double step = (double(inSize) / processCount());
            return int(position/step);
        }

        ////////////////////////////////////////////////////////////
        // Mpi interface functions
        ////////////////////////////////////////////////////////////


        /** Reduce a value for proc == 0 */
        template< class T >
        T reduceSum(T data) const {
            T result(0);
            FMpi::Assert( MPI_Reduce( &data, &result, 1, FMpi::GetType(data), MPI_SUM, 0, communicator ), __LINE__);
            return result;
        }

        /** Reduce a value for all procs */
        template< class T >
        T allReduceSum(T data) const {
            T result(0);
            FMpi::Assert( MPI_Allreduce( &data, &result, 1, FMpi::GetType(data), MPI_SUM, communicator ), __LINE__);
            return result;
        }

        /** Reduce an average */
        template< class T >
        T reduceAverageAll(T data) const {
            T result[processCount()];
            FMpi::Assert( MPI_Allgather( &data, 1, FMpi::GetType(data), result, 1, FMpi::GetType(data), getComm()),  __LINE__ );

            T average = 0;
            for(int idxProc = 0 ; idxProc < processCount() ;++idxProc){
                average += result[idxProc] / processCount();
            }
            return average;
        }

        /** Change the group size */
        void groupReduce(const int from , const int to){
            int * procsIdArray = new int [to - from + 1];
            for(int idxProc = from ;idxProc <= to ; ++idxProc){
                procsIdArray[idxProc - from] = idxProc;
            }

            MPI_Group previousGroup = group;
            FMpi::Assert( MPI_Group_incl(previousGroup, to - from + 1 , procsIdArray, &group),  __LINE__ );

            MPI_Comm previousComm = communicator;
            FMpi::Assert( MPI_Comm_create(previousComm, group, &communicator),  __LINE__ );

            MPI_Comm_free(&previousComm);
            MPI_Group_free(&previousGroup);

            reset();
            delete[]  procsIdArray ;
        }

        /** Change the group, create one groupd where processInGroup[i] != 0
          * and another where processInGroup[i] == 0
          */
        void groupReduce(const int processInGroup[]){
            int * procsIdArray = new int [nbProc];
            int counterNewGroup = 0;
            for(int idxProc = 0 ;idxProc < nbProc ; ++idxProc){
                if(processInGroup[rank] && processInGroup[idxProc]){
                    procsIdArray[counterNewGroup++] = idxProc;
                }
                else if(!processInGroup[rank] && !processInGroup[idxProc]){
                    procsIdArray[counterNewGroup++] = idxProc;
                }
            }

            MPI_Group previousGroup = group;
            FMpi::Assert( MPI_Group_incl(previousGroup, counterNewGroup , procsIdArray, &group),  __LINE__ );

            MPI_Comm previousComm = communicator;
            FMpi::Assert( MPI_Comm_create(previousComm, group, &communicator),  __LINE__ );

            MPI_Comm_free(&previousComm);
            MPI_Group_free(&previousGroup);

            reset();
            FAssertLF(nbProc == counterNewGroup);
            delete[]  procsIdArray ;
        }

        void barrier() const {
            FMpi::Assert(MPI_Barrier(getComm()), __LINE__);
        }

        bool hasPendingMessage() const {
            MPI_Status status;
            int flag = 0;
            MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, getComm(), &flag, &status);
            return (flag != 0);
        }
    };

    ////////////////////////////////////////////////////////
    // FMpi methods
    ////////////////////////////////////////////////////////

    /// Constructor
    /**
     * We use `MPI_Init_thread` because of an openmpi error:
     *
     *     [fourmi062:15896] [[13237,0],1]-[[13237,1],1] mca_oob_tcp_msg_recv: readv failed: Connection reset by peer (104)
     *     [fourmi056:04597] [[13237,0],3]-[[13237,1],3] mca_oob_tcp_msg_recv: readv failed: Connection reset by peer (104)
     *     [fourmi053:08571] [[13237,0],5]-[[13237,1],5] mca_oob_tcp_msg_recv: readv failed: Connection reset by peer (104)
     *    
     * Error for process 1:
     *
     *     [[13237,1],1][btl_openib_component.c:3227:handle_wc] from fourmi062 to: fourmi056 error polling LP CQ with status LOCAL LENGTH ERROR status number 1 for wr_id 7134664 opcode 0  vendor error 105 qp_idx 3
     *                ^
     *
     * All processes raise the same error, the second 1 (see caret) is replaced by the rank.
     */
    FMpi() : communicator(nullptr) {
        if( instanceCount > 0) {
            throw std::logic_error("FMpi should not be instanciated more than once.");
        } else {
            instanceCount++;
        }

        int provided = 0;
        FMpi::Assert( MPI_Init_thread(nullptr,nullptr, MPI_THREAD_SERIALIZED, &provided), __LINE__);
        communicator = new FComm(MPI_COMM_WORLD);
    }

    /// Constructor
    FMpi(int inArgc, char **  inArgv ) : communicator(nullptr) {
        if( instanceCount > 0) {
            throw std::logic_error("FMpi should not be instanciatedmore than once.");
        } else {
            instanceCount++;
        }

        int provided = 0;
        FMpi::Assert( MPI_Init_thread(&inArgc,&inArgv, MPI_THREAD_SERIALIZED, &provided), __LINE__);
        communicator = new FComm(MPI_COMM_WORLD);
    }

    /// Delete the communicator and call `MPI_Finalize`
    ~FMpi(){
        delete communicator;
        MPI_Finalize();
    }

    /// Get the global communicator
    const FComm& global() {
        return (*communicator);
    }

    ////////////////////////////////////////////////////////////
    // Mpi Types meta function
    ////////////////////////////////////////////////////////////

    /// Get the MPI datatype corresponding to a variable.
    static MPI_Datatype GetType(const long long&){
        return MPI_LONG_LONG;
    }
    static int GetTypeCount(const long long&){
        return 1;
    }

    /// Get the MPI datatype corresponding to a variable.
    static MPI_Datatype GetType(const long int&){
        return MPI_LONG;
    }
    static int GetTypeCount(const long int&){
        return 1;
    }

    /// Get the MPI datatype corresponding to a variable.
    static MPI_Datatype GetType(const double&){
        return MPI_DOUBLE;
    }
    static int GetTypeCount(const double&){
        return 1;
    }

    /// Get the MPI datatype corresponding to a variable.
    static MPI_Datatype GetType(const float&){
        return MPI_FLOAT;
    }
    static int GetTypeCount(const float&){
        return 1;
    }

    /// Get the MPI datatype corresponding to a variable.
    static MPI_Datatype GetType(const int&){
        return MPI_INT;
    }
    static int GetTypeCount(const int&){
        return 1;
    }

    /// Get the MPI datatype corresponding to a variable.
    static MPI_Datatype GetType(const char&){
        return MPI_CHAR;
    }
    static int GetTypeCount(const char&){
        return 1;
    }

    /// Get the MPI datatype corresponding to a variable.
    static MPI_Datatype GetType(const unsigned char&){
        return MPI_UNSIGNED_CHAR;
    }
    static int GetTypeCount(const unsigned char&){
        return 1;
    }

    /// Get the MPI datatype corresponding to a variable.
    template <class FReal>
    static MPI_Datatype GetType(const FComplex<FReal>& a){
        return GetType(a.getReal());
    }

    template <class FReal>
    static int GetTypeCount(const FComplex<FReal>& a){
        return 2;
    }

    ////////////////////////////////////////////////////////////
    // Mpi interface functions
    ////////////////////////////////////////////////////////////

    /// Generic mpi assert function
    static void Assert(const int test, const unsigned line, const char* const message = nullptr){
        if(test != MPI_SUCCESS){
            printf("[ERROR-QS] Test failled at line %d, result is %d", line, test);
            if(message) printf(", message: %s",message);
            printf("\n");
            fflush(stdout);
            MPI_Abort(MPI_COMM_WORLD, int(line) );
        }
    }

    /// Compute a left index from data
    template <class T>
    static T GetLeft(const T inSize, const int inIdProc, const int inNbProc) {
        const double step = (double(inSize) / inNbProc);
        return T(ceil(step * inIdProc));
    }

    /// Compute a right index from data
    template <class T>
    static T GetRight(const T inSize, const int inIdProc, const int inNbProc) {
        const double step = (double(inSize) / inNbProc);
        const T res = T(ceil(step * (inIdProc+1)));
        if(res > inSize) return inSize;
        else return res;
    }

    /// Compute a proc id from index & data */
    template <class T>
    static int GetProc(const T position, const T inSize, const int inNbProc) {
        const double step = double(inSize) / double(inNbProc);
        return int(double(position)/step);
    }

    /// assert if mpi error */
    static void MpiAssert(const int test, const unsigned line, const char* const message = nullptr){
        if(test != MPI_SUCCESS){
            printf("[ERROR] Test failed at line %d, result is %d", line, test);
            if(message) printf(", message: %s",message);
            printf("\n");
            fflush(stdout);
            MPI_Abort(MPI_COMM_WORLD, int(line) );
        }
    }

    static const size_t MaxBytesPerDivMess = 20000000;

    template <class ObjectType, class VectorType>
    static int ISendSplit(const ObjectType toSend[], const size_t nbItems,
                          const int dest, const int tagBase, const FMpi::FComm& communicator,
                          VectorType* requestVector){
        const size_t totalByteToSend  = (nbItems*sizeof(ObjectType));
        unsigned char*const ptrDataToSend = (unsigned char*)const_cast<ObjectType*>(toSend);
        for(size_t idxSize = 0 ; idxSize < totalByteToSend ; idxSize += MaxBytesPerDivMess){
            MPI_Request currentRequest;
            const size_t nbBytesInMessage = FMath::Min(MaxBytesPerDivMess, totalByteToSend-idxSize);
            FAssertLF(nbBytesInMessage < std::numeric_limits<int>::max());
            FMpi::Assert( MPI_Isend(&ptrDataToSend[idxSize], int(nbBytesInMessage), MPI_BYTE , dest,
                          tagBase + int(idxSize/MaxBytesPerDivMess), communicator.getComm(), &currentRequest) , __LINE__);

            requestVector->push_back(currentRequest);
        }
        return int((totalByteToSend+MaxBytesPerDivMess-1)/MaxBytesPerDivMess);
    }

    template <class ObjectType, class VectorType>
    static int IRecvSplit(ObjectType toRecv[], const size_t nbItems,
                          const int source, const int tagBase, const FMpi::FComm& communicator,
                          VectorType* requestVector){
        const size_t totalByteToRecv  = (nbItems*sizeof(ObjectType));
        unsigned char*const ptrDataToRecv = (unsigned char*)(toRecv);
        for(size_t idxSize = 0 ; idxSize < totalByteToRecv ; idxSize += MaxBytesPerDivMess){
            MPI_Request currentRequest;
            const size_t nbBytesInMessage = FMath::Min(MaxBytesPerDivMess, totalByteToRecv-idxSize);
            FAssertLF(nbBytesInMessage < std::numeric_limits<int>::max());
            FMpi::Assert( MPI_Irecv(&ptrDataToRecv[idxSize], int(nbBytesInMessage), MPI_BYTE , source,
                          tagBase + int(idxSize/MaxBytesPerDivMess), communicator.getComm(), &currentRequest) , __LINE__);

            requestVector->push_back(currentRequest);
        }
        return int((totalByteToRecv+MaxBytesPerDivMess-1)/MaxBytesPerDivMess);
    }

private:
    /// The original communicator
    FComm* communicator;
    
    /// Counter to avoid several instanciations
    static int instanceCount;
};


#endif //FMPI_HPP


