#ifndef _MPI_HPP_
#define _MPI_HPP_

#include "inria/utils.hpp"

#include "mpi.h"

#include <algorithm>
#include <memory>
#include <ostream>

/**
 * \brief Simple MPI wrapper for idiomatic C++
 *
 * See MPI official documentation for argument description.
 *
 * The aim of this wrapper is not to be exhaustive, it only replaces the most
 * common MPI functions. The MPI objects are made to be 100% layout compatible
 * with the original one (cast is safe).
 *
 */

namespace inria
{
    namespace mpi
    {
        void init(int& argc, char**& argv) { MPI_Init(&argc, &argv); }
        void init() { MPI_Init(nullptr, nullptr); }

        void finalize() { MPI_Finalize(); }

        bool initialized() noexcept
        {
            int res = 0;
            MPI_Initialized(&res);
            return res;
        }

        bool finalized()
        {
            int res;
            MPI_Finalized(&res);
            return res;
        }

        class [[gnu::unused]] environment_setup
        {
            bool already_initialized;

          public:
            explicit environment_setup(int& argc, char**& argv)
            {
                already_initialized = ::inria::mpi::initialized();
                if(!already_initialized)
                {
                    ::inria::mpi::init(argc, argv);
                }
            }
            explicit environment_setup()
            {
                already_initialized = ::inria::mpi::initialized();
                if(!already_initialized)
                {
                    ::inria::mpi::init();
                }
            }
            environment_setup(const environment_setup&) = delete;
            environment_setup(environment_setup&&) = delete;
            environment_setup& operator=(const environment_setup&) = delete;
            environment_setup& operator=(environment_setup&&) = delete;
            ~environment_setup()
            {
                if(!already_initialized)
                {
                    ::inria::mpi::finalize();
                }
            }
        };

        struct datatype
        {
            MPI_Datatype raw_type;
            static datatype create_struct(int count, int array_of_blocklengths[], MPI_Aint array_of_displacements[],
                                          datatype array_of_types[])
            {
                datatype new_type{};
                MPI_Type_create_struct(count, array_of_blocklengths, array_of_displacements,
                                       reinterpret_cast<MPI_Datatype*>(array_of_types),
                                       reinterpret_cast<MPI_Datatype*>(&new_type));
                return new_type;
            }
        };

        static_assert(std::is_standard_layout<datatype>::value && (sizeof(datatype) == sizeof(MPI_Datatype)),
                      "mpi::datatype is not layout compatible with MPI_Status");

        template<class T>
        struct datatype_trait
        {
            using type = T;
            static constexpr MPI_Datatype get_type() { return MPI_BYTE; }
        };

        template<class DatatypeTrait>
        constexpr std::size_t get_msg_count(std::size_t item_count)
        {
            return DatatypeTrait::get_type() == MPI_BYTE ? item_count * sizeof(typename DatatypeTrait::type)
                                                         : item_count;
        }

        struct status
        {
            MPI_Status raw_status;

            int get_count(const MPI_Datatype& datatype) const
            {
                int count = -1;
                MPI_Get_count(const_cast<MPI_Status*>(&this->raw_status), datatype, &count);
                return count;
            }

            int tag() const noexcept { return raw_status.MPI_TAG; }

            int source() const noexcept { return raw_status.MPI_SOURCE; }
        };

        static_assert(std::is_standard_layout<status>::value && (sizeof(status) == sizeof(MPI_Status)),
                      "mpi::status is not layout compatible with MPI_Status");

        struct request
        {
            MPI_Request raw_request;

            request() = default;
            request(const request&) = default;
            request(request&&) = default;
            request& operator=(const request&) = default;
            request& operator=(request&&) = default;

            request(MPI_Request req)
              : raw_request(req)
            {
            }
            request& operator=(const MPI_Request& r)
            {
                raw_request = r;
                return *this;
            }

            void free() { MPI_Request_free(&this->raw_request); }

            bool get_status(status& status)
            {
                int complete = false;
                MPI_Request_get_status(this->raw_request, &complete, &status.raw_status);
                return complete;
            }

            void wait(status& status) { MPI_Wait(&this->raw_request, &status.raw_status); }
            void wait() { MPI_Wait(&this->raw_request, MPI_STATUS_IGNORE); }

            template<class Integer>
            static void waitall(Integer count, request array_of_requests[], status array_of_statuses[])
            {
                // Reinterpret cast is possible because request only attribute is an MPI_Request
                MPI_Waitall(static_cast<int>(count), reinterpret_cast<MPI_Request*>(array_of_requests),
                            reinterpret_cast<MPI_Status*>(array_of_statuses));
            }
            template<class Integer>
            static void waitall(Integer count, request array_of_requests[])
            {
                // Reinterpret cast is possible because request only attribute is an MPI_Request
                MPI_Waitall(static_cast<int>(count), reinterpret_cast<MPI_Request*>(array_of_requests),
                            MPI_STATUSES_IGNORE);
            }

            template<class Range>
            static void waitall(Range&& r)
            {
                using inria::size;
                using std::begin;
                using std::end;
                MPI_Request* reqs = new MPI_Request[size(r)];
                std::transform(begin(r), end(r), reqs, [](const request& req) { return req.raw_request; });
                MPI_Waitall(static_cast<int>(size(r)), reqs, MPI_STATUSES_IGNORE);
                std::copy(reqs, reqs + size(r), begin(r));
                delete[] reqs;
            }

            static int waitany(int count, request array_of_requests[], status& status)
            {
                int index;
                MPI_Waitany(count, reinterpret_cast<MPI_Request*>(array_of_requests), &index, &status.raw_status);
                return index;
            }
            static int waitany(int count, request array_of_requests[])
            {
                int index;
                MPI_Waitany(count, reinterpret_cast<MPI_Request*>(array_of_requests), &index, MPI_STATUS_IGNORE);
                return index;
            }
        };

        static_assert(std::is_standard_layout<request>::value && sizeof(request) == sizeof(MPI_Request),
                      "mpi::request is not layout compatible with MPI_Request");

        struct communicator
        {
            MPI_Comm raw_comm{MPI_COMM_NULL};

            communicator() = default;
            communicator(const communicator&) = default;
            communicator(communicator&&) = default;
            communicator& operator=(const communicator&) = default;
            communicator& operator=(communicator&&) = default;

            communicator(MPI_Comm comm, bool duplicated=false)
            {
              if (duplicated) {
                MPI_Comm_dup(comm, &raw_comm);
              } else {
                raw_comm =  comm;
              }
            }

            operator MPI_Comm() const { return raw_comm; }

            // Accessors

            int size() const
            {
                int s;
                MPI_Comm_size(this->raw_comm, &s);
                return s;
            }

            int rank() const
            {
                int r;
                MPI_Comm_rank(this->raw_comm, &r);
                return r;
            }
            MPI_Comm get_comm() const { return raw_comm; }

            // Modifiers

            void free()
            {
                if(this->raw_comm != MPI_COMM_NULL)
                {
                    MPI_Comm_free(&this->raw_comm);
                }
            }

            communicator split(int color, int key) const
            {
                communicator new_comm;
                MPI_Comm_split(this->raw_comm, color, key, &new_comm.raw_comm);
                return new_comm;
            }

            // Communication

            void send(void* buf, int count, MPI_Datatype datatype, int dest, int tag) const
            {
                MPI_Send(buf, count, datatype, dest, tag, this->raw_comm);
            }

            request isend(void* buf, int count, MPI_Datatype datatype, int dest, int tag) const
            {
                request req;
                MPI_Isend(buf, count, datatype, dest, tag, this->raw_comm, &req.raw_request);
                return req;
            }

            status probe(int source, int tag) const
            {
                status stat;
                MPI_Probe(source, tag, this->raw_comm, &stat.raw_status);
                return stat;
            }

            void recv(void* buf, int count, MPI_Datatype datatype, int source, int tag, status& status) const
            {
                MPI_Recv(buf, count, datatype, source, tag, this->raw_comm, &status.raw_status);
            }
            void recv(void* buf, int count, MPI_Datatype datatype, int source, int tag) const
            {
                MPI_Recv(buf, count, datatype, source, tag, this->raw_comm, MPI_STATUS_IGNORE);
            }

            request irecv(void* buf, int count, MPI_Datatype datatype, int source, int tag) const
            {
                request req;
                MPI_Irecv(buf, count, datatype, source, tag, this->raw_comm, &req.raw_request);
                return req;
            }

            void gather(void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
                        MPI_Datatype recvtype, int root) const
            {
                MPI_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, this->raw_comm);
            }

            void gather(void* buf, int count, MPI_Datatype type, int root) const
            {
                if(rank() == root)
                {
                    gather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, buf, count, type, root);
                }
                else
                {
                    MPI_Aint lb, extent;
                    MPI_Type_get_extent(type, &lb, &extent);
                    gather((char*)buf + rank() * extent, count, type, nullptr, 0, MPI_DATATYPE_NULL, root);
                }
            }

            void allgather(void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
                           MPI_Datatype recvtype) const
            {
                MPI_Allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, this->raw_comm);
            }

            void allgather(void* buf, int count, MPI_Datatype type) const
            {
                allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, buf, count, type);
            }

            void bcast(void* buffer, int count, MPI_Datatype datatype, int root) const
            {
                MPI_Bcast(buffer, count, datatype, root, this->raw_comm);
            }

            void reduce(void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root) const
            {
                MPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, this->raw_comm);
            }

            void reduce(void* buf, int count, MPI_Datatype datatype, MPI_Op op, int root) const
            {
                if(rank() == root)
                {
                    reduce(MPI_IN_PLACE, buf, count, datatype, op, root);
                }
                else
                {
                    reduce(buf, nullptr, count, datatype, op, root);
                }
            }

            void allreduce(void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op) const
            {
                MPI_Allreduce(sendbuf, recvbuf, count, datatype, op, this->raw_comm);
            }

            void allreduce(void* buf, int count, MPI_Datatype datatype, MPI_Op op) const
            {
                this->allreduce(MPI_IN_PLACE, buf, count, datatype, op);
            }

            void alltoall(void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
                          MPI_Datatype recvtype) const
            {
                MPI_Alltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, this->raw_comm);
            }

            void scan(void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op)
            {
                MPI_Scan(sendbuf, recvbuf, count, datatype, op, this->raw_comm);
            }

            void barrier() const { MPI_Barrier(this->raw_comm); }

            friend std::ostream& operator<<(std::ostream& os, const communicator& comm)
            {
                if(comm != MPI_COMM_NULL)
                {
                    os << "<MPI_Comm: " << comm.rank() << '/' << comm.size() << '>';
                }
                else
                {
                    os << "<MPI_Comm: null>";
                }
                return os;
            }

            static communicator world() { return {MPI_COMM_WORLD}; }
        };

        static_assert(std::is_standard_layout<communicator>::value, "communicator is not a standard layout");
        static_assert(sizeof(communicator) == sizeof(MPI_Comm),
                      "mpi::communicator is not layout compatible with MPI_Request");

        bool is_predefined(MPI_Datatype type)
        {
            int a, b, c, combiner;
            MPI_Type_get_envelope(type, &a, &b, &c, &combiner);
            return combiner == MPI_COMBINER_NAMED;
        }

        /**
         * \brief Find out MPI datatype for given type.
         *
         * If the type isn't a predefined type, returns an contiguous byte type of size sizeof(T)
         */
        template<class T = void>
        MPI_Datatype get_datatype()
        {
            static_assert(!std::is_same<T, void>::value, "void type cannot bound to an MPI datatype.");
            MPI_Datatype byte_datatype;
            MPI_Type_contiguous(sizeof(T), MPI_BYTE, &byte_datatype);
            return byte_datatype;
        }

        template<>
        MPI_Datatype get_datatype<char>()
        {
            return MPI_CHAR;
        }

        template<>
        MPI_Datatype get_datatype<unsigned char>()
        {
            return MPI_UNSIGNED_CHAR;
        }

        template<>
        MPI_Datatype get_datatype<std::int16_t>()
        {
            return MPI_INT16_T;
        }

        template<>
        MPI_Datatype get_datatype<std::uint16_t>()
        {
            return MPI_UINT16_T;
        }

        template<>
        MPI_Datatype get_datatype<std::int32_t>()
        {
            return MPI_INT32_T;
        }

        template<>
        MPI_Datatype get_datatype<std::uint32_t>()
        {
            return MPI_UINT32_T;
        }

        template<>
        MPI_Datatype get_datatype<std::int64_t>()
        {
            return MPI_INT64_T;
        }

        template<>
        MPI_Datatype get_datatype<std::uint64_t>()
        {
            return MPI_UINT64_T;
        }

        template<>
        MPI_Datatype get_datatype<float>()
        {
            return MPI_FLOAT;
        }

        template<>
        MPI_Datatype get_datatype<double>()
        {
            return MPI_DOUBLE;
        }

        template<>
        MPI_Datatype get_datatype<long double>()
        {
            return MPI_LONG_DOUBLE;
        }

        struct datatype_commit_guard
        {
            bool predefined = true;
            MPI_Datatype datatype = MPI_DATATYPE_NULL;

            datatype_commit_guard() = default;

            datatype_commit_guard(MPI_Datatype type)
              : datatype(type)
            {
                this->predefined = is_predefined(this->datatype);
                if(!this->predefined)
                {
                    MPI_Type_commit(&this->datatype);
                }
            }

            datatype_commit_guard(const datatype_commit_guard&) = delete;

            datatype_commit_guard(datatype_commit_guard&& other) { this->swap(other); }

            datatype_commit_guard& operator=(const datatype_commit_guard&) = delete;

            datatype_commit_guard& operator=(datatype_commit_guard&& other)
            {
                this->swap(other);
                return *this;
            }

            void swap(datatype_commit_guard& other)
            {
                using std::swap;
                swap(this->predefined, other.predefined);
                swap(this->datatype, other.datatype);
            }

            ~datatype_commit_guard()
            {
                if(!this->predefined)
                {
                    MPI_Type_free(&this->datatype);
                }
            }
        };

        template<class T>
        [[gnu::warn_unused_result]] datatype_commit_guard create_datatype_if_null(MPI_Datatype& datatype)
        {
            mpi::datatype_commit_guard type_guard;
            if(datatype == MPI_DATATYPE_NULL)
            {
                MPI_Datatype byte_datatype = mpi::get_datatype<T>();
                type_guard = mpi::datatype_commit_guard(byte_datatype);
                datatype = type_guard.datatype;
            }
            return type_guard;
        };

    }   // namespace mpi

    /**
     * \brief MPI configuration for distributed algorithms
     *
     * \warning Defined outside the inria::mpi namespace to allow ADL to find the
     * algorithms using tag dispatching
     *
     */
    struct mpi_config
    {
        /// Communicator associated to configuration.
        mpi::communicator comm;
        /// MPI datatype for type to be sent
        MPI_Datatype datatype = MPI_DATATYPE_NULL;
        /// Base to generate in-operation comm tags
        int base_tag = 0;

        mpi_config(mpi::communicator communicator)
          : comm(communicator)
        {
        }
        mpi_config(MPI_Comm communicator)
          : comm(communicator)
        {
        }
        mpi_config(mpi::communicator communicator, MPI_Datatype type)
          : comm(communicator)
          , datatype(type)
        {
        }
        mpi_config(mpi::communicator communicator, MPI_Datatype type, int base_tag_)
          : comm(communicator)
          , datatype(type)
          , base_tag(base_tag_)
        {
        }

        mpi_config(const mpi_config&) = default;
        mpi_config(mpi_config&&) = default;
        mpi_config& operator=(const mpi_config&) = default;
        mpi_config& operator=(mpi_config&&) = default;
    };

}   // end of namespace inria

#endif /* _MPI_HPP_ */
