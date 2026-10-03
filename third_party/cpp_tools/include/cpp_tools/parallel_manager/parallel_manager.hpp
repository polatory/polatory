#ifndef CPP_TOOLS_PARALLEL_MANAGER_PARALLEL_MANAGER_HPP
#define CPP_TOOLS_PARALLEL_MANAGER_PARALLEL_MANAGER_HPP

#include <iostream>

#ifdef CPP_TOOLS_PARALLEL_MANAGER_USE_MPI
#include "cpp_tools/parallel_manager/mpi.hpp"
#include <mpi.h>
#endif

#ifdef CPP_TOOLS_PARALLEL_MANAGER_USE_OMP
#include <omp.h>
#endif
namespace cpp_tools::parallel_manager
{
    class parallel_manager
    {
      private:
        int m_number_processes{};   ///< number of processes
        int m_process_id{};         ///<  The id of the process
        int m_number_threads{};     ///< number of threads
#ifdef CPP_TOOLS_PARALLEL_MANAGER_USE_MPI
        cpp_tools::parallel_manager::mpi::communicator m_communicator;
#else
        struct empty
        {
        };
        empty m_communicator{};
#endif

      public:
#ifdef CPP_TOOLS_PARALLEL_MANAGER_USE_MPI
        using comm_type = cpp_tools::parallel_manager::mpi::communicator;
#else
        using comm_type = empty;
#endif
        int get_num_threads() const { return m_number_threads; };
        void set_num_threads(const int num)
        {
            m_number_threads = num;
#ifdef CPP_TOOLS_PARALLEL_MANAGER_USE_OMP
            omp_set_num_threads(m_number_threads);
#endif
        };
        int get_num_processes() const { return m_number_processes; };
        int get_process_id() const { return m_process_id; };
        bool master() const { return m_process_id == 0; };
        bool io_master() const { return m_process_id == 0; };

        parallel_manager()
          : m_number_processes(1)
          , m_process_id(0)
          , m_number_threads(1)
        {
#ifdef CPP_TOOLS_PARALLEL_MANAGER_USE_OMP
            omp_set_num_threads(1);
#endif
        }
        parallel_manager(const int num_threads, const int num_processes = 1)
          : m_number_processes(num_processes)
          , m_number_threads(num_threads)
#ifdef CPP_TOOLS_PARALLEL_MANAGER_USE_MPI
          , m_communicator(MPI_COMM_NULL)
#endif
        {
#ifdef CPP_TOOLS_PARALLEL_MANAGER_USE_OMP
            omp_set_num_threads(num_threads);
#endif
        }
#ifdef CPP_TOOLS_PARALLEL_MANAGER_USE_MPI
        ///
        /// \brief Constructor with a  MPI communicator
        ///
        ///  To avoid deadlock the communicator HAVE TO BE duplicated
        ///
        /// \param in_communicator MPI communicator
        ///
        parallel_manager(const int num_threads, MPI_Comm in_communicator, bool duplicated = false)
          : m_number_threads(num_threads)
        {
            this->init(in_communicator, duplicated);
#ifdef CPP_TOOLS_PARALLEL_MANAGER_USE_OMP
            omp_set_num_threads(num_threads);
#endif
        }
#endif

        //////////
        void init()
        {
#ifdef CPP_TOOLS_PARALLEL_MANAGER_USE_MPI
            int provided{0}, mpi_is_initialized{0};
            int err = MPI_Initialized(&mpi_is_initialized);
            if(!mpi_is_initialized)
            {
                int err = MPI_Init_thread(nullptr, nullptr, MPI_THREAD_SERIALIZED, &provided);
                if(err != MPI_SUCCESS)
                {
                    char estring[MPI_MAX_ERROR_STRING];
                    int len, eclass;
                    MPI_Error_class(err, &eclass);
                    MPI_Error_string(err, estring, &len);
                    std::string error(estring);
                    printf("Error %d: %s\n", eclass, estring);
                    std::cerr << "Error in MPI_Init_thread,  " << eclass << " " << error << std::endl;
                }
            }
            this->init(cpp_tools::parallel_manager::mpi::communicator::world());
#endif
        }
#ifdef CPP_TOOLS_PARALLEL_MANAGER_USE_MPI
        ///
        /// \brief initialize the communication level
        ///
        ///  To avoid deadlock the communicator HAVE TO BE duplicated
        ///
        /// \param in_communicator MPI communicator
        ///
        void init(MPI_Comm in_communicator, bool duplicated = false)
        {
            int provided;
            int error{};
            error = MPI_Query_thread(&provided);
            if(provided < MPI_THREAD_SERIALIZED)
            {
                std::cerr << " MPI should at least initialized at MPI_THREAD_SERIALIZED level" << std::endl;
                MPI_Finalize();
                std::exit(EXIT_FAILURE);
            }
            m_communicator = cpp_tools::parallel_manager::mpi::communicator(in_communicator, duplicated);
            m_number_processes = m_communicator.size();
            m_process_id = m_communicator.rank();
        }

#endif

        void end()
        {
            //     std::clog << " para::end" << std::endl;

#ifdef CPP_TOOLS_PARALLEL_MANAGER_USE_MPI
            m_communicator.barrier();

            MPI_Finalize();
            //     std::cout << " end parallel version" << std::endl;
#endif
        }
        comm_type get_communicator() const
        {
            // #ifdef CPP_TOOLS_PARALLEL_MANAGER_USE_MPI
            return m_communicator;
            // #else

            //             return empty{};
            // #endif
        }
        comm_type& get_communicator()
        {
            return m_communicator;

            // #ifdef CPP_TOOLS_PARALLEL_MANAGER_USE_MPI
            // #else

            //             return empty{};
            // #endif
        }
    };
}   // namespace cpptools::parallel_manager
#endif
