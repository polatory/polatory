#pragma once

#include <vector>

#ifdef SCALFMM_USE_MPI
#include <mpi.h>
#endif
template<typename morton_type, typename grp_access_type>
class transferDataAccess
{
  private:
    // vector per level and per process
    std::vector<std::vector<std::vector<morton_type>>> m_send_morton;
    std::vector<std::vector<std::vector<grp_access_type>>> m_receive_cells_access;
    std::vector<std::vector<std::vector<grp_access_type>>> m_send_cells_access;

#ifdef SCALFMM_USE_MPI
    std::vector<std::vector<MPI_Datatype>> m_send_multipoles_type;
    std::vector<std::vector<MPI_Datatype>> m_receive_multipoles_type;
#endif

  public:
    auto inline get_send_multipole_types(const int& level) -> std::vector<MPI_Datatype>&
    {
        return m_send_multipoles_type[level];
    }

    auto inline print_send_multipole_types(const int& level) -> void
    {
        auto const& type = m_send_multipoles_type[level];
        for(int p = 0; p < type.size(); ++p)
        {
            std::cout << "   ptr_data_type(" << p << ") " << &(type[p]) << " level: " << level << std::endl
                      << std::flush;
        }
    }
    auto inline get_receive_multipole_types(const int& level) -> std::vector<MPI_Datatype>&
    {
        return m_receive_multipoles_type[level];
    }
    auto inline send_morton_indexes(int const& level, int const& proc) -> std::vector<morton_type>&
    {
        return m_send_morton[level][proc];
    }
    auto inline send_morton_indexes(int const& level) -> std::vector<std::vector<morton_type>>&
    {
        return m_send_morton[level];
    }
    auto inline receive_cells_access(int const& level) -> std::vector<std::vector<grp_access_type>>&
    {
        return m_receive_cells_access[level];
    }

    auto inline send_cells_access(int const& level) -> std::vector<std::vector<grp_access_type>>&
    {
        return m_send_cells_access[level];
    }

    transferDataAccess(const int tree_height, const int nb_proc)
    {
        m_receive_multipoles_type.resize(tree_height);
        m_send_multipoles_type.resize(tree_height);
        m_send_morton.resize(tree_height);
        for(auto& vec: m_send_morton)
        {
            vec.resize(nb_proc);
        }
        m_receive_cells_access.resize(tree_height);
        for(auto& vec: m_receive_cells_access)
        {
            vec.resize(nb_proc);
        }
        m_send_cells_access.resize(tree_height);
        for(auto& vec: m_send_cells_access)
        {
            vec.resize(nb_proc);
        }
    }
};

class UpDownDataAccess
{
    // vector per level and per process
    bool done{false};          // if built
    int nb_child_send{0};      // number of child to send
    int nb_child_receive{0};   // number of child to receive

  public:
    UpDownDataAccess()
      : done(false)
      , nb_child_send(0)
      , nb_child_receive(0)
    {
    }
    bool is_done() { return done; }
    auto get_nb_child_to_send() const { return nb_child_send; }
    void set_nb_child_to_send(const int n) { nb_child_send = n; }
    auto get_nb_child_to_receive() const { return nb_child_receive; }
    void set_nb_child_to_receive(const int n) { nb_child_receive = n; }
};