// --------------------------------
// See LICENCE file at project root
// File : scalfmm/tools/vtk_writer.hpp
// --------------------------------
#ifndef SCALFMM_TOOLS_VTK_WRITER_HPP
#define SCALFMM_TOOLS_VTK_WRITER_HPP

#include "scalfmm/meta/utils.hpp"
#include "scalfmm/tree/for_each.hpp"

#include <cstddef>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace scalfmm::tools::io
{
    /**
     * @brief
     *
     * @param nb_val
     * @param name
     * @return std::string
     */
    std::string writeHeader(const int nb_val, const std::string& name)
    {
        std::string header;
        if(nb_val > 0)
        {
            header = "Scalars=\"";
            for(int i = 0; i < nb_val; ++i)
            {
                header += name + std::to_string(i) + " ";
            }
            header += "\" ";
        }
        return header;
    }

    /**
     * @brief
     *
     * @tparam TreeType
     * @param filename
     * @param tree
     * @param npart
     */
    template<class TreeType>
    void exportVTKxml(std::string const& filename, TreeType const& tree, std::size_t npart)
    {
        using particle_type = typename TreeType::particle_type;
        constexpr int dimension = TreeType::leaf_type::dimension;
        constexpr int nb_input_elements = TreeType::leaf_type::particle_type::inputs_size;
        constexpr int nb_output_elements = TreeType::leaf_type::particle_type::outputs_size;
        std::vector<particle_type> particles(npart);
        std::size_t pos{0};

        scalfmm::component::for_each_leaf(std::cbegin(tree), std::cend(tree),
                                          [&pos, &particles](auto& leaf)
                                          {
                                              const auto container = leaf.cparticles();

                                              // for(std::size_t idx = 0; idx < leaf.cparticles().size(); ++idx)
                                              // {
                                              //     particles[pos++] = leaf.cparticles().particle(idx);
                                              // }
                                              for(auto const& it_p: leaf)
                                              {
                                                  // particle_type& particles_elem = particles[pos++];
                                                  particles[pos++] = typename TreeType::leaf_type::particle_type(it_p);
                                                  //
                                                  // int i = 0;
                                                  // const auto points = p.position();
                                                  // for(int k = 0; k < dimension; ++k, ++i)
                                                  // {
                                                  //     particles_elem[i] = points[k];
                                                  // }
                                                  // // get inputs
                                                  // for(int k = 0; k < nb_input_elements; ++k, ++i)
                                                  // {
                                                  //     particles_elem[i] = p.inputs(k);
                                                  // }
                                                  // // get outputs
                                                  // for(int k = 0; k < nb_output_elements; ++k, ++i)
                                                  // {
                                                  //     particles_elem[i] = p.outputs(k);
                                                  // }
                                              }
                                          });

        if(TreeType::dimension > 3)
        {
            std::cerr << " exportVTKxml works only for dimension <=3 and the "
                         "results are in space 3"
                      << std::endl;
            return;
        }
        std::ofstream VTKfile(filename);
        std::size_t j = 0;
        std::size_t stride = meta::tuple_size_v<particle_type>;
        int nb_output_values = particle_type::outputs_size;
        int nb_input_values = particle_type::inputs_size;
        std::cout << "Write vtk format in " << filename << std::endl;
        std::cout << "   dim=" << TreeType::dimension << " stride =" << stride << " nb_input_values " << nb_input_values
                  << " nb_output_values " << nb_output_values << "  N " << npart << std::endl;
        VTKfile << "<?xml version=\"1.0\"?>" << std::endl
                << "<VTKFile type=\"PolyData\" version=\"0.1\" "
                   "byte_order=\"LittleEndian\"> "
                << std::endl
                << "<PolyData>" << std::endl
                << "<Piece NumberOfPoints=\" " << particles.size() << " \"  NumberOfVerts=\" " << particles.size()
                << " \" NumberOfLines=\" 0\" NumberOfStrips=\"0\" NumberOfPolys=\"0\">" << std::endl
                << "<Points>" << std::endl
                << "<DataArray type=\"Float64\" NumberOfComponents=\"3" /*<< dimension*/
                << "\" "
                   "format=\"ascii\"> "
                << std::endl;
        j = 0;
        for(std::size_t i = 0; i < particles.size(); ++i)
        {
            for(int k = 0; k < TreeType::dimension; ++k)
            {
                VTKfile << particles[i].position(k) << "  ";
            }
            for(int i = 0; i < 3 - TreeType::dimension; ++i)
            {
                VTKfile << "0.0  ";
            }
        }
        if(nb_input_values + nb_output_values > 0)
        {
            std::string header("Scalars=\"");
            if(nb_input_values > 0)
            {
                for(int i = 0; i < nb_input_values; ++i)
                {
                    header += "intputValue" + std::to_string(i) + " ";
                }
                for(int i = 0; i < nb_output_values; ++i)
                {
                    header += "outputValue" + std::to_string(i) + " ";
                }
                header += "\" ";
            }
            VTKfile << std::endl
                    << "</DataArray> " << std::endl
                    << "</Points> " << std::endl
                    << "<PointData " << header << " > " << std::endl;
            for(int k = 0; k < nb_input_values; ++k)
            {
                VTKfile << "<DataArray type=\"Float64\" Name=\"inputValue" + std::to_string(k) +
                             "\"  "
                             "format=\"ascii\">"
                        << std::endl;
                j = 0;
                for(std::size_t i = 0; i < particles.size(); ++i)
                {
                    VTKfile << particles[i].inputs(k) << " ";
                }
                VTKfile << std::endl << "</DataArray>" << std::endl;
            }
            for(int k = 0; k < nb_output_values; ++k)
            {
                VTKfile << "<DataArray type=\"Float64\" Name=\"outputValue" + std::to_string(k - nb_input_values) +
                             "\"  "
                             "format=\"ascii\">"
                        << std::endl;
                j = 0;
                for(std::size_t i = 0; i < particles.size(); ++i)
                {
                    VTKfile << particles[i].outputs(k) << " ";
                }
                VTKfile << std::endl << "</DataArray>" << std::endl;
            }
            VTKfile << "    </PointData>" << std::endl
                    << "    <CellData>"
                    << " </CellData>" << std::endl;
        }
        VTKfile << "    <Verts>" << std::endl
                << "    <DataArray type=\"Int32\" Name=\"connectivity\" "
                   "format=\"ascii\">"
                << std::endl;
        for(std::size_t i = 0; i < particles.size(); ++i)
        {
            VTKfile << i << " ";
        }
        VTKfile << std::endl
                << "</DataArray>" << std::endl
                << "<DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">" << std::endl;
        for(std::size_t i = 1; i < particles.size() + 1; ++i)
        {
            VTKfile << i << " ";
        }
        VTKfile << std::endl
                << "</DataArray>" << std::endl
                << "    </Verts>" << std::endl
                << "<Lines></Lines>" << std::endl
                << "<Strips></Strips>" << std::endl
                << "<Polys></Polys>" << std::endl
                << "</Piece>" << std::endl
                << "</PolyData>" << std::endl
                << "</VTKFile>" << std::endl;
    }

    /**
     * @brief  Export particles in xml polydata VTK  Format
     *
     * Export particles in the xml polydata VTK  Format.
     *   A particle is composed of k fields    pos (dim real) (size/N-dim) values
     * It is useful to plot the distribution with paraView
     *
     * @tparam VectorType
     * @param filename (string) file to save the vtk file.
     * @param  particles vector of particles of type Real (float or double)
     * @param  dimension dimension of the space (2 or 3)
     * @param  nb_input_values number of input values per particles
     * @param  N number of particles
     */
    template<class VectorType>
    void exportVTKxml(std::string& filename, const VectorType& particles, const int dimension,
                      const int nb_input_values, const std::size_t N)
    {
        if(dimension > 3)
        {
            std::cerr << " exportVTKxml works only for dimension <=3 and the "
                         "results are in space 3"
                      << std::endl;
            return;
        }
        std::ofstream VTKfile(filename);
        std::size_t j = 0;
        std::size_t stride = particles.size() / N;
        int nb_output_values = stride - dimension - nb_input_values;
        std::cout << " dimension " << dimension << std::endl;
        std::cout << " stride " << stride << std::endl;
        std::cout << " nb_input_values " << nb_input_values << std::endl;
        std::cout << " nb_output_values " << nb_output_values << std::endl;
        std::cout << " N " << N << std::endl;
        if(nb_output_values < 0)
        {
            std::cerr << "nb_output_values <0, the dimension maybe wrong\n";
        }
        std::cout << "Write vtk format in " << filename << std::endl;
        std::cout << "   dim=" << dimension << " stride =" << stride << " nb_input_values " << nb_input_values
                  << " nb_output_values " << nb_output_values << "  N " << N << std::endl;
        VTKfile << "<?xml version=\"1.0\"?>" << std::endl
                << "<VTKFile type=\"PolyData\" version=\"0.1\" "
                   "byte_order=\"LittleEndian\"> "
                << std::endl
                << "<PolyData>" << std::endl
                << "<Piece NumberOfPoints=\" " << N << " \"  NumberOfVerts=\" " << N
                << " \" NumberOfLines=\" 0\" NumberOfStrips=\"0\" NumberOfPolys=\"0\">" << std::endl
                << "<Points>" << std::endl
                << "<DataArray type=\"Float64\" NumberOfComponents=\"3" /*<< dimension*/
                << "\" "
                   "format=\"ascii\"> "
                << std::endl;
        j = 0;
        for(std::size_t i = 0; i < particles.size(); i += stride)
        {
            for(int k = 0; k < dimension; ++k)
            {
                VTKfile << particles[i + k] << "  ";
            }
            for(int i = 0; i < 3 - dimension; ++i)
            {
                VTKfile << "0.0  ";
            }
        }
        VTKfile << std::endl << "</DataArray> " << std::endl << "</Points> " << std::endl;
        if(nb_input_values + nb_output_values > 0)
        {
            std::string header("Scalars=\"");
            if(nb_input_values > 0)
            {
                for(int i = 0; i < nb_input_values; ++i)
                {
                    header += "intputValue" + std::to_string(i) + " ";
                }
                for(int i = 0; i < nb_output_values; ++i)
                {
                    header += "outputValue" + std::to_string(i) + " ";
                }
                header += "\" ";
            }
            VTKfile << "<PointData " << header << " > " << std::endl;
            for(int k = 0; k < nb_input_values; ++k)
            {
                VTKfile << "<DataArray type=\"Float64\" Name=\"inputValue" + std::to_string(k) +
                             "\"  "
                             "format=\"ascii\">"
                        << std::endl;
                j = 0;
                for(std::size_t i = 0; i < N; ++i, j += stride)
                {
                    VTKfile << particles[j + dimension + k] << " ";
                }
                VTKfile << std::endl << "</DataArray>" << std::endl;
            }
            for(int k = nb_input_values; k < nb_input_values + nb_output_values; ++k)
            {
                VTKfile << "<DataArray type=\"Float64\" Name=\"outputValue" + std::to_string(k - nb_input_values) +
                             "\"  "
                             "format=\"ascii\">"
                        << std::endl;
                j = 0;
                for(std::size_t i = 0; i < N; ++i, j += stride)
                {
                    VTKfile << particles[j + dimension + k] << " ";
                }
                VTKfile << std::endl << "</DataArray>" << std::endl;
            }
            VTKfile << "    </PointData>" << std::endl
                    << "    <CellData>"
                    << " </CellData>" << std::endl;
        }
        else
        {
        }
        VTKfile << "    <Verts>" << std::endl
                << "    <DataArray type=\"Int32\" Name=\"connectivity\" "
                   "format=\"ascii\">"
                << std::endl;
        for(std::size_t i = 0; i < N; ++i)
        {
            VTKfile << i << " ";
        }
        VTKfile << std::endl
                << "</DataArray>" << std::endl
                << "<DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">" << std::endl;
        for(std::size_t i = 1; i < N + 1; ++i)
        {
            VTKfile << i << " ";
        }
        VTKfile << std::endl
                << "</DataArray>" << std::endl
                << "    </Verts>" << std::endl
                << "<Lines></Lines>" << std::endl
                << "<Strips></Strips>" << std::endl
                << "<Polys></Polys>" << std::endl
                << "</Piece>" << std::endl
                << "</PolyData>" << std::endl
                << "</VTKFile>" << std::endl;
    };
}   // namespace scalfmm::tools::io
#endif
