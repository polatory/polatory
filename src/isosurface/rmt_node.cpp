#include <exception>
#include <polatory/isosurface/rmt_node.hpp>

namespace polatory {
namespace isosurface {

namespace detail {

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4297)  // 'function' : function assumed not to throw an exception but does
#endif
neighbor_edge_pairs::neighbor_edge_pairs() noexcept try : base{{{{1, 9}, { 3, 13 }
, { 4, 12 }
}  // namespace detail
, {{2, 0}, {4, 13}}, {{1, 7}, {4, 10}, {5, 13}}, {{0, 6}, {4, 9}}, {{0, 5}, {1, 6}, {2, 3}},
    {{2, 6}, {4, 7}}, {{3, 7}, {4, 8}, {5, 9}}, {{2, 8}, {5, 11}, {6, 10}}, {{6, 11}, {7, 9}},
    {{0, 8}, {3, 11}, {6, 12}}, {{2, 11}, {7, 13}}, {{7, 12}, {8, 13}, {9, 10}}, {{0, 11}, {9, 13}},
{
  {0, 10}, {1, 11}, { 2, 12 }
}
}  // namespace isosurface
}  // namespace polatory
{}
catch (const std::exception&) {
  std::terminate();
}
#ifdef _MSC_VER
#pragma warning(pop)
#endif

}  // namespace detail

const std::array<edge_bitset, 14> NeighborMasks{0x321a, 0x2015, 0x24b2, 0x0251, 0x006f,
                                                0x00d4, 0x03b8, 0x0d64, 0x0ac0, 0x1949,
                                                0x2884, 0x3780, 0x2a01, 0x1c07};

const std::array<edge_bitset, 24> FaceEdges{
    0x0013, 0x0209, 0x0019, 0x1201, 0x3001, 0x2003, 0x0016, 0x2006, 0x0034, 0x00a4, 0x0484, 0x2404,
    0x0058, 0x0248, 0x0070, 0x00e0, 0x01c0, 0x0340, 0x0c80, 0x0980, 0x0b00, 0x1a00, 0x2c00, 0x3800};

const std::array<face_bitset, 24> NeighborFaces{
    0x000064, 0x00200c, 0x001003, 0x200012, 0x800028, 0x000091, 0x000181, 0x000860,
    0x004240, 0x008500, 0x040a00, 0x400480, 0x006004, 0x021002, 0x009100, 0x014200,
    0x0a8000, 0x112000, 0x480400, 0x150000, 0x2a0000, 0x900008, 0x840800, 0x600010};

const detail::neighbor_edge_pairs NeighborEdgePairs;

bool rmt_node::has_neighbor(edge_index edge) const { return neighbors_->at(edge) != nullptr; }

rmt_node& rmt_node::neighbor(edge_index edge) {
  POLATORY_ASSERT(has_neighbor(edge));
  return *neighbors_->at(edge);
}

const rmt_node& rmt_node::neighbor(edge_index edge) const {
  POLATORY_ASSERT(has_neighbor(edge));
  return *neighbors_->at(edge);
}

}  // namespace isosurface
}  // namespace polatory
