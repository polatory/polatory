#pragma once

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace polatory::common {

template <class T>
struct read {
  void operator()(std::istream& is, T& t) const {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    is.read(reinterpret_cast<char*>(&t), sizeof(t));
  }
};

template <class T>
struct write {
  void operator()(std::ostream& os, const T& t) const {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    os.write(reinterpret_cast<const char*>(&t), sizeof(t));
  }
};

template <class T>
void load(const std::string& filename, T& t) {
  std::ifstream ifs(filename, std::ios::binary);
  if (!ifs) {
    throw std::runtime_error{"cannot open file: " + filename};
  }

  read<T>{}(ifs, t);
}

template <class T>
void save(const std::string& filename, const T& t) {
  std::ofstream ofs(filename, std::ios::binary);
  if (!ofs) {
    throw std::runtime_error{"cannot open file: " + filename};
  }

  write<T>{}(ofs, t);
}

}  // namespace polatory::common
