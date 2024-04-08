#pragma once

#include <Eigen/Core>
#include <format>
#include <fstream>
#include <iostream>
#include <polatory/types.hpp>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace polatory::common {

template <class T, class = void>
struct Read;

template <class T, class = void>
struct Write;

template <class T>
void read(std::istream& is, T& t) {
  Read<T>{}(is, t);
}

template <class T>
void write(std::ostream& os, const T& t) {
  Write<T>{}(os, t);
}

template <class T>
struct Read<T, std::enable_if_t<std::is_trivially_copyable_v<T>>> {
  void operator()(std::istream& is, T& t) const {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    is.read(reinterpret_cast<char*>(&t), sizeof(t));
  }
};

template <class T>
struct Write<T, std::enable_if_t<std::is_trivially_copyable_v<T>>> {
  void operator()(std::ostream& os, const T& t) const {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    os.write(reinterpret_cast<const char*>(&t), sizeof(t));
  }
};

template <class T>
struct Read<std::basic_string<T>, std::enable_if_t<std::is_trivially_copyable_v<T>>> {
  void operator()(std::istream& is, std::basic_string<T>& t) const {
    std::size_t size{};
    read(is, size);
    t.resize(size);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    is.read(reinterpret_cast<char*>(t.data()), sizeof(T) * t.size());
  }
};

template <class T>
struct Write<std::basic_string<T>, std::enable_if_t<std::is_trivially_copyable_v<T>>> {
  void operator()(std::ostream& os, const std::basic_string<T>& t) const {
    write(os, t.size());
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    os.write(reinterpret_cast<const char*>(t.data()), sizeof(T) * t.size());
  }
};

template <class T>
struct Read<std::vector<T>, std::enable_if_t<std::is_trivially_copyable_v<T>>> {
  void operator()(std::istream& is, std::vector<T>& t) const {
    std::size_t size{};
    read(is, size);
    t.resize(size);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    is.read(reinterpret_cast<char*>(t.data()), sizeof(T) * t.size());
  }
};

template <class T>
struct Write<std::vector<T>, std::enable_if_t<std::is_trivially_copyable_v<T>>> {
  void operator()(std::ostream& os, const std::vector<T>& t) const {
    write(os, t.size());
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    os.write(reinterpret_cast<const char*>(t.data()), sizeof(T) * t.size());
  }
};

template <class T>
struct Read<std::vector<T>, std::enable_if_t<!std::is_trivially_copyable_v<T>>> {
  void operator()(std::istream& is, std::vector<T>& t) const {
    std::size_t size{};
    read(is, size);
    T value{};
    // The second argument is required because std::vector<T> does not have access
    // to the default constructor of T if it is non-public.
    t.resize(size, value);
    for (auto& elem : t) {
      read(is, elem);
    }
  }
};

template <class T>
struct Write<std::vector<T>, std::enable_if_t<!std::is_trivially_copyable_v<T>>> {
  void operator()(std::ostream& os, const std::vector<T>& t) const {
    write(os, t.size());
    for (const auto& elem : t) {
      write(os, elem);
    }
  }
};

template <class T, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
struct Read<Eigen::Matrix<T, Rows, Cols, Options, MaxRows, MaxCols>,
            std::enable_if_t<std::is_trivially_copyable_v<T>>> {
  void operator()(std::istream& is,
                  Eigen::Matrix<T, Rows, Cols, Options, MaxRows, MaxCols>& t) const {
    index_t rows{Rows};
    index_t cols{Cols};
    if (Rows == Eigen::Dynamic) {
      read(is, rows);
    }
    if (Cols == Eigen::Dynamic) {
      read(is, cols);
    }
    t.resize(rows, cols);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    is.read(reinterpret_cast<char*>(t.data()), sizeof(T) * t.size());
  }
};

template <class T, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
struct Write<Eigen::Matrix<T, Rows, Cols, Options, MaxRows, MaxCols>,
             std::enable_if_t<std::is_trivially_copyable_v<T>>> {
  void operator()(std::ostream& os,
                  const Eigen::Matrix<T, Rows, Cols, Options, MaxRows, MaxCols>& t) const {
    if (Rows == Eigen::Dynamic) {
      write(os, t.rows());
    }
    if (Cols == Eigen::Dynamic) {
      write(os, t.cols());
    }
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    os.write(reinterpret_cast<const char*>(t.data()), sizeof(T) * t.size());
  }
};

template <class T>
void load(const std::string& filename, T& t) {
  std::ifstream ifs(filename, std::ios::binary);
  if (!ifs) {
    throw std::runtime_error{std::format("cannot open file '{}'", filename)};
  }

  read(ifs, t);
}

template <class T>
void save(const std::string& filename, const T& t) {
  std::ofstream ofs(filename, std::ios::binary);
  if (!ofs) {
    throw std::runtime_error{std::format("cannot open file '{}'", filename)};
  }

  write(ofs, t);
}

}  // namespace polatory::common

// NOTE: T must have either a public or non-public default constructor
// for deserializing std::vector<T>, etc.
#define POLATORY_FRIEND_READ_WRITE(T)     \
  template <class, class>                 \
  friend struct ::polatory::common::Read; \
                                          \
  template <class, class>                 \
  friend struct ::polatory::common::Write;

// NOTE: T must have either a public or non-public default constructor.
#define POLATORY_IMPLEMENT_LOAD_SAVE(T)        \
  static T load(const std::string& filename) { \
    T t{};                                     \
    ::polatory::common::load(filename, t);     \
    return t;                                  \
  }                                            \
                                               \
  void save(const std::string& filename) const { ::polatory::common::save(filename, *this); }
