#pragma once

// For _wunlink (Win32) or unlink (POSIX).
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

#include <boost/filesystem.hpp>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace polatory::preconditioner {

class binary_cache {
 public:
  binary_cache() {
    auto filename = boost::filesystem::temp_directory_path() / boost::filesystem::unique_path();
    fs_.open(filename.string(), std::ios::binary | std::ios::out);
    fs_.close();
    fs_.open(filename.string(), std::ios::binary | std::ios::in | std::ios::out);
    unlink(filename);
  }

  void get(std::size_t id, char* data) const {
    std::lock_guard lock(mutex_);

    const auto& record = records_.at(id);
    fs_.clear();
    fs_.seekg(record.offset);
    fs_.read(data, record.size);
  }

  std::size_t put(const char* data, std::size_t size) {
    std::lock_guard lock(mutex_);

    auto id = records_.size();
    auto offset = static_cast<std::size_t>(fs_.tellp());
    fs_.write(data, size);
    records_.push_back({offset, size});
    return id;
  }

 private:
  struct record {
    std::size_t offset;
    std::size_t size;
  };

  static int unlink(const boost::filesystem::path& filename) {
#ifdef _WIN32
    return ::_wunlink(filename.c_str());
#else
    return ::unlink(filename.c_str());
#endif
  }

  std::vector<record> records_;
  mutable std::fstream fs_;
  mutable std::mutex mutex_;
};

}  // namespace polatory::preconditioner
