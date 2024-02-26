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
#include <optional>
#include <stdexcept>
#include <vector>

namespace polatory::preconditioner {

class binary_cache {
 public:
  binary_cache() {
    ofs_.emplace(filename_.string(), std::ios::binary);
    if (!*ofs_) {
      throw std::runtime_error("Failed to open file: " + filename_.string());
    }
  }

  std::size_t put(const char* data, std::size_t size) {
    std::lock_guard lock(mutex_);

    if (finalized_) {
      throw std::runtime_error("cache is finalized");
    }

    auto id = records_.size();
    auto offset = static_cast<std::size_t>(ofs_->tellp());
    ofs_->write(data, size);
    records_.push_back({offset, size});
    return id;
  }

  void finalize() {
    std::lock_guard lock(mutex_);

    if (finalized_) {
      throw std::runtime_error("cache is finalized");
    }

    ofs_.reset();

    ifs_.emplace(filename_.string(), std::ios::binary);
    if (!*ifs_) {
      throw std::runtime_error("Failed to open file: " + filename_.string());
    }

    unlink(filename_);

    finalized_ = true;
  }

  void get(std::size_t id, char* data) const {
    std::lock_guard lock(mutex_);

    if (!finalized_) {
      throw std::runtime_error("cache is not finalized");
    }

    const auto& record = records_.at(id);
    ifs_->clear();
    ifs_->seekg(record.offset);
    ifs_->read(data, record.size);
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

  boost::filesystem::path filename_{boost::filesystem::temp_directory_path() /
                                    boost::filesystem::unique_path()};
  std::vector<record> records_;
  mutable std::optional<std::ifstream> ifs_;
  std::optional<std::ofstream> ofs_;
  bool finalized_{};
  mutable std::mutex mutex_;
};

}  // namespace polatory::preconditioner
