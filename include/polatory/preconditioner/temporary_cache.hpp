#pragma once

// For _wunlink (Win32) or unlink (POSIX).
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

#include <boost/filesystem.hpp>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>

namespace polatory::preconditioner {

class temporary_cache {
 public:
  std::ofstream& begin_write() {
    if (state_ != kInitial) {
      throw std::runtime_error("invalid state");
    }

    ofs_.emplace(filename_.string(), std::ios::binary);
    if (!*ofs_) {
      throw std::runtime_error("Failed to open file: " + filename_.string());
    }

    state_ = kWriting;
    return *ofs_;
  }

  void end_write() {
    if (state_ != kWriting) {
      throw std::runtime_error("invalid state");
    }

    ofs_.reset();

    ifs_.emplace(filename_.string(), std::ios::binary);
    if (!*ifs_) {
      throw std::runtime_error("Failed to open file: " + filename_.string());
    }

    unlink(filename_);

    state_ = kClosed;
  }

  std::ifstream& begin_read() {
    if (state_ != kClosed) {
      throw std::runtime_error("invalid state");
    }

    state_ = kReading;
    return *ifs_;
  }

  void end_read() {
    if (state_ != kReading) {
      throw std::runtime_error("invalid state");
    }

    ifs_->clear();
    ifs_->seekg(0);

    state_ = kClosed;
  }

 private:
  static int unlink(const boost::filesystem::path& filename) {
#ifdef _WIN32
    return ::_wunlink(filename.c_str());
#else
    return ::unlink(filename.c_str());
#endif
  }

  enum state { kInitial, kWriting, kReading, kClosed };

  boost::filesystem::path filename_{boost::filesystem::temp_directory_path() /
                                    boost::filesystem::unique_path()};
  std::optional<std::ifstream> ifs_;
  std::optional<std::ofstream> ofs_;
  state state_{kInitial};
};

}  // namespace polatory::preconditioner
