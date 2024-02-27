#pragma once

#ifdef _WIN32
#include <fileapi.h>
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

#ifdef _WIN32
    file_ = ::CreateFileW(filename.c_str(), GENERIC_READ | GENERIC_WRITE,
                          FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, CREATE_ALWAYS,
                          FILE_FLAG_DELETE_ON_CLOSE, nullptr);
    if (file_ == INVALID_HANDLE_VALUE) {
      throw std::runtime_error("failed to open a temporary file");
    }
#endif

    fs_.open(filename.string(), std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
    if (!fs_.is_open()) {
      throw std::runtime_error("failed to open a temporary file");
    }

#ifndef _WIN32
    ::unlink(filename.c_str());
#endif
  }

  ~binary_cache() {
#ifdef _WIN32
    ::CloseHandle(file_);
#endif
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

#ifdef _WIN32
  HANDLE file_;
#endif

  std::vector<record> records_;
  mutable std::fstream fs_;
  mutable std::mutex mutex_;
};

}  // namespace polatory::preconditioner
