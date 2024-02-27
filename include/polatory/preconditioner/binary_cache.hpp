#pragma once

#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif

#include <boost/filesystem.hpp>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace polatory::preconditioner {

class binary_cache {
 public:
  binary_cache() {
    auto filename = boost::filesystem::temp_directory_path() / boost::filesystem::unique_path();

#ifdef _WIN32
    file_ = ::CreateFileW(filename.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, CREATE_NEW,
                          FILE_FLAG_DELETE_ON_CLOSE, nullptr);
    if (file_ == INVALID_HANDLE_VALUE) {
      throw std::runtime_error("failed to open a temporary file a");
    }
#else
    file_ = ::open(filename.c_str(), O_RDWR | O_CREAT | O_EXCL);
    if (file_ == -1) {
      throw std::runtime_error("failed to open a temporary file b");
    }
    ::unlink(filename.c_str());
#endif

    records_.push_back({0, 0});
  }

  ~binary_cache() {
#ifdef _WIN32
    ::CloseHandle(file_);
#else
    ::close(file_);
#endif
  }

  void get(std::size_t id, char* data) const {
    std::lock_guard lock(mutex_);

    const auto& record = records_.at(id);

#ifdef _WIN32
    LARGE_INTEGER distance;
    distance.QuadPart = record.offset;
    ::SetFilePointerEx(file_, distance, nullptr, FILE_BEGIN);
    ::ReadFile(file_, data, record.size, nullptr, nullptr);
#else
    ::lseek(file_, record.offset, SEEK_SET);
    ::read(file_, data, record.size);
#endif
  }

  std::size_t put(const char* data, std::size_t size) {
    std::lock_guard lock(mutex_);

    auto id = records_.size();
    auto offset = records_.back().offset + records_.back().size;

#ifdef _WIN32
    LARGE_INTEGER distance;
    distance.QuadPart = 0;
    ::SetFilePointerEx(file_, distance, nullptr, FILE_END);
    ::WriteFile(file_, data, size, nullptr, nullptr);
#else
    ::lseek(file_, 0, SEEK_END);
    ::write(file_, data, size);
#endif

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
#else
  int file_;
#endif

  std::vector<record> records_;
  mutable std::mutex mutex_;
};

}  // namespace polatory::preconditioner
