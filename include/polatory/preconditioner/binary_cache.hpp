#pragma once

#ifdef _WIN32
#include <Windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

#include <boost/filesystem.hpp>
#include <format>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace polatory::preconditioner {

class BinaryCache {
 public:
  BinaryCache() {
    auto filename = boost::filesystem::temp_directory_path() / boost::filesystem::unique_path();

#ifdef _WIN32
    file_ = ::CreateFileW(filename.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, CREATE_NEW,
                          FILE_FLAG_DELETE_ON_CLOSE, nullptr);
    if (file_ == INVALID_HANDLE_VALUE) {
      throw std::runtime_error(
          std::format("failed to open a temporary file '{}'", filename.string()));
    }
#else
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
    file_ = ::open(filename.c_str(), O_RDWR | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR);
    if (file_ == -1) {
      throw std::runtime_error(
          std::format("failed to open a temporary file '{}'", filename.string()));
    }
    ::unlink(filename.c_str());
#endif

    records_.emplace_back(0, 0);
  }

  ~BinaryCache() {
#ifdef _WIN32
    ::CloseHandle(file_);
#else
    ::close(file_);
#endif
  }

  BinaryCache(const BinaryCache&) = delete;
  BinaryCache(BinaryCache&&) = delete;
  BinaryCache& operator=(const BinaryCache&) = delete;
  BinaryCache& operator=(BinaryCache&&) = delete;

  void get(std::size_t id, void* data) const {
    std::lock_guard lock(mutex_);

    const auto& record = records_.at(id);

#ifdef _WIN32
    LARGE_INTEGER distance;
    distance.QuadPart = record.offset;
    ::SetFilePointerEx(file_, distance, nullptr, FILE_BEGIN);
    ::ReadFile(file_, data, record.size, nullptr, nullptr);
#else
    ::lseek(file_, static_cast<::off_t>(record.offset), SEEK_SET);
    ::read(file_, data, record.size);
#endif
  }

  std::size_t put(const void* data, std::size_t size) {
    std::lock_guard lock(mutex_);

#ifdef _WIN32
    LARGE_INTEGER distance;
    distance.QuadPart = 0;
    ::SetFilePointerEx(file_, distance, nullptr, FILE_END);
    ::WriteFile(file_, data, size, nullptr, nullptr);
#else
    ::lseek(file_, 0, SEEK_END);
    ::write(file_, data, size);
#endif

    auto id = records_.size();
    auto offset = records_.back().offset + records_.back().size;
    records_.emplace_back(offset, size);
    return id;
  }

 private:
  struct Record {
    std::size_t offset{};
    std::size_t size{};
  };

#ifdef _WIN32
  HANDLE file_{};
#else
  int file_{};
#endif

  std::vector<Record> records_;
  mutable std::mutex mutex_;
};

}  // namespace polatory::preconditioner
