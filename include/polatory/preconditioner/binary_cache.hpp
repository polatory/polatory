#pragma once

#ifdef _WIN32
#include <Windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

#include <boost/filesystem.hpp>
#include <cstddef>
#include <format>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace polatory::preconditioner {

class BinaryCache {
 public:
  BinaryCache() {
    auto filename = boost::filesystem::temp_directory_path() / boost::filesystem::unique_path();

#ifdef _WIN32
    // Without FILE_FLAG_OVERLAPPED, all I/O on the handle is serialized even at explicit offsets.
    file_ = ::CreateFileW(filename.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, CREATE_NEW,
                          FILE_FLAG_DELETE_ON_CLOSE | FILE_FLAG_OVERLAPPED, nullptr);
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
    Record record;
    {
      std::scoped_lock lock(mutex_);
      record = records_.at(id);
    }

    if (!read_at(data, record.size, record.offset)) {
      throw std::runtime_error("failed to read from the cache file");
    }
  }

  std::size_t put(const void* data, std::size_t size) {
    std::scoped_lock lock(mutex_);

    auto id = records_.size();
    auto offset = records_.back().offset + records_.back().size;

    if (!write_at(data, size, offset)) {
      throw std::runtime_error("failed to write to the cache file");
    }

    records_.emplace_back(offset, size);
    return id;
  }

 private:
  struct Record {
    std::size_t offset{};
    std::size_t size{};
  };

  bool read_at(void* data, std::size_t size, std::size_t offset) const {
#ifdef _WIN32
    auto ov = overlapped(offset);
    DWORD n{};
    auto ok = ::ReadFile(file_, data, static_cast<DWORD>(size), &n, &ov) || wait(ov, n);
    return ok && n == size;
#else
    auto n = ::pread(file_, data, size, static_cast<::off_t>(offset));
    return std::cmp_equal(n, size);
#endif
  }

  bool write_at(const void* data, std::size_t size, std::size_t offset) const {
#ifdef _WIN32
    auto ov = overlapped(offset);
    DWORD n{};
    auto ok = ::WriteFile(file_, data, static_cast<DWORD>(size), &n, &ov) || wait(ov, n);
    return ok && n == size;
#else
    auto n = ::pwrite(file_, data, size, static_cast<::off_t>(offset));
    return std::cmp_equal(n, size);
#endif
  }

#ifdef _WIN32
  static OVERLAPPED overlapped(std::size_t offset) {
    thread_local HANDLE event = ::CreateEventW(nullptr, TRUE, FALSE, nullptr);

    OVERLAPPED ov{};
    ov.Offset = static_cast<DWORD>(offset);
    ov.OffsetHigh = static_cast<DWORD>(offset >> 32);
    ov.hEvent = event;
    return ov;
  }

  bool wait(OVERLAPPED& ov, DWORD& n) const {
    return ::GetLastError() == ERROR_IO_PENDING && ::GetOverlappedResult(file_, &ov, &n, TRUE) != 0;
  }

  HANDLE file_{};
#else
  int file_{};
#endif

  std::vector<Record> records_;
  mutable std::mutex mutex_;
};

}  // namespace polatory::preconditioner
