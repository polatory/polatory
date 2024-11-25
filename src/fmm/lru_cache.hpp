#pragma once

#include <list>
#include <unordered_map>
#include <utility>

namespace polatory::fmm {

template <class Key, class T>
class LruCache {
  using KeyValuePair = std::pair<const Key, T>;
  using List = std::list<KeyValuePair>;
  using Iterator = typename List::iterator;
  using ConstIterator = typename List::const_iterator;

 public:
  explicit LruCache(std::size_t capacity) : capacity_{capacity} {}

  Iterator begin() { return list_.begin(); }

  ConstIterator begin() const { return list_.begin(); }

  void clear() {
    list_.clear();
    map_.clear();
  }

  Iterator end() { return list_.end(); }

  ConstIterator end() const { return list_.end(); }

  Iterator find(const Key& key) {
    auto it = map_.find(key);
    if (it == map_.end()) {
      return end();
    }

    return it->second;
  }

  ConstIterator find(const Key& key) const {
    auto it = map_.find(key);
    if (it == map_.end()) {
      return end();
    }

    return it->second;
  }

  std::size_t size() const { return list_.size(); }

  void touch(ConstIterator it) { list_.splice(list_.begin(), list_, it); }

  template <typename... Args>
  std::pair<Iterator, bool> try_emplace(const Key& key, Args&&... args) {
    auto it = find(key);
    if (it != end()) {
      return {it, false};
    }

    list_.emplace_front(std::piecewise_construct, std::forward_as_tuple(key),
                        std::forward_as_tuple(std::forward<Args>(args)...));
    map_.try_emplace(key, list_.begin());

    if (list_.size() > capacity_) {
      map_.erase(list_.back().first);
      list_.pop_back();
    }

    return {begin(), true};
  }

 private:
  std::size_t capacity_;
  List list_;
  std::unordered_map<Key, Iterator> map_;
};

}  // namespace polatory::fmm
