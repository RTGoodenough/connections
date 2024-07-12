#pragma once

#include <concepts>
#include <utility>
#include "util/self_crtp.hpp"

namespace cntns::util {

template <typename class_t, typename other_t>
concept has_subtract = requires(class_t instance, other_t other)
{
  {instance - other};
};

template <typename class_t, typename other_t>
concept has_addition = requires(class_t instance, other_t other)
{
  {instance + other};
};

template <typename class_t, typename other_t>
concept has_multiply = requires(class_t instance, other_t other)
{
  {instance * other};
};

template <typename class_t, typename other_t>
concept has_divide = requires(class_t instance, other_t other)
{
  {instance / other};
};

template <typename class_t>
class Operators : public util::Self<Operators<class_t>> {
  using util::Self<Operators<class_t>>::self;

 public:
  template <typename other_t>
  requires has_addition<class_t, other_t>
  constexpr auto operator+=(other_t&& other) -> decltype(auto)
  {
    self() = self() + std::forward<other_t>(other);
    return self();
  }

  template <typename other_t>
  requires has_subtract<class_t, other_t>
  constexpr auto operator-=(other_t&& other) -> decltype(auto)
  {
    self() = self() - std::forward<other_t>(other);
    return self();
  }

  template <typename other_t>
  requires has_multiply<class_t, other_t>
  constexpr auto operator*=(other_t&& other) -> decltype(auto)
  {
    self() = self() * std::forward<other_t>(other);
    return self();
  }

  template <typename other_t>
  requires has_divide<class_t, other_t>
  constexpr auto operator/=(other_t&& other) -> decltype(auto)
  {
    self() = self() / std::forward<other_t>(other);
    return self();
  }
};
}  // namespace cntns::util