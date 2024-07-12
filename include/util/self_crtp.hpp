#pragma once

namespace cntns::util {
template <typename>
class Self;

template <template <typename> typename crtp_t, typename derived_t>
class Self<crtp_t<derived_t>> {
 public:
  constexpr auto self() -> auto& { return static_cast<derived_t&>(*this); }
  constexpr auto self() const -> auto const& { return static_cast<derived_t&>(*this); }
};
}  // namespace cntns::util