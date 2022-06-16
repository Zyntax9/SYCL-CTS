/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2020-2022 The Khronos Group Inc.
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
//  This file contains helper functions for modifying and comparing values,
//  arrays, and objects that implements operator[]
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_COMMON_VALUE_OPERATIONS_H
#define __SYCLCTS_TESTS_COMMON_VALUE_OPERATIONS_H
#include "../../util/type_traits.h"

#include <cassert>

namespace value_operations {

template <typename T, size_t N>
using ArrayT = T[N];

template <typename T>
struct IsArrayT : std::false_type {};
template <typename T, size_t N>
struct IsArrayT<ArrayT<T, N>> : std::true_type {};

template <typename T> struct IsNonArrayWithSubscriptOperator {
  static constexpr bool value =
      !IsArrayT<T>::value && has_subscript_operator_v<T>;
};

// Modify functions
template <typename T, size_t N, typename RightNonArrT = T>
inline void assign(ArrayT<T, N>& left, const RightNonArrT& right) {
  for (size_t i = 0; i < N; ++i) {
    left[i] = right;
  }
}

template <typename LeftArrT, size_t LeftArrN, typename RightArrT,
          size_t RightArrN>
inline void assign(ArrayT<LeftArrT, LeftArrN>& left,
                   const ArrayT<RightArrT, RightArrN>& right) {
  static_assert(LeftArrN == RightArrN, "Arrays have to be the same size");
  for (size_t i = 0; i < LeftArrN; ++i) {
    left[i] = right[i];
  }
}

template <typename LeftArrT, typename RightNonArrT>
inline typename std::enable_if_t<
    IsNonArrayWithSubscriptOperator<LeftArrT>::value &&
    !has_subscript_operator_v<RightNonArrT>>
assign(LeftArrT& left, const RightNonArrT& right) {
  for (size_t i = 0; i < left.size(); ++i) {
    left[i] = right;
  }
}

template <typename LeftArrT, typename RightArrT>
inline typename std::enable_if_t<
    IsNonArrayWithSubscriptOperator<LeftArrT>::value &&
    IsNonArrayWithSubscriptOperator<RightArrT>::value>
assign(LeftArrT& left, const RightArrT& right) {
  assert((left.size() == right.size()) && "Arrays have to be the same size");
  for (size_t i = 0; i < left.size(); ++i) {
    left[i] = right[i];
  }
}

template <typename LeftNonArrT, typename RightNonArrT = LeftNonArrT>
inline typename std::enable_if_t<!has_subscript_operator_v<LeftNonArrT> &&
                                 !has_subscript_operator_v<RightNonArrT>>
assign(LeftNonArrT& left, const RightNonArrT& right) {
  left = right;
}

#if defined(SYCL_EXT_ONEAPI_PROPERTIES) && \
    defined(SYCL_EXT_ONEAPI_DEVICE_GLOBAL)

template <typename T, typename Props, typename RightNonArrT>
void assign(sycl::ext::oneapi::experimental::device_global<T, Props>& left,
            const RightNonArrT& right) {
  assign(left.get(), right);
}

#endif
/////////////////////////// Modify functions

// Compare functions
template <typename T, size_t N>
inline bool are_equal(const ArrayT<T, N>& left, const T& right) {
  for (size_t i = 0; i < N; ++i) {
    if (left[i] != right) return false;
  }
  return true;
}

template <typename LeftArrT, size_t LeftArrN, typename RightArrT,
          size_t RightArrN>
inline bool are_equal(const ArrayT<LeftArrT, LeftArrN>& left,
                      const ArrayT<RightArrT, RightArrN>& right) {
  static_assert(LeftArrN == RightArrN, "Arrays have to be the same size");
  for (size_t i = 0; i < LeftArrN; ++i) {
    if (left[i] != right[i]) return false;
  }
  return true;
}

template <typename LeftArrT, typename RightNonArrT>
inline typename std::enable_if_t<
    IsNonArrayWithSubscriptOperator<LeftArrT>::value &&
        !has_subscript_operator_v<RightNonArrT>,
    bool>
are_equal(const LeftArrT& left, const RightNonArrT& right) {
  for (size_t i = 0; i < left.size(); ++i) {
    if (left[i] != right) return false;
  }
  return true;
}

template <typename LeftArrT, typename RightArrT>
inline typename std::enable_if_t<
    IsNonArrayWithSubscriptOperator<LeftArrT>::value &&
        IsNonArrayWithSubscriptOperator<RightArrT>::value,
    bool>
are_equal(const LeftArrT& left, const RightArrT& right) {
  assert((left.size() == right.size()) && "Arrays have to be the same size");
  for (size_t i = 0; i < left.size(); ++i) {
    if (left[i] != right[i]) return false;
  }
  return true;
}

template <typename LeftNonArrT, typename RightNonArrT = LeftNonArrT>
inline typename std::enable_if_t<!has_subscript_operator_v<LeftNonArrT> &&
                                     !has_subscript_operator_v<RightNonArrT>,
                                 bool>
are_equal(const LeftNonArrT& left, const RightNonArrT& right) {
  return (left == right);
}

#if defined(SYCL_EXT_ONEAPI_PROPERTIES) && \
    defined(SYCL_EXT_ONEAPI_DEVICE_GLOBAL)

template <typename T, typename Props, typename RightT>
bool are_equal(sycl::ext::oneapi::experimental::device_global<T, Props>& left,
               const RightT& right) {
  return are_equal(left.get(), right);
}

template <typename LeftT, typename LeftProps, typename RightT,
          typename RightProps>
bool are_equal(
    sycl::ext::oneapi::experimental::device_global<LeftT, LeftProps>& left,
    const sycl::ext::oneapi::experimental::device_global<RightT, RightProps>&
        right) {
  return are_equal(left.get(), right.get());
}

#endif
//////////////////////////// Compare functions

}  // namespace value_operations
#endif  //__SYCLCTS_TESTS_COMMON_VALUE_OPERATIONS_H
