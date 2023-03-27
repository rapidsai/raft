# =============================================================================
# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

INCLUDE(CheckCXXSourceRuns)

SET(AVX_CODE
    "
  #include <immintrin.h>

  int main()
  {
    __m256 a;
    a = _mm256_set1_ps(0);
    return 0;
  }
"
)

SET(AVX512_CODE
    "
  #include <immintrin.h>

  int main()
  {
    __m512i a = _mm512_set_epi8(0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0);
    __m512i b = a;
    __mmask64 equality_mask = _mm512_cmp_epi8_mask(a, b, _MM_CMPINT_EQ);
    return 0;
  }
"
)

SET(AVX2_CODE
    "
  #include <immintrin.h>

  int main()
  {
    __m256i a = {0};
    a = _mm256_abs_epi16(a);
    __m256i x;
    _mm256_extract_epi64(x, 0); // we rely on this in our AVX2 code
    return 0;
  }
"
)

MACRO(CHECK_SSE lang type flags)
  SET(__FLAG_I 1)
  SET(CMAKE_REQUIRED_FLAGS_SAVE ${CMAKE_REQUIRED_FLAGS})
  FOREACH(__FLAG ${flags})
    IF(NOT ${lang}_${type}_FOUND)
      SET(CMAKE_REQUIRED_FLAGS ${__FLAG})
      CHECK_CXX_SOURCE_RUNS("${${type}_CODE}" ${lang}_HAS_${type}_${__FLAG_I})
      IF(${lang}_HAS_${type}_${__FLAG_I})
        SET(${lang}_${type}_FOUND
            TRUE
            CACHE BOOL "${lang} ${type} support"
        )
        SET(${lang}_${type}_FLAGS
            "${__FLAG}"
            CACHE STRING "${lang} ${type} flags"
        )
      ENDIF()
      MATH(EXPR __FLAG_I "${__FLAG_I}+1")
    ENDIF()
  ENDFOREACH()
  SET(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_SAVE})

  IF(NOT ${lang}_${type}_FOUND)
    SET(${lang}_${type}_FOUND
        FALSE
        CACHE BOOL "${lang} ${type} support"
    )
    SET(${lang}_${type}_FLAGS
        ""
        CACHE STRING "${lang} ${type} flags"
    )
  ENDIF()

  MARK_AS_ADVANCED(${lang}_${type}_FOUND ${lang}_${type}_FLAGS)

ENDMACRO()

# CHECK_SSE(C "AVX" " ;-mavx;/arch:AVX") CHECK_SSE(C "AVX2" " ;-mavx2 -mfma;/arch:AVX2") CHECK_SSE(C
# "AVX512" " ;-mavx512f -mavx512dq -mavx512vl -mavx512bw -mfma;/arch:AVX512")
#
CHECK_SSE(CXX "AVX" " ;-mavx;/arch:AVX")
CHECK_SSE(CXX "AVX2" " ;-mavx2 -mfma;/arch:AVX2")
CHECK_SSE(CXX "AVX512" " ;-mavx512f -mavx512dq -mavx512vl -mavx512bw -mfma;/arch:AVX512")
