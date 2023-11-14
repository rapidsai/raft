#include <gtest/gtest.h>

#include "../ann_mg.cuh"

namespace raft::neighbors::mg {

typedef AnnMGTest<float, float, uint32_t> AnnMGTestF_float;
    TEST_P(AnnMGTestF_float, AnnMG) { this->testAnnMG(); }
    INSTANTIATE_TEST_CASE_P(AnnMGTest, AnnMGTestF_float, ::testing::ValuesIn(inputs));
}
