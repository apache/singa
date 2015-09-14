#include "gtest/gtest.h"
#include "utils/common.h"
#include <unordered_map>
#include <string>
#include <vector>

using std::string;
using std::vector;
using namespace singa;

TEST(CommonTest, TestIntVecToString) {

    vector<int> num_vec {2, 3, 5, 7, 11};
    string str = "(2, 3, 5, 7, 11, )";
    ASSERT_EQ(str, IntVecToString(num_vec));
}

TEST(CommonTest, TestStringPrintf) {
    const char* str_a = "abc";
    const char* str_b = "edfgh";
    const char* str_c = " !@#";
    const char* str_d = "1";
    const char* str_e = "2";
    const char* str_f = "3";

    string fmt_a = "%s%s%s";
    string fmt_b = "[%s] [%s] [%s] ";

    string str_d_a = "abcedfgh !@#";
    string str_d_b = "[1] [2] [3] ";

    ASSERT_EQ(str_d_a, StringPrintf(fmt_a, str_a, str_b, str_c));
    ASSERT_EQ(str_d_b, StringPrintf(fmt_b, str_d, str_e, str_f));
}

TEST(CommonTest, TestGCDLCM) {
    int a = 2, b = 5, c = 10, d = 15;

    ASSERT_EQ(1, gcd(a, b));
    ASSERT_EQ(5, gcd(c, d));
    ASSERT_EQ(10, LeastCommonMultiple(b, c));
    ASSERT_EQ(30, LeastCommonMultiple(c, d));
}

TEST(CommonTest, TestMetric) {
    string str, msg;
    Metric metric;
    metric.Add("a", 0.5);
    metric.Add("b", 0.5);
    metric.Add("a", 1.5);
    str = metric.ToLogString();
    msg = metric.ToString();
    metric.Reset();
    metric.ParseFrom(msg);
    ASSERT_EQ(str, metric.ToLogString());
}

TEST(CommonTest, TestSlice) {
    vector<vector<int>> slices_0;
    vector<int> sizes {14112, 96, 256, 884736, 384};
    ASSERT_EQ(slices_0, Slice(0, sizes));
    
    vector<vector<int>> slices_1 {
        { 14112 },
        { 96 },
        { 256 },
        { 884736 },
        { 384 },
    };
    
    vector<vector<int>> slices_2 {
        { 14112 },
        { 96 },
        { 256 },
        { 435328, 449408 },
        { 384 },
    };
        
    vector<vector<int>> slices_4 {
        { 14112 },
        { 96 },
        { 256 },
        { 210432,224896,224896,224512 },
        { 384 },
    };
    
    vector<vector<int>> slices_8 {
        { 14112 },
        { 96 },
        { 256 },
        { 97984,112448,112448,112448,112448,112448,112448,112064 },
        { 384 },
    };
    
    ASSERT_EQ(slices_1, Slice(1, sizes));
    ASSERT_EQ(slices_2, Slice(2, sizes));
    ASSERT_EQ(slices_4, Slice(4, sizes));
    ASSERT_EQ(slices_8, Slice(8, sizes));
}

TEST(CommonTest, TestPartitionSlices) {
    vector<int> slices {
         97984,112448,112448,112448,112448,112448,112448,112064
    };
    vector<int> box_1 { 0, 0, 0, 0, 0, 0, 0, 0 };
    vector<int> box_2 { 0, 0, 0, 0, 1, 1, 1, 1 };
    vector<int> box_4 { 0, 0, 1, 1, 2, 2, 3, 3 };
    vector<int> box_8 { 0, 1, 2, 3, 4, 5, 6, 7 };
    ASSERT_EQ(box_1, PartitionSlices(1, slices));
    ASSERT_EQ(box_2, PartitionSlices(2, slices));
    ASSERT_EQ(box_4, PartitionSlices(4, slices));
    ASSERT_EQ(box_8, PartitionSlices(8, slices));
}
