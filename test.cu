#include <stdio.h>
#include <gtest/gtest.h>
#include "hybrid_memory.hxx"

template <typename T>
class HybridMemoryTest : public ::testing::Test {
protected:
    virtual void SetUp() {
        _ptr = new T[100];
        _hm = utils::hybrid_memory<T>(_ptr, 100);
        _hm_null_init = utils::hybrid_memory<T>();
    }

    virtual void TearDown() {
        delete[] _ptr;
    }

    T* _ptr; 
    utils::hybrid_memory<T> _hm;
    utils::hybrid_memory<T> _hm_null_init;
};

typedef testing::Types<uint32_t, int, float> TestingTypes;

template <typename T>
T* get_fill_values();

template<>
uint32_t* get_fill_values<uint32_t>() {
    uint32_t* vals_ptr = new uint32_t[3];
    vals_ptr[0] = 0;
    vals_ptr[1] = 1;
    vals_ptr[2] = 2;
    return vals_ptr;
}

template<>
int* get_fill_values<int>() {
    int* vals_ptr = new int[3];
    vals_ptr[0] = -1;
    vals_ptr[1] = 0;
    vals_ptr[2] = 1;
    return vals_ptr;
}

template<>
float* get_fill_values<float>() {
    float* vals_ptr = new float[3];
    vals_ptr[0] = -0.1;
    vals_ptr[1] = 0.0;
    vals_ptr[2] = 0.1;
    return vals_ptr;
}

TYPED_TEST_CASE(HybridMemoryTest, TestingTypes);

TYPED_TEST(HybridMemoryTest, HostPointer) {
    auto host_ptr = this->_hm.host();
    auto null_host_ptr = this->_hm_null_init.host();
    EXPECT_EQ(this->_ptr, host_ptr);
    EXPECT_EQ(nullptr, null_host_ptr);
}

TYPED_TEST(HybridMemoryTest, Size) {
    EXPECT_EQ(this->_hm.size(), 100);
    EXPECT_EQ(this->_hm_null_init.size(), 0);
}

TYPED_TEST(HybridMemoryTest, FillOnHost) {
    auto fill_values = get_fill_values<TypeParam>();

    for (int i = 0; i < 3; i++) {
        auto fval = fill_values[i];
        this->_hm.fill(fval);
        for (int k = 0; k < this->_hm.size(); k++) {
            EXPECT_EQ(fval, this->_hm.host()[k]);
        }
    }
}

TYPED_TEST(HybridMemoryTest, DeviceNotActive) {
    try {
        auto device_ptr = this->_hm.device();
        FAIL() << "Expected std::runtime_error";
    } catch (const std::runtime_error& err) {
        EXPECT_EQ(err.what(), std::string("not active on the device!"));
    } catch(...) {
        FAIL() << "Expected std::runtime_error";
    }
}

TYPED_TEST(HybridMemoryTest, HostNotActive) {
    this->_hm.upload();
    try {
        auto host_ptr = this->_hm.host();
        FAIL() << "Expected std::runtime_error";
    } catch (const std::runtime_error& err) {
        EXPECT_EQ(err.what(), std::string("not active on the host!"));
    } catch(...) {
        FAIL() << "Expected std::runtime_error";
    }
}

TYPED_TEST(HybridMemoryTest, UploadDownloadIntegrity) {
    auto fill_values = get_fill_values<TypeParam>();
    
    for (int i = 0; i < 3; i++) {
        auto fval = fill_values[i];
        this->_hm.fill(fval);
        this->_hm.upload();
        this->_hm.download();
        for (int k = 0; k < this->_hm.size(); k++) {
            EXPECT_EQ(fval, this->_hm.host()[k]);
        }
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
