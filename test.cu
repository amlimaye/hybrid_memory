#include <stdio.h>
#include <gtest/gtest.h>
#include "hybrid_memory.hxx"

class HybridMemoryTest : public ::testing::Test {
protected:
    virtual void SetUp() {
        _ptr = new int[100];
        _hm = utils::hybrid_memory<int>(_ptr, 100);
        _hm_null_init = utils::hybrid_memory<int>();
    }

    virtual void TearDown() {
        delete[] _ptr;
    }

    int* _ptr; 
    utils::hybrid_memory<int> _hm;
    utils::hybrid_memory<int> _hm_null_init;
};

TEST_F(HybridMemoryTest, HostPointer) {
    int* host_ptr = _hm.host();
    int* null_host_ptr = _hm_null_init.host();
    EXPECT_EQ(_ptr, host_ptr);
    EXPECT_EQ(nullptr, null_host_ptr);
}

TEST_F(HybridMemoryTest, Size) {
    EXPECT_EQ(_hm.size(), 100);
    EXPECT_EQ(_hm_null_init.size(), 0);
}

TEST_F(HybridMemoryTest, FillOnHost) {
    _hm.fill(0);
    for (int k = 0; k < _hm.size(); k++) {
        EXPECT_EQ(0, _hm.host()[k]);
    }

    _hm.fill(10);
    for (int k = 0; k < _hm.size(); k++) {
        EXPECT_EQ(10, _hm.host()[k]);
    }

    _hm.fill(100);
    for (int k = 0; k < _hm.size(); k++) {
        EXPECT_EQ(100, _hm.host()[k]);
    }
}

TEST_F(HybridMemoryTest, DeviceNotActive) {
    try {
        int* device_ptr = _hm.device();
        FAIL() << "Expected std::runtime_error";
    } catch (const std::runtime_error& err) {
        EXPECT_EQ(err.what(), std::string("not active on the device!"));
    } catch(...) {
        FAIL() << "Expected std::runtime_error";
    }
}

TEST_F(HybridMemoryTest, HostNotActive) {
    _hm.upload();
    try {
        int* host_ptr = _hm.host();
        FAIL() << "Expected std::runtime_error";
    } catch (const std::runtime_error& err) {
        EXPECT_EQ(err.what(), std::string("not active on the host!"));
    } catch(...) {
        FAIL() << "Expected std::runtime_error";
    }
}

TEST_F(HybridMemoryTest, UploadDownloadIntegrity) {
    _hm.fill(0);
    _hm.upload();
    _hm.download();
    for (int k = 0; k < _hm.size(); k++) {
        EXPECT_EQ(0, _hm.host()[k]);
    }

    _hm.fill(10);
    _hm.upload();
    _hm.download();
    for (int k = 0; k < _hm.size(); k++) {
        EXPECT_EQ(10, _hm.host()[k]);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
