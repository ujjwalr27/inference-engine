#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "io/safetensors.h"

namespace {

namespace fs = std::filesystem;

class SafeTensorsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
    path_ = (fs::temp_directory_path() / (std::string("gpt2_st_") + info->name() + ".safetensors")).string();
  }
  void TearDown() override { fs::remove(path_); }

  // Writes [u64 header_len][header][data]; header_len_override lets tests corrupt the length field.
  void write(const std::string& header, const std::vector<char>& data, int64_t header_len_override = -1) {
    std::ofstream out(path_, std::ios::binary);
    uint64_t len = header_len_override >= 0 ? static_cast<uint64_t>(header_len_override) : header.size();
    for (int i = 0; i < 8; ++i) out.put(static_cast<char>((len >> (8 * i)) & 0xFF));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    out.write(data.data(), static_cast<std::streamsize>(data.size()));
  }

  static std::vector<char> bytes_of(const std::vector<float>& f, const std::vector<int64_t>& i) {
    std::vector<char> out(f.size() * sizeof(float) + i.size() * sizeof(int64_t));
    std::memcpy(out.data(), f.data(), f.size() * sizeof(float));
    std::memcpy(out.data() + f.size() * sizeof(float), i.data(), i.size() * sizeof(int64_t));
    return out;
  }

  std::string path_;
};

const char* kValidHeader =
    R"({"__metadata__":{"source":"unit-test"},)"
    R"("a":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},)"
    R"("b":{"dtype":"I64","shape":[3],"data_offsets":[16,40]}})";

TEST_F(SafeTensorsTest, LoadsTensorsAndMetadata) {
  write(kValidHeader, bytes_of({1.0f, 2.0f, 3.0f, 4.0f}, {7, -8, 9}));
  const auto st = gpt2::SafeTensors::open(path_);

  EXPECT_EQ(st.names(), (std::vector<std::string>{"a", "b"}));
  EXPECT_EQ(st.metadata().at("source"), "unit-test");
  EXPECT_EQ(st.dtype("a"), torch::kFloat32);
  EXPECT_EQ(st.shape("a"), (std::vector<int64_t>{2, 2}));

  const auto a = st.load("a");
  EXPECT_TRUE(torch::equal(a, torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}})));
  const auto b = st.load("b");
  EXPECT_EQ(b.scalar_type(), torch::kInt64);
  EXPECT_TRUE(torch::equal(b, torch::tensor({7, -8, 9}, torch::kInt64)));
}

TEST_F(SafeTensorsTest, UnknownNameThrows) {
  write(kValidHeader, bytes_of({1.0f, 2.0f, 3.0f, 4.0f}, {7, -8, 9}));
  const auto st = gpt2::SafeTensors::open(path_);
  EXPECT_FALSE(st.contains("missing"));
  EXPECT_THROW(st.load("missing"), std::runtime_error);
}

TEST_F(SafeTensorsTest, RejectsTruncatedData) {
  write(kValidHeader, bytes_of({1.0f, 2.0f, 3.0f, 4.0f}, {7}));  // b needs 24 bytes, only 8 present
  EXPECT_THROW(gpt2::SafeTensors::open(path_), std::runtime_error);
}

TEST_F(SafeTensorsTest, RejectsSizeShapeMismatch) {
  write(R"({"a":{"dtype":"F32","shape":[3],"data_offsets":[0,8]}})", bytes_of({1.0f, 2.0f}, {}));
  EXPECT_THROW(gpt2::SafeTensors::open(path_), std::runtime_error);
}

TEST_F(SafeTensorsTest, RejectsBadHeaderLength) {
  write(kValidHeader, bytes_of({1.0f, 2.0f, 3.0f, 4.0f}, {7, -8, 9}), 1LL << 40);
  EXPECT_THROW(gpt2::SafeTensors::open(path_), std::runtime_error);
}

TEST_F(SafeTensorsTest, RejectsBadJsonAndDtype) {
  write("{not json", {});
  EXPECT_THROW(gpt2::SafeTensors::open(path_), std::runtime_error);
  write(R"({"a":{"dtype":"F8","shape":[1],"data_offsets":[0,1]}})", {'x'});
  EXPECT_THROW(gpt2::SafeTensors::open(path_), std::runtime_error);
}

TEST(SafeTensors, MissingFileThrows) {
  EXPECT_THROW(gpt2::SafeTensors::open("/nonexistent/file.safetensors"), std::runtime_error);
}

}  // namespace
