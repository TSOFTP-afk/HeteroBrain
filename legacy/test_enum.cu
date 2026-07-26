// 测试 enum class
enum class TestEnum : unsigned char {
    A = 0,
    B = 1,
};

int main() {
    TestEnum e = TestEnum::A;
    return (int)e;
}
