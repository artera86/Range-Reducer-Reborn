// Unit tests for host-compatible functions from secp256k1.cuh and main.cu
// Compiles with: g++ -std=c++17 -O2 -o test_host_functions tests/test_host_functions.cpp
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <cassert>

// ── Portable redefinitions of CUDA-only types/qualifiers ──────────────────
#define BIGINT_WORDS 8

struct BigInt {
    uint32_t data[BIGINT_WORDS];
};

// ── Host-compatible functions extracted from secp256k1.cuh ────────────────

static inline void init_bigint(BigInt *x, uint32_t val) {
    x->data[0] = val;
    for (int i = 1; i < BIGINT_WORDS; i++) x->data[i] = 0;
}

static inline void copy_bigint(BigInt *dest, const BigInt *src) {
    for (int i = 0; i < BIGINT_WORDS; i++) {
        dest->data[i] = src->data[i];
    }
}

static inline int compare_bigint(const BigInt *a, const BigInt *b) {
    for (int i = BIGINT_WORDS - 1; i >= 0; i--) {
        if (a->data[i] > b->data[i]) return 1;
        if (a->data[i] < b->data[i]) return -1;
    }
    return 0;
}

static inline bool is_zero(const BigInt *a) {
    for (int i = 0; i < BIGINT_WORDS; i++) {
        if (a->data[i]) return false;
    }
    return true;
}

static inline int get_bit(const BigInt *a, int i) {
    int word_idx = i >> 5;
    int bit_idx = i & 31;
    if (word_idx >= BIGINT_WORDS) return 0;
    return (a->data[word_idx] >> bit_idx) & 1;
}

// ── Host-compatible functions extracted from main.cu ──────────────────────

static inline uint8_t hex_char_to_byte(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return 0;
}

static void hex_string_to_bytes(const char* hex_str, uint8_t* bytes, int num_bytes) {
    for (int i = 0; i < num_bytes; i++) {
        bytes[i] = (hex_char_to_byte(hex_str[i * 2]) << 4) |
                   hex_char_to_byte(hex_str[i * 2 + 1]);
    }
}

static void hex_to_bigint(const char* hex_str, BigInt* bigint) {
    for (int i = 0; i < 8; i++) {
        bigint->data[i] = 0;
    }

    int len = 0;
    while (hex_str[len] != '\0' && len < 64) len++;

    int word_idx = 0;
    int bit_offset = 0;

    for (int i = len - 1; i >= 0 && word_idx < 8; i--) {
        uint8_t val = hex_char_to_byte(hex_str[i]);
        bigint->data[word_idx] |= ((uint32_t)val << bit_offset);
        bit_offset += 4;
        if (bit_offset >= 32) {
            bit_offset = 0;
            word_idx++;
        }
    }
}

static void clear_last_6_hex(BigInt* num) {
    num->data[0] &= 0xFF000000;
}

// cpu_u256Sub from secp256k1.cuh
static inline void cpu_u256Sub(BigInt* res, const BigInt* a, const BigInt* b) {
    uint64_t borrow = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t diff = (uint64_t)a->data[i] - (uint64_t)b->data[i] - borrow;
        res->data[i] = (uint32_t)diff;
        borrow = (diff >> 63) & 1;
    }
}

// ── Minimal test harness ──────────────────────────────────────────────────

static int g_tests_run = 0;
static int g_tests_passed = 0;

#define TEST(name) \
    static void test_##name(); \
    static struct Register_##name { \
        Register_##name() { run_test(#name, test_##name); } \
    } reg_##name; \
    static void test_##name()

static void run_test(const char* name, void (*fn)()) {
    g_tests_run++;
    printf("  %-50s ", name);
    try {
        fn();
        g_tests_passed++;
        printf("PASSED\n");
    } catch (...) {
        printf("FAILED\n");
    }
}

#define ASSERT_TRUE(cond) \
    do { if (!(cond)) { printf("ASSERT_TRUE failed: %s (line %d)\n", #cond, __LINE__); throw 1; } } while(0)

#define ASSERT_EQ(a, b) \
    do { if ((a) != (b)) { printf("ASSERT_EQ failed: %s != %s (line %d)\n", #a, #b, __LINE__); throw 1; } } while(0)

// ── Tests: init_bigint ────────────────────────────────────────────────────

TEST(init_bigint_zero) {
    BigInt x;
    init_bigint(&x, 0);
    for (int i = 0; i < BIGINT_WORDS; i++)
        ASSERT_EQ(x.data[i], 0u);
}

TEST(init_bigint_one) {
    BigInt x;
    init_bigint(&x, 1);
    ASSERT_EQ(x.data[0], 1u);
    for (int i = 1; i < BIGINT_WORDS; i++)
        ASSERT_EQ(x.data[i], 0u);
}

TEST(init_bigint_large) {
    BigInt x;
    init_bigint(&x, 0xDEADBEEF);
    ASSERT_EQ(x.data[0], 0xDEADBEEFu);
    for (int i = 1; i < BIGINT_WORDS; i++)
        ASSERT_EQ(x.data[i], 0u);
}

// ── Tests: copy_bigint ────────────────────────────────────────────────────

TEST(copy_bigint_basic) {
    BigInt src, dest;
    init_bigint(&src, 42);
    copy_bigint(&dest, &src);
    ASSERT_EQ(compare_bigint(&src, &dest), 0);
}

TEST(copy_bigint_full) {
    BigInt src = {{ 0x11111111, 0x22222222, 0x33333333, 0x44444444,
                    0x55555555, 0x66666666, 0x77777777, 0x88888888 }};
    BigInt dest;
    copy_bigint(&dest, &src);
    for (int i = 0; i < BIGINT_WORDS; i++)
        ASSERT_EQ(dest.data[i], src.data[i]);
}

// ── Tests: compare_bigint ─────────────────────────────────────────────────

TEST(compare_bigint_equal) {
    BigInt a, b;
    init_bigint(&a, 100);
    init_bigint(&b, 100);
    ASSERT_EQ(compare_bigint(&a, &b), 0);
}

TEST(compare_bigint_less) {
    BigInt a, b;
    init_bigint(&a, 50);
    init_bigint(&b, 100);
    ASSERT_EQ(compare_bigint(&a, &b), -1);
}

TEST(compare_bigint_greater) {
    BigInt a, b;
    init_bigint(&a, 200);
    init_bigint(&b, 100);
    ASSERT_EQ(compare_bigint(&a, &b), 1);
}

TEST(compare_bigint_high_word_differs) {
    BigInt a = {{}}, b = {{}};
    a.data[7] = 1;
    b.data[0] = 0xFFFFFFFF;
    ASSERT_EQ(compare_bigint(&a, &b), 1);
}

TEST(compare_bigint_both_zero) {
    BigInt a, b;
    init_bigint(&a, 0);
    init_bigint(&b, 0);
    ASSERT_EQ(compare_bigint(&a, &b), 0);
}

// ── Tests: is_zero ────────────────────────────────────────────────────────

TEST(is_zero_true) {
    BigInt x;
    init_bigint(&x, 0);
    ASSERT_TRUE(is_zero(&x));
}

TEST(is_zero_false_low) {
    BigInt x;
    init_bigint(&x, 1);
    ASSERT_TRUE(!is_zero(&x));
}

TEST(is_zero_false_high) {
    BigInt x;
    init_bigint(&x, 0);
    x.data[7] = 1;
    ASSERT_TRUE(!is_zero(&x));
}

TEST(is_zero_all_bits_set) {
    BigInt x;
    for (int i = 0; i < BIGINT_WORDS; i++) x.data[i] = 0xFFFFFFFF;
    ASSERT_TRUE(!is_zero(&x));
}

// ── Tests: get_bit ────────────────────────────────────────────────────────

TEST(get_bit_zero_value) {
    BigInt x;
    init_bigint(&x, 0);
    for (int i = 0; i < 256; i++)
        ASSERT_EQ(get_bit(&x, i), 0);
}

TEST(get_bit_one) {
    BigInt x;
    init_bigint(&x, 1);
    ASSERT_EQ(get_bit(&x, 0), 1);
    ASSERT_EQ(get_bit(&x, 1), 0);
}

TEST(get_bit_high_bit) {
    BigInt x;
    init_bigint(&x, 0);
    x.data[7] = 0x80000000;
    ASSERT_EQ(get_bit(&x, 255), 1);
    ASSERT_EQ(get_bit(&x, 254), 0);
}

TEST(get_bit_pattern) {
    BigInt x;
    init_bigint(&x, 0xA5);  // 10100101
    ASSERT_EQ(get_bit(&x, 0), 1);
    ASSERT_EQ(get_bit(&x, 1), 0);
    ASSERT_EQ(get_bit(&x, 2), 1);
    ASSERT_EQ(get_bit(&x, 3), 0);
    ASSERT_EQ(get_bit(&x, 4), 0);
    ASSERT_EQ(get_bit(&x, 5), 1);
    ASSERT_EQ(get_bit(&x, 6), 0);
    ASSERT_EQ(get_bit(&x, 7), 1);
}

TEST(get_bit_out_of_range) {
    BigInt x;
    for (int i = 0; i < BIGINT_WORDS; i++) x.data[i] = 0xFFFFFFFF;
    ASSERT_EQ(get_bit(&x, 256), 0);
    ASSERT_EQ(get_bit(&x, 1000), 0);
}

// ── Tests: hex_char_to_byte ───────────────────────────────────────────────

TEST(hex_char_to_byte_digits) {
    ASSERT_EQ(hex_char_to_byte('0'), 0);
    ASSERT_EQ(hex_char_to_byte('5'), 5);
    ASSERT_EQ(hex_char_to_byte('9'), 9);
}

TEST(hex_char_to_byte_lowercase) {
    ASSERT_EQ(hex_char_to_byte('a'), 10);
    ASSERT_EQ(hex_char_to_byte('c'), 12);
    ASSERT_EQ(hex_char_to_byte('f'), 15);
}

TEST(hex_char_to_byte_uppercase) {
    ASSERT_EQ(hex_char_to_byte('A'), 10);
    ASSERT_EQ(hex_char_to_byte('C'), 12);
    ASSERT_EQ(hex_char_to_byte('F'), 15);
}

TEST(hex_char_to_byte_invalid) {
    ASSERT_EQ(hex_char_to_byte('g'), 0);
    ASSERT_EQ(hex_char_to_byte('z'), 0);
    ASSERT_EQ(hex_char_to_byte(' '), 0);
}

// ── Tests: hex_string_to_bytes ────────────────────────────────────────────

TEST(hex_string_to_bytes_basic) {
    uint8_t bytes[2];
    hex_string_to_bytes("abcd", bytes, 2);
    ASSERT_EQ(bytes[0], 0xAB);
    ASSERT_EQ(bytes[1], 0xCD);
}

TEST(hex_string_to_bytes_zeros) {
    uint8_t bytes[4];
    hex_string_to_bytes("00000000", bytes, 4);
    for (int i = 0; i < 4; i++)
        ASSERT_EQ(bytes[i], 0);
}

TEST(hex_string_to_bytes_ff) {
    uint8_t bytes[3];
    hex_string_to_bytes("ffffff", bytes, 3);
    for (int i = 0; i < 3; i++)
        ASSERT_EQ(bytes[i], 0xFF);
}

TEST(hex_string_to_bytes_hash160) {
    uint8_t bytes[20];
    hex_string_to_bytes("29a78213caa9eea824acf08022ab9dfc83414f56", bytes, 20);
    ASSERT_EQ(bytes[0], 0x29);
    ASSERT_EQ(bytes[1], 0xa7);
    ASSERT_EQ(bytes[19], 0x56);
}

// ── Tests: hex_to_bigint ──────────────────────────────────────────────────

TEST(hex_to_bigint_zero) {
    BigInt x;
    hex_to_bigint("0", &x);
    ASSERT_TRUE(is_zero(&x));
}

TEST(hex_to_bigint_one) {
    BigInt x;
    hex_to_bigint("1", &x);
    BigInt expected;
    init_bigint(&expected, 1);
    ASSERT_EQ(compare_bigint(&x, &expected), 0);
}

TEST(hex_to_bigint_small) {
    BigInt x;
    hex_to_bigint("ff", &x);
    BigInt expected;
    init_bigint(&expected, 0xFF);
    ASSERT_EQ(compare_bigint(&x, &expected), 0);
}

TEST(hex_to_bigint_word_boundary) {
    BigInt x;
    hex_to_bigint("100000000", &x);
    ASSERT_EQ(x.data[0], 0u);
    ASSERT_EQ(x.data[1], 1u);
    for (int i = 2; i < BIGINT_WORDS; i++)
        ASSERT_EQ(x.data[i], 0u);
}

TEST(hex_to_bigint_secp256k1_p) {
    BigInt x;
    hex_to_bigint("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F", &x);
    ASSERT_EQ(x.data[0], 0xFFFFFC2Fu);
    ASSERT_EQ(x.data[1], 0xFFFFFFFEu);
    for (int i = 2; i < BIGINT_WORDS; i++)
        ASSERT_EQ(x.data[i], 0xFFFFFFFFu);
}

TEST(hex_to_bigint_lowercase) {
    BigInt a, b;
    hex_to_bigint("abcdef", &a);
    hex_to_bigint("ABCDEF", &b);
    ASSERT_EQ(compare_bigint(&a, &b), 0);
}

TEST(hex_to_bigint_puzzle_range) {
    BigInt x;
    hex_to_bigint("1fffff", &x);
    ASSERT_EQ(x.data[0], 0x1FFFFFu);
    for (int i = 1; i < BIGINT_WORDS; i++)
        ASSERT_EQ(x.data[i], 0u);
}

// ── Tests: clear_last_6_hex ───────────────────────────────────────────────

TEST(clear_last_6_hex_basic) {
    BigInt x;
    init_bigint(&x, 0xFFFFFFFF);
    clear_last_6_hex(&x);
    ASSERT_EQ(x.data[0], 0xFF000000u);
}

TEST(clear_last_6_hex_preserves_upper) {
    BigInt x;
    hex_to_bigint("123456789ABCDEF0", &x);
    uint32_t saved_word1 = x.data[1];
    clear_last_6_hex(&x);
    ASSERT_EQ(x.data[0] & 0x00FFFFFF, 0u);
    ASSERT_EQ(x.data[1], saved_word1);
}

TEST(clear_last_6_hex_zero) {
    BigInt x;
    init_bigint(&x, 0);
    clear_last_6_hex(&x);
    ASSERT_EQ(x.data[0], 0u);
}

TEST(clear_last_6_hex_only_low_bits) {
    BigInt x;
    init_bigint(&x, 0x00FFFFFF);
    clear_last_6_hex(&x);
    ASSERT_EQ(x.data[0], 0u);
}

// ── Tests: cpu_u256Sub ────────────────────────────────────────────────────

TEST(cpu_u256Sub_simple) {
    BigInt a, b, res;
    init_bigint(&a, 10);
    init_bigint(&b, 3);
    cpu_u256Sub(&res, &a, &b);
    BigInt expected;
    init_bigint(&expected, 7);
    ASSERT_EQ(compare_bigint(&res, &expected), 0);
}

TEST(cpu_u256Sub_same) {
    BigInt a, b, res;
    init_bigint(&a, 42);
    init_bigint(&b, 42);
    cpu_u256Sub(&res, &a, &b);
    ASSERT_TRUE(is_zero(&res));
}

TEST(cpu_u256Sub_borrow) {
    BigInt a = {{ 0, 1, 0, 0, 0, 0, 0, 0 }};    // 0x100000000
    BigInt b;
    init_bigint(&b, 1);
    BigInt res;
    cpu_u256Sub(&res, &a, &b);
    ASSERT_EQ(res.data[0], 0xFFFFFFFF);
    ASSERT_EQ(res.data[1], 0u);
}

TEST(cpu_u256Sub_p_minus_2) {
    // Reproduce init_gpu_constants logic: p - 2
    const BigInt p = {{
        0xFFFFFC2F, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF,
        0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
    }};
    BigInt two;
    init_bigint(&two, 2);
    BigInt p_minus_2;
    cpu_u256Sub(&p_minus_2, &p, &two);
    ASSERT_EQ(p_minus_2.data[0], 0xFFFFFC2Du);
    ASSERT_EQ(p_minus_2.data[1], 0xFFFFFFFEu);
    for (int i = 2; i < BIGINT_WORDS; i++)
        ASSERT_EQ(p_minus_2.data[i], 0xFFFFFFFFu);
}

TEST(cpu_u256Sub_large) {
    BigInt a, b, res;
    for (int i = 0; i < BIGINT_WORDS; i++) {
        a.data[i] = 0xFFFFFFFF;
        b.data[i] = 0xFFFFFFFF;
    }
    cpu_u256Sub(&res, &a, &b);
    ASSERT_TRUE(is_zero(&res));
}

TEST(cpu_u256Sub_multi_borrow) {
    BigInt a = {{ 0, 0, 1, 0, 0, 0, 0, 0 }};   // 2^64
    BigInt b;
    init_bigint(&b, 1);
    BigInt res;
    cpu_u256Sub(&res, &a, &b);
    ASSERT_EQ(res.data[0], 0xFFFFFFFF);
    ASSERT_EQ(res.data[1], 0xFFFFFFFF);
    ASSERT_EQ(res.data[2], 0u);
}

// ── Tests: round-trip hex_to_bigint consistency ───────────────────────────

TEST(hex_to_bigint_roundtrip_compare) {
    BigInt a, b;
    hex_to_bigint("100000", &a);
    hex_to_bigint("1fffff", &b);
    ASSERT_EQ(compare_bigint(&a, &b), -1);  // a < b
}

TEST(hex_to_bigint_n_value) {
    // secp256k1 order n
    BigInt n;
    hex_to_bigint("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141", &n);
    ASSERT_EQ(n.data[0], 0xD0364141u);
    ASSERT_EQ(n.data[1], 0xBFD25E8Cu);
    ASSERT_EQ(n.data[2], 0xAF48A03Bu);
    ASSERT_EQ(n.data[3], 0xBAAEDCE6u);
    ASSERT_EQ(n.data[4], 0xFFFFFFFEu);
    for (int i = 5; i < BIGINT_WORDS; i++)
        ASSERT_EQ(n.data[i], 0xFFFFFFFFu);
}

// ── Tests: edge cases & integration ───────────────────────────────────────

TEST(bigint_max_value) {
    BigInt x;
    for (int i = 0; i < BIGINT_WORDS; i++) x.data[i] = 0xFFFFFFFF;
    ASSERT_TRUE(!is_zero(&x));
    ASSERT_EQ(get_bit(&x, 0), 1);
    ASSERT_EQ(get_bit(&x, 255), 1);
}

TEST(clear_last_6_hex_then_compare) {
    BigInt a, b;
    hex_to_bigint("FFFFFFFF", &a);
    hex_to_bigint("FF000000", &b);
    clear_last_6_hex(&a);
    ASSERT_EQ(compare_bigint(&a, &b), 0);
}

TEST(sub_then_compare) {
    BigInt a, b, diff;
    hex_to_bigint("1fffff", &a);
    hex_to_bigint("100000", &b);
    cpu_u256Sub(&diff, &a, &b);
    BigInt expected;
    init_bigint(&expected, 0xFFFFF);
    ASSERT_EQ(compare_bigint(&diff, &expected), 0);
}

TEST(copy_then_modify_independent) {
    BigInt src, dest;
    init_bigint(&src, 99);
    copy_bigint(&dest, &src);
    src.data[0] = 0;
    ASSERT_EQ(dest.data[0], 99u);
}

TEST(hex_to_bigint_single_digit) {
    BigInt x;
    hex_to_bigint("a", &x);
    ASSERT_EQ(x.data[0], 10u);
}

TEST(hex_to_bigint_odd_length) {
    BigInt x;
    hex_to_bigint("abc", &x);
    ASSERT_EQ(x.data[0], 0xABCu);
}

// ── Main ──────────────────────────────────────────────────────────────────

// Test registration: we call each test function explicitly since we don't have
// a real test framework with auto-registration via static constructors in a
// single TU.

typedef void (*TestFunc)();
struct TestEntry { const char* name; TestFunc fn; };

// Forward declarations
void test_init_bigint_zero();
void test_init_bigint_one();
void test_init_bigint_large();
void test_copy_bigint_basic();
void test_copy_bigint_full();
void test_compare_bigint_equal();
void test_compare_bigint_less();
void test_compare_bigint_greater();
void test_compare_bigint_high_word_differs();
void test_compare_bigint_both_zero();
void test_is_zero_true();
void test_is_zero_false_low();
void test_is_zero_false_high();
void test_is_zero_all_bits_set();
void test_get_bit_zero_value();
void test_get_bit_one();
void test_get_bit_high_bit();
void test_get_bit_pattern();
void test_get_bit_out_of_range();
void test_hex_char_to_byte_digits();
void test_hex_char_to_byte_lowercase();
void test_hex_char_to_byte_uppercase();
void test_hex_char_to_byte_invalid();
void test_hex_string_to_bytes_basic();
void test_hex_string_to_bytes_zeros();
void test_hex_string_to_bytes_ff();
void test_hex_string_to_bytes_hash160();
void test_hex_to_bigint_zero();
void test_hex_to_bigint_one();
void test_hex_to_bigint_small();
void test_hex_to_bigint_word_boundary();
void test_hex_to_bigint_secp256k1_p();
void test_hex_to_bigint_lowercase();
void test_hex_to_bigint_puzzle_range();
void test_clear_last_6_hex_basic();
void test_clear_last_6_hex_preserves_upper();
void test_clear_last_6_hex_zero();
void test_clear_last_6_hex_only_low_bits();
void test_cpu_u256Sub_simple();
void test_cpu_u256Sub_same();
void test_cpu_u256Sub_borrow();
void test_cpu_u256Sub_p_minus_2();
void test_cpu_u256Sub_large();
void test_cpu_u256Sub_multi_borrow();
void test_hex_to_bigint_roundtrip_compare();
void test_hex_to_bigint_n_value();
void test_bigint_max_value();
void test_clear_last_6_hex_then_compare();
void test_sub_then_compare();
void test_copy_then_modify_independent();
void test_hex_to_bigint_single_digit();
void test_hex_to_bigint_odd_length();

static const TestEntry all_tests[] = {
    // init_bigint
    { "init_bigint_zero", test_init_bigint_zero },
    { "init_bigint_one", test_init_bigint_one },
    { "init_bigint_large", test_init_bigint_large },
    // copy_bigint
    { "copy_bigint_basic", test_copy_bigint_basic },
    { "copy_bigint_full", test_copy_bigint_full },
    // compare_bigint
    { "compare_bigint_equal", test_compare_bigint_equal },
    { "compare_bigint_less", test_compare_bigint_less },
    { "compare_bigint_greater", test_compare_bigint_greater },
    { "compare_bigint_high_word_differs", test_compare_bigint_high_word_differs },
    { "compare_bigint_both_zero", test_compare_bigint_both_zero },
    // is_zero
    { "is_zero_true", test_is_zero_true },
    { "is_zero_false_low", test_is_zero_false_low },
    { "is_zero_false_high", test_is_zero_false_high },
    { "is_zero_all_bits_set", test_is_zero_all_bits_set },
    // get_bit
    { "get_bit_zero_value", test_get_bit_zero_value },
    { "get_bit_one", test_get_bit_one },
    { "get_bit_high_bit", test_get_bit_high_bit },
    { "get_bit_pattern", test_get_bit_pattern },
    { "get_bit_out_of_range", test_get_bit_out_of_range },
    // hex_char_to_byte
    { "hex_char_to_byte_digits", test_hex_char_to_byte_digits },
    { "hex_char_to_byte_lowercase", test_hex_char_to_byte_lowercase },
    { "hex_char_to_byte_uppercase", test_hex_char_to_byte_uppercase },
    { "hex_char_to_byte_invalid", test_hex_char_to_byte_invalid },
    // hex_string_to_bytes
    { "hex_string_to_bytes_basic", test_hex_string_to_bytes_basic },
    { "hex_string_to_bytes_zeros", test_hex_string_to_bytes_zeros },
    { "hex_string_to_bytes_ff", test_hex_string_to_bytes_ff },
    { "hex_string_to_bytes_hash160", test_hex_string_to_bytes_hash160 },
    // hex_to_bigint
    { "hex_to_bigint_zero", test_hex_to_bigint_zero },
    { "hex_to_bigint_one", test_hex_to_bigint_one },
    { "hex_to_bigint_small", test_hex_to_bigint_small },
    { "hex_to_bigint_word_boundary", test_hex_to_bigint_word_boundary },
    { "hex_to_bigint_secp256k1_p", test_hex_to_bigint_secp256k1_p },
    { "hex_to_bigint_lowercase", test_hex_to_bigint_lowercase },
    { "hex_to_bigint_puzzle_range", test_hex_to_bigint_puzzle_range },
    // clear_last_6_hex
    { "clear_last_6_hex_basic", test_clear_last_6_hex_basic },
    { "clear_last_6_hex_preserves_upper", test_clear_last_6_hex_preserves_upper },
    { "clear_last_6_hex_zero", test_clear_last_6_hex_zero },
    { "clear_last_6_hex_only_low_bits", test_clear_last_6_hex_only_low_bits },
    // cpu_u256Sub
    { "cpu_u256Sub_simple", test_cpu_u256Sub_simple },
    { "cpu_u256Sub_same", test_cpu_u256Sub_same },
    { "cpu_u256Sub_borrow", test_cpu_u256Sub_borrow },
    { "cpu_u256Sub_p_minus_2", test_cpu_u256Sub_p_minus_2 },
    { "cpu_u256Sub_large", test_cpu_u256Sub_large },
    { "cpu_u256Sub_multi_borrow", test_cpu_u256Sub_multi_borrow },
    // integration / round-trip
    { "hex_to_bigint_roundtrip_compare", test_hex_to_bigint_roundtrip_compare },
    { "hex_to_bigint_n_value", test_hex_to_bigint_n_value },
    { "bigint_max_value", test_bigint_max_value },
    { "clear_last_6_hex_then_compare", test_clear_last_6_hex_then_compare },
    { "sub_then_compare", test_sub_then_compare },
    { "copy_then_modify_independent", test_copy_then_modify_independent },
    { "hex_to_bigint_single_digit", test_hex_to_bigint_single_digit },
    { "hex_to_bigint_odd_length", test_hex_to_bigint_odd_length },
};

int main() {
    printf("Running %zu tests...\n\n", sizeof(all_tests) / sizeof(all_tests[0]));
    for (const auto& t : all_tests) {
        run_test(t.name, t.fn);
    }
    printf("\n%d/%d tests passed.\n", g_tests_passed, g_tests_run);
    return (g_tests_passed == g_tests_run) ? 0 : 1;
}
