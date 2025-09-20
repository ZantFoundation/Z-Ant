#include "semihost.h"

#include <float.h>
#include <stddef.h>
#include <stdint.h>

#define HEAP_SIZE (384u * 1024u)
#define ALIGNMENT 8u

typedef struct BlockHeader {
    struct BlockHeader *next;
    struct BlockHeader *prev;
    size_t size;
    uint8_t free_flag;
    uint8_t padding[sizeof(size_t) - 1u];
} BlockHeader;

static uint8_t g_heap[HEAP_SIZE] __attribute__((aligned(ALIGNMENT)));
static BlockHeader *g_heap_head = NULL;

static size_t g_current_usage = 0u;
static size_t g_peak_usage = 0u;
static size_t g_alloc_failures = 0u;
static uint8_t g_warned_high_watermark = 0u;

static inline size_t align_up(size_t value) {
    return (value + (ALIGNMENT - 1u)) & ~(ALIGNMENT - 1u);
}

static void heap_init(void) {
    g_heap_head = (BlockHeader *)g_heap;
    g_heap_head->next = NULL;
    g_heap_head->prev = NULL;
    g_heap_head->size = HEAP_SIZE - sizeof(BlockHeader);
    g_heap_head->free_flag = 1u;
}

static BlockHeader *block_from_ptr(void *ptr) {
    if (ptr == NULL) {
        return NULL;
    }
    return (BlockHeader *)((uint8_t *)ptr - sizeof(BlockHeader));
}

static void split_block(BlockHeader *block, size_t size) {
    const size_t available = block->size;
    if (available <= size + sizeof(BlockHeader) + ALIGNMENT) {
        block->size = available;
        return;
    }

    uint8_t *base = (uint8_t *)block;
    BlockHeader *new_block = (BlockHeader *)(base + sizeof(BlockHeader) + size);
    new_block->size = available - size - sizeof(BlockHeader);
    new_block->free_flag = 1u;
    new_block->prev = block;
    new_block->next = block->next;
    if (new_block->next) {
        new_block->next->prev = new_block;
    }

    block->size = size;
    block->next = new_block;
}

static void coalesce(BlockHeader *block) {
    if (block == NULL) {
        return;
    }

    if (block->next && block->next->free_flag) {
        BlockHeader *next = block->next;
        block->size += sizeof(BlockHeader) + next->size;
        block->next = next->next;
        if (block->next) {
            block->next->prev = block;
        }
    }

    if (block->prev && block->prev->free_flag) {
        BlockHeader *prev = block->prev;
        prev->size += sizeof(BlockHeader) + block->size;
        prev->next = block->next;
        if (block->next) {
            block->next->prev = prev;
        }
    }
}

static BlockHeader *find_suitable_block(size_t size) {
    BlockHeader *curr = g_heap_head;
    while (curr != NULL) {
        if (curr->free_flag && curr->size >= size) {
            return curr;
        }
        curr = curr->next;
    }
    return NULL;
}

static void *do_memcpy(void *dest, const void *src, size_t n) {
    uint8_t *d = (uint8_t *)dest;
    const uint8_t *s = (const uint8_t *)src;
    for (size_t i = 0; i < n; ++i) {
        d[i] = s[i];
    }
    return dest;
}

static void *do_memmove(void *dest, const void *src, size_t n) {
    uint8_t *d = (uint8_t *)dest;
    const uint8_t *s = (const uint8_t *)src;
    if (d == s || n == 0u) {
        return dest;
    }
    if (d < s) {
        for (size_t i = 0; i < n; ++i) {
            d[i] = s[i];
        }
    } else {
        for (size_t i = n; i != 0u; --i) {
            d[i - 1u] = s[i - 1u];
        }
    }
    return dest;
}

static void *do_memset(void *dest, int value, size_t n) {
    uint8_t *d = (uint8_t *)dest;
    uint8_t v = (uint8_t)value;
    for (size_t i = 0; i < n; ++i) {
        d[i] = v;
    }
    return dest;
}

static void write_decimal(size_t value) {
    char buffer[32];
    size_t index = sizeof(buffer);
    buffer[--index] = '\0';
    if (value == 0u) {
        buffer[--index] = '0';
    } else {
        while (value != 0u && index > 0u) {
            buffer[--index] = (char)('0' + (value % 10u));
            value /= 10u;
        }
    }
    semihost_write0(&buffer[index]);
}

static void account_increase(size_t delta) {
    g_current_usage += delta;
    if (g_current_usage > g_peak_usage) {
        g_peak_usage = g_current_usage;
        if (g_peak_usage > (512u * 1024u) && !g_warned_high_watermark) {
            g_warned_high_watermark = 1u;
            semihost_write0("[heap] usage exceeded 512 KiB: ");
            write_decimal(g_peak_usage);
            semihost_write0(" bytes\n");
        }
    }
}

static void account_decrease(size_t delta) {
    if (g_current_usage >= delta) {
        g_current_usage -= delta;
    } else {
        g_current_usage = 0u;
    }
}

void *malloc(size_t size) {
    if (size == 0u) {
        return NULL;
    }

    if (g_heap_head == NULL) {
        heap_init();
    }

    size = align_up(size);
    BlockHeader *block = find_suitable_block(size);
    if (block == NULL) {
        if (g_alloc_failures == 0u) {
            semihost_write0("[heap] malloc failure requesting ");
            write_decimal(size);
            semihost_write0(" bytes\n");
        }
        ++g_alloc_failures;
        return NULL;
    }

    split_block(block, size);
    block->free_flag = 0u;
    account_increase(block->size);
    return (uint8_t *)block + sizeof(BlockHeader);
}

void free(void *ptr) {
    BlockHeader *block = block_from_ptr(ptr);
    if (block == NULL) {
        return;
    }
    account_decrease(block->size);
    block->free_flag = 1u;
    coalesce(block);
}

void *calloc(size_t nmemb, size_t size) {
    if (nmemb == 0u || size == 0u) {
        return NULL;
    }
    const size_t total = nmemb * size;
    void *ptr = malloc(total);
    if (ptr != NULL) {
        do_memset(ptr, 0, total);
    }
    return ptr;
}

void *memcpy(void *dest, const void *src, size_t n) {
    return do_memcpy(dest, src, n);
}

void *memmove(void *dest, const void *src, size_t n) {
    return do_memmove(dest, src, n);
}

void *memset(void *dest, int value, size_t n) {
    return do_memset(dest, value, n);
}

int memcmp(const void *lhs, const void *rhs, size_t n) {
    const uint8_t *a = (const uint8_t *)lhs;
    const uint8_t *b = (const uint8_t *)rhs;
    for (size_t i = 0; i < n; ++i) {
        if (a[i] != b[i]) {
            return (int)a[i] - (int)b[i];
        }
    }
    return 0;
}

void *realloc(void *ptr, size_t size) {
    if (ptr == NULL) {
        return malloc(size);
    }
    if (size == 0u) {
        free(ptr);
        return NULL;
    }

    size = align_up(size);
    BlockHeader *block = block_from_ptr(ptr);
    if (block == NULL) {
        return NULL;
    }

    const size_t old_size = block->size;

    if (block->size >= size) {
        split_block(block, size);
        if (block->size < old_size) {
            account_decrease(old_size - block->size);
        }
        return ptr;
    }

    if (block->next && block->next->free_flag &&
        (block->size + sizeof(BlockHeader) + block->next->size) >= size) {
        BlockHeader *next = block->next;
        block->size += sizeof(BlockHeader) + next->size;
        block->next = next->next;
        if (block->next) {
            block->next->prev = block;
        }
        split_block(block, size);
        if (block->size > old_size) {
            account_increase(block->size - old_size);
        }
        return ptr;
    }

    void *new_ptr = malloc(size);
    if (new_ptr == NULL) {
        return NULL;
    }
    const size_t copy_size = old_size < size ? old_size : size;
    do_memcpy(new_ptr, ptr, copy_size);
    free(ptr);
    return new_ptr;
}

static inline int float_is_nan(float x) {
    const uint32_t bits = ((const union { float f; uint32_t u; }){ .f = x }).u;
    return (bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0u;
}

float fmaxf(float a, float b) {
    if (float_is_nan(a)) {
        return b;
    }
    if (float_is_nan(b)) {
        return a;
    }
    return (a > b) ? a : b;
}

float fminf(float a, float b) {
    if (float_is_nan(a)) {
        return b;
    }
    if (float_is_nan(b)) {
        return a;
    }
    return (a < b) ? a : b;
}

static int fast_floor_to_int(float x) {
    const int i = (int)x;
    return ((float)i > x) ? (i - 1) : i;
}

float roundf(float x) {
    if (x >= 0.0f) {
        return (float)fast_floor_to_int(x + 0.5f);
    }
    return (float)fast_floor_to_int(x - 0.5f);
}

static float exp_approx(float r) {
    const float c1 = 1.0f;
    const float c2 = 1.0f;
    const float c3 = 0.5f;
    const float c4 = 0.16666667f;
    const float c5 = 0.04166667f;
    const float c6 = 0.00833333f;
    return (((((c6 * r + c5) * r + c4) * r + c3) * r + c2) * r + c1);
}

float expf(float x) {
    const float LN2 = 0.6931471805599453f;
    const float INV_LN2 = 1.4426950408889634f;

    if (x > 88.0f) {
        return FLT_MAX;
    }
    if (x < -103.0f) {
        return 0.0f;
    }

    const int n = fast_floor_to_int(x * INV_LN2);
    const float r = x - (float)n * LN2;
    float result = exp_approx(r);

    if (n > 0) {
        for (int i = 0; i < n; ++i) {
            result *= 2.0f;
        }
    } else if (n < 0) {
        for (int i = 0; i > n; --i) {
            result *= 0.5f;
        }
    }

    return result;
}

float fabsf(float x) {
    return (x < 0.0f) ? -x : x;
}

void __aeabi_memcpy(void *dest, const void *src, size_t n) {
    do_memcpy(dest, src, n);
}

void __aeabi_memcpy4(void *dest, const void *src, size_t n) {
    do_memcpy(dest, src, n);
}

void __aeabi_memcpy8(void *dest, const void *src, size_t n) {
    do_memcpy(dest, src, n);
}

void __aeabi_memset(void *dest, size_t n, int c) {
    do_memset(dest, c, n);
}

void __aeabi_memset4(void *dest, size_t n, int c) {
    do_memset(dest, c, n);
}

void __aeabi_memset8(void *dest, size_t n, int c) {
    do_memset(dest, c, n);
}

void __aeabi_memclr(void *dest, size_t n) {
    do_memset(dest, 0, n);
}

void __aeabi_memclr4(void *dest, size_t n) {
    do_memset(dest, 0, n);
}

void __aeabi_memclr8(void *dest, size_t n) {
    do_memset(dest, 0, n);
}

uintptr_t __stack_chk_guard = 0x2a2a2a2au;

static void handle_ubsan_trap(void) {
    semihost_write0("[ubsan] trap\n");
    semihost_exit(-2);
    for (;;) {
    }
}

void __stack_chk_fail(void) {
    semihost_write0("[stack] check failed\n");
    semihost_exit(-1);
    for (;;) {
    }
}

void __ubsan_handle_add_overflow(void) {
    handle_ubsan_trap();
}

void __ubsan_handle_sub_overflow(void) {
    handle_ubsan_trap();
}

void __ubsan_handle_divrem_overflow(void) {
    handle_ubsan_trap();
}

void __ubsan_handle_pointer_overflow(void) {
    handle_ubsan_trap();
}

void __ubsan_handle_load_invalid_value(void) {
    handle_ubsan_trap();
}

void __ubsan_handle_type_mismatch_v1(void) {
    handle_ubsan_trap();
}

void __ubsan_handle_mul_overflow(void) {
    handle_ubsan_trap();
}

void __ubsan_handle_negate_overflow(void) {
    handle_ubsan_trap();
}

static void emit_final_heap_report_impl(void) {
    semihost_write0("[heap] peak usage: ");
    write_decimal(g_peak_usage);
    semihost_write0(" bytes\n");
    semihost_write0("[heap] current usage: ");
    write_decimal(g_current_usage);
    semihost_write0(" bytes\n");
    semihost_write0("[heap] alloc failures: ");
    write_decimal(g_alloc_failures);
    semihost_write0("\n");
}

void support_emit_heap_report(void) {
    emit_final_heap_report_impl();
}

void _exit(int status) {
    emit_final_heap_report_impl();
    semihost_exit(status);
    for (;;) {
    }
}

void abort(void) {
    semihost_exit(1);
    for (;;) {
    }
}
