#include <stddef.h>
#include <stdint.h>

#define HEAP_SIZE 4096u

static uint8_t g_heap[HEAP_SIZE];
static size_t g_heap_offset = 0u;

void *malloc(size_t size)
{
    if (size == 0u) {
        return NULL;
    }
    size_t aligned = (size + 7u) & ~((size_t)7u);
    if (aligned > HEAP_SIZE - g_heap_offset) {
        return NULL;
    }
    void *ptr = &g_heap[g_heap_offset];
    g_heap_offset += aligned;
    return ptr;
}

void free(void *ptr)
{
    (void)ptr;
}

void *memcpy(void *dest, const void *src, size_t n)
{
    uint8_t *d = (uint8_t *)dest;
    const uint8_t *s = (const uint8_t *)src;
    for (size_t i = 0; i < n; ++i) {
        d[i] = s[i];
    }
    return dest;
}

void *memmove(void *dest, const void *src, size_t n)
{
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

void *memset(void *dest, int value, size_t n)
{
    uint8_t *d = (uint8_t *)dest;
    uint8_t v = (uint8_t)value;
    for (size_t i = 0; i < n; ++i) {
        d[i] = v;
    }
    return dest;
}

int memcmp(const void *lhs, const void *rhs, size_t n)
{
    const uint8_t *a = (const uint8_t *)lhs;
    const uint8_t *b = (const uint8_t *)rhs;
    for (size_t i = 0; i < n; ++i) {
        if (a[i] != b[i]) {
            return (int)a[i] - (int)b[i];
        }
    }
    return 0;
}

void __aeabi_memcpy(void *dest, const void *src, size_t n)
{
    memcpy(dest, src, n);
}

void __aeabi_memcpy4(void *dest, const void *src, size_t n)
{
    memcpy(dest, src, n);
}

void __aeabi_memcpy8(void *dest, const void *src, size_t n)
{
    memcpy(dest, src, n);
}

void __aeabi_memset(void *dest, size_t n, int c)
{
    memset(dest, c, n);
}

void __aeabi_memset4(void *dest, size_t n, int c)
{
    memset(dest, c, n);
}

void __aeabi_memset8(void *dest, size_t n, int c)
{
    memset(dest, c, n);
}

void __aeabi_memclr(void *dest, size_t n)
{
    memset(dest, 0, n);
}

void __aeabi_memclr4(void *dest, size_t n)
{
    memset(dest, 0, n);
}

void __aeabi_memclr8(void *dest, size_t n)
{
    memset(dest, 0, n);
}
