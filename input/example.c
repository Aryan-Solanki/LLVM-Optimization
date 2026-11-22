/* example.c - a program the optimizer cannot simply pre-compute.
 *
 * Everything depends on argc (a run-time value), so the -O levels and the LLM's pass pipeline have to make real
 * choices (inlining, loop unrolling, switch lowering, ...) and their results differ.
 * Deterministic output: run it with no arguments.
 */
#include <stdio.h>

#define N 24

static unsigned crc_step(unsigned crc, unsigned char byte) {
    crc ^= byte;
    for (int k = 0; k < 8; k++)
        crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
    return crc;
}

static unsigned checksum(const int *v, int n) {
    unsigned crc = 0xFFFFFFFFu;
    for (int i = 0; i < n; i++)
        for (int b = 0; b < 4; b++)
            crc = crc_step(crc, (unsigned char)(v[i] >> (8 * b)));
    return ~crc;
}

static void insertion_sort(int *v, int n) {
    for (int i = 1; i < n; i++) {
        int key = v[i], j = i - 1;
        while (j >= 0 && v[j] > key) {
            v[j + 1] = v[j];
            j--;
        }
        v[j + 1] = key;
    }
}

static const char *classify(int x) {
    switch (x % 7) {
    case 0:  return "zero";
    case 1:  return "one";
    case 2:  return "two";
    case 3:  return "three";
    case 4:  return "four";
    case 5:  return "five";
    default: return "six";
    }
}

static int gcd(int a, int b) { return b == 0 ? a : gcd(b, a % b); }

static long long dot(const int *a, const int *b, int n) {
    long long s = 0;
    for (int i = 0; i < n; i++)
        s += (long long)a[i] * b[i];
    return s;
}

static int count_primes(int limit) {
    int count = 0;
    for (int x = 2; x <= limit; x++) {
        int prime = 1;
        for (int d = 2; d * d <= x; d++)
            if (x % d == 0) { prime = 0; break; }
        count += prime;
    }
    return count;
}

int main(int argc, char **argv) {
    (void)argv;
    int seed = argc * 7919;                 /* run-time value: argc is 1 with no arguments */
    int a[N], b[N];
    for (int i = 0; i < N; i++) {
        seed = seed * 1103515245 + 12345;
        a[i] = (seed >> 8) % 1000;
        b[i] = (seed >> 4) % 100;
    }

    printf("dot      = %lld\n", dot(a, b, N));
    insertion_sort(a, N);
    printf("min/max  = %d %d\n", a[0], a[N - 1]);
    printf("checksum = %08x\n", checksum(a, N));
    printf("gcd      = %d\n", gcd(a[N - 1], a[N / 2] + argc));

    int hist[7] = {0};
    for (int i = 0; i < N; i++) {
        const char *name = classify(a[i]);
        hist[(unsigned char)name[1] % 7]++;
    }
    for (int i = 0; i < 7; i++)
        printf("hist[%d]  = %d\n", i, hist[i]);

    printf("primes   = %d\n", count_primes(200 + argc));
    return 0;
}
