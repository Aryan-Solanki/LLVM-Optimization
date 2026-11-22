/* Differential-test driver for multi_function.c (linked with each optimized variant). */
#include <stdio.h>

int weighted_sum(const int *x, int n);
int max_of(const int *x, int n);
extern int call_count;

int main(void)
{
    int a[] = {5, -200, 7, 300, 0, 1, 2, 3, 4, -9, 11, 100, 101, -100, -101, 42};
    for (int n = 1; n <= 16; n++)
        printf("%d %d %d\n", weighted_sum(a, n), max_of(a, n), call_count);
    return 0;
}
