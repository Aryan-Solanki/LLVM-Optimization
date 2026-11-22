/* Test input for the 2024 workflow: one translation unit with several functions and module-level state
   (a constant table, a mutable global, a static helper that the inliner can remove). */
static const int weights[8] = {3, 1, 4, 1, 5, 9, 2, 6};
int call_count = 0;

static int clamp(int v, int lo, int hi)
{
    if (v < lo)
        return lo;
    if (v > hi)
        return hi;
    return v;
}

int weighted_sum(const int *x, int n)
{
    int s = 0;
    for (int i = 0; i < n; i++)
        s += clamp(x[i], -100, 100) * weights[i & 7];
    call_count++;
    return s;
}

int max_of(const int *x, int n)
{
    int m = x[0];
    for (int i = 1; i < n; i++)
        if (x[i] > m)
            m = x[i];
    call_count++;
    return m;
}
