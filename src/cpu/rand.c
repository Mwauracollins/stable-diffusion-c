#include <stdlib.h>
#include <math.h>

float rand_normal() {
    float u1 = ((float) rand() / RAND_MAX);
    float u2 = ((float) rand() / RAND_MAX);

    float z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    return z0;
}