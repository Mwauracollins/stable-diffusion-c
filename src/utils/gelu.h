#include <math.h>

float gelu(float x) {
    return 0.5 * x * (1 + tanh(sqrt(2 / M_PI) * (x + 0.044715 * x * x * x)));
}
