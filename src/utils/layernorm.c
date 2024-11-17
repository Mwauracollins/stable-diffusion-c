void layernorm_forward(
    float* input, // (B, T, C)
    float* output, // Normalized output (B, T, C)
    float* gamma, // Scaling params (C)
    float* beta, // Shift params (C)
    float* rmean,
    float* rstd_dev,
    int B,
    int T,
    int C
) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float mean = 0.0f;
            float variance = 0.0f;

            // calculate the mean
            for (int c = 0; c < C; c++) {
                mean += input[b * T * C + t * C + c];
            }
            mean /= C;

            // calculate the variance
            for (int c = 0; c < C; c++) {
                float xshift = input[b * T * C + t * C + c] - mean;
                variance += xshift * xshift;
            }
            variance /= C;

            float std_dev = sqrf(variance + 1e-5f);

            //norm and apply gamma/beta
            for (int c = 0; c < C; c++) {
                output[b * T * C + t * C + c] = gamma[c] * ((input[b * T * C + t * C + c] - mean) / std_dev) + beta[c];
            }
            //save the mean and std_dev for backward pass
            rmean[b * T + t] = mean;
            rstd_dev[b * T + t] = std_dev;
        }
    }
}

void layernorm_backward(
    float* input, // (B, T, C)
    float* gamma, // (C)
    float* beta, // (C)
    float* dinput, // (B, T, C)
    float* doutput, // (B, T, C)
    float* dgamma, // (C)
    float* dbeta, // (C)
    float* rmean, // (B, T)
    float* rstd_dev,// (B, T)
    int B, int T, int C
) {
    //TODO: CHECK THIS MATH; YOU WERE A LIL TIPSY
    // initialize the gradients
    memset(dgamma, 0, C * sizeof(float));
    memset(dbeta, 0, C * sizeof(float));

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float mean = rmean[b * T + t];
            float std_dev = rstd_dev[b * T + t];

            for (int c = 0; c < C; c++) {
                float x_hat = (input[b * T * C + t * C + c] - mean) / std_dev;
                dgamma[c] += doutput[b * T * C + t * C + c] * x_hat;
                dbeta[c] += doutput[b * T * C + t * C + c];
            }

            float sum_doutput = 0.0f;
            float sum_doutput_xhat = 0.0f;

            for (int c = 0; c < C; c++) {
                float x_hat = (input[b * T * C + t * C + c] - mean) / std_dev;
                sum_doutput += doutput[b * T * C + t * C + c];
                sum_doutput_xhat += doutput[b * T * C + t * C + c] * x_hat;
            }

            // gradient wrt to inputs
            for (int c = 0; c < C; c++) {
                float x_hat = (input[b * T * C + t * C + c] - mean) / std_dev;

                dinput[b * T * C + t * C + c] = gamma[c] / std_dev * (
                    doutput[b * T * C + t * C + c] - (sum_doutput / C) - (x_hat * sum_doutput_xhat / C)
                );
            }
        }
    }
}