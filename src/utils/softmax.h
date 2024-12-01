void softmax_forward(float* logits, int T) {
    for (int i = 0; i < T; i++) 
    {
        float max_logit = logits[i * T];
        for (int j = 1; j < T; j++) {
            if (logits[i * T + j] > max_logit) {
                max_logit = logits[i * T + j];
            }
        }

        // Subtract max_logit and exponentiate
        float sum_exp = 0.0f;
        for (int j = 0; j < T; j++) {
            logits[i * T + j] = expf(logits[i * T + j] - max_logit);
            sum_exp += logits[i * T + j];
        }

        // Normalize to get probabilities
        for (int j = 0; j < T; j++) {
            logits[i * T + j] /= sum_exp;
        }
    }
}

// Softmax backward function for each row
void softmax_backward(float* probs, float* d_probs, int T) {
    for (int i = 0; i < T; i++) {
        for (int j = 0; j < T; j++) {
            float delta = (i == j) ? 1.0f : 0.0f;
            d_probs[i * T + j] = probs[i] * (delta - probs[j]);
        }
    }
}
