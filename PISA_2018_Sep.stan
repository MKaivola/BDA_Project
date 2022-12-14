data {
  int<lower=0> N; // Number of observations
  int<lower=0> P; // Number of expl. variables
  matrix[N,2] y_std; // Response variable (PISA for boys and girls) (standardized)
  real X_std[N,P,2]; // Data matrix for both groups (standardized)
  
  real y_mean[2]; // Mean of the response variables
  real y_sd[2]; // Standard deviation of response variables
  
  real alpha_mu_prior; // Mean of the prior dist. of alpha
  real<lower=0> alpha_sd_prior; // Scale of the prior dist. of alpha
  
  vector[P] beta_mu_prior; // Mean vector of the prior dist. of beta
  matrix[P,P] beta_sigma_prior; // Cov. matrix of the prior dist. of beta
  
  real<lower=0> sigma_sd_prior; // SCale of the prior dist. of sigma
}

parameters {
  matrix[P,2] beta; // Parameters for the expl. variables per group
  vector[2] alpha; // Intercept per group
  vector<lower=0>[2] sigma; // Measurement SD per group
}

transformed parameters {
  matrix[N, 2] mu_std; // Linear model for the mean
  for (i in 1:2) {
    mu_std[,i] = alpha[i] + to_matrix(X_std[,,i]) * beta[,i];
  }
}

model {
  // Priors
  for (i in 1:2) {
    beta[,i] ~ multi_normal(beta_mu_prior, beta_sigma_prior);
    alpha[i] ~ normal(alpha_mu_prior, alpha_sd_prior);
    sigma[i] ~ normal(0, sigma_sd_prior);
  }
  // Likelihood
  for (i in 1:2) {
    y_std[,i] ~ normal(mu_std[, i], sigma[i]);
  }
}

generated quantities {
  matrix[N, 2] log_lik;
  matrix[N, 2] y_rep;
  
  for (i in 1:2) {
    for (j in 1:N) {
      log_lik[j,i] = normal_lpdf(y_std[j,i] | mu_std[j,i], sigma[i]);
      y_rep[j,i] = normal_rng(mu_std[j,i] * y_sd[i] + y_mean[i], sigma[i] * y_sd[i]);
    }
  }
}
