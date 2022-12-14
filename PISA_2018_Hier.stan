data {
  int<lower=0> N; // Number of observations
  int<lower=0> P; // Number of expl. variables
  matrix[N,2] y_std; // Response variable (PISA for boys and girls) (standardized)
  real X_std[N,P,2]; // Data matrix for both groups (standardized)
  
  real y_mean[2]; // Mean of the response variables
  real y_sd[2]; // Standard deviation of response variables
  
  real alpha_mu_prior; // Prior mean for the intercept pop. mean
  real<lower=0> alpha_sd_prior; // Prior scale for the intercept pop. mean
  real<lower=0> alpha_tau_scale; // Prior scale for the intercept pop. scale
  
  vector[P] beta_mu_prior; // Prior means for the covariate param. pop. means
  vector<lower=0>[P] beta_sigma_prior; // Prior scales for the covariate param. pop. means
  real<lower=0> beta_tau_scale; // Prior scale for the  covariate param. pop. scales
  
  real<lower=0> sigma_sd_prior;
}

parameters {
  real mu_j[P]; // Covariate param. population means
  real<lower=0> tau_j[P]; // Covariate param. population scales
  real mu_alpha; // Intercept population mean
  real<lower=0> tau_alpha; // Intercept population scale
  real<lower=0> sigma; // Shared measurement scale
  
  real z_hat[P,2]; // Latent variables for the covariate parameters
  real z[2]; // Latent variables for the intercepts
}

transformed parameters {
  matrix[P,2] beta; // Parameters for the expl. variables per group
  vector[2] alpha; // Intercept per group
  matrix[N, 2] mu_std; // Linear model for the mean
  
  // Generation of the group-wise parameters from the latent variables
  for (i in 1:2) {
    alpha[i] = tau_alpha * z[i] + mu_alpha;
    for (j in 1:P) {
      beta[j,i] = tau_j[j] * z_hat[j,i] + mu_j[j];
    }
  }
  
  for (i in 1:2) {
    mu_std[,i] = alpha[i] + to_matrix(X_std[,,i]) * beta[,i];
  }
}

model {
  // Hyperpriors
  mu_alpha ~ normal(alpha_mu_prior, alpha_sd_prior);
  tau_alpha ~ normal(0, alpha_tau_scale);

  for (j in 1:P) {
    mu_j[j] ~ normal(beta_mu_prior[j], beta_sigma_prior[j]);
    tau_j[j] ~ normal(0, beta_tau_scale);
  }
  
  // Prior
  sigma ~ normal(0, sigma_sd_prior);
  
  // Latent variables
  for (i in 1:2) {
    z[i] ~ normal(0, 1);
    for (j in 1:P) {
      z_hat[j,i] ~ normal(0, 1);
    }
  }
  
  // Likelihood
  for (i in 1:2) {
    y_std[,i] ~ normal(mu_std[, i], sigma);
  }
}

generated quantities {
  matrix[N, 2] log_lik;
  matrix[N, 2] y_rep;
  
  for (i in 1:2) {
    for (j in 1:N) {
      log_lik[j,i] = normal_lpdf(y_std[j,i] | mu_std[j,i], sigma);
      y_rep[j,i] = normal_rng(mu_std[j,i] * y_sd[i] + y_mean[i], sigma * y_sd[i]);
    }
  }
}
