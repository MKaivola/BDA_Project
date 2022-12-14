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
  
  vector<lower=1>[2] nu; // dof
  
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
    alpha[i] ~ student_t(3, alpha_mu_prior, alpha_sd_prior);
    sigma[i] ~ student_t(3, 0, sigma_sd_prior);
    nu[i] ~ gamma(2,0.1);
    for (j in 1:P) {
      beta[j,i] ~ student_t(3, beta_mu_prior[j], beta_sigma_prior[j,j]);
    }
  }
  // Likelihood
  for (i in 1:2) {
    y_std[,i] ~ student_t(nu[i], mu_std[, i], sigma[i]);
  }
}

generated quantities {
  matrix[N, 2] log_lik;
  matrix[N, 2] y_rep;
  
  for (i in 1:2) {
    for (j in 1:N) {
      log_lik[j,i] = student_t_lpdf(y_std[j,i] | nu[i],  mu_std[j,i], sigma[i]);
      y_rep[j,i] = student_t_rng(nu[i], mu_std[j,i] * y_sd[i] + y_mean[i], sigma[i] * y_sd[i]);
    }
  }
}
