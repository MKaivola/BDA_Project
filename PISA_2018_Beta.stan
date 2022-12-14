data {
  int<lower=0> N; // Number of observations
  int<lower=0> P; // Number of expl. variables
  matrix<lower = 0, upper = 1>[N, 2] y; // Response variable (PISA for boys and girls) 
  real X_std[N,P,2]; // Data matrix for both groups (standardized)
 
  real loc_params_prior; // Location of the prior dist. of covariate params. and intercept
  real<lower=0> nu_params_prior; // dof of the prior dist. of covariate params. and intercept
  real<lower=0> scale_params_prior; // Scale of the prior dist. of covariate params. and intercept
  
  real<lower=0> nu_precision_prior; // dof of the prior dist. of precision
  real<lower=0> scale_precision_prior; // Scale of the prior dist. of precision
}

parameters {
  matrix[P,2] beta; // Parameters for the expl. variables per group
  vector[2] alpha; // Intercept per group
  vector<lower=0>[2] phi; // Precision per group
}

transformed parameters {
  matrix[N, 2] mu; // Expected values of the Beta variables
  for (i in 1:2) {
    mu[,i] = inv_logit(alpha[i] + to_matrix(X_std[,,i]) * beta[,i]);
  }
}

model {
  // Priors
  for (i in 1:2) {
    alpha[i] ~ student_t(nu_params_prior, loc_params_prior, scale_params_prior);
    phi[i] ~ student_t(nu_precision_prior, 0, scale_precision_prior);
    for (j in 1:P) {
      beta[j,i] ~ student_t(nu_params_prior, loc_params_prior, scale_params_prior);
    }
  }
  // Likelihood
  for (i in 1:2) {
    y[,i] ~ beta(mu[,i] * phi[i], (1 - mu[,i]) * phi[i]);
  }
}

generated quantities {
  matrix[N, 2] log_lik;
  matrix[N, 2] y_rep;
  
  for (i in 1:2) {
    for (j in 1:N) {
      log_lik[j,i] = beta_lpdf(y[j,i] | mu[j,i] * phi[i], (1 - mu[j,i]) * phi[i]);
      y_rep[j,i] = beta_rng(mu[j,i] * phi[i], (1 - mu[j,i]) * phi[i]);
    }
  }
}
