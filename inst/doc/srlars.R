## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(collapse = TRUE, comment = "#>")

## ----setup--------------------------------------------------------------------
library(srlars)
library(mvnfast)
library(cellWise)

## ----truth--------------------------------------------------------------------
set.seed(100)

n <- 50              # training observations
m <- 2000            # test observations
p <- 500             # candidate predictors
p.active <- 75       # truly active predictors, in blocks below
group.size <- 15     # active predictors per correlated block
n_models <- 10       # ensemble size (K)

# Active predictors sit in correlated blocks; everything else is independent noise.
sigma.mat <- matrix(0, p, p)
sigma.mat[1:p.active, 1:p.active] <- 0.1 # weak correlation across blocks
for (g in 0:(p.active / group.size - 1)) {
  idx <- (g * group.size + 1):(g * group.size + group.size)
  sigma.mat[idx, idx] <- 0.7 # stronger correlation within a block
}
diag(sigma.mat) <- 1

# A sparse, moderate-signal true coefficient vector
true.beta <- c(runif(p.active, 0, 5) * (-1) ^ rbinom(p.active, 1, 0.7),
              rep(0, p - p.active))
sigma <- as.numeric(sqrt(t(true.beta) %*% sigma.mat %*% true.beta)) # signal-to-noise = 1

## ----generate-data------------------------------------------------------------
x_train <- mvnfast::rmvn(n, mu = rep(0, p), sigma = sigma.mat)
y_train <- as.numeric(x_train %*% true.beta + rnorm(n, 0, sigma))
colnames(x_train) <- paste0("V", 1:p)

x_test <- mvnfast::rmvn(m, mu = rep(0, p), sigma = sigma.mat)
y_test <- as.numeric(x_test %*% true.beta + rnorm(m, 0, sigma))
colnames(x_test) <- colnames(x_train)

## ----contaminate--------------------------------------------------------------
contam_correlation <- function(X, prop, sigma_mat, gamma = 3) {
  n <- nrow(X); p <- ncol(X)
  idx <- sample.int(n * p, size = round(n * p * prop))
  rows <- ((idx - 1) %% n) + 1
  cols <- ((idx - 1) %/% n) + 1

  for (i in 1:n) {
    J <- cols[rows == i]
    if (length(J) == 0) next
    if (length(J) == 1) { X[i, J] <- gamma * 3; next }
    SigmaJ <- sigma_mat[J, J, drop = FALSE]
    vmin <- eigen(SigmaJ, symmetric = TRUE)$vectors[, length(J)]
    denom <- mahalanobis(t(vmin), center = rep(0, length(J)), cov = SigmaJ)
    X[i, J] <- gamma * sqrt(length(J)) * (vmin / sqrt(denom))
  }
  X
}

x_train <- contam_correlation(x_train, prop = 0.15, sigma_mat = sigma.mat)

## ----metrics-helper-----------------------------------------------------------
get_metrics <- function(fit) {
  coefs <- as.numeric(coef(fit))[-1]
  sel <- which(coefs != 0)
  truth <- which(true.beta != 0)

  preds <- as.numeric(predict(fit, x_test))

  c(Precision = length(intersect(sel, truth)) / max(length(sel), 1),
   Recall = length(intersect(sel, truth)) / length(truth),
   MSPE = mean((y_test - preds)^2) / sigma^2,
   `Mean sub-model size` = mean(vapply(fit$active.sets, length, integer(1))))
}

## ----fit-default--------------------------------------------------------------
fit_default <- srlars(x_train, y_train,
                      n_models = n_models,
                      tolerance = 1e-4,
                      x_preprocess = "ddc",
                      y_preprocess = "wrap",
                      cor_estimator = "wrap",
                      cv_preprocess = "global",
                      cv_fit = "huber",
                      cv_loss = "huber",
                      cv_folds = 5,
                      compute_coef = TRUE)

## ----active-sizes-------------------------------------------------------------
knitr::kable(
  data.frame(`Sub-model` = seq_len(n_models),
            `Variables selected` = vapply(fit_default$active.sets, length, integer(1))),
  align = "c"
)

## ----metrics-default----------------------------------------------------------
metrics_default <- get_metrics(fit_default)
knitr::kable(t(round(metrics_default, 3)), caption = "srlars() at the default max_share = 1")

## ----coef-predict-------------------------------------------------------------
knitr::kable(t(round(coef(fit_default)[1:6], 3)),
            col.names = c("Intercept", paste0("V", 1:5)))
knitr::kable(t(round(predict(fit_default, x_test[1:5, ]), 2)),
            col.names = paste("Test row", 1:5))

## ----fit-shared---------------------------------------------------------------
fit_shared <- srlars(x_train, y_train,
                     n_models = n_models,
                     max_share = n_models,
                     tolerance = 1e-4,
                     x_preprocess = "ddc",
                     y_preprocess = "wrap",
                     cor_estimator = "wrap",
                     cv_preprocess = "global",
                     cv_fit = "huber",
                     cv_loss = "huber",
                     cv_folds = 5,
                     compute_coef = TRUE)

metrics_shared <- get_metrics(fit_shared)
knitr::kable(
  rbind(`max_share = 1 (default)` = round(metrics_default, 3),
       `max_share = n_models`    = round(metrics_shared, 3))
)

## ----fit-nmin-----------------------------------------------------------------
fit_floor <- srlars(x_train, y_train,
                    n_models = n_models,
                    n_min = 10,
                    tolerance = 1e-4,
                    x_preprocess = "ddc",
                    y_preprocess = "wrap",
                    cor_estimator = "wrap",
                    cv_preprocess = "global",
                    cv_fit = "huber",
                    cv_loss = "huber",
                    cv_folds = 5,
                    compute_coef = TRUE)

metrics_floor <- get_metrics(fit_floor)
knitr::kable(
  rbind(`n_min = NULL (default)` = round(metrics_default, 3),
       `n_min = 10`               = round(metrics_floor, 3))
)

