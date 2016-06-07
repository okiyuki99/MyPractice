# RStanを使ってみる
## 参考：R stan導入公開版：http://www.slideshare.net/KojiKosugi/r-stan

library(rstan)

# 公式サイトのeight school とやらのデータを用意
schools_dat <- list(J = 8, 
                    y = c(28,  8, -3,  7, -1,  1, 18, 12),
                    sigma = c(15, 10, 16, 11,  9, 11, 10, 18))

# stanをやってみる
## stan関数でstanを呼んでいる
## デフォルトでは、サンプリング回数の半分がWarmup
fit <- stan(file = '8schools.stan',  # stanファイル
            data = schools_dat,      # data
            iter = 1000,             # 1000サンプリング（反復回数）
            chains = 4)              # 4セットやる

# 結果を確認
print(fit)

# 可視化
traceplot(fit)
plot(fit)

# 世界一簡単な例
n <- 100
mu <- 50
sig <- 10
y <- rnorm(n, mu, sig)

# 平均と分散を推定するstanコード
stancode <- '
data{
  int<lower=0> T;
  real N[T];
}
parameters {
  real mu;
  real<lower=0> s2;
}
model{
  N ~ normal(mu, sqrt(s2));
  s2 ~ cauchy(0, 5);
}
'
datastan <- list(N=y, T=n)
fit <- stan(model_code = stancode,
            data = datastan,
            iter = 1000,
            chain = 4)

# 平均と分散の推定結果可視化
print(fit)
traceplot(fit)
mean(y)
var(y)
