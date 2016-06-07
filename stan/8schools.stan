# 4つのセクションに分かれている
# データセクション
## 外部から受け取るデータに対応
data {
  int<lower=0> J; // number of schools 
  real y[J]; // estimated treatment effects
  real<lower=0> sigma[J]; // s.e. of effect estimates 
}
# パラメタセクション
## 今から組みたいモデルで使うパラメタを宣言
parameters {
  real mu; 
  real<lower=0> tau;
  real eta[J];
}
# パラメタ変換セクション
## データとパラメタの橋渡し
transformed parameters {
  real theta[J];
  for (j in 1:J)
    theta[j] <- mu + tau * eta[j];
}
# モデルセクション
## パラメタセクションと変換セクションで宣言したものでモデルを書く
model {
  eta ~ normal(0, 1);
  y ~ normal(theta, sigma);
}