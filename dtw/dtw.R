library(dplyr)
library(tidyr)
library(ggplot2)
library(MSBVAR)

ts_a <- AirPassengers[31:45]
ts_b <- AirPassengers[41:55]
plot(AirPassengers)

plot(ts_a)
plot(ts_b)

## dtw 距離関数の実装例
dtw_distance <- function(ts_a, ts_b, d = function(x, y) abs(x-y),
                         window = max(length(ts_a), length(ts_b))) {
  ts_a_len <- length(ts_a)
  ts_b_len <- length(ts_b)
  
  # コスト行列 (ts_a と ts_b のある2点間の距離を保存)
  cost <- matrix(NA, nrow = ts_a_len, ncol = ts_b_len)
  
  # 距離行列 (ts_a と ts_b の最短距離を保存)
  dist <- matrix(NA, nrow = ts_a_len, ncol = ts_b_len)
  
  # cost と distance 行列を初期化
  cost[1, 1] <- d(ts_a[1], ts_b[1])
  dist[1, 1] <- cost[1, 1]
  
  # 1列目を繰り返し計算で求める
  for (i in 2:ts_a_len) {
    cost[i, 1] <- d(ts_a[i], ts_b[1])
    dist[i, 1] <- dist[i-1, 1] + cost[i, 1] # dist[2,1] <- dist[1,1] + cost[2,1]
  }
  
  # 1行目を繰り返し計算で求める
  for (j in 2:ts_b_len) {
    cost[1, j] <- d(ts_a[1], ts_b[j])
    dist[1, j] <- dist[1, j-1] + cost[1, j]
  }
  
  for (i in 2:ts_a_len) {
    # 最短距離を探索する範囲 (ウィンドウサイズ = ラグ)
    window.start <- max(2, i - window)
    window.end <- min(ts_b_len, i + window)
    
    for (j in window.start:window.end) {
      # dtw::symmetric1 と同じパターン
      choices <- c(dist[i-1, j], dist[i, j-1], dist[i-1, j-1])
      cost[i, j] <- d(ts_a[i], ts_b[j])
      dist[i, j] <- min(choices) + cost[i, j]
    }
  } 
  return(dist[nrow(dist), ncol(dist)])
}

set.seed(1)

# 各グループの系列数
N = 20
# 系列の長さ
SPAN = 24
# トレンドが上昇/ 下降する時の平均値
TREND = 0.5

generate_ts <- function(m, label) {
  library(dplyr)
  # ランダムな AR 成分を追加
  .add.ar <- function(x) {
    x + arima.sim(n = SPAN, list(ar = runif(2, -0.5, 0.5)))
  }
  # 平均が偏った 乱数を cumsum してトレンドとする
  d <- matrix(rnorm(SPAN * N, mean = m, sd = 1), ncol = N) %>%
    data.frame() %>%
    cumsum()
  d <- apply(d, 2, .add.ar) %>%
    data.frame()
  colnames(d) <- paste0(label, seq(1, N))
  d
}

group1 = generate_ts(TREND, label = 'U')
group2 = generate_ts(0, label = 'N')
group3 = generate_ts(-TREND, label = 'D')

data <- cbind(group1, group2, group3)
data <- as.data.frame(data)

# DTW 距離で距離行列を作成
library(TSclust)
d <- diss(data, "DTWARP")
# hclust は既定で実行 = 最遠隣法
h <- hclust(d)
par(cex=0.6)
plot(h, hang = -1)

# クラスタ数をもとめる
clusters <- cutree(h, 5)

# クラスタ数に応じて可視化
data %>%
  mutate(time = seq(1,nrow(data))) %>%
  gather(ip, volume, -time) %>%
  mutate(group = as.character(clusters[ip])) %>%
  filter(group == "1") %>%
  ggplot(aes(x=time, y=volume, group = ip, color=group)) +
    geom_line()

# マルコフ転換モデル
x1<-ts(sin(seq(from=0,to=0.5*pi,length.out=390)))
x2<-ts(sin(seq(from=0,to=pi,length.out=390)))

#model1 <- lm(stock1~x1+x2)
#msmModel1<-msmFit(model1,k=2,sw=rep(T,4))

data(IsraelPalestineConflict) 
plot(IsraelPalestineConflict)
set.seed(123)
xm <- msbvar(IsraelPalestineConflict, p=3, h=2,
             lambda0=0.8, lambda1=0.15,
             lambda3=1, lambda4=1, lambda5=0, mu5=0,
             mu6=0, qm=12,
             alpha.prior=matrix(c(10,5,5,9), 2, 2))