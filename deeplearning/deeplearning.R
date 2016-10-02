# Deep Learning の練習
# 参考ページ：http://tjo.hatenablog.com/entry/2016/07/21/190000
# 上記を一からかいて学ぶ

## ライブラリ読み込み
library(glmnet)  # L1正則化回帰（リッジ回帰）
library(Metrics) # RMSEを求める
library(randomForest)
#install.packages("drat", repos="https://cran.rstudio.com")
#drat:::addRepo("dmlc")
#install.packages("mxnet")
library(mxnet)   # DeepLearning MXNET

## データ読み込み
df <- read.csv('/data/uci/OnlineNewsPopularity/OnlineNewsPopularity.csv')

## 必要なデータを準備
df <- df[,-c(1,2)]
idx <- sample(nrow(df),5000, replace=F)
df.train <- df[-idx,]
df.test <- df[idx,]

## L1正則化回帰
df.train.glmnet <- cv.glmnet(as.matrix(df.train[,-59]),
                             log(df.train[,59]),
                             family='gaussian',
                             alpha=1)
plot(df.train.glmnet)
plot(log(df.test$shares), predict(df.train.glmnet, 
                                  newx=as.matrix(df.test[,-59]),
                                  s=df.train.glmnet$lambda.min))
## RMSE で評価
Metrics::rmse(log(df.test$shares), 
              predict(df.train.glmnet, 
                      newx=as.matrix(df.test[,-59]), 
                      s=df.train.glmnet$lambda.min))

## ランダムフォレスト
#df.train.rf <- randomForest(log(df.train[,59])~.,df.train)
#Metrics::rmse(log(d_test$shares), predict(d_train.rf,newdata=d_test[,-59]))


## Deep Learning
# フォーマッティング
train <- mxnet::data.matrix(df.train)
test <- data.matrix(df.test)
train.x <- train[,-59]
train.y <- train[,59]
test.x <- test[,-59]
test.y <- test[,59]

# 正規化
train_means <- apply(train.x, 2, mean)
train_stds <- apply(train.x, 2, sd)
test_means <- apply(test.x, 2, mean)
test_stds <- apply(test.x, 2, sd)
train.x <- t((t(train.x)-train_means)/train_stds)
test.x <- t((t(test.x)-test_means)/test_stds)

# 目的変数を対数変換
train.y <- log(train.y)
test.y <- log(test.y)

# モデルづくり
### 中間層3層
### ユニット数は特徴次元数の4倍-4倍-2倍ぐらいで適当に決め打ち
### 今回は線形回帰を出力するため、出力はmx.symbol.LinearRegressionOutput関数
### 直前の全結合層のユニット数は1 にする

data <- mx.symbol.Variable("data")
fc1 <- mx.symbol.FullyConnected(data, name="fc1", num_hidden=220)
act1 <- mx.symbol.Activation(fc1, name="tanh1", act_type="tanh")
drop1 <- mx.symbol.Dropout(act1, p=0.5)
fc2 <- mx.symbol.FullyConnected(drop1, name="fc2", num_hidden=220)
act2 <- mx.symbol.Activation(fc2, name="tanh2", act_type="tanh")
drop2 <- mx.symbol.Dropout(act2, p=0.5)
fc3 <- mx.symbol.FullyConnected(drop2, name="fc3", num_hidden=110)
act3 <- mx.symbol.Activation(fc3, name="tanh3", act_type="tanh")
drop3 <- mx.symbol.Dropout(act3, p=0.5)
fc4 <- mx.symbol.FullyConnected(drop3, name="fc4", num_hidden=1)
output <- mx.symbol.LinearRegressionOutput(fc4, name="linreg")
devices <- mx.cpu()
mx.set.seed(71)

model <- mx.model.FeedForward.create(output, 
                                     X=train.x, 
                                     y=train.y, 
                                     ctx=devices, 
                                     num.round=100, # 訓練する回数
                                     array.batch.size=100, 
                                     learning.rate=1e-5, 
                                     momentum=0.99, 
                                     eval.metric=mx.metric.rmse, 
                                     initializer=mx.init.uniform(0.5), 
                                     array.layout = "rowmajor", 
                                     epoch.end.callback=mx.callback.log.train.metric(100))
preds <- predict(model, test.x, array.layout='rowmajor')
Metrics::rmse(preds, test.y)

## チューニング1
### 前回色々試してみた印象では「中間層第1層のユニット数を極端に大きくし、
### 以後の中間層では急激に絞っていく」戦略が良さそうだという感じでした。
### そこで特徴次元数の6倍-1/2倍-1/6倍というようにしてみます。
### またMXnetのdocを見ているとDropoutをp=0.2ぐらいで突っ込むと良いらしい
### 教科書通りのp=0.5よりも低く設定して各層に追加してみます。

data <- mx.symbol.Variable("data")
fc1 <- mx.symbol.FullyConnected(data, name="fc1", num_hidden=360)
act1 <- mx.symbol.Activation(fc1, name="tanh1", act_type="tanh")
drop1 <- mx.symbol.Dropout(act1, p=0.2)
fc2 <- mx.symbol.FullyConnected(drop1, name="fc2", num_hidden=30)
act2 <- mx.symbol.Activation(fc2, name="tanh2", act_type="tanh")
drop2 <- mx.symbol.Dropout(act2, p=0.2)
fc3 <- mx.symbol.FullyConnected(drop2, name="fc3", num_hidden=10)
act3 <- mx.symbol.Activation(fc3, name="tanh3", act_type="tanh")
drop3 <- mx.symbol.Dropout(act3, p=0.2)
fc4 <- mx.symbol.FullyConnected(drop3, name="fc4", num_hidden=1)
output <- mx.symbol.LinearRegressionOutput(fc4, name="linreg")
devices <- mx.cpu()
mx.set.seed(71)
model <- mx.model.FeedForward.create(output, 
                                     X=train.x, 
                                     y=train.y, 
                                     ctx=devices, 
                                     num.round=250, 
                                     array.batch.size=200, 
                                     learning.rate=2e-4, 
                                     momentum=0.99,
                                     eval.metric=mx.metric.rmse,
                                     initializer=mx.init.uniform(0.5), 
                                     array.layout = "rowmajor", 
                                     epoch.end.callback=mx.callback.log.train.metric(20))
preds <- predict(model, test.x, array.layout='rowmajor')
Metrics::rmse(preds, test.y)
