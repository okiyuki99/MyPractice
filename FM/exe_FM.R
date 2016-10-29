# Factorization Machines
library(libFMexe)
Sys.setenv(PATH=paste(Sys.getenv("PATH"),"/usr/local/share/libfm-1.42.src/bin",sep=":"))

# データ読み込み : Movie Lens
data(movie_lens)
# > head(movie_lens)
# User                                                Movie Rating
# 1    1                                     Toy Story (1995)      5
# 2    1                                     GoldenEye (1995)      3
# 3    1                                    Four Rooms (1995)      4
# 4    1                                    Get Shorty (1995)      3
# 5    1                                       Copycat (1995)      3
# 6    1 Shanghai Triad (Yao a yao yao dao waipo qiao) (1995)      5

# 訓練データを2/3, テストデータを1/3
train_rows <- sample.int(nrow(movie_lens), nrow(movie_lens) * 2 / 3)
train <- movie_lens[train_rows, ]
test <- movie_lens[-train_rows, ]

# Factorization Machines
predFM <- libFM(train, 
                test, 
                Rating ~ User + Movie,
                task = "r", # task : 回帰"r" or 分類"c"
                dim = 10, 
                iter = 500)

# 予測誤差
mean((predFM - test$Rating)^2)


data(iris)

# 訓練データを2/3, テストデータを1/3
X <- dplyr::select(iris, -Species)
y <- iris[,"Species"]
y <- iris[, stringr::str_detect(colnames(iris), "Species")]
train_rows <- sample.int(nrow(iris), nrow(iris) * 2 / 3)
train <- iris[train_rows, ] 
test <- iris[-train_rows, ]

# Factorization Machines
# データの形式がだめなのか。。
predFM <- libFM(train = train, 
                test = test,
                formula = Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width,
                task = "c", # task : 回帰"r" or 分類"c"
                dim = 10, # 近似した行列のランクの大きさ（パラメータ）
                iter = 500)

# データ読み込み
hayes.roth <- data.table::fread("/mnt/data/uci/hayes-roth/hayes-roth.tsv", data.table = F)
names(hayes.roth) <- c("hobby", "age", "educational_level", "marital_status", "class")
head(hayes.roth)

y <- hayes.roth[,"class"]
train_rows <- sample.int(nrow(hayes.roth), nrow(hayes.roth) * 2 / 3)
train <- hayes.roth[train_rows, ] 
test <- hayes.roth[-train_rows, ]

# Factorization Machines
predFM <- libFM(train = train, 
                test = test,
                formula = class ~ hobby + age + educational_level + marital_status,
                task = "c", # task : 回帰"r" or 分類"c"
                dim = 3, # 近似した行列のランクの大きさ（パラメータ）
                iter = 500)
# 予測誤差
# predFM の値がおかしい
mean((predFM - test$class)^2)
