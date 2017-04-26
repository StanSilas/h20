# Energy effiency h2o neural network for deep learning
library(h2o)
library(readr)
h2o.init(nthreads = -1)

# datasets="https://raw.githubusercontent.com/DarrenCook/h2o/bk/datasets/ENB2012_data.csv"
# data = h2o.importFile(dataset)
# data <- h2o.importFile("S:/My Data Projects in R/H2O/Energy Efficiency/ENB2012_data.csv")
data <- read_csv("S:/My Data Projects in R/H2O/Energy Efficiency/ENB2012_data.csv")

# X1	Relative Compactness 
# X2	Surface Area 
# X3	Wall Area 
# X4	Roof Area 
# X5	Overall Height 
# X6	Orientation 
# X7	Glazing Area 
# X8	Glazing Area Distribution 
# y1	Heating Load 
# y2	Cooling Load

data$X6 <- as.factor(data$X6)
data$X8 <- as.factor(data$X8)

#converting our data frame to a h2o data frame
data<-as.h2o(data)
#splitting into train and test
splits <- h2o.splitFrame(data, 0.8)
train <- splits[[1]]
test <- splits[[2]]

#Seperating predictors predicted
x <- c("X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8")
y <- "Y2" 

m_nnet <- h2o.deeplearning(x, y, train, nfolds = 10, model_id = "DL_defaults")

res_m_nnet<-h2o.performance(m_nnet,test)
res_m_nnet
# very bad results!!!!!!!!!!
# MSE:  8.63743
# RMSE:  2.93895
# MAE:  2.11464
# RMSLE:  0.1060086
# Mean Residual Deviance :  8.63743

m_nnet2 <- h2o.deeplearning(x, y, train, nfolds = 10,
stopping_metric = "MSE",
stopping_tolerance = 0.005,
stopping_rounds = 3,
epochs = 1000,
train_samples_per_iteration = 0,
score_interval = 3)


res_m_nnet2<-h2o.performance(m_nnet2,test)
res_m_nnet2


#H2ORegressionMetrics: deeplearning
# Fantastic results 
# MSE:  0.5385381
# RMSE:  0.7338516
# MAE:  0.4742203
# RMSLE:  0.02596564
# Mean Residual Deviance :  0.5385381


#future prospects
# 
# L1 and L2 regularization. L1
# regularization, in neural nets, causes the neurons to use fewer of their inputs (the most
#significant ones, hopefully!); this might make them more resistant to noise (not an issue in
#this data set, so the expectation is that L1 regularization will not help-that is why I only try
#one value). L2 regularization reminds me of Tall Poppy Syndrome, because the biggest
# weights get knocked down, and the smaller weights survive unscathed. It encourages the
# network to use all its inputs a bit, and not just use a few of them. L1 and L2 seem in conflict,
# yet one-third of the models in this grid will try both together. We could use two grids to avoid
# this, but how about we just try it and see what happens?



m_best <- h2o.deeplearning(
  x, y, train,
  nfolds = 10,
  model_id = "DL_best",
  activation = "Tanh",
  l2 = 0.00001, #1e-05
  hidden = c(162,162),
  stopping_metric = "MSE",
  stopping_tolerance = 0.0005,
  stopping_rounds = 5,
  epochs = 2000,
  train_samples_per_iteration = 0,
  score_interval = 3
)

m_best<-h2o.performance(m_best,test)
m_best


# H2ORegressionMetrics: deeplearning
# 
# MSE:  0.4310551
# RMSE:  0.6565479
# MAE:  0.4952717
# RMSLE:  0.02618314
# Mean Residual Deviance :  0.4310551
