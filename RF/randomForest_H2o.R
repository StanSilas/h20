#h20 credit card fraud randomForest
library(readr)
creditcard <- read_csv("C:/Users/vivek/Downloads/creditcardfraud/creditcard.csv")
View(creditcard)

creditcard$Class = as.factor(creditcard$Class)

library(h2o)
h2o.init(nthreads = -1)


credit_card<-creditcard
h2o_credit = as.h2o (credit_card)


#
divide = h2o.splitFrame(h2o_credit, ratios = 0.7)

train = divide[[1]]
test = divide[[2]]

rf1 = h2o.randomForest(x=1:30,y=31, training_frame = train, validation_frame = test, ntrees = 500)
rf1
h2o.gainsLift(rf1,train)

rf2 = h2o.randomForest(x=1:30,y=31, training_frame = train, validation_frame = test, ntrees = 500,score_each_iteration = T)
rf2
h2o.gainsLift(rf1,train)
#this has high AUc 0.98 , but can take forever to run


# now performing grid search, for best set of parameters , choose params wisely

# g <- h2o.grid("randomForest",
#               hyper_params = list(
#                 ntrees = c(1, 5),
#                 max_depth = c(2, 5),
#                 min_rows = c(10, 30)
#               ),
#               x = 1:30, y = 31, training_frame = train, nfolds = 10
# )

#g
h2o.shutdown()
g <- h2o.grid("randomForest",
              hyper_params = list(
                ntrees = c(50, 100, 120),
                max_depth = c(40, 60),
                min_rows = c(1, 2)
              ),
              x = x, y = y, training_frame = train, nfolds = 10
)