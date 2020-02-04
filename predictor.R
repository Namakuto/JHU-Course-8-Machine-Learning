library(caret); library(FSelector); library(randomForest)
setwd("~/=== LU 3211 Biostats/COURSERA/Course 8 - Machine Learning")
train<-read.csv("pmltraining.csv", na.strings=c("#DIV/0!", "NA", "")) 
test<-read.csv("pmltesting.csv", na.strings=c("#DIV/0!", "NA", ""))

part<-createDataPartition(y=train$classe, p=0.60, list=FALSE)
train1<-train[part,]
train2<-train[-part,]

#///////// Trying caret package

numeric.col<-train1[,8:160] # Exclude non-numerics
numeric.col<-numeric.col[ , apply(numeric.col, 2, function(x) !any(is.na(x)))]
#REMOVE NA's

best.correlation<-cfs(classe~., data=numeric.col) # Let's try correlations first

par(mar=c(4,4,2,1), mfcol=c(1,2))
plot(roll_belt~classe, data=numeric.col, main="roll_belt vs classe", cex.main=1)
plot(pitch_belt~classe, data=numeric.col, main="pitch_belt vs classe", cex.main=1)

set.seed(1)
forest<-randomForest(classe~., data=numeric.col, importance=TRUE, ntree=50)
varImpPlot(forest, cex=0.7) # uh oh, looks like some things may be redundant.


#library(rpart)
#model.forest<-rpart(classe~., data=numeric.col, method="class")
#par(mar=c(1,1,1,1))
#plot(model.forest)
#text(model.forest,use.n = TRUE,all=FALSE, cex=0.5)


gain<-gain.ratio(classe~., data=numeric.col)
gain$names<-row.names(gain); row.names(gain)<-NULL
list.gain<-gain[order(gain$attr_importance, decreasing=TRUE),]


cor.mat<-cor(numeric.col[,1:52], use="pairwise.complete.obs")
cor.matnew<-findCorrelation(cor.mat, cutoff = 0.7, names=FALSE) # Returns columns that
# are highly co-correlated (multicollinearity!!)-- to remove
numeric.col2<-numeric.col[,-cor.matnew] # 40 variables --> 34 variables

diag(cor.mat)<-0; max(abs(cor.mat)) # Max is 0.992...


set.seed(2)
forest2<-randomForest(classe~., data=numeric.col2, importance=TRUE, ntree=50)
varImpPlot(forest2, cex=0.7) # uh oh, looks like some things may be redundant.


best.correlation2<-cfs(classe~., data=numeric.col2) # new correlations
best.indices2<-grep(paste(best.correlation2, collapse = "|"), names(numeric.col2), value=FALSE)
numeric.colbest<-numeric.col2[,best.indices2]

cor.mat2<-cor(numeric.col2[,1:33])
diag(cor.mat2)<-0; max(abs(cor.mat2)) # Max is 0.775... high! Let's reduce our cutoff. 
# Max is now 0.34... really good!

cor.mat2<-cor(numeric.colbest)
diag(cor.mat2)<-0; max(abs(cor.mat2)) # Max is 0.775... high! Let's reduce our cutoff. 


library(doParallel)
cl<-makeCluster(2)
registerDoParallel(cl)
mymodel<-train(classe~gyros_belt_z+magnet_belt_y+gyros_arm_y+magnet_arm_x+
               roll_dumbbell+gyros_dumbbell_y+magnet_dumbbell_z+roll_forearm+pitch_forearm,
             
             data=numeric.col2, method="rf", 
             trControl=trainControl(method="cv", number=2),
             prox=TRUE, verbose=TRUE, ntree=100)
stopCluster(cl)
saveRDS(mymodel, "mymodel.rds")
#mymodel<-readRDS("mymodel.Rds")

pred<-predict(mymodel, newdata=train2)
confusionMatrix(pred, train2$classe)
# Accuracy of 95.81%. Out-of-sample error rate of 100%-95.91%

newpred<-predict(mymodel,newdata=test)
test$classe<-newpred # 100% correct!


library(rpart); library(rpart.plot)
tree.plot<-rpart(classe~., method="class", data=numeric.col2)
prp(tree.plot, type=0, ycompress=FALSE, cex=0.5, compress=TRUE, branch=1, Margin = -0.05)

text(tree.plot, labels=names(numeric.col2), cex=1, all=FALSE)


#lm.model<-lm(classe~gyros_belt_z+magnet_belt_y+gyros_arm_y+magnet_arm_x+
#                                roll_dumbbell+gyros_dumbbell_y+magnet_dumbbell_z+roll_forearm+pitch_forearm,
#data=numeric.col2)
#sort(lm.model$coefficients, decreasing = TRUE)

