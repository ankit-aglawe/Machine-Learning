library(caret)
library(e1071)
dataset<- iris
head(iris)
dataset$Species

t1<- train(Species~.,data=dataset, method="knn")
predict(t1, dataset[1,1:4])



#-------------------------------------------------------------------

v<- createDataPartition(dataset$Species, p=0.8, list=FALSE)
trainData= dataset[v,]
testData=dataset[-v,]

t2<- train(Species~., data=trainData, method="knn")
pred<-predict(t2, testData)
confusionMatrix(testData[,5],pred)

#-----------------------------------------------------------------------


e<- read.table(file.choose())
head(e)

t3<- train(V9~., data=e[,-1], method="knn")
pred1<- predict(t3, e[1,2:8])


#-------------------------------------------------------------------------


newData<- read.table(file.choose())        

head(newData)

t4<- train(V8~., data=newData, method="knn")

pred2<- predict(t4, newData[1,1:7])
