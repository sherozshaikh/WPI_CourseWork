

library(ISLR)
library(MASS)
library(class)
library(boot)

set.seed(1)

weekly<-read.csv("Weekly.csv")

names(weekly)

dim(weekly)

summary(weekly)




View(weekly)




pairs(weekly[,1:8])




plot(weekly$Volume,weekly$Year,xlab="Volume",ylab="Years")



cor(weekly[,-9],)



table(weekly$Direction)



sum(is.na(weekly$Direction))




str(weekly$Direction)



weekly$Direction<-as.factor(weekly$Direction)




log_reg_fit=glm(Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume,data=weekly,family=binomial)
summary(log_reg_fit)

contrasts(weekly$Direction)




log_reg_fit_probs=predict(log_reg_fit,type="response")
log_reg_fit_pred=rep("Down",dim(weekly)[1])
log_reg_fit_pred[log_reg_fit_probs>0.5]="Up"
table(log_reg_fit_pred,weekly$Direction)
mean(log_reg_fit_pred==weekly$Direction)
(54 + 557) / (54 + 557 + 430 + 48)




train_set<-weekly[(weekly$Year<2009),]
dim(train_set)
test_set<-weekly[!(weekly$Year<2009),]
dim(test_set)




log_reg_fit_till_2008=glm(Direction~Lag2,data=train_set,family=binomial)
summary(log_reg_fit_till_2008)
log_reg_fit_till_2008_probs=predict(log_reg_fit_till_2008,data.frame(Lag2=test_set$Lag2),type="response")
log_reg_fit_till_2008_pred=rep("Down",dim(test_set)[1])
log_reg_fit_till_2008_pred[log_reg_fit_till_2008_probs>0.5]="Up"
table(log_reg_fit_till_2008_pred,test_set$Direction)
mean(log_reg_fit_till_2008_pred==test_set$Direction)





lda_fit_till_2008=lda(Direction~Lag2,data=train_set)
summary(lda_fit_till_2008)
lda_fit_till_2008_pred=predict(lda_fit_till_2008,data.frame(Lag2=test_set$Lag2),type="response")
table(lda_fit_till_2008_pred$class,test_set$Direction)
mean(lda_fit_till_2008_pred$class==test_set$Direction)





qda_fit_till_2008=qda(Direction~Lag2,data=train_set)
summary(qda_fit_till_2008)
qda_fit_till_2008_pred=predict(qda_fit_till_2008,data.frame(Lag2=test_set$Lag2),type="response")
table(qda_fit_till_2008_pred$class,test_set$Direction)
mean(qda_fit_till_2008_pred$class==test_set$Direction)





knn_fit_till_2008_pred=knn((data.frame(Lag2=train_set$Lag2)),(data.frame(Lag2=test_set$Lag2)),train_set$Direction,k=1)
table(knn_fit_till_2008_pred,test_set$Direction)
mean(knn_fit_till_2008_pred==test_set$Direction)







names(weekly)

weekly$Year_scaled<-scale(weekly$Year)
weekly$Lag1_scaled<-scale(weekly$Lag1)
weekly$Lag2_scaled<-scale(weekly$Lag2)
weekly$Lag3_scaled<-scale(weekly$Lag3)
weekly$Lag4_scaled<-scale(weekly$Lag4)
weekly$Lag5_scaled<-scale(weekly$Lag5)
weekly$Volume_scaled<-scale(weekly$Volume)
weekly$Today_scaled<-scale(weekly$Today)
weekly$Lag2_Lag1<-weekly$Lag2*weekly$Lag1
weekly$Lag3_Lag1<-weekly$Lag3*weekly$Lag1
weekly$Lag4_Lag1<-weekly$Lag4*weekly$Lag1
weekly$Lag5_Lag1<-weekly$Lag5*weekly$Lag1
weekly$Volume_Lag1<-weekly$Volume*weekly$Lag1

names(weekly)

# var(weekly[c("Year_scaled","Lag1_scaled","Lag2_scaled","Lag3_scaled","Lag4_scaled","Lag5_scaled","Volume_scaled","Today_scaled","Year","Lag1","Lag2","Lag3","Lag4","Lag5","Volume","Today")])


train_set<-weekly[(weekly$Year<2009),]
dim(train_set)
test_set<-weekly[!(weekly$Year<2009),]
dim(test_set)


# transformations

log_reg_fit_till_2008=glm(Direction~Lag2_scaled+Lag3_scaled+Volume_scaled,data=train_set,family=binomial)
summary(log_reg_fit_till_2008)
log_reg_fit_till_2008_probs=predict(log_reg_fit_till_2008,test_set[,c("Lag1_scaled","Lag2_scaled","Lag3_scaled","Lag4_scaled","Lag5_scaled","Volume_scaled","Today_scaled")],type="response")
log_reg_fit_till_2008_pred=rep("Down",nrow(test_set))
log_reg_fit_till_2008_pred[log_reg_fit_till_2008_probs>0.5]="Up"
table(log_reg_fit_till_2008_pred,test_set$Direction)
mean(log_reg_fit_till_2008_pred==test_set$Direction)



lda_fit_till_2008=lda(Direction~Lag2_scaled+Lag3_scaled+Volume_scaled,data=train_set)
summary(lda_fit_till_2008)
lda_fit_till_2008_pred=predict(lda_fit_till_2008,test_set[,c("Lag1_scaled","Lag2_scaled","Lag3_scaled","Lag4_scaled","Lag5_scaled","Volume_scaled","Today_scaled")],type="response")
table(lda_fit_till_2008_pred$class,test_set$Direction)
mean(lda_fit_till_2008_pred$class==test_set$Direction)



qda_fit_till_2008=qda(Direction~Lag2_scaled+Lag3_scaled+Volume_scaled,data=train_set)
summary(qda_fit_till_2008)
qda_fit_till_2008_pred=predict(qda_fit_till_2008,test_set[,c("Lag1_scaled","Lag2_scaled","Lag3_scaled","Lag4_scaled","Lag5_scaled","Volume_scaled","Today_scaled")],type="response")
table(qda_fit_till_2008_pred$class,test_set$Direction)
mean(qda_fit_till_2008_pred$class==test_set$Direction)



k_values<-c(1,3,5,7,9,10)
for (k_clust in k_values) {cat("K =",k_clust,",Training Error =",mean(knn(train_set[,c("Lag1_scaled","Lag2_scaled","Lag3_scaled","Lag4_scaled","Lag5_scaled","Volume_scaled","Today_scaled")],test_set[,c("Lag1_scaled","Lag2_scaled","Lag3_scaled","Lag4_scaled","Lag5_scaled","Volume_scaled","Today_scaled")],train_set$Direction,k=k_clust)!=test_set$Direction),"\n")}


# interactions


log_reg_fit_till_2008=glm(Direction~Lag2_scaled+Lag2_Lag1+Lag3_Lag1+Volume_scaled,data=train_set,family=binomial)
summary(log_reg_fit_till_2008)
log_reg_fit_till_2008_probs=predict(log_reg_fit_till_2008,test_set[,c("Lag2_scaled","Lag2_Lag1","Lag3_Lag1","Volume_scaled")],type="response")
log_reg_fit_till_2008_pred=rep("Down",nrow(test_set))
log_reg_fit_till_2008_pred[log_reg_fit_till_2008_probs>0.5]="Up"
table(log_reg_fit_till_2008_pred,test_set$Direction)
mean(log_reg_fit_till_2008_pred==test_set$Direction)




lda_fit_till_2008=lda(Direction~Lag2_scaled+Lag2_Lag1+Lag3_Lag1+Volume_scaled,data=train_set)
summary(lda_fit_till_2008)
lda_fit_till_2008_pred=predict(lda_fit_till_2008,test_set[,c("Lag2_scaled","Lag2_Lag1","Lag3_Lag1","Volume_scaled")],type="response")
table(lda_fit_till_2008_pred$class,test_set$Direction)
mean(lda_fit_till_2008_pred$class==test_set$Direction)



qda_fit_till_2008=qda(Direction~Lag2_scaled+Lag2_Lag1+Lag3_Lag1+Volume_scaled,data=train_set)
summary(qda_fit_till_2008)
qda_fit_till_2008_pred=predict(qda_fit_till_2008,test_set[,c("Lag2_scaled","Lag2_Lag1","Lag3_Lag1","Volume_scaled")],type="response")
table(qda_fit_till_2008_pred$class,test_set$Direction)
mean(qda_fit_till_2008_pred$class==test_set$Direction)



k_values<-c(1,3,5,7,9,10)
for (k_clust in k_values) {cat("K =",k_clust,",Training Error =",mean(knn(train_set[,c("Lag2_scaled","Lag2_Lag1","Lag3_Lag1","Volume_scaled")],test_set[,c("Lag2_scaled","Lag2_Lag1","Lag3_Lag1","Volume_scaled")],train_set$Direction,k=k_clust)!=test_set$Direction),"\n")}

















auto_df<-read.csv("Auto.csv")

names(auto_df)

dim(auto_df)

summary(auto_df)

str(auto_df)

unique(auto_df$horsepower)

auto_df<-auto_df[auto_df$horsepower!="?",]

auto_df$horsepower<-as.integer(auto_df$horsepower)

str(auto_df)

summary(auto_df)

mpg_median<-median(auto_df$mpg)

auto_df$mpg01<-ifelse(auto_df$mpg>mpg_median,1,0)


boxplot(mpg01~cylinders,data=auto_df,main="mpg01 vs cylinders")


boxplot(mpg01~displacement,data=auto_df,main="mpg01 vs displacement")


boxplot(mpg01~horsepower,data=auto_df,main="mpg01 vs horsepower")


boxplot(mpg01~weight,data=auto_df,main="mpg01 vs weight")


boxplot(mpg01~acceleration,data=auto_df,main="mpg01 vs acceleration")


pairs(auto_df[,c("acceleration","cylinders","displacement","horsepower","mpg01","weight")])






auto_df_train_indices<-sample(1:nrow(auto_df),0.8*nrow(auto_df))
train_set<-auto_df[auto_df_train_indices,]
test_set<-auto_df[-auto_df_train_indices,]
dim(train_set)
dim(test_set)




log_reg_fit=glm(mpg01~cylinders+displacement+horsepower+weight+acceleration+year+origin,data=train_set,family=binomial)
summary(log_reg_fit)
log_reg_fit_probs=predict(log_reg_fit,test_set[,c("cylinders","displacement","horsepower","weight","acceleration","year","origin")],type="response")
log_reg_fit_pred=rep(0,nrow(test_set))
log_reg_fit_pred[log_reg_fit_probs>0.5]=1
table(log_reg_fit_pred,test_set$mpg01)
mean(log_reg_fit_pred==test_set$mpg01)
mean(log_reg_fit_pred!=test_set$mpg01)





lda_fit=lda(mpg01~cylinders+displacement+horsepower+weight+acceleration+year+origin,data=train_set)
summary(lda_fit)
lda_fit_pred=predict(lda_fit,test_set[,c("cylinders","displacement","horsepower","weight","acceleration","year","origin")],type="response")
table(lda_fit_pred$class,test_set$mpg01)
mean(lda_fit_pred$class==test_set$mpg01)
mean(lda_fit_pred$class!=test_set$mpg01)





qda_fit=qda(mpg01~cylinders+displacement+horsepower+weight+acceleration+year+origin,data=train_set)
summary(qda_fit)
qda_fit_pred=predict(qda_fit,test_set[,c("cylinders","displacement","horsepower","weight","acceleration","year","origin")],type="response")
table(qda_fit_pred$class,test_set$mpg01)
mean(qda_fit_pred$class==test_set$mpg01)
mean(qda_fit_pred$class!=test_set$mpg01)




k_values<-c(1,3,5,7,9,10)
for (k_clust in k_values) {cat("K =",k_clust,",Training Error =",mean(knn(train_set[,c("cylinders","displacement","horsepower","weight","acceleration","year","origin")],test_set[,c("cylinders","displacement","horsepower","weight","acceleration","year","origin")],train_set$mpg01,k=k_clust)!=test_set$mpg01),"\n")}







(4/5)^5


(1 - 1/100)^100


(1 - 1/10000)^10000


(1 - 1/100000)^100000
n_range_values<-1:100000
n_range_values_prob<-(1 - 1/n_range_values)^n_range_values
plot(n_range_values,n_range_values_prob,type = "l",xlab = "Number of Observations (n)",ylab = "Probability",main = "Probability of jth Observation in the Bootstrap Sample")




store=rep(NA,10000)
for(i in 1:10000){store[i]=sum(sample(1:100,rep=TRUE)==4)>0}
mean(store)





default_df<-read.csv("Default.csv")

View(default_df)

default_df$default<-as.factor(default_df$default)

log_reg_fit=glm(default~income+balance,data=default_df,family=binomial)
summary(log_reg_fit)





train_indices<-sample(nrow(default_df),0.5*nrow(default_df))
train_set<-default_df[train_indices,]
validation_set<-default_df[-train_indices,]
log_reg_fit=glm(default~income+balance,data=train_set,family=binomial)
log_reg_fit_probs<-predict(log_reg_fit,validation_set,type="response")
log_reg_fit_pred<-ifelse(log_reg_fit_probs>0.5,"Yes","No")
cat("Validation Error =",mean(log_reg_fit_pred!=validation_set$default),"\n")
table(log_reg_fit_pred,validation_set$default)






train_indices<-sample(nrow(default_df),0.7*nrow(default_df))
train_set<-default_df[train_indices,]
validation_set<-default_df[-train_indices,]
log_reg_fit=glm(default~income+balance,data=train_set,family=binomial)
log_reg_fit_probs<-predict(log_reg_fit,validation_set,type="response")
log_reg_fit_pred<-ifelse(log_reg_fit_probs>0.5,"Yes","No")
cat("Validation Error =",mean(log_reg_fit_pred!=validation_set$default),"\n")
table(log_reg_fit_pred,validation_set$default)





train_indices<-sample(nrow(default_df),0.8*nrow(default_df))
train_set<-default_df[train_indices,]
validation_set<-default_df[-train_indices,]
log_reg_fit=glm(default~income+balance,data=train_set,family=binomial)
log_reg_fit_probs<-predict(log_reg_fit,validation_set,type="response")
log_reg_fit_pred<-ifelse(log_reg_fit_probs>0.5,"Yes","No")
cat("Validation Error =",mean(log_reg_fit_pred!=validation_set$default),"\n")
table(log_reg_fit_pred,validation_set$default)





train_indices<-sample(nrow(default_df),0.9*nrow(default_df))
train_set<-default_df[train_indices,]
validation_set<-default_df[-train_indices,]
log_reg_fit=glm(default~income+balance,data=train_set,family=binomial)
log_reg_fit_probs<-predict(log_reg_fit,validation_set,type="response")
log_reg_fit_pred<-ifelse(log_reg_fit_probs>0.5,"Yes","No")
cat("Validation Error =",mean(log_reg_fit_pred!=validation_set$default),"\n")
table(log_reg_fit_pred,validation_set$default)







train_indices<-sample(nrow(default_df),0.5*nrow(default_df))
train_set<-default_df[train_indices,]
validation_set<-default_df[-train_indices,]
log_reg_fit=glm(default~income+balance+student,data=train_set,family=binomial)
log_reg_fit_probs<-predict(log_reg_fit,validation_set,type="response")
log_reg_fit_pred<-ifelse(log_reg_fit_probs>0.5,"Yes","No")
cat("Validation Error =",mean(log_reg_fit_pred!=validation_set$default),"\n")
table(log_reg_fit_pred,validation_set$default)



train_indices<-sample(nrow(default_df),0.7*nrow(default_df))
train_set<-default_df[train_indices,]
validation_set<-default_df[-train_indices,]
log_reg_fit=glm(default~income+balance+student,data=train_set,family=binomial)
log_reg_fit_probs<-predict(log_reg_fit,validation_set,type="response")
log_reg_fit_pred<-ifelse(log_reg_fit_probs>0.5,"Yes","No")
cat("Validation Error =",mean(log_reg_fit_pred!=validation_set$default),"\n")
table(log_reg_fit_pred,validation_set$default)


train_indices<-sample(nrow(default_df),0.8*nrow(default_df))
train_set<-default_df[train_indices,]
validation_set<-default_df[-train_indices,]
log_reg_fit=glm(default~income+balance+student,data=train_set,family=binomial)
log_reg_fit_probs<-predict(log_reg_fit,validation_set,type="response")
log_reg_fit_pred<-ifelse(log_reg_fit_probs>0.5,"Yes","No")
cat("Validation Error =",mean(log_reg_fit_pred!=validation_set$default),"\n")
table(log_reg_fit_pred,validation_set$default)


train_indices<-sample(nrow(default_df),0.9*nrow(default_df))
train_set<-default_df[train_indices,]
validation_set<-default_df[-train_indices,]
log_reg_fit=glm(default~income+balance+student,data=train_set,family=binomial)
log_reg_fit_probs<-predict(log_reg_fit,validation_set,type="response")
log_reg_fit_pred<-ifelse(log_reg_fit_probs>0.5,"Yes","No")
cat("Validation Error =",mean(log_reg_fit_pred!=validation_set$default),"\n")
table(log_reg_fit_pred,validation_set$default)








log_reg_fit=glm(default~income+balance,data=default_df,family=binomial)
summary(log_reg_fit)$coef[,"Std. Error"]


boot.fn<-function(df,indices) {
  log_reg_fit<-glm(default~income+balance,data=df[indices,],family=binomial)
  return(coef(log_reg_fit))
}

boot_results<-boot(default_df,boot.fn,R=1000)
cat("Standard Errors from glm() function:","\n")
summary(log_reg_fit)    
summary(log_reg_fit)$coef[,"Std. Error"]

cat("Standard Errors from bootstrap function:","\n")
boot_results


