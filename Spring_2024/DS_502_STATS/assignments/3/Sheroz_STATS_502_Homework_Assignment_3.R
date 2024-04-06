# suppose
y=2
lambda=2

betas=seq(-10,10,0.8)

# equation 6.12
func1=((y - betas)^2) + (lambda * (betas^2))

plot(betas,func1,pch=20,xlab='Beta',ylab='Ridge')

# equation 6.14
estimation_beta=y/(1+lambda)
print(estimation_beta)

estimation_function=((y - estimation_beta)^2) + (lambda * (estimation_beta^2))
print(estimation_function)

points(estimation_beta,estimation_function,col='red',pch=4,lwd=2,cex=estimation_beta)





# suppose
y=2
lambda=2

betas=seq(-10,10,0.8)

# equation 6.13
func1=((y - betas)^2) + (lambda * abs(betas))

plot(betas,func1,pch=20,xlab='Beta',ylab='Lasso')

# equation 6.15
# since y > (lambda / 2) i.e 2 > (2/2) so
estimation_beta=y-(lambda/2)
print(estimation_beta)

estimation_function=((y - estimation_beta)^2) + (lambda * abs(estimation_beta))
print(estimation_function)

points(estimation_beta,estimation_function,col='red',pch=4,lwd=2,cex=estimation_beta)






set.seed(123)

x_n=100
e_n=100
x=rnorm(x_n)
print(x)
e_noise=rnorm(e_n)
print(e_noise)


beta_0=1
beta_1=2
beta_2=3
beta_3=4


y=beta_0 + (beta_1 * x) + (beta_2 * (x^2)) + (beta_3 * (x^3)) + e_noise

y



library(leaps)
question_3_data=data.frame(x,x^2,x^3,x^4,x^5,x^6,x^7,x^8,x^9,x^10,y)
model_bss=regsubsets(y~.,data=question_3_data,nvmax=10)
model_bss_summary=summary(model_bss)
model_bss_summary
names(model_bss)
names(model_bss_summary)

par(mfrow=c(3,1))
plot(model_bss_summary$cp,xlab='number of variables',ylab='cp',type='l')
plot(model_bss_summary$bic,xlab='number of variables',ylab='bic',type='l')
plot(model_bss_summary$adjr2,xlab='number of variables',ylab='adjr2',type='l')

c(which.min(model_bss_summary$cp),which.min(model_bss_summary$bic),which.max(model_bss_summary$adjr2))
print(coef(model_bss,3))
print(coef(model_bss,7))

# forward stepwise selection
model_fds=regsubsets(y~.,data=question_3_data,nvmax=10,method='forward')
model_fds_summary=summary(model_fds)
model_fds_summary

par(mfrow=c(3,1))
plot(model_fds_summary$cp,xlab='number of variables',ylab='cp',type='l')
plot(model_fds_summary$bic,xlab='number of variables',ylab='bic',type='l')
plot(model_fds_summary$adjr2,xlab='number of variables',ylab='adjr2',type='l')

c(which.min(model_fds_summary$cp),which.min(model_fds_summary$bic),which.max(model_fds_summary$adjr2))

indices<-c(which.min(model_fds_summary$cp),which.min(model_fds_summary$bic),which.max(model_fds_summary$adjr2))

for (index in indices) {
  cat("Number of variables:",index,"\n")
  cat("Coefficients:\n")
  print(coef(model_fds,id=index))
  cat("\n")
}





# backward stepwise selection
model_bds=regsubsets(y~.,data=question_3_data,nvmax=10,method='backward')
model_bds_summary=summary(model_bds)
model_bds_summary

par(mfrow=c(3,1))
plot(model_bds_summary$cp,xlab='number of variables',ylab='cp',type='l')
plot(model_bds_summary$bic,xlab='number of variables',ylab='bic',type='l')
plot(model_bds_summary$adjr2,xlab='number of variables',ylab='adjr2',type='l')

c(which.min(model_bds_summary$cp),which.min(model_bds_summary$bic),which.max(model_bds_summary$adjr2))

indices<-c(which.min(model_bds_summary$cp),which.min(model_bds_summary$bic),which.max(model_bds_summary$adjr2))

for (index in indices) {
  cat("Number of variables:",index,"\n")
  cat("Coefficients:\n")
  print(coef(model_bds,id=index))
  cat("\n")
}








library(glmnet)

xmat=model.matrix(y~.,data=question_3_data)[,-1]
model_lasso=cv.glmnet(xmat,y,alpha=1)
model_lasso_best_lambda=model_lasso$lambda.min
print(model_lasso_best_lambda)
plot(model_lasso)

model_best_lasso=glmnet(xmat,y,alpha=1)
predict(model_best_lasso,s=model_lasso_best_lambda,type="coefficients")

beta_7=6

y=beta_0 + (beta_7 * (x^7)) + e_noise

question_3_f_data=data.frame(x,x^2,x^3,x^4,x^5,x^6,x^7,x^8,x^9,x^10,y)

# best subset selection method
model_bss_q3f=regsubsets(y~.,data=question_3_f_data,nvmax=10)

model_bss_q3f_summary=summary(model_bss_q3f)
model_bss_q3f_summary

par(mfrow=c(3,1))
plot(model_bss_q3f_summary$cp,xlab='number of variables',ylab='cp',type='l')
plot(model_bss_q3f_summary$bic,xlab='number of variables',ylab='bic',type='l')
plot(model_bss_q3f_summary$adjr2,xlab='number of variables',ylab='adjr2',type='l')

indices<-c(which.min(model_bss_q3f_summary$cp),which.min(model_bss_q3f_summary$bic),which.max(model_bss_q3f_summary$adjr2))

for (index in indices) {
  cat("Number of variables:",index,"\n")
  cat("Coefficients:\n")
  print(coef(model_bss_q3f,id=index))
  cat("\n")
}





# lasso selection method
xmat=model.matrix(y~.,data=question_3_f_data)[,-1]
model_lasso=cv.glmnet(xmat,y,alpha=1)
model_lasso_best_lambda=model_lasso$lambda.min
print(model_lasso_best_lambda)

plot(model_lasso)

model_best_lasso=glmnet(xmat,y,alpha=1)
predict(model_best_lasso,s=model_lasso_best_lambda,type="coefficients")

model_best_lasso_summary=summary(model_best_lasso)

model_best_lasso_summary






library(ISLR)

set.seed(123)

sum(is.na(College))

round((nrow(College)*0.2),0)

test_index=sample(nrow(College),round((nrow(College)*0.2),0),replace=FALSE)
train_set=College[-test_index,]
test_set=College[test_index,]

c(nrow(train_set),nrow(test_set),nrow(train_set)+nrow(test_set),nrow(College))

names(College)

lm_model=lm(Apps~.,data=train_set)
lm_model_pred=predict(lm_model,test_set)
# RSS
print(mean((test_set[,"Apps"]-lm_model_pred)^2))





library(glmnet)

train_xmat=model.matrix(Apps~.,data=train_set)
test_xmat=model.matrix(Apps~.,data=test_set)

# Ridge
model_ridge=cv.glmnet(train_xmat,train_set[,'Apps'],alpha=0)
model_ridge_best_lambda=model_ridge$lambda.min
print(model_ridge_best_lambda)

plot(model_ridge)

ridge_model_pred=predict(model_ridge,test_xmat,s=model_ridge_best_lambda)
# RSS for Ridge
print(mean((test_set[,"Apps"]-ridge_model_pred)^2))




# Lasso
model_lasso=cv.glmnet(train_xmat,train_set[,'Apps'],alpha=1)
model_lasso_best_lambda=model_lasso$lambda.min
print(model_lasso_best_lambda)

plot(model_lasso)

lasso_model_pred=predict(model_lasso,test_xmat,s=model_lasso_best_lambda)
# RSS for Lasso
print(mean((test_set[,"Apps"]-lasso_model_pred)^2))

predict(model_lasso,test_xmat,s=model_lasso_best_lambda,type='coefficients')





library(pls)

set.seed(123)

model_pcr=pcr(Apps~.,data=train_set,scale=TRUE,validation="CV")

summary(model_pcr)

validationplot(model_pcr,val.type="MSEP")

names(model_pcr)

model_pcr$validation

model_pcr_pred=predict(model_pcr,test_set,ncomp=17)
round((mean((test_set[,'Apps'] - model_pcr_pred)^2)),0)




model_plsr=plsr(Apps~.,data=train_set,scale=TRUE,validation="CV")

summary(model_plsr)

validationplot(model_plsr,val.type="MSEP")

model_plsr$validation

model_plsr_pred=predict(model_plsr,test_set,ncomp=13)
round((mean((test_set[,'Apps'] - model_plsr_pred)^2)),0)





barplot(c(726062.2,799031.9,750801.5,726062,727135),col="red",names.arg=c("OLS","Ridge","Lasso","PCR","PLSR"),main="Test RSS")



