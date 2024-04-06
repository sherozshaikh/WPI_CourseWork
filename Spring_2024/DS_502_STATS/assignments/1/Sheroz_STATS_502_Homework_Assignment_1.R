library(ggplot2)
library(cowplot)
library(corrplot)


college<-read.csv("./College.csv")
rownames(college)=college[,1]
fix(college)


college=college[,-1]
fix(college)
summary(college)


college[,1]=as.numeric(factor(college[,1]))
pairs(college[,1:10])


boxplot(college$Outstate~college$Private)


Elite<-rep("No",nrow(college ))
Elite[college$Top10perc > 50]<-"Yes"
Elite<-as.factor(Elite)
college<-data.frame(college,Elite)
summary(college$Elite)


boxplot(college$Outstate~college$Elite)


par(mfrow=c(4,4))
hist(college$Apps)
hist(college$Accept)
hist(college$Enroll)
hist(college$Top10perc)
hist(college$Top25perc)
hist(college$F.Undergrad)
hist(college$P.Undergrad)
hist(college$Outstate)
hist(college$Room.Board)
hist(college$Books)
hist(college$Personal)
hist(college$PhD)
hist(college$Terminal)
hist(college$S.F.Ratio)
hist(college$Expend)
hist(college$perc.alumni)


str(college)

num_vars<-c("Apps","Accept","Enroll","Top10perc","Top25perc","F.Undergrad","P.Undergrad","Outstate","Room.Board","Books","Personal","PhD","Terminal","S.F.Ratio","perc.alumni","Expend","Grad.Rate")



par(mfrow=c(4,5))
for (var in num_vars) {hist(college[[var]],main=var,xlab="",col="skyblue",border="white")}




density_plots<-list()
for (var in num_vars) {
  density_plot<-ggplot(college,aes(x=.data[[var]])) + geom_density(fill="skyblue",color="blue") + labs(title=var)
  density_plots[[var]]<-density_plot
}
plot_grid(plotlist=density_plots,nrow=6,ncol=3)



cor_matrix<-cor(college[num_vars])
corrplot(cor_matrix,method="color",type="upper",tl.col="black",tl.srt=45)



plot(college$Apps,college$Accept,xlab="Number of Applications",ylab="Number of Acceptances",main="Applications vs. Acceptances",col="skyblue")



boxplot(college$Grad.Rate ~ college$Private,main="Graduation Rate by Private/Public Institution",xlab="Private",ylab="Graduation Rate",col=c("skyblue","lightgreen"))



hist(college$S.F.Ratio,main="Student-Faculty Ratio Distribution",xlab="Student-Faculty Ratio",col="skyblue",border="white")



# ================================================================================================================================


auto<-read.csv("./Auto.csv",header=TRUE,na.strings="?")
summary(auto)



auto<-na.omit(auto)
head(auto)



str(auto)



sapply(auto[,c("mpg","cylinders","displacement","horsepower","weight","acceleration","year")],range)



sapply(auto[,c("mpg","cylinders","displacement","horsepower","weight","acceleration","year")],mean)



sapply(auto[,c("mpg","cylinders","displacement","horsepower","weight","acceleration","year")],sd)





auto_subset<-auto[-c(10:85),]


sapply(auto_subset[,c("mpg","cylinders","displacement","horsepower","weight","acceleration","year")],range)



sapply(auto_subset[,c("mpg","cylinders","displacement","horsepower","weight","acceleration","year")],mean)



sapply(auto_subset[,c("mpg","cylinders","displacement","horsepower","weight","acceleration","year")],sd)




pairs(auto[,c("mpg","cylinders","displacement","horsepower","weight","acceleration","year")])



cor_matrix<-cor(auto[,c("mpg","cylinders","displacement","horsepower","weight","acceleration","year")])
corrplot(cor_matrix,method="color",type="upper",tl.col="black",tl.srt=45)





par(mfrow=c(3,3))
hist(auto$mpg)
hist(auto$cylinders)
hist(auto$displacement)
hist(auto$horsepower)
hist(auto$weight)
hist(auto$acceleration)
hist(auto$year)




boxplot(auto$mpg~auto$cylinders)




lm_auto_fit<-lm(mpg~horsepower,data=auto)
summary(lm_auto_fit)



predict(lm_auto_fit,newdata=data.frame(horsepower=98),interval="confidence",level=0.95)


predict(lm_auto_fit,newdata=data.frame(horsepower=98),interval="prediction",level=0.95)



plot(auto$horsepower,auto$mpg,main="Linear Regression: MPG vs Horsepower",xlab="horsepower",ylab="mpg",col="black")
abline(lm_auto_fit,col="red")



plot(lm_auto_fit)


pairs(auto[1:8])



cor(auto[1:8])



lm_auto_fit<-lm(mpg ~ . - name,data=auto)
summary(lm_auto_fit)



plot(lm_auto_fit)





lm_auto_fit<-lm(mpg~horsepower*displacement,data=auto[,1:8])
summary(lm_auto_fit)



lm_auto_fit<-lm(mpg~horsepower*weight,data=auto[,1:8])
summary(lm_auto_fit)


lm_auto_fit<-lm(mpg~horsepower*displacement,data=auto[,1:8])
summary(lm_auto_fit)



lm_auto_fit<-lm(mpg~horsepower*weight+horsepower*displacement,data=auto[,1:8])
summary(lm_auto_fit)




lm_auto_fit<-lm(mpg ~ log(horsepower) + sqrt(displacement),data=auto[,1:8])
summary(lm_auto_fit)



lm_auto_fit<-lm(mpg ~ log(horsepower) + sqrt(horsepower),data=auto[,1:8])
summary(lm_auto_fit)



lm_auto_fit<-lm(mpg ~ log(horsepower) + sqrt(weight),data=auto[,1:8])
summary(lm_auto_fit)


lm_auto_fit<-lm(mpg ~ log(horsepower) + acceleration^2,data=auto[,1:8])
summary(lm_auto_fit)


lm_auto_fit<-lm(mpg ~ log(horsepower) + sqrt(displacement) + sqrt(horsepower) + sqrt(weight) + acceleration^2,data=auto[,1:8])
summary(lm_auto_fit)


