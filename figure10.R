library(readxl)
data_b <- read_excel("C:/Users/mjgeng/Desktop/比特币及其相关数据/1.xlsx")
#数据预处理
data.01<-ts(data_b$V1, frequency = 365,start = 2018)
par(cex=1.5)
plot(data.01,ylab="BTC",col="blue",lwd=2)
legend(2018,65000,"BTC",lty=1,text.col="blue",col="blue",lwd = 2)
factor1<-c("MD","F&G","ETH","LTC","BCH","USDT")
par(mfrow=c(1,1),cex=1)
par(mfrow=c(1,5))
par(mar=c(10,10,10,10))
data.02<-ts(data_b$V2/1000000000000, frequency = 365,start = 2018)
plot(data.02,ylab="MD",col="red",lwd=2)
legend(2018,65000,"BTC",lty=1,text.col="red",col="red",lwd = 2)
par(mar=c(10,10,10,10)
par(pin = c(5,1))
par(mfrow=c(7,1),oma=c(0.2,0.2,0.2,0.2))
for(i in c(1:1)){
  factor<-factor1[i]
  j=i+1
  data_t<-data_b[j]
  data_t<-(data_t-min(data_t,na.rm = T))/(max(data_t,na.rm = T)-min(data_t,na.rm = T))
  data_ts<-ts(data_t, frequency = 365,start = 2018)
  plot(data_ts,ylab=factor,col="blue",lwd=2)
  legend("topleft",factor,lty=1,text.col="blue",col="blue",lwd=2)
}
for(i in c(2:2)){
  factor<-factor1[i]
  j=i+1
  data_t<-data_b[j]
  data_t<-(data_t-min(data_t,na.rm = T))/(max(data_t,na.rm = T)-min(data_t,na.rm = T))
  data_ts<-ts(data_t, frequency = 365,start = 2018)
  plot(data_ts,ylab=factor,col="red",lwd=2)
  legend("topleft",factor,lty=1,text.col="red",col="red",lwd=2)
}
for(i in c(3:6)){
  factor<-factor1[i]
  j=i+1
  data_t<-data_b[j]
  data_t<-(data_t-min(data_t,na.rm = T))/(max(data_t,na.rm = T)-min(data_t,na.rm = T))
  data_ts<-ts(data_t, frequency = 365,start = 2018)
  plot(data_ts,ylab=factor,col="purple",lwd=2)
  legend("topleft",factor,lty=1,text.col="purple",col="purple",lwd=2)
}

factor2<-c("DJIA","NASDAQ","S&P 500","VIXCLS","FTSE 100","HSCEI","HSI","AHP","SSEC")
for(i in c(1:4)){
  factor<-factor2[i]
  j=i+7
  data_t<-data_b[j]
  data_t<-(data_t-min(data_t,na.rm = T))/(max(data_t,na.rm = T)-min(data_t,na.rm = T))
  data_ts<-ts(data_t, frequency = 365,start = 2018)
  plot(data_ts,ylab=factor,col="blue",lwd=2)
  legend("topleft",factor,lty=1,text.col="blue",col="blue",lwd=2)
}
for(i in c(5:5)){
  factor<-factor2[i]
  j=i+7
  data_t<-data_b[j]
  data_t<-(data_t-min(data_t,na.rm = T))/(max(data_t,na.rm = T)-min(data_t,na.rm = T))
  data_ts<-ts(data_t, frequency = 365,start = 2018)
  plot(data_ts,ylab=factor,col="red",lwd=2)
  legend("bottomleft",factor,lty=1,text.col="red",col="red",lwd=2)
}
for(i in c(6:9)){
  factor<-factor2[i]
  j=i+7
  data_t<-data_b[j]
  data_t<-(data_t-min(data_t,na.rm = T))/(max(data_t,na.rm = T)-min(data_t,na.rm = T))
  data_ts<-ts(data_t, frequency = 365,start = 2018)
  plot(data_ts,ylab=factor,col="purple",lwd=2)
  legend("bottomleft",factor,lty=1,text.col="purple",col="purple",lwd=2)
}
factor3<-c("GOLD","SILVER","WTI","EFFR","T10YIE")
for(i in c(1:2)){
  factor<-factor3[i]
  j=i+16
  data_t<-data_b[j]
  data_t<-(data_t-min(data_t,na.rm = T))/(max(data_t,na.rm = T)-min(data_t,na.rm = T))
  data_ts<-ts(data_t, frequency = 365,start = 2018)
  plot(data_ts,ylab=factor,col="blue",lwd=2)
  legend("topleft",factor,lty=1,text.col="blue",col="blue",lwd=2)
}
for(i in c(3:3)){
  factor<-factor3[i]
  j=i+16
  data_t<-data_b[j]
  data_t<-(data_t-min(data_t,na.rm = T))/(max(data_t,na.rm = T)-min(data_t,na.rm = T))
  data_ts<-ts(data_t, frequency = 365,start = 2018)
  plot(data_ts,ylab=factor,col="red",lwd=2)
  legend("bottomleft",factor,lty=1,text.col="red",col="red",lwd=2)
}
for(i in c(4:4)){
  factor<-factor3[i]
  j=i+16
  data_t<-data_b[j]
  data_t<-(data_t-min(data_t,na.rm = T))/(max(data_t,na.rm = T)-min(data_t,na.rm = T))
  data_ts<-ts(data_t, frequency = 365,start = 2018)
  plot(data_ts,ylab=factor,col="purple",lwd=2)
  legend("bottomleft",factor,lty=1,text.col="purple",col="purple",lwd=2)
}
for(i in c(5:5)){
  factor<-factor3[i]
  j=i+16
  data_t<-data_b[j]
  data_t<-(data_t-min(data_t,na.rm = T))/(max(data_t,na.rm = T)-min(data_t,na.rm = T))
  data_ts<-ts(data_t, frequency = 365,start = 2018)
  plot(data_ts,ylab=factor,col="green",lwd=2)
  legend("bottomleft",factor,lty=1,text.col="green",col="green",lwd=2)
}

factor4<-c("USDX","AUD/USD","EUR/USD","GBP/USD","100JPY/USD","CHY/USD","CAD/USD")


for(i in c(1:1)){
  factor<-factor4[i]
  j=i+21
  data_t<-data_b[j]
  data_t<-(data_t-min(data_t,na.rm = T))/(max(data_t,na.rm = T)-min(data_t,na.rm = T))
  data_ts<-ts(data_t, frequency = 365,start = 2018)
  plot(data_ts,ylab=factor,col="red",lwd=2)
  legend("topleft",factor,lty=1,text.col="red",col="red",lwd=2)
}

for(i in c(1:7)){
  factor<-factor4[i]
  j=i+21
  data_t<-data_b[j]
  data_t<-(data_t-min(data_t,na.rm = T))/(max(data_t,na.rm = T)-min(data_t,na.rm = T))
  data_ts<-ts(data_t, frequency = 365,start = 2018)
  plot(data_ts,ylab=factor,col="purple",lwd=2)
  legend("bottomleft",factor,lty=1,text.col="purple",col="purple",lwd=2)
}

#归一化
data.gui<-function(data){
  result<-(data-min(data,na.rm = T))/(max(data,na.rm = T)-min(data,na.rm = T))
  return(result)
}

dt1<-c(1:1517)
for (i in c(1:29)) {
  dt <- data_b[,i]
  dt <- data.gui(dt)
  dt1 <- as.data.frame(cbind(dt1,dt))
}
dt1
write.csv(dt1,"归一化后数据.csv")






#xgboost
install.packages("xgboost")
install.packages("Matrix")
install.packages("Ckmeans.1d.dp")
library(xgboost)
library(Matrix)
library(Ckmeans.1d.dp)
#模型训练,将数据修改成稀疏矩阵
data_b<-read.csv("C:/Users/mjgeng/Desktop/比特币及其相关数据/归一化后数据.csv")
train_matrix<-sparse.model.matrix(V1 ~ .,data = data_b)
train_label<-as.numeric(data_b$V1)
train_label<-data_b$V1[1:1514]
#分离变量和标签
fin<-list(data=train_matrix,label=train_label)
dtrain<-xgb.DMatrix(data = fin$data,label = fin$label)
#运用xgboost函数对数据进行处理，寻找到影响比特币价格的关键性变量
xgb <- xgboost(data = dtrain,nround=250)
#重要重要性排序 
importance <- xgb.importance(train_matrix@Dimnames[[2]], model = xgb)
sum.xgb<-sum(importance$Importance)
importance$Importance<-importance$Importance/sum.xgb
head(importance)
#绘制变量重要性图
xgb.ggplot.importance(importance)
#选择重要性高的变量
xgb.ggplot.importance(importance[c(1,2,3,4,5,6,7,8,9,11,14,
                                   36,48,65),])
xgb.ggplot.importance(importance[1:30,])
#随机森林回归
library(randomForest)
library(foreign)
gx.rf<-randomForest(V1~V2+V3+V4+V5+V6+V7+V8+
                      V9+V10+V11+V12+V13+V14+V15+
                      V16+V17+V18+V19+V20+V21+V22+V23+
                      V24+V25+V26+V27+V28,data_b,importance=TRUE, ntree=100,
                    na.action = na.omit)
print(gx.rf)

gx.rf_im<-importance(gx.rf)
sum <- sum(gx.rf_im[,2])
gx.rf_im[,2] <- gx.rf_im[,2]/sum
gx.rf_im
varImpPlot(gx.rf,main = "")
?varImpPlot
#画火柴棒图
install.packages("ggplot2")
library(ggplot2)
data_pic<-read_excel("C:/Users/mjgeng/Desktop/比特币及其相关数据/5.xlsx")
par(mfrow = c(1,2))
par(mfrow = c(2,1))
ggplot(data_pic,aes(IncNodePurity,a))+
  geom_segment(aes(x=0, 
                   xend=IncNodePurity,
                   y=a, 
                   yend=a))+
  geom_point(shape=21,size=3,colour="black",fill="#FC4E07")+
  theme(
    axis.title=element_text(size=12,face="plain",color="black"),
    axis.text = element_text(size=12,face="plain",color="black"),
    legend.title=element_text(size=12,face="plain",color="black")
  )
data_pic2<-read_excel("C:/Users/mjgeng/Desktop/比特币及其相关数据/4.xlsx")
ggplot(data_pic2,aes(Importance,Feature))+
  geom_segment(aes(x=0, 
                   xend=Importance,
                   y=Feature, 
                   yend=Feature))+
  geom_point(shape=21,size=3,colour="black",fill="#FC4E07")+
  theme(
    axis.title=element_text(size=12,face="plain",color="black"),
    axis.text = element_text(size=12,face="plain",color="black"),
    legend.title=element_text(size=12,face="plain",color="black")
  )
#玫瑰图
data_gg_a<-read_excel("C:/Users/mjgeng/Desktop/比特币及其相关数据/6.xlsx")
data_gg_b<-read_excel("C:/Users/mjgeng/Desktop/比特币及其相关数据/7.xlsx")
colour <- rainbow(11)
library(ggplot2)
plot_xg<-ggplot(data_gg_a,aes(x=factor,y=Importance*600,label=factor))+
  geom_bar(aes(fill=factor(factor)),width=1,stat = 'identity')+
  geom_label(aes(x=factor),
             nudge_y = 250)+
  scale_fill_manual(values = colour)+
  coord_polar(theta = 'x',start = 0,direction = 1)+
  theme(legend.position = "top")+
  theme_void()+ggtitle("XGBoost")#+guides(fill=FALSE)
plot_rf<-ggplot(data_gg_b,aes(x=factor,y=IncNodePurity*1500,label=factor))+
  geom_bar(aes(fill=factor(factor)),width=1,stat = 'identity')+
  geom_label(aes(x=factor),
             nudge_y = 500)+
  scale_fill_manual(values = colour)+
  coord_polar(theta = 'x',start = 0,direction = 1)+
  theme(legend.position = "top")+
  theme_void()+ggtitle("Random forest")
install.packages("ggpubr")
install.packages("gridExtra")
install.packages("cowplot")
library(ggpubr)


ggarrange(plot_xg,plot_rf)

