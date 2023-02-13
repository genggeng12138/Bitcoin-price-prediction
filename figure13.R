library(readxl)
col_lssvr<-c("#00008B","#8B2323","#006400","#EEAD0E","#458B00","#483D8B")
data_lssvr <- read_excel("C:/Users/mjgeng/Desktop/比特币及其相关数据/适应度曲线/附件2lssvr.xlsx")
#data_lssvr <- read_excel("D:/研究生/2022深圳杯/pso.xlsx")
pso_lssvr_6 <- data_lssvr$psolssvr6
pso_lssvr_11 <- data_lssvr$psolssvr11
pso_lssvr_27 <- data_lssvr$psolssvr27
woa_lssvr_6 <- data_lssvr$woalssvr6
woa_lssvr_11 <- data_lssvr$woalssvr11
woa_lssvr_27 <- data_lssvr$woalssvr27
par(cex=1.3)
plot(c(1:100),pso_lssvr_6,type = "S",xlab = "Iteration",ylab = "Fitnes value",
     ylim=c(0.8,0.915),lwd=6,col=col_lssvr[1])
lines(c(1:100),pso_lssvr_11,lwd=6,col=col_lssvr[2],type = "S")
lines(c(1:100),pso_lssvr_27,lwd=6,col=col_lssvr[3],type = "S")
lines(c(1:100),woa_lssvr_6,lwd=6,col=col_lssvr[4],type = "S")
lines(c(1:100),woa_lssvr_11,lwd=6,col=col_lssvr[5],type = "S")
lines(c(1:100),woa_lssvr_27,lwd=6,col=col_lssvr[6],type = "S")
points(c(1:100),pso_lssvr_6,pch=0,col=col_lssvr[1],lwd=3)
points(c(1:100),pso_lssvr_11,pch=1,col=col_lssvr[2],lwd=3)
points(c(1:100),pso_lssvr_27,pch=2,col=col_lssvr[3],lwd=3)
points(c(1:100),woa_lssvr_6,pch=3,col=col_lssvr[4],lwd=3)
points(c(1:100),woa_lssvr_11,pch=4,col=col_lssvr[5],lwd=3)
points(c(1:100),woa_lssvr_27,pch=5,col=col_lssvr[6],lwd=3)
factor_lssvr<-c("PSO-RF-GSA-LSSVR","PSO-RF-LSSVR","PSO-XGboost-LSSVR",
                "WOA-RF-GSA-LSSVR","WOA-RF-LSSVR","WOA-XGboost-LSSVR")

legend("bottomright",factor_lssvr,text.col = col_lssvr,lwd=4,col = col_lssvr,pch = c(0:5))
?legend

plot(c(1:15),pso_lssvr_6[1:15],type = "S",xlab = "Iteration",ylab = "Fitnes value",
     ylim=c(0.84,0.915),lwd=8,col=col_lssvr[1])
lines(c(1:100),pso_lssvr_11,lwd=8,col=col_lssvr[2],type = "S")
lines(c(1:100),pso_lssvr_27,lwd=8,col=col_lssvr[3],type = "S")
lines(c(1:100),woa_lssvr_6,lwd=8,col=col_lssvr[4],type = "S")
lines(c(1:100),woa_lssvr_11,lwd=8,col=col_lssvr[5],type = "S")
lines(c(1:100),woa_lssvr_27,lwd=8,col=col_lssvr[6],type = "S")
points(c(1:100),pso_lssvr_6,pch=0,col=col_lssvr[1],lwd=4)
points(c(1:100),pso_lssvr_11,pch=1,col=col_lssvr[2],lwd=4)
points(c(1:100),pso_lssvr_27,pch=2,col=col_lssvr[3],lwd=4)
points(c(1:100),woa_lssvr_6,pch=3,col=col_lssvr[4],lwd=4)
points(c(1:100),woa_lssvr_11,pch=4,col=col_lssvr[5],lwd=4)
points(c(1:100),woa_lssvr_27,pch=5,col=col_lssvr[6],lwd=4)







data_svr <- read_excel("C:/Users/mjgeng/Desktop/比特币及其相关数据/适应度曲线/附件2svr.xlsx")
#data_svr <- read_excel("D:/研究生/2022深圳杯/pso.xlsx")
pso_svr_6 <- data_svr$psosvr6
pso_svr_11 <- data_svr$psosvr11
pso_svr_27 <- data_svr$psosvr27
woa_svr_6 <- data_svr$woasvr6
woa_svr_11 <- data_svr$woasvr11
woa_svr_27 <- data_svr$woasvr27
par(cex=1.3)
plot(c(1:length(pso_svr_6)),pso_svr_6,type = "S",xlab = "Iteration",ylab = "Fitnes value",
     ylim=c(0.5,0.91),lwd=6,col=col_lssvr[1])
lines(c(1:100),pso_svr_11,lwd=6,col=col_lssvr[2],type = "S")
lines(c(1:100),pso_svr_27,lwd=6,col=col_lssvr[3],type = "S")
lines(c(1:100),woa_svr_6,lwd=6,col=col_lssvr[4],type = "S")
lines(c(1:100),woa_svr_11,lwd=6,col=col_lssvr[5],type = "S")
lines(c(1:100),woa_svr_27,lwd=6,col=col_lssvr[6],type = "S")
points(c(1:100),pso_svr_6,pch=0,lwd=3,col=col_lssvr[1])
points(c(1:100),pso_svr_11,pch=1,lwd=3,col=col_lssvr[2])
points(c(1:100),pso_svr_27,pch=2,lwd=3,col=col_lssvr[3])
points(c(1:100),woa_svr_6,pch=3,lwd=3,col=col_lssvr[4])
points(c(1:100),woa_svr_11,pch=4,lwd=3,col=col_lssvr[5])
points(c(1:100),woa_svr_27,pch=5,lwd=3,col=col_lssvr[6])
factor_svr<-c("PSO-RF-GSA-SVR","PSO-RF-SVR","PSO-XGBoost-SVR",
              "WOA-RF-GSA-SVR","WOA-RF-SVR","WOA-XGBoost-SVR")
#col_svr<-c("#FAEBD7","#7FFFD4","#EEE98F","#FFD39B","#C1FFC1","#8DEEEE")
legend("bottomright",factor_svr,text.col = col_lssvr,lwd=4,col = col_svr,pch = c(0:5))

plot(c(1:length(pso_svr_6))[1:25],pso_svr_6[1:25],type = "S",xlab = "Iteration",ylab = "Fitnes value",
     ylim=c(0.55,0.91),lwd=8,col=col_lssvr[1])
lines(c(1:100),pso_svr_11,lwd=8,col=col_lssvr[2],type = "S")
lines(c(1:100),pso_svr_27,lwd=8,col=col_lssvr[3],type = "S")
lines(c(1:100),woa_svr_6,lwd=8,col=col_lssvr[4],type = "S")
lines(c(1:100),woa_svr_11,lwd=8,col=col_lssvr[5],type = "S")
lines(c(1:100),woa_svr_27,lwd=8,col=col_lssvr[6],type = "S")
points(c(1:100),pso_svr_6,pch=0,lwd=4,col=col_lssvr[1])
points(c(1:100),pso_svr_11,pch=1,lwd=4,col=col_lssvr[2])
points(c(1:100),pso_svr_27,pch=2,lwd=4,col=col_lssvr[3])
points(c(1:100),woa_svr_6,pch=3,lwd=4,col=col_lssvr[4])
points(c(1:100),woa_svr_11,pch=4,lwd=4,col=col_lssvr[5])
points(c(1:100),woa_svr_27,pch=5,lwd=4,col=col_lssvr[6])



data <- read_excel("C:/Users/mjgeng/Desktop/2.xlsx")
par(mfrow=c(2,3))

plot(c(1:20),data$RF,type = "S",xlab = "Number",ylab = "MAE",
     lwd=3,col=col_lssvr[1],main="RF")
points(c(1:20),data$RF,pch=0,lwd=3,col=col_lssvr[1])
abline(h=0.1,lwd=2,lty=2)
plot(c(1:20),data$XGBoost,type = "S",xlab = "Number",ylab = "MAE",
     lwd=3,col=col_lssvr[2],main="XGBoost")
points(c(1:20),data$XGBoost,pch=1,lwd=3,col=col_lssvr[2])
abline(h=0.1,lwd=2,lty=2)
plot(c(1:20),data$`RF-GSA`,type = "S",xlab = "Number",ylab = "MAE",
     lwd=3,col=col_lssvr[3],main="RF-GSA")
points(c(1:20),data$`RF-GSA`,pch=2,lwd=3,col=col_lssvr[3])
abline(h=0.1,lwd=2,lty=2)

data <- read_excel("C:/Users/mjgeng/Desktop/1.xlsx")
par(mfrow=c(2,3))

plot(c(1:20),data$RF,type = "S",xlab = "Number",ylab = "MAE",
     lwd=3,col=col_lssvr[4],main="RF")
points(c(1:20),data$RF,pch=3,lwd=3,col=col_lssvr[4])
abline(h=0.1,lwd=2,lty=2)
plot(c(1:20),data$XGBoost,type = "S",xlab = "Number",ylab = "MAE",
     lwd=3,col=col_lssvr[5],main="XGBoost")
points(c(1:20),data$XGBoost,pch=4,lwd=3,col=col_lssvr[5])
abline(h=0.1,lwd=2,lty=2)
plot(c(1:20),data$`RF-GSA`,type = "S",xlab = "Number",ylab = "MAE",
     lwd=3,col=col_lssvr[6],main="RF-GSA")
points(c(1:20),data$`RF-GSA`,pch=5,lwd=3,col=col_lssvr[6])
abline(h=0.1,lwd=2,lty=2)

col_lssvr<-c("#00008B","#8B2323","#006400","#EEAD0E","#458B00","#483D8B")


data_twsvr <- read_excel("C:/Users/mjgeng/Desktop/比特币及其相关数据/适应度曲线/附件2twsvr.xlsx")
#data_svr <- read_excel("D:/研究生/2022深圳杯/pso.xlsx")
pso_twsvr_6 <- data_twsvr$psotwsvr6
pso_twsvr_11 <- data_twsvr$psotwsvr11
pso_twsvr_27 <- data_twsvr$psotwsvr27
woa_twsvr_6 <- data_twsvr$woatwsvr6
woa_twsvr_11 <- data_twsvr$woatwsvr11
woa_twsvr_27 <- data_twsvr$woatwsvr27
par(cex=1.3)
plot(c(1:100),pso_twsvr_6,type = "S",xlab = "Iteration",ylab = "Fitnes value",
     ylim=c(0.85,0.955),lwd=6,col=col_lssvr[1])
lines(c(1:100),pso_twsvr_11,lwd=6,col=col_lssvr[2],type = "S")
lines(c(1:100),pso_twsvr_27,lwd=6,col=col_lssvr[3],type = "S")
lines(c(1:100),woa_twsvr_6,lwd=6,col=col_lssvr[4],type = "S")
lines(c(1:100),woa_twsvr_11,lwd=6,col=col_lssvr[5],type = "S")
lines(c(1:100),woa_twsvr_27,lwd=6,col=col_lssvr[6],type = "S")
points(c(1:100),pso_twsvr_6,pch=0,lwd=3,col=col_lssvr[1])
points(c(1:100),pso_twsvr_11,pch=1,lwd=3,col=col_lssvr[2])
points(c(1:100),pso_twsvr_27,pch=2,lwd=3,col=col_lssvr[3])
points(c(1:100),woa_twsvr_6,pch=3,lwd=3,col=col_lssvr[4])
points(c(1:100),woa_twsvr_11,pch=4,lwd=3,col=col_lssvr[5])
points(c(1:100),woa_twsvr_27,pch=5,lwd=3,col=col_lssvr[6])
factor_twsvr<-c("PSO-RF-GSA-TWSVR","PSO-RF-TWSVR","PSO-XGBoost-TWSVR",
                "WOA-RF-GSA-TWSVR","WOA-RF-TWSVR","WOA-XGBoost-TWSVR")
#col_twsvr<-c("#6B238E","#5F9F9F","#9F5F9F","#FF2400","#FF7F00","#BC1717")
legend("bottomright",factor_twsvr,text.col = col_twsvr,lwd=4,col = col_twsvr,pch = c(0:5))

plot(c(1:15),pso_twsvr_6[1:15],type = "S",xlab = "Iteration",ylab = "Fitnes value",
     ylim=c(0.88,0.955),lwd=8,col=col_lssvr[1])
lines(c(1:100),pso_twsvr_11,lwd=8,col=col_lssvr[2],type = "S")
lines(c(1:100),pso_twsvr_27,lwd=8,col=col_lssvr[3],type = "S")
lines(c(1:100),woa_twsvr_6,lwd=8,col=col_lssvr[4],type = "S")
lines(c(1:100),woa_twsvr_11,lwd=8,col=col_lssvr[5],type = "S")
lines(c(1:100),woa_twsvr_27,lwd=8,col=col_lssvr[6],type = "S")
points(c(1:100),pso_twsvr_6,pch=0,lwd=4,col=col_lssvr[1])
points(c(1:100),pso_twsvr_11,pch=1,lwd=4,col=col_lssvr[2])
points(c(1:100),pso_twsvr_27,pch=2,lwd=4,col=col_lssvr[3])
points(c(1:100),woa_twsvr_6,pch=3,lwd=4,col=col_lssvr[4])
points(c(1:100),woa_twsvr_11,pch=4,lwd=4,col=col_lssvr[5])
points(c(1:100),woa_twsvr_27,pch=5,lwd=4,col=col_lssvr[6])







pso_twsvr_6 <- data_twsvr$psotwsvr6[1:20]
pso_twsvr_11 <- data_twsvr$psotwsvr11[1:20]
pso_twsvr_27 <- data_twsvr$psotwsvr27[1:20]
woa_twsvr_6 <- data_twsvr$woatwsvr6[1:20]
woa_twsvr_11 <- data_twsvr$woatwsvr11[1:20]
woa_twsvr_27 <- data_twsvr$woatwsvr27[1:20]
plot(c(1:20),pso_twsvr_6,type = "S",xlab = "Iteration",ylab = "Fitnes value",
     ylim=c(0.88,0.96),lwd=8,col="#6B238E")
lines(c(1:20),pso_twsvr_11,lwd=8,col="#5F9F9F",type = "S")
lines(c(1:20),pso_twsvr_27,lwd=8,col="#9F5F9F",type = "S")
lines(c(1:20),woa_twsvr_6,lwd=8,col="#FF2400",type = "S")
lines(c(1:20),woa_twsvr_11,lwd=8,col="#FF7F00",type = "S")
lines(c(1:20),woa_twsvr_27,lwd=8,col="#BC1717",type = "S")

pso_svr_6 <- data_svr$psosvr6[1:20]
pso_svr_11 <- data_svr$psosvr11[1:20]
pso_svr_27 <- data_svr$psosvr27[1:20]
woa_svr_6 <- data_svr$woasvr6[1:20]
woa_svr_11 <- data_svr$woasvr11[1:20]
woa_svr_27 <- data_svr$woasvr27[1:20]
plot(c(1:20),pso_svr_6,type = "S",xlab = "Iteration",ylab = "Fitnes value",
     ylim=c(0.65,0.92),lwd=8,col="#6B238E")
lines(c(1:20),pso_svr_11,lwd=8,col="#5F9F9F",type = "S")
lines(c(1:20),pso_svr_27,lwd=8,col="#9F5F9F",type = "S")
lines(c(1:20),woa_svr_6,lwd=8,col="#FF2400",type = "S")
lines(c(1:20),woa_svr_11,lwd=8,col="#FF7F00",type = "S")
lines(c(1:20),woa_svr_27,lwd=8,col="#BC1717",type = "S")

pso_lssvr_6 <- data_lssvr$psolssvr6[1:20]
pso_lssvr_11 <- data_lssvr$psolssvr11[1:20]
pso_lssvr_27 <- data_lssvr$psolssvr27[1:20]
woa_lssvr_6 <- data_lssvr$woalssvr6[1:20]
woa_lssvr_11 <- data_lssvr$woalssvr11[1:20]
woa_lssvr_27 <- data_lssvr$woalssvr27[1:20]
par(cex=1.3)
plot(c(1:20),pso_lssvr_6,type = "S",xlab = "Iteration",ylab = "Fitnes value",
     ylim=c(0.85,0.92),lwd=5,col="#6B238E")
lines(c(1:20),pso_lssvr_11,lwd=5,col="#5F9F9F",type = "S")
lines(c(1:20),pso_lssvr_27,lwd=5,col="#9F5F9F",type = "S")
lines(c(1:20),woa_lssvr_6,lwd=5,col="#FF2400",type = "S")
lines(c(1:20),woa_lssvr_11,lwd=5,col="#FF7F00",type = "S")
lines(c(1:20),woa_lssvr_27,lwd=5,col="#BC1717",type = "S")