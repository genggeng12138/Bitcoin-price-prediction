library(readxl)
col_lssvr<-c("#00008B","#8B2323","#006400","#EEAD0E","#458B00","#483D8B")
data_lssvr <- read_excel("C:/Users/mjgeng/Desktop/比特币及其相关数据/适应度曲线/lssvr.xlsx")
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
factor_lssvr<-c("PSO-XGBoost-LSSVR","PSO-RF-LSSVR","PSO-Complete-LSSVR",
                "WOA-XGBoost-LSSVR","WOA-RF-LSSVR","WOA-Complete-LSSVR")

legend("bottomright",factor_lssvr,text.col = col_lssvr,lwd=4,col = col_lssvr,pch = c(0:5))


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







data_svr <- read_excel("C:/Users/mjgeng/Desktop/比特币及其相关数据/适应度曲线/svr.xlsx")
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
factor_svr<-c("PSO-XGBoost-SVR","PSO-RF-SVR","PSO-Complete-SVR",
              "WOA-XGBoost-SVR","WOA-RF-SVR","WOA-Complete-SVR")
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


data_twsvr <- read_excel("C:/Users/mjgeng/Desktop/比特币及其相关数据/适应度曲线/twsvr.xlsx")
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
factor_twsvr<-c("PSO-XGBoost-TWSVR","PSO-RF-TWSVR","PSO-Complete-TWSVR",
                "WOA-XGBoost-TWSVR","WOA-RF-TWSVR","WOA-Complete-TWSVR")

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





