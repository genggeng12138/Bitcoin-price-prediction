library(ggplot2)
library(RColorBrewer)
library(readxl)
install.packages("ggpattern")
library(ggpattern)
#生成模拟的数据
Group <- c(rep("Complete Features(22)",6),rep("XGBoost Features(10)",6),rep("RF Features(7)",6)) #组别变量
a<-c("WOA-SVR","WOA-LSSVR","WOA-TWSVR","PSO-SVR","PSO-LSSVR","PSO-TWSVR")
Attribute <- c(a,a,a) #每个组别的属性
ACC <- c(0.9022,0.8868,0.9484,0.8964,0.8835,0.9484,
         0.9161,0.9041,0.9547,0.9074,0.9041,0.9239,
         0.8067,0.9015,0.9502,0.8041,0.8994,0.9491) 
#不同组别不同属性的取值
Data <-data.frame(Group=Group,Attribute=Attribute,ACC=ACC) 
#生成模拟数据对应的数据框
Data$ACC <- round(Data$ACC,digits = 4) #保留四位小数
#因子化属性，调整levels中属性的顺序可相应修改不同属性显示时在x轴中的排序
Data$Attribute <-factor(Data$Attribute,levels=a) 

P <- ggplot(data=Data,aes(x=Attribute,y=ACC,fill=Group))+ #创建绘图的基本图层，"data="指定需要显示数据的数据框，"x="指定x轴的取值，"y="指定Y轴的取值,"fill="指定图填充的颜色
  geom_bar(stat = "identity",position = "dodge")+ #绘制条形图，position = "dodge"设置条形图不堆叠显示
  scale_fill_manual(values = c("#98F5FF","#FFD39B","#BCEE68")) +
  scale_pattern_manual(values = c("none","stripe","stripe")) +
  #scale_fill_manual(values = brewer.pal(12, "Paired")[c(1,12)])+ #设置填充的颜色
  geom_text(aes(label=ACC),position=position_dodge(width = 1),vjust=-0.5,color="black",size=5) + #在条形图上方0.5处(vjust=-0.5)以黑色(color="black")字体大小为5显示(size=5)数值大小
  theme_bw()+ #让刻度线和网格线的颜色更协调一些
  theme(axis.text.x=element_text(colour="black",family="Times",size=18), #设置x轴刻度标签的字体显示倾斜角度为15度，并向下调整1(hjust = 1)，字体簇为Times大小为20
        axis.text.y=element_text(family="Times",size=18,face="plain"), #设置y轴刻度标签的字体簇，字体大小，字体样式为plain
        axis.title.y=element_text(family="Times",size = 24,face="plain"), #设置y轴标题的字体属性
        panel.border = element_blank(),axis.line = element_line(colour = "black",size=1), #去除默认填充的灰色，并将x=0轴和y=0轴加粗显示(size=1)
        legend.text=element_text(face="italic", family="Times", colour="black",  #设置图例的子标题的字体属性
                                 size=16),
        legend.title=element_text(face="italic", family="Times", colour="black", #设置图例的总标题的字体属性
                                  size=16))+ 
  ylab("EVS")+xlab("") #设置x轴和y轴的标题
P
jpeg(file = "results_Value.jpg",width =1800,height = 3000,units = "px",res =300) #结果保存保存为results_Value.jpg，宽高为4800*3000像素，分辨率为300dpi
print(P)
dev.off()
?geom_col_pattern()
