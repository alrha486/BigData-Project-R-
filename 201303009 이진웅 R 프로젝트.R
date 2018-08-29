install.packages("caret")
install.packages("foreach")
install.packages("party")
install.packages("DEoptim")
install.packages("Hmisc")
install.packages("ellipse")
install.packages("mlbench")
install.packages("lattice")
install.packages("corrgram")
library(mlbench)
library(corrgram)
library(lattice)
library(caret)
library(Hmisc)
library(ellipse)
library(rpart)
library(DEoptim)
library(party)
library(foreach)
library(dplyr)
library(magrittr)
library(ggplot2)
library(corrgram)
library(gridExtra) 
library(pscl)
library(corrplot) 
library(tree)
library(rpart)



gun <- read.csv("C:/Users/kslbs/Desktop/guns1.csv", header = T, stringsAsFactors = FALSE) 
#read.csv로 csv파일 불러오기 


gun <- gun[, !names(gun) %in% c("X","hispanic")]
#불필요 컬럼 제거

gun <- na.omit(gun)
#gun 데이터에 결측치 제거

gun$sex <- as.factor(gun$sex)
gun$police <- as.numeric(gun$police)
gun$race <- as.factor(gun$race)
gun$place <- as.factor(gun$place)
gun$intent <- as.factor(gun$intent)
gun$year <- as.numeric(gun$year)
gun$month <- as.numeric(gun$month)
gun$age <- as.numeric(gun$age)
gun$education <- as.numeric(gun$education)
#데이터 타입 지정

gun
#gun 데이터 
str(gun)
#guns 데이터 구조


summary(gun)
#guns데이터의 분포정보
set.seed(137)


install.packages("caret")
library(caret) #패키지 설치
test_idx <- createDataPartition(gun$intent, p=0.1)$Resample1
#Y 값을 고려한 데이터의 분할
gun.test <- gun[test_idx,]
gun.train <- gun[-test_idx,]
nrow(gun.test)
nrow(gun.train)
#test데이터와 train데이터로 분리


prop.table(table(gun.train$intent))
#gun데이터의 사망유형 비유
save(gun, gun.test, gun.train, file="gun.RData")
#파일 저장
createFolds(gun.train$intent, k=10)
#데이터 분리
create_ten_fold_cv <- function() {
  set.seed(137)
  lapply(createFolds(gun.train$intent , k=10), function(idx) {
    return(list(train=gun.train[-idx, ],
                validation=gun.train[idx, ]))
  })
}
#10겹 교차 검증 데이터를 만드는 함수
x <- create_ten_fold_cv()
str(x) 
#훈련데이터와 검증데이터 저장
head(x$Fold01$train)
#부분조회(Fold01$train)
data <- gun.train
#변수 저자
str(gun)
#gun데이터 구조
summary(intent ~ year + month + police + sex + race + education, data = data, method = "reverse")
# 각 변수 값에 따른 사망유형 종류
data.complete <- data[complete.cases(data), ]
# 각 행에 NA값이 하나도 없는지 여부를 테스트해주는 변수
featurePlot(
  data.complete[,sapply(names(data.complete),function(n) { is.numeric(data.complete [, n]) })],
  data.complete [, c("intent")], "ellipse")
#featureplot을 이용한 데이터 시각화
str(gun)
# 원래 표가 나와야되는데 왜 안나오냐 389페이지
mosaicplot(intent ~ race + sex, data = data, color=TRUE, main="guns")
#인종과 성별로 구분한 사망유형(mosaicplot)
par("mar")
par(mar=c(1,1,1,1))
#margin오류 수정 코드


xtabs( ~ intent + race , data=data)
# 사망사유, 인종별 분할표(xtabs)

xtabs( ~ year + intent , data=data)
# 연도, 사망사유별 분할표(xtabs)

xtabs( ~ sex + year , data=data)
# 연도, 성별 별 사망자 수 분할표(xtabs)
xtabs( ~ year+ race , data=data)
# 연도, 인종별 사망자 수 분할표(xtabs)
xtabs(intent == "Suicide" ~ year+ race , data=data)
# 연도, 인종별 사망자 수 분할표(사망 사유가 '자살')
xtabs(race == "White" ~ sex + year , data=data) / xtabs(race == "Black" ~ sex + year , data=data)
# 연도, 성별 별 사망자 수 분할표(백인 사망 수/흑인 사망 수  >>> 남자는 4~5배, 여자는 2~3배 차이)

predicted <- c(1,0,0,1,1)
actual <- c(1,0,0,0,0)
sum(predicted == actual) / NROW(predicted)
# 예측한 값 중 정확히 예측한 값의 비율(정확도 = 0.6)

m <- rpart(intent ~ year + month + police + sex + age + race + place + education, data=gun.train)
p <- predict(m, newdata = gun.train, type = "class")
head(p)
# rpart 모델만듬


folds <- create_ten_fold_cv()
rpart_result <- foreach(f=folds) %do% {
  model_rpart <- rpart(intent ~ year + month + police + sex + age + race + place + education,
                       data=f$train)
  predicted <- predict(model_rpart, newdata=f$validation, type="class")
  return(list(actual=f$validation$intent, predicted=predicted))}
# folds 전체에 대한 결과를 리스트로 묶어서 변수에 저장

head(rpart_result)
#변수 확인

evaluation <- function(lst) {
  accuracy <- sapply(lst, function(one_result) { 
    return(sum(one_result$predicted == one_result$actual)
           / NROW(one_result$actual)) 
    })
  print(sprintf("MEAN +/- SD: %.3f +/- %.3f", mean(accuracy), sd(accuracy)))
  return(accuracy)
}
#평균과 표준편차를 계산한 뒤 Accuracy의 벡터를 결과로 반호
evaluation(rpart_result)
rpart_accuracy <- evaluation(rpart_result)
#rpart 모델의 성능 : 82.3%, 오차범위 : 0.003

ctree_result <- foreach(f=folds) %do% {
  model_ctree <- ctree(intent ~ year + month + police + sex + age + race + place + education,
                       data=f$train)
  predicted <- predict(model_ctree , newdata=f$validation, type="response") 
  return(list(actual=f$validation$intent , predicted=predicted))
}

str(gun)
#ctree :type에 response를 지정해야 class가 반환
ctree_accuracy <- evaluation(ctree_result)
#cpart 모델의 성능: 83.7%, 오차범위 : 0.002(rpart보다 미세하게 높다)

plot(density(rpart_accuracy), main="rpart VS ctree", xlim=c(0.8,0.85))
lines(density(ctree_accuracy), col="red", lty="dashed",xlim=c(0.8,0.85))
# rpart 모델과 ctree모델의 정확도 비교

gun <- transform(gun, age = ifelse(age < 20, "15",
                            ifelse(age >=20 & age < 30, "25",
                            ifelse(age >= 30 & age < 40, "35",
                            ifelse(age >= 40 & age < 50, "45",
                            ifelse(age >= 50 & age < 60, "55",
                           ifelse(age >= 60 & age < 70, "65","75")))))))
#age를 범주형 변수로 바꾸었을 경우 각 모델별 성능을 비교하기위해 바꾸었음
                                          
sv <- subset ( gun , race == " White " | race == " Black " )
sv$race <- factor ( sv$ race )
boxplot(gun$intent ~ gun$race, data=sv, notch = TRUE) 


corrg <-cor (gun[,c("intent","police","sex","age","race","education")])
#상관계수 비교할 열만 추출하여 변수에 저장
corrgram(corrg,type = "corr",upper.panel = panel.conf)
#corr상관계수 그래프 그리기