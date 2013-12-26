library(pscl)
library(MASS)
library(boot)
library(stargazer)

user_data1 <- read.csv("/Users/georgeberry/Documents/user_1_out.csv")
user_data2 <- read.csv("/Users/georgeberry/Documents/user_2_out.csv")

remove_outliers <- function(x, na.rm = TRUE) {
  qnt <- quantile(x, probs=c(.25, .75), na.rm = na.rm)
  H <- 1.5 * IQR(x, na.rm = na.rm)
  y <- x
  y[x < (qnt[1] - H)] <- NA
  y[x > (qnt[2] + H)] <- NA
  y
}

neg_bn <- function(x){
  
  print(nrow(x))
  x[,'components'] <- x[,'components'] - x[,'counts']
  x[,'neighbors'] <- x[,'neighbors']/x[,'counts']
  x[,"kmfromvenue"] <- x[,"kmfromvenue"]/x[,'counts']
  x[,"kmfromvenue"] <- log(x[,"kmfromvenue"])
  x[,"triangles"] <- log1p(x[,"triangles"])
  x[,"components"] <- log1p(x[,"components"])
  x[,"counts"] <- log1p(x[,"counts"])
  x[,"neighbors"] <- log1p(x[,"neighbors"])
  x[,"kcore"] <- log1p(x[,"kcore"])
  
  
  #x[,"kmfromvenue"] <- remove_outliers(x[,"kmfromvenue"])
  x[,"components"] <- remove_outliers(x[,"components"])
  x[,"triangles"] <- remove_outliers(x[,"triangles"])
  x <- x[complete.cases(x),]
  
  print(nrow(x))
  
  
  
  user_samp <- x[sample(nrow(x), 5000), c('egocheckin','components','triangles','counts','kmfromvenue','degree','egoperiodcheckins','kcore','venueperiodcheckins','neighbors')]
  
  print(cor(user_samp))
  
  m <- glm.nb(egocheckin ~ components + triangles + counts + components*triangles + egoperiodcheckins + neighbors + kmfromvenue, data = user_samp)
  
  m
}

m1 <- neg_bn(user_data1) 
m2 <- neg_bn(user_data2)

stargazer(m1, m2)

vuong(m1,m3)