library(pscl)
library(MASS)
library(boot)

user_data <- read.csv("/Users/georgeberry/Documents/info6850/observation/results/user_1_out.csv")
venue_data <- read.csv("/Users/georgeberry/Documents/info6850/observation/results/venue_1_out.csv")

user_data[,"kmfromvenue"] <- user_data[,"kmfromvenue"]/1000

m1 <- zeroinfl(egocheckin ~ components + triangles + kcore | kmfromvenue, data = user_data, dist = "negbin", 
               EM = TRUE)

m2 <- glm.nb(egocheckin ~ components + triangles + kcore, data = user_data)

vuong(m1, m2)


m3 <- zeroinfl(egocheckin ~ components + triangles + kcore | kmfromvenue, data = venue_data, dist = "negbin", 
               EM = TRUE)