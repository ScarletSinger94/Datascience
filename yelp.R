library (ggplot2)
business_data = read.csv(file="Documents/Datascience/yelp_academic_dataset_business.json.csv")
ggplot(business_data) + geom_bar(aes(x=state), fill="gray")
cor(business_data$review_count,business_data$stars)#yas
ggplot(business_data,	aes(stars,review_count,colour =	businessCluster$cluster))	+	geom_point()

ggplot(business_data) + geom_bar(aes(x=stars), fill="gray")
ggplot(data=business_data,aes(x=factor(1),fill=factor(state))) + geom_bar(width=1)+coord_polar(theta="y")
user_data = read.csv(file="Documents/Datascience/yelp_academic_dataset_user.json.csv")
user_votes = user_data[,c("cool_votes","funny_votes","useful_votes")]
user_pop = user_data[,c("average_stars","review_count","fans")]
user_vote = user_data[,c("cool_votes","funny_votes","useful_votes","average_stars","review_count","fans")]
cor(user_votes)
cor(user_vote)

cor(user_data$review_count,user_data$fans)#yas
# cor(user_data$review_count,user_data$cool_votes)
# cor(user_data$review_count,user_data$funny_votes)
cor(user_data$review_count,user_data$useful_votes)#best
cor(user_data$average_stars,user_data$fans)#sad
# cor(user_data$average_stars,user_data$useful_votes)#sad
# cor(user_data$average_stars,user_data$funny_votes)#sad
# cor(user_data$average_stars,user_data$cool_votes)#sad
cor(user_data$review_count,user_data$average_stars)#also sad
cor(user_data$useful_votes,user_data$fans)#high
# cor(user_data$cool_votes,user_data$fans)
# cor(user_data$funny_votes,user_data$fans)

my.lm= lm(useful_votes~ review_count + fans, data=user_data)
my.lm= lm(review_count~ fans + average_stars, data=user_data)
my.lm= lm(useful_votes~ review_count + fans, data=user_data)

coeffs=coefficients(my.lm)
coeffs

ggplot(user_data)+geom_bar(aes(x=review_count), fill="gray")
ggplot(user_data)+geom_bar(aes(x=average_stars), fill="gray")
ggplot(user_data)+geom_bar(aes(x=fans), fill="gray")

set.seed(20)
userCluster <- kmeans(user_data[,	c(3,11)],	3,	nstart =	20)
businessCluster <- kmeans(business_data[,	c(9,10)],	3,	nstart =	20)

#ggplot(user_data,	aes(review_count,useful_votes,colour =	userCluster$cluster))	+	geom_point()
#ggplot(user_data,	aes(cool_votes,useful_votes,colour =	userCluster$cluster))	+	geom_point()
#ggplot(user_data,	aes(fans,useful_votes,colour =	userCluster$cluster))	+	geom_point()
ggplot(user_data,	aes(review_count,useful_votes,colour =	userCluster$cluster))	+	geom_point()
#ggplot(user_data,	aes(fans,average_stars,colour =	userCluster$cluster))	+	geom_point()
