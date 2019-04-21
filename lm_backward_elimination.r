# checking the availablity of libraries, if not available, then install
packages <- c("tm", "RTextTools", "e1071", "naivebayes","dplyr","caret","doMC", "caTools")
if (length(setdiff(packages, rownames(installed.packages()))) > 0) {
  install.packages(setdiff(packages, rownames(installed.packages())))
}
# Load required libraries
library(tm)
library(RTextTools)
library(e1071)
library(naivebayes)
library(dplyr)
library(caret)
library(MASS)
library(caTools)
# Library for parallel processing
library(doMC)
registerDoMC(cores=detectCores())  # Use all available cores



# Importing the dataset
dataset = read.csv('social_network_ads.csv')
dataset = dataset[2:5]

# Encoding the target feature as factor
dataset$Purchased = factor(dataset$Purchased, levels = c(0, 1))
dataset$Gender = factor(dataset$Gender,
                        levels = c('Male', 'Female'),
                        labels = c(1, 2))

dataset$Gender = as.numeric(dataset$Gender)
# Splitting the dataset into the Training set and Test set
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling 
training_set[-4] = scale(training_set[-4])
test_set[-4] = scale(test_set[-4])

classifier = naiveBayes(x = training_set[-4],
                        y = training_set$Purchased)

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-4])

# Making the Confusion Matrix
cm = table(test_set[, 4], y_pred)
(accuracy = sum(diag(cm))/sum(cm)*100)


NaiveBayesModel <- naive_bayes(training_set$Purchased ~ ., data = training_set[-4])
Pred_class <- predict(NaiveBayesModel, newdata = test_set[-4])
tab <- table(Pred_class, test_set[,4])
(accuracy <- sum(diag(tab))/sum(tab))

lda.fit <- lda(training_set$Purchased ~., data=training_set[-4])
lda.pred <- predict(lda.fit, test_set[-4])
lda.Pred_class <- lda.pred$class
tab <- table(lda.Pred_class, test_set$Purchased)
(accuracy <- sum(diag(tab))/sum(tab))

qda.fit <- qda(training_set$Purchased ~., data=training_set[-4])
qda.pred <- predict(qda.fit, test_set[-4])
qda.Pred_class <- qda.pred$class
tab <- table(qda.Pred_class, test_set$Purchased)
(accuracy <- sum(diag(tab))/sum(tab))


# NLP - sentiment classification using Naive Bayes

df<- read.csv("movie-pang02.csv", stringsAsFactors = FALSE)
glimpse(df)

# Randomise dataset
set.seed(1)
df <- df[sample(nrow(df)), ]
glimpse(df)

# Convert the 'class' variable from character to factor.
df$class <- as.factor(df$class)

corpus <- Corpus(VectorSource(df$text))
corpus.clean <- corpus %>%
  tm_map(content_transformer(tolower)) %>% 
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords(kind="en")) %>%
  tm_map(stripWhitespace)
dtm <- DocumentTermMatrix(corpus.clean)
# Inspect the dtm
inspect(dtm[40:50, 1000:1010])

# partitioning data
df.train <- df[1:1500,]
df.test <- df[1501:2000,]

dtm.train <- dtm[1:1500,]
dtm.test <- dtm[1501:2000,]

corpus.clean.train <- corpus.clean[1:1500]
corpus.clean.test <- corpus.clean[1501:2000]


# feature selection
dim(dtm.train)

# restrict the DTM to use only the frequent words using the ‘dictionary’ option
fivefreq <- findFreqTerms(dtm.train, 5)
length((fivefreq))

# Use only 5 most frequent words (fivefreq) to build the DTM
dtm.train.nb <- DocumentTermMatrix(corpus.clean.train, control=list(dictionary = fivefreq))
dim(dtm.train.nb)

dtm.test.nb <- DocumentTermMatrix(corpus.clean.test, control=list(dictionary = fivefreq))
dim(dtm.test.nb)


# Function to convert the word frequencies to yes (presence) and no (absence) labels
convert_count <- function(x) {
  y <- ifelse(x > 0, 1,0)
  y <- factor(y, levels=c(0,1), labels=c("No", "Yes"))
  y
}

# Apply the convert_count function to get final training and testing DTMs
trainNB <- apply(dtm.train.nb, 2, convert_count)
testNB <- apply(dtm.test.nb, 2, convert_count)

# Train the classifier
system.time( classifier <- naiveBayes(trainNB, df.train$class, laplace = 1) )
system.time( classifier_nv <- naive_bayes(df.train$class ~ ., data = as.data.frame(trainNB), laplace = 1))


# Use the NB classifier we built to make predictions on the test set.
system.time( pred <- predict(classifier, newdata=testNB) )
system.time( pred_2 <- predict(classifier_nv, newdata=testNB) )

# Create a truth table by tabulating the predicted class labels with the actual class labels 
table("Predictions"= pred,  "Actual" = df.test$class )
table("Predictions"= pred_2,  "Actual" = df.test$class )

# Prepare the confusion matrix
conf.mat <- confusionMatrix(pred, df.test$class)
conf.mat
conf.mat_2 <- confusionMatrix(pred_2, df.test$class)
conf.mat_2
