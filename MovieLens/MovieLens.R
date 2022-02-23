##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

library(tidyverse)
library(caret)
library(data.table)
library(ggplot2)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

##########################################################
# Data Exploration
##########################################################

# General overview of the dataset:
head(edx)

# To understand the structure of the data.
str(edx)

# Number of different movies are in the edx dataset.
n_distinct(edx$movieId)

# Number of different users are in the edx dataset.
n_distinct(edx$userId)

# Number of movie ratings in each of the following genres in the edx dataset
genres <- c("Drama", "Comedy", "Thriller", "Romance")
sapply(genres, function(g) {
  sum(str_detect(edx$genres, g))
})

# Movies with most ratings:
edx %>% 
  group_by(movieId, title) %>% 
  summarize(count = n()) %>% 
  arrange(desc(count))

# Histogram representing the top rated movies
edx %>% 
  group_by(title) %>% 
  summarize(count = n()) %>% 
  arrange(desc(count)) %>%
  top_n(10, count) %>% 
  ggplot(aes(count, reorder(title, count))) + 
  geom_bar(stat = "identity") + 
  ggtitle(" Most popular movies") 

# Most common ratings:
edx %>% 
  group_by(rating) %>% 
  summarize(count=n()) %>% 
  top_n(5, count) %>% 
  arrange(desc(count))

# Visualizing the number of occurrence of each rating
edx %>% group_by(rating) %>% 
  summarize(count = n()) %>% 
  ggplot(aes(x = rating, y = count)) + 
  geom_line(color="black") + 
  ggtitle("Number of occurence of each rating") 
# The graph clearly shows that the common rating is 4 and most of the users are giving 
# a positive rating rather than a negative rating


##########################################################
# Residual Mean Squared Error (RMSE)
##########################################################
# Define Residual Mean Squared Error (RMSE)
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}


##########################################################
# Partitioning edx into test set and train set
##########################################################
# Preparing data
# We use the same procedure used to create edx and validation sets.
# Validation set will be 10% of edx data
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train_set <- edx[-test_index,]
temp <- edx[test_index,]

# Make sure userId and movieId in test set are also in train set
test_set <- temp %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Add rows removed from test set back into train set
removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)

rm(test_index, temp, removed)

##########################################################
# Data Cleaning
##########################################################

# We will be using only userID, movieID, rating and title for our model,
# so we are removing timestamp and genres.
train_set <- train_set %>% 
  select(userId, movieId, rating, title)
test_set  <- test_set  %>% 
  select(userId, movieId, rating, title)


##########################################################
# Methods used to train and test the algorithm
##########################################################
# Model 1 : Using Mean
# Model 2 : Using Mean + Movie Effects
# Model 3 : Mean + Movie Effects + User Effects
# Model 4 : Regularization: Movie + User Effects


# Model 1 : Using Mean
# Here we are using only the average of all ratings for making a prediction.
mu_hat <- mean(train_set$rating)

rmse_results <- tibble(Method = "Mean", RMSE = RMSE(test_set$rating, mu_hat))


#  Model 2 : Mean + Movie Effects
# We can augment our previous model by taking movie bias, 
# by adding the term bi to represent average ranking for movie i

# Estimating movie effects
bi <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu_hat))

# Movie effects distribution:
bi %>% 
  ggplot(aes(b_i)) + 
  geom_histogram(bins=10, color = "black") + 
  ggtitle("Movie effects distribution")

# Predicted ratings for Movie effects:
predict_bi <- mu_hat + 
  test_set %>% 
  left_join(bi, by = "movieId") %>% 
  pull(b_i)

# RMSE for Mean + Movie effects
rmse_results <- bind_rows(rmse_results, 
                          tibble(Method = "Mean + Movie effects", 
                                 RMSE = RMSE(test_set$rating, predict_bi)))


#  Model 3 : Mean + Movie Effects + User Effects
# Visualizing the average rating for user u 
# for those that have rated 100 or more movies
train_set %>% 
  group_by(userId) %>% 
  filter(n()>=100) %>% 
  summarize(b_u = mean(rating)) %>% 
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, color = "black")+ 
  ggtitle("Average rating for users")

# Estimate User Effects:
bu <- train_set %>% 
  left_join(bi, by = 'movieId') %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating - mu_hat - b_i))

# Predicted ratings for movie + user effects
predict_bu <- test_set %>% 
  left_join(bi, by="movieId") %>% 
  left_join(bu, by="userId") %>% 
  mutate(pred = mu_hat + b_i + b_u) %>% 
  pull(pred)

# RMSE for Mean + Movie effects + User effects:
rmse_results <- bind_rows(rmse_results, 
                          tibble(Method = "Mean + Movie effects + User effects", 
                                 RMSE = RMSE(test_set$rating, predict_bu)))


# Model 4: Regularization: Movie + User Effects
# λ is a tuning parameter. We can use cross-validation to choose it.
# Generate a sequence of values for lambda ranging from 3 to 6 with 0.25 inc.

lambdas <- seq(3, 6, .25)
# Regularized model, predict ratings and calculate RMSE for each value of lambda
rmses <- sapply(lambdas, function(l){
  
  mu <- mean(train_set$rating)
  
  b_i <- train_set %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- 
    test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(test_set$rating, predicted_ratings))
})

# Assign optimal tuning parameter (lambda)
lambda <- lambdas[which.min(rmses)]

# Minimum RMSE achieved
regularised_rmse <- min(rmses)

# Results table
rmse_results <- bind_rows(rmse_results, 
                          tibble(Method = "Regularised Movie + User effects", 
                                 RMSE = regularised_rmse))


##########################################################
# Final validation: Linear Model with Regularization.
##########################################################

mu <- mean(edx$rating)

# Movie effect (bi)
b_i <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda))

# User effect (bu)
b_u <- edx %>%
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))

# Prediction
predicted_ratings_final <- validation %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

# Results table
rmse_results <- bind_rows(rmse_results,
                    tibble(Method = "Final Regularization (edx vs validation)",
                           RMSE = RMSE(validation$rating, predicted_ratings_final)))


##########################################################
# The RMSE returned by testing the algorithm on the validation set
##########################################################

# Project objective & Final validation RMSE result

final_results <- bind_rows(tibble(Method = "Project objective", 
                                  RMSE = 0.86490), rmse_results)

final_results %>% knitr::kable()
