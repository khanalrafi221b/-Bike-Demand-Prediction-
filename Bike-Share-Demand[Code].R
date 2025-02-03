rm(list = ls()) # Clearing the environment

# Load necessary libraries
library(dplyr)
library(ggplot2)
library(corrplot)
library(caret)
library(forecast)
library(cluster)
library(factoextra)
library(dendextend)


# Reading the dataset
daily_data <- read.csv("day.csv")
hourly_data <- read.csv("hour.csv")

# Ensure date formats are consistent
daily_data$dteday <- as.Date(daily_data$dteday)
hourly_data$dteday <- as.Date(hourly_data$dteday)

# Merging datasets on 'dteday'
merged_data <- merge(daily_data, hourly_data, by = "dteday")
View(merged_data)

# Merging datasets and resolving overlapping variables
combined_data <- merge(daily_data, hourly_data, by = "dteday", suffixes = c(".daily", ".hourly"))
View(combined_data)

# Resolve overlapping variables, prefer hourly data over daily
columns_to_resolve <- names(daily_data)[names(daily_data) %in% names(hourly_data)]
for (col in columns_to_resolve) {
  if (col != "dteday") {  
    combined_data[[col]] <- combined_data[[paste0(col, ".hourly")]]
    combined_data[[paste0(col, ".daily")]] <- NULL
    combined_data[[paste0(col, ".hourly")]] <- NULL
  }
}
View(combined_data)

# Save the combined data to a CSV file
write.csv(combined_data, "combined_data.csv", row.names = FALSE)

final_data <- read.csv("combined_data.csv")
View(final_data)
# Pre-Processing and Exploratory Data Analysis

# General overview
summary(final_data)
str(final_data)

# Data Cleaning

# Plotting the total counts of rentals per day
ggplot(final_data, aes(x = dteday, y = cnt)) +
  geom_line(group = 1, color = "blue") +
  labs(title = "Daily Bike Rentals", x = "Date", y = "Total Rentals")

# Boxplot of hourly rentals by season
ggplot(final_data, aes(x = factor(season), y = cnt, fill = factor(season))) +
  geom_boxplot() +
  labs(title = "Hourly Bike Rentals by Season", x = "Season", y = "Hourly Rentals")

# Histogram of temperatures (considering hourly temperature is more relevant)
ggplot(final_data, aes(x = temp)) +
  geom_histogram(bins = 30, fill = "red", color = "black") +
  labs(title = "Distribution of Temperature", x = "Temperature", y = "Frequency")

# Checking usage patterns by hour of the day
ggplot(final_data, aes(x = hr, y = cnt)) +
  geom_line(stat = "summary", fun = "mean", color = "darkgreen") +
  labs(title = "Average Bike Rentals by Hour", x = "Hour of Day", y = "Average Rentals")

# Extract only numeric columns from combined_data
numeric_data <- final_data %>%
  select_if(is.numeric)

# Calculate the correlation matrix
cor_matrix <- cor(numeric_data, use = "complete.obs")

# Print the correlation matrix
print(cor_matrix)

# Plot the correlation matrix
corrplot(cor_matrix, method = "color", type = "full", tl.col = "black", tl.cex = 0.8)

# Linear regression model
# Read and preprocess the data and remove instant and dteday columns(not needed for analysis)
hour.df <- final_data[, -c(1, 3)]  # Drop instant and dteday columns

# Convert season and year to factors
hour.df$season <- factor(hour.df$season, levels = c(1, 2, 3, 4), labels = c("Winter", "Spring", "Summer", "Fall"))
hour.df$yr <- factor(hour.df$yr, levels = c(0, 1), labels = c("2011", "2012"))

# Partition data into training and validation sets
set.seed(2)
train.index <- sample(c(1:dim(hour.df)[1]), dim(hour.df)[1] * 0.6)
train.df <- hour.df[train.index, ]
valid.df <- hour.df[-train.index, ]

train.df$hr <- as.factor(train.df$hr)
valid.df$hr <- as.factor(valid.df$hr)

train.df$weekday <- as.factor(train.df$weekday)
valid.df$weekday <- as.factor(valid.df$weekday)

train.df$mnth <- as.factor(train.df$mnth)
valid.df$mnth <- as.factor(valid.df$mnth)

train.df$weathersit <- as.factor(train.df$weathersit)
valid.df$weathersit <- as.factor(valid.df$weathersit)


# Fit linear regression model
linear_model <- lm(cnt ~ temp + hum + windspeed + season + weathersit + holiday + hr*workingday + mnth, data = train.df)

# Summary of the linear regression model
summary(linear_model)

# Predict on validation data
linear_pred <- predict(linear_model, valid.df, type="response")

# Evaluate model performance using RMSE
rmse <- sqrt(mean((linear_pred - valid.df$cnt)^2))
cat("Root Mean Squared Error (RMSE):", rmse)

# Model for casual users
casual_model <- lm(casual ~ temp + hum + windspeed + season + weathersit + holiday +
                     hr * workingday + weekday + season:temp + hum:windspeed, data = train.df)

# Summary of the model for casual users
summary(casual_model)

# Model for registered users
registered_model <- lm(registered ~ temp + hum + windspeed + season + weathersit + holiday +
                         hr * workingday + weekday + season:temp + hum:windspeed, data = train.df)

# Summary of the model for registered users
summary(registered_model)

casual_pred <- predict(casual_model, valid.df)

registered_pred <- predict(registered_model, valid.df)

casual_rmse <- sqrt(mean((casual_pred - valid.df$casual)^2))
cat("RMSE for Casual Model:", casual_rmse, "\n")

registered_rmse <- sqrt(mean((registered_pred - valid.df$registered)^2))
cat("RMSE for Registered Model:", registered_rmse, "\n")

# Aggregate hourly data to daily level to create time series data
daily_totals <- final_data %>%
  group_by(dteday) %>%
  summarise(total_rentals = sum(cnt))

# Create a time series object
ts_data <- ts(daily_totals$total_rentals, frequency = 365, start = c(2011, 1))

# Plot the time series data
plot(ts_data, main = "Time Series of Daily Bike Rentals", ylab = "Total Rentals", xlab = "Time")

# Decompose the time series into trend, seasonal, and random components
decomposed <- decompose(ts_data)
plot(decomposed)

plot.ts(ts_data)
decomposed_ts <- stl(ts_data, s.window = "periodic")
plot(decomposed_ts)
# Fit an ARIMA model for forecasting
fit <- auto.arima(ts_data)

# Forecast the next 30 days
forecast_data <- forecast(fit, h = 30)
plot(forecast_data)

# Summary of the ARIMA model
summary(fit)

# clustering
clustering_data <- final_data %>%
  select(casual, registered, temp, hum, windspeed, workingday, holiday, season) %>%
  na.omit()

clustering_data_scaled <- scale(clustering_data)

set.seed(123)
fviz_nbclust(clustering_data_scaled, kmeans, method = "wss",k.max=21)
kmeans_result <- kmeans(clustering_data_scaled, centers = 5)  # 5 clusters as an example
clustering_data$cluster <- as.factor(kmeans_result$cluster)

# View the cluster assignment
table(clustering_data$cluster)

ggplot(clustering_data, aes(x = casual, y = registered, color = cluster)) +
  geom_point() +
  labs(title = "Clustering of Casual vs Registered Users", x = "Casual Users", y = "Registered Users")

cluster_summary <- clustering_data %>%
  group_by(cluster) %>%
  summarise(across(c(casual, registered, temp, hum, windspeed), mean))
print(cluster_summary)

#hclust

dist_matrix <- dist(clustering_data_scaled)
hclust_result <- hclust(dist_matrix, method = "ward.D2")
hc2 <- hclust(dist_matrix, method = "complete")
plot(hclust_result)
plot(hc2)

clustering_data$cluster <- cutree(hclust_result, k = 5)

dend_wardSquare <- as.dendrogram(hclust_result)
dend_complete <- as.dendrogram(hc2)





