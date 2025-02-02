# -Bike-Demand-Prediction-
This project explores the key factors influencing bike-sharing demand using machine learning and statistical modeling in R. By analyzing a two-year dataset from the UCI repository, I uncover demand patterns, user segmentation, and environmental influences affecting bike rentals.
---

## 📌 Project Overview
- **Predictive Modeling**: Developed **multiple linear regression models** and **ARIMA time-series forecasting** to predict bike rental demand.
- **Exploratory Data Analysis (EDA)**: Visualized seasonal patterns, commuting behaviors, and environmental impacts on demand.
- **Clustering Analysis**: Implemented **K-means and hierarchical clustering** to segment users based on rental behavior.
- **Business Insights**: Formulated **dynamic pricing, fleet optimization, and marketing strategies** based on findings.

---

## 📊 Dataset
This project utilizes a **bike-sharing dataset from the UCI Machine Learning Repository**, containing **daily and hourly rental data** over two years. The dataset includes:
- **Temporal variables**: Hour, day, month, year
- **Seasonal indicators**: Summer, winter, fall, spring
- **Environmental factors**: Temperature, humidity, wind speed
- **User segmentation**: Casual vs. registered riders

🔗 **Dataset Source**: [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset)

---

## ⚙️ Technologies & Tools
- **Programming Language**: R
- **Libraries Used**:  
  - `dplyr` – Data manipulation  
  - `ggplot2` – Data visualization  
  - `forecast` – Time-series forecasting  
  - `caret` – Machine learning  
  - `corrplot` – Correlation analysis  

---

## 📈 Key Findings
### 🔹 Demand Patterns
- **Temperature & Seasonality**: Rentals peak between **15°C-27°C**, with higher usage in **spring and summer**.
- **Hourly Trends**: Two demand peaks exist—**8 AM (morning commute)** and **6 PM (evening commute)**.
- **Casual vs. Registered Users**: **Registered users dominate** overall rentals, while casual users prefer weekends.

### 🔹 Predictive Models
- **Regression Models**: Temperature is the strongest predictor of demand. High humidity negatively impacts rentals.
- **Time Series Analysis**: ARIMA models predict rental demand with a strong upward trend in ridership over time.
- **Clustering Analysis**: Segmented users into **five behavioral groups** to optimize promotions and fleet allocation.

---

### 📢 Business Recommendations
- **Dynamic Pricing**: Lower prices during extreme weather; surge pricing during peak hours.
- **Fleet Optimization**: Align maintenance schedules with off-peak hours.
- **Targeted Marketing**: Offer discounts to casual users on weekends; loyalty programs for registered users.

📝 Contributors
[Al Rafi] (@khanalrafi221b)
📧 For questions, contact: khanalrafi221b@gmail.com

