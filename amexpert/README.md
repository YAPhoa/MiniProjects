# Analytics Vidhya AMexpert 2019

Link : https://datahack.analyticsvidhya.com/contest/amexpert-2019-machine-learning-hackathon/ 

## Project Description

We are provided with the following information :
* User Demographic Details
* Campaign and coupon Details
* Product details
* Customer Previous transactions

Based on previous transaction & performance data from the last 18 campaigns, **predict the probability for the next 10 campaigns in the test set for each coupon and customer combination, whether the customer will redeem the coupon or not?**

## Dataset Description

Below is the schema of the dataset for this challenge
![Dataset Schema](https://github.com/YAPhoa/MiniProjects/blob/master/amexpert/assets/schema.png)
The files that were given and some brief description :
* **train.csv**: Train data containing the coupons offered to the given customers under the 18 campaigns
* **campaign_data.csv**: Campaign information for each of the 28 campaigns
* **coupon_item_mapping.csv**: Mapping of coupon and items valid for discount under that coupon
* **customer_demographics.csv**: Customer demographic information for some customers
* **customer_transaction_data.csv**: Transaction data for all customers for duration of campaigns in the train data
* **test.csv**: Contains the coupon customer combination for which redemption status is to be predicted

To summarise the entire process:

* Customers receive coupons under various campaigns and may choose to redeem it.
* They can redeem the given coupon for any valid product for that coupon as per coupon item mapping within the duration between campaign start date and end date
* Next, the customer will redeem the coupon for an item at the retailer store and that will reflect in the transaction table in the column coupon_discount.

## Approaches
The main challenge for this competition is feature engineering, since the data is provided are scattered in different part of dataset. The model that I used are catboost and xgboost(with dart and without dart). To optimize I am using hyperopt which implement bayesian optimization.

## Evaluation
Metrics for this competition is using AUC-ROC curve.
Local Evaluation using 8-fold group cross validation. Grouping based on customer (customer_id).
Current local CV is around 0.91 for optimized model.

## Comments
My current public leaderboard standing is top 15% for this challenge
I have yet to find the "magic feature" that could boost my score
