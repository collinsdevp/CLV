%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lifetimes 
import os

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

os.chdir(r"C:\Users\UCO\Desktop\impl\main_system")

df = pd.read_csv("Online+Retail.csv")

# get rows and columns
df.shape 

df.dtypes
# count empty fields
df.isnull().sum() 
#percentage of missing values
135080/df.shape[0]
df['CustomerID'].nunique()
dfnew = df[(df.Quantity>0) & (df.CustomerID.isnull() == False)]
dfnew.shape
#count the customers in each country
df['Country'].nunique()
dfnew.groupby(['Country'])['CustomerID'].aggregate('count').reset_index().sort_values('CustomerID',ascending=False)
#dfnew.Country.value_counts()[:].plot(kind='bar')
dfnew['amt'] = dfnew['Quantity'] * dfnew['UnitPrice']
dfnew['InvoiceDate'] = pd.to_datetime(dfnew['InvoiceDate']).dt.date
dfnew['InvoiceDate'].min()
dfnew['InvoiceDate'].max()

from lifetimes.plotting import *
from lifetimes.utils import *

modeldata = summary_data_from_transaction_data(dfnew, 'CustomerID', 'InvoiceDate', monetary_value_col='amt', observation_period_end='2011-12-9')
modeldata.head()
#modeldata['frequency'].plot(kind='hist', bins=50)
#print(modeldata['frequency'].describe())
#percentage of customer with no [repeat] order
print(sum(modeldata['frequency'] == 0)/float(len(modeldata)))
#dfnew[dfnew.CustomerID == 12346.0]

from lifetimes import BetaGeoFitter
#parameter eliminates overfitting and noise and robust
bgf = BetaGeoFitter(penalizer_coef=0.0)
bgf.fit(modeldata['frequency'], modeldata['recency'], modeldata['T'])
print(bgf)

# create frequency recency matrix
from lifetimes.plotting import plot_frequency_recency_matrix

plot_frequency_recency_matrix(bgf) #plot for predict purchase

from lifetimes.plotting import plot_probability_alive_matrix

#probability of being alive
modeldata['Churn_probability'] = bgf.conditional_probability_alive(modeldata['frequency'], modeldata['recency'], modeldata['T'])

#plot for churn or probability of being alive
fig = plt.figure(figsize=(12,8))
plot_probability_alive_matrix(bgf)


# predict number of purchase customer will make
t = 30  #number of days to predict customer will make purchase
modeldata['predicted_num_purchases'] = bgf.conditional_expected_number_of_purchases_up_to_time(t, modeldata['frequency'], modeldata['recency'], modeldata['T'])
modeldata.sort_values(by='predicted_num_purchases').tail(5)


from lifetimes.plotting import plot_period_transactions
#used to validate the model
plot_period_transactions(bgf) 

#another type of model validation
summary_cal_holdout = calibration_and_holdout_data(df, 'CustomerID', 'InvoiceDate',
                                        calibration_period_end='2011-06-08',
                                        observation_period_end='2011-12-9' )   
print(summary_cal_holdout.head())

from lifetimes.plotting import plot_calibration_purchases_vs_holdout_purchases

bgf.fit(summary_cal_holdout['frequency_cal'], summary_cal_holdout['recency_cal'], summary_cal_holdout['T_cal'])
plot_calibration_purchases_vs_holdout_purchases(bgf, summary_cal_holdout)


#predict the number of purchase made with t =days for single customer
t = 30
individual = modeldata.loc[12380]
bgf.predict(t, individual['frequency'], individual['recency'], individual['T'])


from lifetimes.plotting import plot_history_alive
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12,8)) # plot setting width and height
id = 14620  # id = 18074  id = 14606
days_since_birth = 365
sp_trans = df.loc[df['CustomerID'] == id]
plot_history_alive(bgf, days_since_birth, sp_trans, 'InvoiceDate')


# customers who had at least one repeat purchase with us
returning_customers_summary = modeldata[modeldata['frequency']>0]

print(len(returning_customers_summary))
returning_customers_summary.shape

from lifetimes import GammaGammaFitter

ggf = GammaGammaFitter(penalizer_coef = 0)
ggf.fit(returning_customers_summary['frequency'],
        returning_customers_summary['monetary_value'])

print(ggf)

returning_customers_summary = returning_customers_summary[returning_customers_summary['monetary_value']>0]

returning_customers_summary['predicted_avg_sales']=ggf.conditional_expected_average_profit(returning_customers_summary['frequency'],returning_customers_summary['monetary_value'])



# checking the expevred average value and the actual average value in the data to make sure the values are good
print(f"Expected Average sales: {returning_customers_summary['predicted_avg_sales'].mean()}")
print(f"Actual Average sales: {returning_customers_summary['monetary_value'].mean()}")
# The values seem to be fine

#calculating CLV for 1 month
returning_customers_summary['Predicted_CLV'] = ggf.customer_lifetime_value(bgf,
                                                                           returning_customers_summary['frequency'],
                                                                           returning_customers_summary['recency'],
                                                                           returning_customers_summary['T'],
                                                                           returning_customers_summary['monetary_value'],
                                                                           time=1, # lifetime in months
                                                                           freq='D', # frequency in which data is present (T),
                                                                           discount_rate=0.01 #discount rate
                                                                           )                                                   
# calculate CLV manual
#returning_customers_summary['manual_predict_clv']= returning_customers_summary['predicted_num_purchases'] * returning_customers_summary['predicted_avg_sales']
#calculate CLV profit
profit_margin=0.05
returning_customers_summary['profit_CLV'] =returning_customers_summary['Predicted_CLV'] * profit_margin
 

############ THE END OF CLV ###################
