import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet
import datetime

df = pd.read_csv('GlobalLandTemperaturesByCountry.csv')


plt.style.use('fivethirtyeight')

#function that cleans dataset
def clean_df(df):
    #convert to datetime
    df['dt'] = pd.to_datetime(df['dt'])
    #filter for USA and after 1900
    df = df.loc[(df['Country'] == 'United States') & (df['dt'] >= '1900-01-01')]
    #drop unnneeded columns
    df = df.drop(columns = ['AverageTemperatureUncertainty', 'Country'])
    df = df.set_index('dt')

    return df

df_cleaned = clean_df(df)
print(df_cleaned.info())

#plot the temperature
df_cleaned.plot(
        figsize= (15, 5),
        style= '-',
        title= 'USA Temperature Over the Years',
        legend= False,
        xlabel= '',
        ylabel= 'Temp in Celcius',
        ylim =  (-10,30)
    )


#plt.show()


#create a training dataset

num_train = round(len(df_cleaned) * 0.75)

train = df_cleaned.iloc[:num_train]
test = df_cleaned.iloc[num_train:]


#visualize the different pieces (training/testing sets)
fig, ax= plt.subplots(figsize= (15, 5))
train.plot(
    ax= ax,
    label= 'Training Set'
)
test.plot(
    ax= ax,
    label= 'Test Set'
)

ax.set_title('USA Temperature Over the Years')
ax.set_xlabel('')
ax.set_ylabel('Temp in Celcius')
ax.legend(['Training Set', 'Test Set'])
#plt.show()

#create new dataframes for prophet fit
def create_prophet_set(df):
    return (df
        .reset_index()
        .rename(columns= {
            'dt': 'ds',
            'AverageTemperature': 'y'
        })
    )

train_prophet = create_prophet_set(train)
test_prophet = create_prophet_set(test)

model_prophet = Prophet()
model_prophet.fit(train_prophet)
preds= model_prophet.predict(test_prophet)
fig, ax= plt.subplots(figsize= (15, 5))
model_prophet.plot(
    preds,
    ax= ax
)
ax.set_title('Prophet Forecast')





#plot forecasted vs actual
fig, ax= plt.subplots(figsize= (15, 5))
model_prophet.plot(preds, ax= ax)
ax.scatter(test_prophet['ds'], test_prophet['y'], color= 'red')
ax.set_title('Prophet Forecast vs Actual Value')
ax.set_xbound(
    lower= test_prophet['ds'].min(),
    #upper= test_prophet['ds'].max()
    upper = datetime.date(2013, 1, 1)
)
plt.show()
