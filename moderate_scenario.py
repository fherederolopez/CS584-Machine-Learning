import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
import joblib
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np

df = pd.read_csv(r"C:\Users\ferna\OneDrive - Universidad Politécnica de Madrid\IIT\Machine Learning\Project\ProjectCSV\S&P 500 Historical Data V2.csv")

# Drop columns that we don't need
df.drop(columns=["Vol.","Change %","Open","High","Low"], inplace=True)

# Change column types
df["Price"] = df["Price"].str.replace(',', '')
df["Price"] = df["Price"].astype(float)
df["Date"] =  pd.to_datetime(df["Date"]).dt.date

df = df.reindex(index=df.index[::-1])
df.reset_index(inplace=True)
df.drop(columns="index", inplace=True)

plt.plot(df["Date"], df["Price"], c="b")
dtFmt = mdates.DateFormatter('%Y-%b')
plt.gca().xaxis.set_major_formatter(dtFmt)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.xticks(rotation=90, fontweight='light', fontsize='x-small')
plt.show()

eur = pd.read_csv(r"C:\Users\mende\OneDrive\Escritorio\CS584 - Machine Learning\Project\europe_cases.csv")

eur.rename(columns={"date":"Date", "new":"eur_deaths"}, inplace=True)
eur.drop(columns=["deaths"], inplace=True)

plt.plot(eur["Date"], eur["eur_deaths"], c="r")
dtFmt = mdates.DateFormatter('%Y-%b')
plt.gca().xaxis.set_major_formatter(dtFmt)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.xticks(rotation=90, fontweight='light', fontsize='x-small')
plt.show()

plt.plot(eur["eur_deaths"])
plt.show()

eur["Date"] = pd.to_datetime(eur["Date"])
df["Date"] = pd.to_datetime(df["Date"])

deaths = eur["eur_deaths"][640:820].reset_index()

data = df.merge(eur, on="Date")

plt.plot(data["Date"], data["Price"], c="b")
plt.plot(data["Date"], data["eur_deaths"], c="r")
dtFmt = mdates.DateFormatter('%Y-%b')
plt.gca().xaxis.set_major_formatter(dtFmt)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.xticks(rotation=90, fontweight='light', fontsize='x-small')
plt.show()

# SECOND WAVE SIMULATION
deaths.drop(columns="index", inplace=True)
columns = {"Price":[], "eur_deaths":[]}
catas_data = pd.DataFrame(data=columns)
catas_data["Price"] = data["Price"][:180]
catas_data["eur_deaths"] = deaths
data = catas_data

data = data[:180]

# LOAD SCALER
scaler = joblib.load(r"C:\Users\mende\OneDrive\Escritorio\CS584 - Machine Learning\Project\scaler.save")

scaled = scaler.fit_transform(data)
scaled_data = pd.DataFrame(scaled, columns=data.columns)

# NEW DEATH VALUES - MODERATE(1/2 TIMES than the first covid wave) 
scaled_data["eur_deaths"] = scaled_data["eur_deaths"]*0.5

# Plot both Price and Deaths after normalization
plt.plot(scaled_data["Price"], color="b")
plt.plot(scaled_data["eur_deaths"], color="r")
plt.show()

test = scaled_data["eur_deaths"]
final = scaled_data

model = load_model(r"C:\Users\mende\OneDrive\Escritorio\CS584 - Machine Learning\Project\Project\model.h5")

yhat = model.predict(test)

plt.plot(yhat)
plt.show()

plt.plot(final["Price"])
plt.show()

label = final["Price"].to_numpy()
yhat = yhat.reshape(len(yhat),)
seq = np.append(label, yhat)

seq = pd.DataFrame(seq)

x = seq.index.values
color_list=['red','blue']

plt.plot(seq[:180], color="blue")
plt.plot(seq[179:], color="red")
plt.show()

# Invert scaling and plot

deaths_scaled = scaled_data["eur_deaths"]
deaths_scaled = deaths_scaled.append(deaths_scaled)
deaths_scaled = deaths_scaled.reset_index()
deaths_scaled.drop(columns="index", inplace=True)
scaled_dataframe = seq.join(deaths_scaled)

inverse_dataframe = scaler.inverse_transform(scaled_dataframe)
inverse_dataframe = pd.DataFrame(inverse_dataframe)

plt.plot(inverse_dataframe[0][:180], c="b")
plt.plot(inverse_dataframe[0][179:], c="r")
frame1 = plt.gca()
frame1.axes.xaxis.set_ticklabels([])
plt.title("S&P 500 value - Possible moderate scenario")
plt.show()
