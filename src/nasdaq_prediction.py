import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

def main():      
    path_to_files = str("C:/Users/mende/OneDrive/Escritorio/CS584 - Machine Learning/Project/Delivery/data/")

    df = pd.read_csv(path_to_files + "NASDAQ Composite Historical Data.csv")

    # Drop columns that we don't need
    df = df.drop(columns=["Vol.","Change %","Open","High","Low"])

    # Change column types
    df["Price"] = df["Price"].str.replace(',', '')
    df["Price"] = df["Price"].astype(float)
    df["Date"] =  pd.to_datetime(df["Date"]).dt.date

    # Plot the SP500 Price & Date 
    plt.plot(df["Date"], df["Price"])
    dtFmt = mdates.DateFormatter('%Y-%b')
    plt.gca().xaxis.set_major_formatter(dtFmt)
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=90, fontweight='light', fontsize='x-small')
    plt.ylabel("Price")
    plt.xlabel("Date")
    plt.title("S&P 500 Price")
    plt.show()

    eur = pd.read_csv(path_to_files + "europe_cases.csv")
    eur = eur.rename(columns={"date":"Date", "new":"eur_deaths"})
    eur["Date"] =  pd.to_datetime(eur["Date"]).dt.date

    data = eur.merge(df, on="Date")
    data.set_index("Date", inplace=True)
    data.sort_index(inplace=True)

    data = data[:100]
    data.drop(columns=["deaths"],inplace=True)

    plt.plot(data["eur_deaths"], c="b")
    plt.show()

    scaler = MinMaxScaler()
    scaler.fit(data)
    scaled = scaler.fit_transform(data)
    scaled_data = pd.DataFrame(scaled, columns=data.columns)

    # Plot both Price and Deaths after normalization
    plt.plot(scaled_data["Price"], color="b")
    plt.plot(scaled_data["eur_deaths"], color="r")
    plt.show()

    # Load our model
    model = load_model(path_to_files + "model.h5")

    yhat = model.predict(scaled_data["eur_deaths"])

    plt.title("NASDAQ real and predicted values for the COVID-19 first wave")
    plt.plot(yhat, c="r", label="Prediction")
    plt.plot(scaled_data["Price"],c ="b", label="Real value")
    plt.legend()
    plt.xticks([], [])
    plt.show()

if __name__ == "__main__":
    main()