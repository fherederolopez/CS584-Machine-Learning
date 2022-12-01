import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

def main():
    # Read SP500 dataframe
    path_to_files = str("C:/Users/ferna/OneDrive - Universidad Politécnica de Madrid/IIT/Machine Learning/Project/Delivery/data/")

    sp500 = pd.read_csv(path_to_files + "/raw_data/S&P 500 Historical Data.csv")

    # Drop columns that we don't need
    sp500 = sp500.drop(columns=["Vol.","Change %","Open","High","Low"])

    # Change column types
    sp500["Price"] = sp500["Price"].str.replace(',', '')
    sp500["Price"] = sp500["Price"].astype(float)
    sp500["Date"] =  pd.to_datetime(sp500["Date"]).dt.date

    # Plot the SP500 Price & Date 
    plt.plot(sp500["Date"], sp500["Price"])
    dtFmt = mdates.DateFormatter('%Y-%b')
    plt.gca().xaxis.set_major_formatter(dtFmt)
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=90, fontweight='light', fontsize='x-small')
    plt.ylabel("Price")
    plt.xlabel("Date")
    plt.title("S&P 500 Price")
    plt.show()

    # Read covid dataframe
    covid = pd.read_csv(path_to_files + "/raw_data/Country.csv")

    # Rename columns for future merge
    covid.rename(columns={"date": "Date", "cases": "Cases", "deaths": "Deaths"}, inplace=True)

    # Change column type
    covid["Date"] =  pd.to_datetime(covid["Date"]).dt.date

    # Plot the Covid deaths
    plt.plot(covid["Date"], covid["Deaths"])
    dtFmt = mdates.DateFormatter('%Y-%b')
    plt.gca().xaxis.set_major_formatter(dtFmt)
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=90, fontweight='light', fontsize='x-small')
    plt.ylabel("Deaths")
    plt.xlabel("Date")
    plt.title("Covid deaths in the United States")
    plt.show()

    # Merge both dataframes into one 
    df = sp500.merge(covid, on="Date")

    # Plot both Price and Deaths without normalization
    plt.plot(df["Date"], df["Price"], color="b")
    plt.plot(df["Date"], df["Deaths"], color="r")
    plt.show()

    # We need to normalize both values
    # First, we need to set Date as index
    df.set_index("Date", inplace=True)
    scaler = MinMaxScaler()
    scaler.fit(df)
    scaled = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled, columns=df.columns)

    # Plot both Price and Deaths after normalization
    plt.plot(scaled_df["Price"], color="b")
    plt.plot(scaled_df["Deaths"], color="r")
    plt.show()
    # We have the total values of deaths, but what we need is dialy deaths

    # Sort dataframe by index date ascending
    df.sort_index(ascending=True, inplace=True)

    # Obtain Dialy deaths
    df["Daily_deaths"] = 0
    df["Daily_cases"] = 0
    for i in range(len(df)):
        if i == 0:
            df["Daily_deaths"][i] = df["Deaths"][i]
            df["Daily_cases"][i] = df["Cases"][i]
        else:
            df["Daily_deaths"][i] = df["Deaths"][i] - df["Deaths"][i-1]
            df["Daily_cases"][i] = df["Cases"][i] - df["Cases"][i-1]

    # Drop non daily columns
    df.drop(columns=["Deaths", "Cases"], inplace=True)

    # Plot daily deaths and daily cases to see outliers
    plt.plot(df["Daily_deaths"])
    dtFmt = mdates.DateFormatter('%Y-%b')
    plt.gca().xaxis.set_major_formatter(dtFmt)
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=90, fontweight='light', fontsize='x-small')
    plt.ylabel("Deaths")
    plt.show()

    plt.plot(df["Daily_cases"])
    dtFmt = mdates.DateFormatter('%Y-%b')
    plt.gca().xaxis.set_major_formatter(dtFmt)
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=90, fontweight='light', fontsize='x-small')
    plt.ylabel("Cases")
    plt.show()
    # As we can see in the plot, there are some outliers which
    # correspond to negative daily cases and negative daily deaths

    # Drop outliers
    df_filtered = df[(df["Daily_cases"] >= 0) & (df["Daily_deaths"] >= 0)]

    # Normalize values
    scaler.fit(df_filtered)
    scaled_filtered = scaler.fit_transform(df_filtered)
    scaled_df_filtered = pd.DataFrame(scaled_filtered, columns=df_filtered.columns)

    # Set index to date
    scaled_df_filtered["Date"] = df_filtered.index
    scaled_df_filtered.set_index("Date", inplace=True)

    # Plot both Price and Deaths after normalization
    plt.plot(scaled_df_filtered.index, scaled_df_filtered["Price"], color="b")
    plt.plot(scaled_df_filtered.index, scaled_df_filtered["Daily_deaths"], color="r")
    plt.gca().xaxis.set_major_formatter(dtFmt)
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=90, fontweight='light', fontsize='x-small')
    plt.title("Deaths vs Price (Normalized)")
    plt.show()

    # Plot both Price and Cases after normalization
    plt.plot(scaled_df_filtered.index, scaled_df_filtered["Price"], color="b")
    plt.plot(scaled_df_filtered.index, scaled_df_filtered["Daily_cases"], color="r")
    plt.gca().xaxis.set_major_formatter(dtFmt)
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=90, fontweight='light', fontsize='x-small')
    plt.title("Cases vs Price (Normalized)")
    plt.show()

    # Reduce dataframe to first 280 values
    scaled_df_filtered_reduced = scaled_df_filtered[:200]

    # Plot both Price and Deaths after normalization
    plt.plot(scaled_df_filtered_reduced.index, scaled_df_filtered_reduced["Price"], color="b")
    plt.plot(scaled_df_filtered_reduced.index, scaled_df_filtered_reduced["Daily_deaths"], color="r")
    plt.gca().xaxis.set_major_formatter(dtFmt)
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=90, fontweight='light', fontsize='x-small')
    plt.title("Deaths vs Price (Normalized and Reduced in time)")
    plt.show()

    # Plot both Price and Cases after normalization
    plt.plot(scaled_df_filtered_reduced.index, scaled_df_filtered_reduced["Price"], color="b")
    plt.plot(scaled_df_filtered_reduced.index, scaled_df_filtered_reduced["Daily_cases"], color="r")
    plt.gca().xaxis.set_major_formatter(dtFmt)
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=90, fontweight='light', fontsize='x-small')
    plt.title("Cases vs Price (Normalized and Reduced in time)")
    plt.show()

    # See correlation

    final_df = scaled_df_filtered_reduced
    correlation_matrix = final_df.corr()
    sns.heatmap(correlation_matrix, annot=True)
    # There is no a big correlation. This is because "Cases" were not counted at the beginning
    # in a good way. In addition, there are peaks since cases and deaths were not count every day
    # exactly, so we are going to plot cases and deaths as a smoother curve.

    # We use a Savitzky Golay filter.
    from scipy.signal import savgol_filter
    final_df["Deaths_smooth"] = savgol_filter(final_df["Daily_deaths"], 51, 3)
    final_df["Cases_smooth"] = savgol_filter(final_df["Daily_cases"], 51, 3)

    # Plot both Price and Deaths after normalization and with Savgol filter
    plt.plot(final_df.index, final_df["Price"], color="b")
    plt.plot(final_df.index, final_df["Deaths_smooth"], color="r")
    plt.gca().xaxis.set_major_formatter(dtFmt)
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=90, fontweight='light', fontsize='x-small')
    plt.title("Deaths vs Price (Normalized and Reduced in time) using Savgol filter")
    plt.show()

    # Plot both Price and Cases after normalization and with Savgol filter
    plt.plot(final_df.index, final_df["Price"], color="b")
    plt.plot(final_df.index, final_df["Cases_smooth"], color="r")
    plt.gca().xaxis.set_major_formatter(dtFmt)
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=90, fontweight='light', fontsize='x-small')
    plt.title("Cases vs Price (Normalized and Reduced in time) using Savgol filter")
    plt.show()

    final_df_smooth = final_df.drop(columns=["Daily_deaths", "Daily_cases"])
    # See new correlation matrix
    correlation_matrix = final_df_smooth.corr()
    sns.heatmap(correlation_matrix, annot=True)

    # Shift deaths // Delete cases since they aren't relevant
    new_df = pd.DataFrame()
    new_df["Date"] = scaled_df_filtered.index
    new_df.set_index("Date", inplace=True)
    new_df["Price"] = scaled_df_filtered["Price"]
    new_df["Deaths_shifted"] = scaled_df_filtered["Daily_deaths"]

    new_df["Deaths_shifted"] = new_df["Deaths_shifted"].shift(-15)

    # We want only the first 200 values
    new_df = new_df[:200]

    # We use the Savgol filter to smooth the curve
    new_df["Deaths_shifted"] = savgol_filter(new_df["Deaths_shifted"], 51, 3)

    # Plot both Price and Deaths shifted after normalization and with Savgol filter
    plt.plot(new_df.index, new_df["Price"], color="b")
    plt.plot(new_df.index, new_df["Deaths_shifted"], color="r")
    plt.gca().xaxis.set_major_formatter(dtFmt)
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=90, fontweight='light', fontsize='x-small')
    plt.title("Deaths shifted vs Price (Normalized and Reduced in time) using Savgol filter")
    plt.show()

    # Plot new correlation matrix
    new_corr_matrix = new_df.corr()
    sns.heatmap(new_corr_matrix, annot=True)

if __name__ == "__main__":
    main()
