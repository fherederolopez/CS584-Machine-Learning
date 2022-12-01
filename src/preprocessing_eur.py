import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    path_to_files = str("C:/Users/ferna/OneDrive - Universidad Politécnica de Madrid/IIT/Machine Learning/Project/Delivery/data/")

    us = pd.read_csv(path_to_files + "/SP500&us_covid.csv", index_col=0)
    eur = pd.read_csv(path_to_files + "/europe_cases.csv")

    us.rename(columns={"Deaths":"us_deaths"}, inplace=True)
    eur.rename(columns={"date":"Date", "new":"eur_deaths"}, inplace=True)
    eur.drop(columns=["deaths"], inplace=True)

    df = us.merge(eur, on="Date")
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)

    # We need to normalize both values
    # We reduce up to the first 180 days
    # Without Weekends
    df.drop(columns=["Cases", "us_deaths"], inplace=True)
    df = df[:180]
    scaler = MinMaxScaler()
    scaler.fit(df)
    scaled = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled, columns=df.columns)

    # Plot both Price and Deaths after normalization
    plt.plot(scaled_df["Price"], color="b")
    plt.plot(scaled_df["eur_deaths"], color="r")
    plt.show()

    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True)

if __name__ == "__main__":
    main()
    





