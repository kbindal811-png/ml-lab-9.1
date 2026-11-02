
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

os.makedirs("dataset", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Dataset URLs
base_url = (
    "https://raw.githubusercontent.com/CSSEGISandData/"
    "COVID-19/master/csse_covid_19_data/"
    "csse_covid_19_time_series/"
)

urls = {
    "confirmed": base_url + "time_series_covid19_confirmed_global.csv",
    "deaths": base_url + "time_series_covid19_deaths_global.csv",
    "recovered": base_url + "time_series_covid19_recovered_global.csv"
}

# Download and save datasets
data = {}

for name, url in urls.items():
    print("Downloading", name, "dataset...")
    data[name] = pd.read_csv(url)
    data[name].to_csv(
        f"dataset/{name}.csv", index=False
    )

# Convert wide data into country-wise time series
def prepare_data(df, value_name):
    date_columns = df.columns[4:]

    result = df.groupby("Country/Region")[date_columns].sum()
    result = result.T.reset_index()

    result = result.rename(columns={"index": "Date"})
    result["Date"] = pd.to_datetime(result["Date"])

    result = result.melt(
        id_vars="Date",
        var_name="Country",
        value_name=value_name
    )

    return result

confirmed = prepare_data(data["confirmed"], "Confirmed")
deaths = prepare_data(data["deaths"], "Deaths")
recovered = prepare_data(data["recovered"], "Recovered")

# Combine COVID datasets
covid = confirmed.merge(
    deaths, on=["Date", "Country"]
)

covid = covid.merge(
    recovered, on=["Date", "Country"]
)

# Convert cumulative figures into numeric values
for column in ["Confirmed", "Deaths", "Recovered"]:
    covid[column] = pd.to_numeric(
        covid[column], errors="coerce"
    ).fillna(0)

# Save combined dataset
covid.to_csv("dataset/combined_covid_data.csv", index=False)

# Latest country-wise statistics
latest_date = covid["Date"].max()

latest = covid[covid["Date"] == latest_date].copy()

latest = latest.sort_values(
    "Confirmed", ascending=False
)

latest["Recovery Rate"] = np.where(
    latest["Confirmed"] > 0,
    latest["Recovered"] / latest["Confirmed"] * 100,
    0
)

latest["Death Rate"] = np.where(
    latest["Confirmed"] > 0,
    latest["Deaths"] / latest["Confirmed"] * 100,
    0
)

latest.to_csv(
    "outputs/country_wise_statistics.csv",
    index=False
)

print("\nLatest COVID-19 Statistics")
print(latest.head(10).to_string(index=False))

# Worldwide daily totals
worldwide = covid.groupby("Date")[
    ["Confirmed", "Deaths", "Recovered"]
].sum().reset_index()

# Calculate new cases
worldwide["New Cases"] = (
    worldwide["Confirmed"].diff().fillna(0).clip(lower=0)
)

# Plot worldwide trends
plt.figure(figsize=(12, 6))

plt.plot(
    worldwide["Date"],
    worldwide["Confirmed"],
    label="Confirmed Cases"
)

plt.plot(
    worldwide["Date"],
    worldwide["Recovered"],
    label="Recovered"
)

plt.plot(
    worldwide["Date"],
    worldwide["Deaths"],
    label="Deaths"
)

plt.xlabel("Date")
plt.ylabel("Total Cases")
plt.title("Worldwide COVID-19 Trends")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("outputs/covid_trends.png")
plt.show()

# Top 10 countries by confirmed cases
top_countries = latest.head(10)

plt.figure(figsize=(12, 6))

sns.barplot(
    data=top_countries,
    x="Confirmed",
    y="Country"
)

plt.title("Top 10 Countries by Confirmed Cases")
plt.xlabel("Confirmed Cases")
plt.ylabel("Country")
plt.tight_layout()
plt.savefig("outputs/top_countries.png")
plt.show()

# Death rate comparison
plt.figure(figsize=(12, 6))

sns.barplot(
    data=top_countries,
    x="Death Rate",
    y="Country"
)

plt.title("COVID-19 Death Rate by Country")
plt.xlabel("Death Rate (%)")
plt.ylabel("Country")
plt.tight_layout()
plt.savefig("outputs/death_rate.png")
plt.show()

# Interactive worldwide trend
fig1 = px.line(
    worldwide,
    x="Date",
    y=["Confirmed", "Recovered", "Deaths"],
    title="Interactive Worldwide COVID-19 Analysis"
)

fig1.write_html(
    "outputs/covid_dashboard.html"
)

# Interactive country comparison
fig2 = px.bar(
    top_countries,
    x="Country",
    y=["Confirmed", "Recovered", "Deaths"],
    title="Country-wise COVID-19 Comparison",
    barmode="group"
)

fig2.write_html(
    "outputs/country_comparison.html"
)

# Vaccination dataset
print("\nDownloading vaccination dataset...")

vaccination_url = (
    "https://raw.githubusercontent.com/owid/"
    "covid-19-data/master/public/data/"
    "owid-covid-data.csv"
)

vaccination = pd.read_csv(
    vaccination_url,
    usecols=[
        "location",
        "date",
        "people_vaccinated",
        "people_fully_vaccinated"
    ]
)

vaccination.to_csv(
    "dataset/vaccination.csv",
    index=False
)

# Remove regional and world aggregates
vaccination = vaccination[
    ~vaccination["location"].isin(
        ["World", "Asia", "Europe", "Africa",
         "North America", "South America", "Oceania"]
    )
]

vaccination["date"] = pd.to_datetime(
    vaccination["date"]
)

vaccination = vaccination.dropna(
    subset=["people_vaccinated"]
)

# Top 10 countries by latest available vaccinated population
latest_vaccination = (
    vaccination.sort_values("date")
    .groupby("location")
    .tail(1)
    .sort_values(
        "people_vaccinated",
        ascending=False
    )
    .head(10)
)

fig3 = px.bar(
    latest_vaccination,
    x="location",
    y="people_vaccinated",
    title="Top 10 Countries by People Vaccinated"
)

fig3.write_html(
    "outputs/vaccination_dashboard.html"
)

print("\nCOVID-19 Analysis Completed Successfully!")
print("Check the dataset and outputs folders.")
