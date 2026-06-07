import sqlite3
import pandas as pd

def clean_data():
    
    # Load the Database
    print("Loading data from SQLite database...")
    conn = sqlite3.connect("data/gas_monitoring.db")

    df = pd.read_sql_query(
        "SELECT * FROM gas_monitoring",
        conn
    )
    conn.close()
    print(df.head())
    
    # Clean categorical columns
    print("Formatting categorical columns...")
    df['HVAC Operation Mode'] = (
    df['HVAC Operation Mode']
    .str.strip()
    .str.lower()
    )

    df['Activity Level'] = (
        df['Activity Level']
        .str.strip()
        .str.lower()
        .str.replace('_', ' ', regex=False)
        .str.replace('lowactivity', 'low activity', regex=False)
        .str.replace('moderateactivity', 'moderate activity', regex=False)
    )
    
    # Format rows
    # make all the words in the rows lowercase and replace underscores with spaces
    print("Formatting rows...")
    text_cols = df.select_dtypes(include=["object", "string"]).columns

    for col in text_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace("_", " ", regex=False)
        )

    # Fix specific activity level issues
    df["Activity Level"] = (
        df["Activity Level"]
        .str.replace("lowactivity", "low activity", regex=False)
        .str.replace("moderateactivity", "moderate activity", regex=False)
        .str.replace("highactivity", "high activity", regex=False)
    )
    
    # Remove duplicates
    print("Removing duplicates...")
    df = df.drop_duplicates()
    print("Duplicates removed.")

    
    # remove invalid temperatures
    # we are removing values above 200 degrees Celsius
    print("Removing invalid temperatures...")
    df = df[df["Temperature"] <= 200]
    print("Invalid temperatures removed.")

    # remove humidity outliers
    print("Removing humidity outliers...")
    Q1 = df['Humidity'].quantile(0.25)
    Q3 = df['Humidity'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Keep only rows within the IQR bounds for humidity
    df = df[
        (df['Humidity'] >= lower_bound) &
        (df['Humidity'] <= upper_bound)
    ]
    
    # Fill in the missing values
    # To fill missing values in Humidity Columns
    print("Handling missing values...")
    humidity_median = df["Humidity"].median()

    df["Humidity"] = df["Humidity"].fillna(humidity_median)
    
    # To fill in missing values in metal oxide sensor unit 2 column
    df["MetalOxideSensor_Unit2"] = df["MetalOxideSensor_Unit2"].fillna(
        df[
            [
                "MetalOxideSensor_Unit1",
                "MetalOxideSensor_Unit3",
                "MetalOxideSensor_Unit4"
            ]
        ].mean(axis=1)
    )
    
    # To fill in missing values in CO gas sensor column
    co_mode = df["CO_GasSensor"].mode()[0]

    df["CO_GasSensor"] = df["CO_GasSensor"].fillna(co_mode)
    
    # To fill in missing values in Ambient light column 
    # Find the most common Ambient Light Level for each Time of Day

    ambient_mode_by_time = df.groupby("Time of Day")["Ambient Light Level"].agg(
        lambda x: x.mode()[0] if not x.mode().empty else pd.NA
    )

    df["Ambient Light Level"] = df["Ambient Light Level"].fillna(
        df["Time of Day"].map(ambient_mode_by_time)
    )

    # If any missing values still remain, fill with overall mode
    overall_ambient_mode = df["Ambient Light Level"].mode()[0]

    df["Ambient Light Level"] = df["Ambient Light Level"].fillna(overall_ambient_mode)
    
    # format decimal values to 2dp
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    df[numeric_cols] = df[numeric_cols].round(2)
    
    # format columns
    df.columns = (
        df.columns
        .str.strip()
        .str.upper()
        .str.replace(" ", "_", regex=False)
    )
    
    # save cleaned csv 
    print("saving cleaned dataset...")
    df.to_csv( "data/cleaned_gas_monitoring.csv", index=False)
    print(f"Cleaning complete. Final shape: {df.shape}")
    
    return df

if __name__ == "__main__":
    clean_data()

