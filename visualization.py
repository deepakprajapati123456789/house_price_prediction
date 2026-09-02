import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("house_price_dataset_10000 (1).csv")

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

print("Columns in Dataset:")
print(df.columns)

# ==========================================
# Create Graph Folder
# ==========================================

os.makedirs("graphs", exist_ok=True)

# ==========================================
# Set Graph Style
# ==========================================

sns.set_style("whitegrid")

# ==========================================
# 1. House Price Distribution
# ==========================================

plt.figure(figsize=(10,6))

sns.histplot(df["price"], bins=40, kde=True, color="skyblue")

plt.title("House Price Distribution")
plt.xlabel("Price")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig("graphs/price_distribution.png")

plt.close()

print("✓ Price Distribution Saved")

# ==========================================
# 2. Price vs Area
# ==========================================

if "area" in df.columns:

    plt.figure(figsize=(10,6))

    sns.scatterplot(
        data=df,
        x="area",
        y="price",
        color="red"
    )

    plt.title("Price vs Area")
    plt.xlabel("Area")
    plt.ylabel("Price")

    plt.tight_layout()

    plt.savefig("graphs/price_vs_area.png")

    plt.close()

    print("✓ Price vs Area Saved")

# ==========================================
# 3. Correlation Heatmap
# ==========================================

numeric_df = df.select_dtypes(include=["int64", "float64"])

plt.figure(figsize=(10,8))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig("graphs/correlation_heatmap.png")

plt.close()

print("✓ Correlation Heatmap Saved")

# ==========================================
# 4. City-wise Average Price
# ==========================================

if "city" in df.columns:

    plt.figure(figsize=(12,6))

    city_avg = df.groupby("city")["price"].mean().sort_values()

    city_avg.plot(kind="bar")

    plt.title("Average House Price by City")
    plt.xlabel("City")
    plt.ylabel("Average Price")

    plt.tight_layout()

    plt.savefig("graphs/city_average_price.png")

    plt.close()

    print("✓ City-wise Average Price Saved")

# ==========================================
# 5. Bedrooms vs Price
# ==========================================

if "bedrooms" in df.columns:

    plt.figure(figsize=(8,6))

    sns.boxplot(
        data=df,
        x="bedrooms",
        y="price"
    )

    plt.title("Bedrooms vs Price")

    plt.tight_layout()

    plt.savefig("graphs/bedrooms_vs_price.png")

    plt.close()

    print("✓ Bedrooms vs Price Saved")

# ==========================================
# 6. Bathrooms vs Price
# ==========================================

if "bathrooms" in df.columns:

    plt.figure(figsize=(8,6))

    sns.boxplot(
        data=df,
        x="bathrooms",
        y="price"
    )

    plt.title("Bathrooms vs Price")

    plt.tight_layout()

    plt.savefig("graphs/bathrooms_vs_price.png")

    plt.close()

    print("✓ Bathrooms vs Price Saved")

# ==========================================
# Finished
# ==========================================

print("\n===================================")
print("All Graphs Generated Successfully!")
print("Graphs are saved in the 'graphs' folder.")
print("===================================")

####################################################
