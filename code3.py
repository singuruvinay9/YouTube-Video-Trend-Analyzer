# ============================================================
# 🎥 YouTube Video Trend Analyzer (Auto-Fix Version)
# Author: Singuru Vinay
# Dataset: YouTube Trending Videos (US)
# Source: https://www.kaggle.com/datasets/datasnaek/youtube-new
# ============================================================

# 📦 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 🧹 2. Load Dataset
df = pd.read_csv(
    r"C:\Users\Vinay\OneDrive\Desktop\youtube video trend analyzer\data\youtube.csv",
    encoding="latin1"
)

# 👀 3. Display available columns to debug
print("📄 Columns in your dataset:")
print(df.columns.tolist())

# ============================================================
# 🔧 AUTO-DETECT 'publish_time' COLUMN
# ============================================================
possible_names = ['publish_time', 'publishedAt', 'publish date', 'upload_time', 'upload_date']
publish_col = None

for col in df.columns:
    if any(name.lower() in col.lower() for name in possible_names):
        publish_col = col
        break

if publish_col:
    print(f"\n✅ Using '{publish_col}' as publish time column.")
    df['publish_time'] = pd.to_datetime(df[publish_col], errors='coerce')
else:
    print("\n⚠️ Could not find a publish time column automatically.")
    print("Please check your CSV and tell me the correct column name.")
    df['publish_time'] = pd.NaT

# Continue only if timestamps are valid
if df['publish_time'].notna().sum() > 0:
    df['publish_date'] = df['publish_time'].dt.date
    df['publish_hour'] = df['publish_time'].dt.hour
else:
    df['publish_date'] = np.nan
    df['publish_hour'] = np.nan

print("\n✅ Time columns prepared successfully.")
print(df[['publish_time', 'publish_date', 'publish_hour']].head())

# ============================================================
# 🧹 DATA CLEANING
# ============================================================
print("\nMissing Values in Each Column:")
print(df.isnull().sum())

df = df.drop_duplicates()

# ============================================================
# 🧠 BASIC INFO
# ============================================================
print("\nBasic Info:")
print(df.info())

# ============================================================
# 📈 ANALYSIS 1: Most Popular Video Categories
# ============================================================

category_map = {
    1: "Film & Animation", 2: "Autos & Vehicles", 10: "Music",
    15: "Pets & Animals", 17: "Sports", 20: "Gaming", 22: "People & Blogs",
    23: "Comedy", 24: "Entertainment", 25: "News & Politics",
    26: "Howto & Style", 27: "Education", 28: "Science & Technology",
    29: "Nonprofits & Activism"
}

if 'category_id' in df.columns:
    df['category_name'] = df['category_id'].map(category_map)
else:
    print("⚠️ No 'category_id' column found; assigning 'Unknown'.")
    df['category_name'] = "Unknown"

category_stats = (
    df.groupby('category_name')[['views', 'likes', 'comment_count']]
    .mean()
    .sort_values(by='views', ascending=False)
)

print("\nAverage Performance by Category:")
print(category_stats)

plt.figure(figsize=(10,6))
sns.barplot(y=category_stats.index, x=category_stats['views'])
plt.title("Average Views per Category")
plt.xlabel("Average Views")
plt.ylabel("Category")
plt.tight_layout()
plt.show()

# ============================================================
# 📊 ANALYSIS 2: Top 10 Most Engaging Channels
# ============================================================

top_channels = (
    df.groupby('channel_title')[['views', 'likes', 'comment_count']]
    .mean()
    .sort_values(by='views', ascending=False)
    .head(10)
)

print("\nTop 10 Engaging channels:")
print(top_channels)

plt.figure(figsize=(10,6))
sns.barplot(x=top_channels['views'], y=top_channels.index, palette="viridis")
plt.title("Top 10 Engaging channels")
plt.xlabel("Average Views")
plt.ylabel("Channel")
plt.tight_layout()
plt.show()

# ============================================================
# 💬 ANALYSIS 3: Like-to-View Ratio (Engagement Rate)
# ============================================================

df['like_ratio'] = df['likes'] / df['views']
df['comment_ratio'] = df['comment_count'] / df['views']

engagement = (
    df.groupby('category_name')[['like_ratio', 'comment_ratio']]
    .mean()
    .sort_values(by='like_ratio', ascending=False)
)

print("\nEngagement by Category (Like & Comment Ratios):")
print(engagement)

plt.figure(figsize=(10,5))
sns.barplot(x=engagement['like_ratio'], y=engagement.index, color='orange')
plt.title("Average Like Ratio by Category")
plt.xlabel("Likes per View")
plt.ylabel("Category")
plt.tight_layout()
plt.show()

# ============================================================
# 🔍 ANALYSIS 4: Keyword Frequency in Tags
# ============================================================

if 'tags' in df.columns:
    all_tags = " ".join(df['tags'].dropna().astype(str)).replace("|", " ")
    word_list = all_tags.split()
    tag_series = pd.Series(word_list).value_counts().head(15)

    print("\nTop 15 Most Frequent Tags:")
    print(tag_series)

    plt.figure(figsize=(10,5))
    sns.barplot(x=tag_series.values, y=tag_series.index, color='purple')
    plt.title("Top 15 Most Frequent Tags in Trending Videos")
    plt.xlabel("Frequency")
    plt.ylabel("Tag")
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ No 'tags' column found; skipping tag frequency analysis.")


