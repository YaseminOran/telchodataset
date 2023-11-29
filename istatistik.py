import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Veri setini yükleme
file_path = 'data/Telco-Customer-Churn.csv'
data = pd.read_csv(file_path)

# 'TotalCharges' sütununu sayısal veriye dönüştürme
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

# 'customerID' sütununu analiz için gereksiz olduğu için çıkar
data = data.drop(columns=['customerID'])

# Temel istatistiksel özet
statistical_summary = data.describe(include='all')

# Korelasyon matrisini hesaplama
corr = data.corr()

# Korelasyon haritasını çizme
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# İstatistiksel özet ve korelasyon matrisini yazdırma
print(statistical_summary)
print(corr)
