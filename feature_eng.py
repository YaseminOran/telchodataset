import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_and_clean_data(file_path):
    churn_data = pd.read_csv(file_path)
    churn_data['TotalCharges'] = pd.to_numeric(churn_data['TotalCharges'], errors='coerce')
    churn_data = churn_data.dropna(subset=['TotalCharges'])
    churn_data = churn_data.drop(['customerID'], axis=1)
    return churn_data


def feature_engineering(churn_data):
    le = LabelEncoder()
    binary_columns = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    for col in binary_columns:
        churn_data[col] = le.fit_transform(churn_data[col])

    categorical_columns = ['gender', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                           'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                           'StreamingMovies', 'Contract', 'PaymentMethod']
    churn_data = pd.get_dummies(churn_data, columns=categorical_columns)

    churn_data['UsageRate'] = churn_data['MonthlyCharges'] / churn_data['tenure'].replace(0, 1)
    churn_data['TotalChargeTenureRatio'] = churn_data['TotalCharges'] / churn_data['tenure'].replace(0, 1)
    return churn_data


"""Öznitelik (feature) üretimi, veri bilimi ve makine öğrenimi projelerinde oldukça önemli bir adımdır. 
İyi tasarlanmış öznitelikler, modelin performansını önemli ölçüde artırabilir. Öznitelik üretimi yaparken 
izlenebilecek bazı temel adımlar ve yöntemler şunlardır:

Veri Setini Anlamak:
İlk adım, veri setinizi derinlemesine anlamaktır. Her sütunun ne anlama geldiğini, hangi tür veriler içerdiğini ve 
potansiyel ilişkileri anlamak önemlidir.
Ayrıca, veri setindeki eksik veya anomali değerleri de tespit etmek gerekir.

İş Problemini Anlamak:
Hangi soruna çözüm bulmaya çalıştığınızı ve hangi tür tahminler yapmak istediğinizi netleştirin. 
Örneğin, müşteri ayrılmasını tahmin etmek istiyorsanız, müşteri davranışları ve abonelik detaylarıyla ilgili 
öznitelikler faydalı olabilir.

Eksploratif Veri Analizi (EDA):
Veri setinizin istatistiksel özetini çıkarmak ve görselleştirmek, hangi özniteliklerin önemli olabileceği konusunda 
fikir verir.
Korelasyon analizi, önemli öznitelikleri belirlemede yardımcı olabilir.

Domain Bilgisi:
Alanınızda (domain) uzman bilgisine sahip olmak, hangi özniteliklerin önemli olabileceğini anlamada kritik öneme sahiptir. Örneğin, telekom sektöründe çalışıyorsanız, müşteri ayrılmasını etkileyebilecek özel sektör dinamiklerini bilmek faydalıdır.
Öznitelik Mühendisliği Yöntemleri:

Dönüşüm Uygulama: Log dönüşümü, karekök dönüşümü gibi matematiksel dönüşümler.

Birleştirme ve Ayrıştırma: Birden fazla özniteliği birleştirerek yeni öznitelikler oluşturma veya tek bir öznitelikten birden fazla öznitelik çıkarma.

Kategorik Verileri İşleme: One-hot encoding, label encoding gibi yöntemler.

Zaman Serisi Verileri İşleme: Zaman damgalarından yeni öznitelikler çıkarma (gün, ay, yıl, haftanın günü vb.).

Eksik Verileri Doldurma: Eksik verileri doldurmak veya bu verileri işaretlemek için öznitelikler oluşturma.

Polinom Öznitelikler: Mevcut özniteliklerin polinom kombinasyonları.

Model Geri Bildirimi:
Başlangıçta oluşturduğunuz özniteliklerle bir model eğitin ve performansını değerlendirin.
Modelin performansı, hangi özniteliklerin daha fazla işe yaradığını anlamada size yardımcı olabilir.

Deneme ve Yanılma:
Öznitelik mühendisliği sürecinde deneme-yanılma yöntemi sıkça kullanılır. Farklı öznitelik kombinasyonlarını deneyerek hangilerinin en iyi sonucu verdiğini görebilirsiniz."""
