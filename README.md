# LayerLords 
#### TEKNOFEST 2024 Sağlıkta Yapay Zeka Yarışması Finalisti LayerLords Takımı Kodları
Projemizde, bilgisayarlı görü teknikleri kullanarak mamografi görüntülerinde kitle ve kalsifikasyon tespiti yapmayı ve BI-RADS (1, 2, 4, 5) kategorilerini tahmin etmeyi başardık. Mamografi radyoloji raporlarından varlık ismi çıkarımı (Named Entity Recognition) gerçekleştirerek, raporlardaki önemli tıbbi terimleri otomatik olarak tanımladık. Ayrıca, bu raporlardan BI-RADS kategorilerini doğru bir şekilde tahmin eden modeller geliştirdik. Bu yaklaşımlar ile mamografi değerlendirmelerinin hızını ve tutarlılığını artırarak sağlık profesyonellerine teşhis ve tedavi süreçlerinde önemli bir destek sağlamayı hedefliyoruz.


## Takım Üyeleri
---

-  Ahmet Akay - [LinkedIn](https://www.linkedin.com/in//), [Github](https://github.com/Aakayy7) <br/>
-  Zafer Khaliqi - [LinkedIn](https://www.linkedin.com/in//), [Github](https://github.com/zaferkhaliqi) <br/>
-  Ayşe Sude Erzurumlu - [LinkedIn](https://www.linkedin.com/in//), [Github](https://github.com/SudeErzurumlu) <br/>


## Bölümler

1. [Veri Seti](#introduction)
2. [V-BIRADS](#installation)
3. [V-YOLO](#usage)
4. [NER](#contributing)
5. [TEXT- CLASFFICATION](#usage)



## Veri Seti




## V-BIRADS




## V-YOLO




## Varlık İsmi Çıkarımı (NER) 

Etiketler şunları ifade etmektedir; 

- ANAT: Anatomik bir kısım, lokalizasyon,  histolojik antite, anatomik dağılım ifadeleri, taraf bulgusu

- OBS-PRESENT: Radyolojik bir özelliğin varlığının olma durumu, tanımlanabilir patofizyolojik 
süreç, veya tanısal hastalık olma durumu, bir bulgunun olma durumu, normal bir sürecin olma 
durumu  

- OBS-ABSENT: Radyolojik bir özelliğin bulunmaması, tanımlanabilir patofizyolojik süreç, 
veya tanısal hastalık olmama durumu, bir bulgunun olmama durumu 

- OBS-UNCERTAIN: Şüphe var ama bir konuda kesinlik  yok, belirsizlik içeren bir bulgu, ayırıcı tanı, 
belirsiz net olmayan patofizyolojik süreç veya teşhis hastalık


Aşağıdaki kod parçasını çalıştırarak gerekli olan bütün kütüphaneleri indirin
```python
pip install -r ner_requirements.txt
```

### 4.1 Annotation with AI

### 4.2 Augmentation ( Word Level )

Veri setini genişletmek ve model performansını artırmak amacıyla çeşitli veri artırma (Data Augmen
tation) teknikleri uygulanmıştır. Karakter değişimi, karakter silme, karakter ekleme ve eşanlamlı 
değiştirme gibi yöntemler kullanılmıştır. Karakter değişimi ile metindeki belirli karakterler farklı karak
terlerle değiştirilirken, karakter silme yöntemiyle belirli karakterler silinerek veri setinde çeşitlilik 
sağlanmıştır. Karakter ekleme ile metne ekstra karakterler eklenmiş ve eşanlamlı değiştirme ile be
lirli kelimeler eşanlamlılarıyla değiştirilmiştir. Bu teknikler, modelin küçük yazım hatalarına, farklı 
karakter varyasyonlarına, eksik veya hatalı karakter içeren verilerle, fazladan karakter içeren ver
ilerle ve farklı kelime kullanımlarıyla başa çıkma yeteneğini artırarak dayanıklılığını ve genelleme 
yeteneğini güçlendirmiştir. Bu teknikler, modelin daha geniş bir veri senaryosu yelpazesiyle eğitilme
sine ve test edilmesine olanak tanıyarak, gerçek dünya verilerindeki varyasyonlara karşı daha esnek 
ve güvenilir olmasını sağlamıştır. 

Gerekli kodlara [Augmentation.py](https://github.com/Aakayy7/SYZ-2024/blob/main/NER-CODES/augment_data.py)

#### Orjinal Metin

```python
Orjinal Metin

Bu örnek metin, karakter düzeyinde veri çoğaltma yöntemlerini göstermektedir.

```

#### character_swap

```python

# Define all the augmentation methods
def character_swap(text, swap_prob=0.1):
    text_chars = list(text)
    for i in range(len(text_chars) - 1):
        if random.random() < swap_prob:
            text_chars[i], text_chars[i + 1] = text_chars[i + 1], text_chars[i]
    return ''.join(text_chars)
```

![Alt Text](https://github.com/Aakayy7/SYZ-2024/raw/main/images/character-swap.png)


#### character_replacement

```python

def character_replacement(text, replace_prob=0.1):
    text_chars = list(text)
    for i in range(len(text_chars)):
        if random.random() < replace_prob:
            text_chars[i] = random.choice('abcdefghijklmnoprstuvyzABCDEFGHIJKLMNOPRSTUVYZ')
    return ''.join(text_chars)
```

![Alt Text](https://github.com/Aakayy7/SYZ-2024/raw/main/images/character_replacemant.png)


#### character_deletion

```python

def character_deletion(text, delete_prob=0.1):
    text_chars = list(text)
    new_text_chars = [ch for ch in text_chars if random.random() > delete_prob]
    return ''.join(new_text_chars)

```

![Alt Text](https://github.com/Aakayy7/SYZ-2024/raw/main/images/Character_deletion.png)


#### character_insertion

```python

def character_insertion(text, insert_prob=0.1):
    text_chars = list(text)
    new_text_chars = []
    for ch in text_chars:
        new_text_chars.append(ch)
        if random.random() < insert_prob:
            new_text_chars.append(random.choice('abcdefghijklmnoprstuvyzABCDEFGHIJKLMNOPRSTUVYZ'))
    return ''.join(new_text_chars)

```

![Alt Text](https://github.com/Aakayy7/SYZ-2024/raw/main/images/character_insertion.png)



### 4.3 Fix_it.py

Verilen JSONL dosyasının içindeki türkçe olmayan karakterleri türkçe karşılıkları ile değiştirir
eğer karşılığı yoksa siler.

Gerekli kodlara [Fix_İT.py](https://github.com/Aakayy7/SYZ-2024/blob/main/NER-CODES/fix_it.py)




### 4.4 Preprocessing

JSONL formatında alınan veriler dataframe çevrilir .

Gerekli Preprocessing kodlarına [NER_Preprocessing.py](https://github.com/Aakayy7/SYZ-2024/blob/main/NER-CODES/NER_Preprocessing.py)


<br>

Örnek Dataframe:
| sentence_id | word | label |
| --- | --- | --- |
| 0 | BİLATERAL | O | 
| 0 | MAMOGRAFİ | O |
| 0 | İNCELEMESİ | O |
| 0 | Her | O |
| 0 | iki | ANAT |



<br>



### 4.5 MODELS

Üretilen ve etiketlenen veriler ön işleme adımlarından geçirilerek SpaCy, CRF, LSTM ve BERT Base 
Turkish Cased modelleri eğitildi. LSTM modeli, diğer modellere göre sekans bazlı verilerde üstün 
performans sergileyerek karmaşık ilişkileri daha etkili bir şekilde işleyebilme yeteneğiyle öne çıktı. 
Türkçe metinlerdeki dil yapısını anlama ve uzun dönem bağımlılıkları yönetme konusundaki 
başarısı, LSTM modelinin seçilmesinde etkili oldu. Ayrıca, LSTM'nin sürekli olarak geliştirilen ve 
geniş kullanıcı kitlesi tarafından benimsenen bir model olması, güvenilirliğini ve etkinliğini pekiştirdi

<br>

Model çıktıları:
| Model Adı                 | Doğruluk (Acc) | macro avg |
|---------------------------|----------|----------------|
| [CRF](https://github.com/Aakayy7/SYZ-2024/blob/main/NER-CODES/CRF_NER.ipynb)| %80 | %75  |
| [SpaCy](https://github.com/Aakayy7/SYZ-2024/blob/main/NER-CODES/NER_SPACY.ipynb)| %75,9   | %72,3 |
| [LSTM](https://github.com/Aakayy7/SYZ-2024/blob/main/NER-CODES/ltsm.ipynb)| %95,3 | %84,3|



<br>

## BI-RADS Kategorisi Tahminleme 


### 5.1 Make Dataset

### 5.2 data visalization

### 5.3 Augmentation

### 5.4 MODELS

### 5.5 Transformers-İnterpret