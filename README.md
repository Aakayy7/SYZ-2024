# LayerLords

<img src="https://github.com/Aakayy7/SYZ-2024/raw/main/images/logo.jpeg" alt="Logo" width="700" height="450"/>


#### TEKNOFEST 2024 Sağlıkta Yapay Zeka Yarışması Türkiye 2. LayerLords Takımı Kodları
Projemizde, bilgisayarlı görü teknikleri kullanarak mamografi görüntülerinde kitle ve kalsifikasyon tespiti yapmayı ve BI-RADS (1, 2, 4, 5) kategorilerini tahmin etmeyi başardık. Mamografi radyoloji raporlarından varlık ismi çıkarımı (Named Entity Recognition) gerçekleştirerek, raporlardaki önemli tıbbi terimleri otomatik olarak tanımladık. Ayrıca, bu raporlardan BI-RADS kategorilerini doğru bir şekilde tahmin eden modeller geliştirdik. Bu yaklaşımlar ile mamografi değerlendirmelerinin hızını ve tutarlılığını artırarak sağlık profesyonellerine teşhis ve tedavi süreçlerinde önemli bir destek sağlamayı hedefliyoruz.


## Takım Üyeleri

---

-  Ahmet Akay - [LinkedIn](https://www.linkedin.com/in/ahmet-akayy/), [Github](https://github.com/Aakayy7) <br/>
-  Zafer Khaliqi - [LinkedIn](https://www.linkedin.com/in/zafer-khaliqi-a50464286/), [Github](https://github.com/zaferkhaliqi) <br/>



## Bölümler

1. [Görüntü Ön İşleme Adımları](#preprocec)
2. [BI-RADS Sınıflandırma ](#birads)
3. [Kitle ve Kalsfikasyon Tespiti](#kitle)
4. [Varlık İsmi Çıkarımı(NER)](#ner)
5. [BIRADS Kategori Tahmini (TEXT CLASFFICATION) ](#tclassfication)




## Görüntü Ön İşleme Adımları

- Dicom to PNG

Veri kümesi boyutu 70 GB idi. Daha sonra, DICOM formatındaki veriler Pydicom kütüphanesi kullanılarak PNG formatına dönüştürüldü ve böylece işlenmesi daha kolay hale geldi.

![Alt Text](https://github.com/Aakayy7/SYZ-2024/raw/main/images/0.png)


- Siyah ve Beyaz Görüntüleri Ayarlama

Görüntüler DICOM'dan PNG'ye dönüştürüldükten sonra, bazıları beyaz modda dönüştürülürken diğerleri siyah moddaydı. Bu karışıklığı gidermek için tüm görüntülerin ya beyaz ya da siyah modda olması gerekir. Tıbbi görüntülemede genellikle siyah mod tercih edildiği için siyah mod seçilmiştir.

![Alt Text](https://github.com/Aakayy7/SYZ-2024/raw/main/images/1.jpg)


- Görüntü Kırpma

Görüntü kırpma işlemi, MakeSenseAI platformunda nesne algılama modelini (YOLOv8s) eğitmek için meme bölgeleri etiketlenerek manuel olarak gerçekleştirilmiştir. Etiketlenen veriler modeli eğitmek için kullanılmış ve daha sonra yerel bilgisayara indirilmiştir. Daha sonra, meme bölgeleri için sınırlayıcı kutu bilgileri elde edilmiş ve görüntüler bu alanlardan kırpılmıştır.


![Alt Text](https://github.com/Aakayy7/SYZ-2024/raw/main/images/2.jpg)


- Yeniden Boyutlandırma

Görüntüler 512x512 olarak yeniden boyutlandırılarak, sinir ağlarının giriş katmanındaki nöron sayısı sabit tutulmuştur.

![Alt Text](https://github.com/Aakayy7/SYZ-2024/raw/main/images/3.jpg)




##  BI-RADS Sınıflandırması

BIRADS için YOLO sınıflandırma modellerinden biri kullanılmıştır. YOLO öncelikle nesne tespiti için bilinmesine rağmen, geleneksel sinir ağı mimarisi nedeniyle sınıflandırma yapma kabiliyetine de sahiptir.


![Alt Text](https://github.com/Aakayy7/SYZ-2024/raw/main/images/7.jpg)



## Kitle ve Kalsifikasyon Tespiti 


![Alt Text](https://github.com/Aakayy7/SYZ-2024/raw/main/images/combined_kitle_kals.jpeg)


Kitleler ve kalsifikasyonlar için görüntüler önce kırpılmış ve modele (YOLOv10x) beslenmiştir. Ardından görüntüler basitçe yeniden boyutlandırılarak modele sunulmuş ve BIRADS modelinin aksine doğrulukta bir fark gözlenmemiştir. Bu nedenle test sırasında karmaşıklığı ve hata oranlarını azaltmak için görüntüler sadece yeniden boyutlandırılarak modele sunulmuştur.



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


### 4.1 Preprocessing

JSONL formatında alınan veriler dataframe (csv) ve Conll formatına çevrilir .


Gerekli Preprocessing kodlarına [NER_Preprocessing.py](https://github.com/Aakayy7/SYZ-2024/blob/main/NER-CODES/NER_Preprocessing.py)


[csv to conll](https://github.com/Aakayy7/SYZ-2024/blob/main/NER-CODES/conl_maker.py)



<br>

Örnek Dataframe:
| sentence_id | word | label |
| --- | --- | --- |
| 0 | BİLATERAL | O | 
| 0 | MAMOGRAFİ | O |
| 0 | İNCELEMESİ | O |
| 0 | Her | O |
| 0 | iki | ANAT |


Örnek conll

```python

Meme O
parankimi O
heterojen O
yoğun O
olduğu O
için O
küçük O
boyutlu OBS-PRESENT
lezyonlar OBS-PRESENT
mamografide OBS-PRESENT
saptanamayabilir OBS-PRESENT

```

<br>



### 4.2 MODELS

Üretilen ve etiketlenen veriler ön işleme adımlarından geçirilerek SpaCy, CRF, LSTM ve bert-base-turkish-ner-cased modelleri eğitildi. BERT modeli, diğer modellere göre sekans bazlı verilerde üstün performans sergileyerek karmaşık ilişkileri daha etkili bir şekilde işleyebilme yeteneğiyle öne çıktı. 
Türkçe metinlerdeki dil yapısını anlama ve uzun dönem bağımlılıkları yönetme konusundaki 
başarısı, BERT modelinin seçilmesinde etkili oldu. Ayrıca, BERT'ün sürekli olarak geliştirilen ve 
geniş kullanıcı kitlesi tarafından benimsenen bir model olması, güvenilirliğini ve etkinliğini pekiştirdi

<br>

Model çıktıları:
| Model Adı                 | Doğruluk (Acc) | macro avg |
|---------------------------|----------|----------------|
| [CRF](https://github.com/Aakayy7/SYZ-2024/blob/main/NER-CODES/CRF_NER.ipynb)| %80 | %75  |
| [SpaCy](https://github.com/Aakayy7/SYZ-2024/blob/main/NER-CODES/NER_SPACY.ipynb)| %75,9   | %72,3 |
| [LSTM](https://github.com/Aakayy7/SYZ-2024/blob/main/NER-CODES/ltsm.ipynb)| %95,3 | %84,3|
| [BERT Base NER Turkish Cased ](https://github.com/Aakayy7/SYZ-2024/blob/main/NER-CODES/BERT_NER.ipynb)| %92 | %86 |


[BERT Modeli Linki](https://huggingface.co/AAkay/bert_ner_model_for_syz_2024)


<br>


## BI-RADS Kategorisi Tahminleme

Mamografi raporlarından BIRADS sınıflarını tahmin eden bir yapay zeka modeli geliştirildi .Bu model, raporları analiz ederek BIRADS 1, 2, 3, 4, ve 5 kategorilerine sınıflandırma yapabiliyor.Amacımız, doktorlara destek olarak, meme kanseri risk seviyelerinin belirlenmesine katkı sağlamak ve erken teşhis süreçlerinde daha hızlı ve güvenilir sonuçlar elde edilmesine yardımcı olmaktır. 


Kategoriler :

- BIRADS 1 (Negatif): Meme dokusu tamamen normaldir. Görüntülemede hiçbir anormal bulgu, kitle, kireçlenme veya lezyon tespit edilmemiştir. Bu durumda, hastanın mamografisi tamamen temiz olarak değerlendirilir.

- BIRADS 2 (Benign): Meme dokusunda, kansersiz ve tamamen iyi huylu bir durum tespit edilmiştir. Örneğin, bir kist, fibrokistik değişiklikler, veya iyi huylu bir kireçlenme gibi bulgular görülebilir. Bu tür bulgular, kansere işaret etmez ve genellikle tedavi veya takip gerektirmez.

- BIRADS 3 (Muhtemelen Benign): Küçük bir olasılıkla kötü huylu olabilecek bir bulgu tespit edilmiştir. Genellikle 6 ay sonra tekrar kontrol edilmesi önerilir.

- BIRADS 4 (Şüpheli Malignite): Kötü huylu, yani kanser olma olasılığı olan bir bulgu mevcuttur. Bu durum genellikle biyopsi gibi ileri tetkikler gerektirir. BIRADS 4, kendi içinde kanser olasılığına göre 4A (düşük), 4B (orta), ve 4C (yüksek) olarak alt kategorilere ayrılabilir.

- BIRADS 5 (Yüksek Olasılıkla Malignite): Kanser olma olasılığı çok yüksektir. Bu durumda acil biyopsi ve tedavi gereklidir.


```python
pip install -r tc-requirements.txt
```


### 5.1 Veri Seti Hazırlama


[Bu Python scripti](https://github.com/Aakayy7/SYZ-2024/blob/main/TEXT-CLASSFICATION/convert_dataset.py), mamografi raporları içeren bir veri setini işleyerek BIRADS sınıflarını tahmin edebilen bir makine öğrenmesi modeli için veri hazırlamak amacıyla tasarlanmıştır. Script şu işlemleri gerçekleştirir:

1 - Veri Setinin Gezinmesi: Ana dizin altında bulunan alt klasörlerdeki metin dosyalarını (mamografi raporları) rekürsif olarak tarar.

2 - BIRADS Skorlarının tespit edilmesi : Metin dosyalarında "BIRADS" veya "SONUÇ" gibi anahtar kelimeler aranarak raporların BIRADS skorları tespit edilir ve csv dosyasına dosya adı ile beraber ait oldupu sınıf kaydedilir  .

3 - Skorların Normalizasyonu: Çıkarılan BIRADS skorları, tutarlılık sağlamak amacıyla normalize edilir. Bu işlem, raporda farklı şekillerde yazılmış (örneğin "BIRADS 4", "BIRADS Düzey 4" gibi) BIRADS skorlarının, standart bir formatta sunulmasını sağlar. Script, metin içerisindeki sayıları algılar ve sadece skoru temsil eden numarayı alarak diğer gereksiz karakterleri temizler. Bu sayede, modelin eğitiminde kullanılacak verilerdeki BIRADS skorları tekdüze hale getirilmiş olur.

4 - Metin Verisinin Temizlenmesi: BIRADS skorları tespit edildikten sonra, bu skorların bulunduğu satırlar metin dosyasından kaldırılır. Bu işlem, raporun kalan kısmını temizleyerek, eğitime uygun hale getirilmesini sağlar. Böylece, modelin yalnızca raporun geri kalan kısmından öğrenme yapması sağlanır, bu da modelin daha doğru tahminler yapmasına katkıda bulunur.

5 - Veri Hazırlığı: Çıkarılan skorlar ve ilgili dosya adlarını ve içeriklerini bir csv dosyası halinde derleyerek, model eğitimi veya daha ileri işleme için hazır hale getirir.


Örnek Dataset:
| Content | BIRADS SCORE |
| --- | --- |
| BİLATERAL MAMOGRAFİ İNCELEMESİ:\nMeme parankim...	 | 1 | 
| BİLATERAL MAMOGRAFİ İNCELEMESİ\nMeme paterni:T...	 | 3 |
| Her iki memeye yönelik Mammografi İncelemesi,\...	 | 5 |
| BİLATERAL MAMOGRAFİ İNCELEMESİ:\nCilt-cilt alt...	 | 4 |


### 5.2 Veri Görselleştirme

![Alt Text](https://github.com/Aakayy7/SYZ-2024/raw/main/images/analyzegraph.png)

Grafik incelendiğinde BIRADS 1 kategorisine ait yetersiz veri olduğu tespit edilmiştir . Bu durum veri dengesizliğine yol açmakta ve Modelin BIRADS1 kategorisine ait verilerde kötü sonuçlar doğurmasına sebep olmuştur .

### 5.3 Gemini ile Veri Arttırımı (Data Augmentation)

Bi önceki bölümde bahsedilen Veri dengesizliği sorununa çözüm olarak [veri arttırımı yöntemi](https://github.com/Aakayy7/SYZ-2024/blob/main/TEXT-CLASSFICATION/Data_Production.py) geliştirilmiştir . 


Bu yöntemde amaç Gemini 1.5 Flash modeli kullanılarak 1 tane rapordan 4 tane yeni rapor üretmektir . Projede kullanılan yöntem, prompt tabanlıdır ve bu prompt başka kategoriler (labeller) için de uygulanabilir.


Kullanılan Gemini 1.5 Flash modeli için ücretsiz olarak [buradan](https://aistudio.google.com/app/apikey) API KEY alabilirsiniz . Bu kodu çalıştırmadan önce kendi API KEY girmeyi unutmayın 

```python

genai.configure(api_key="WRİTE_YOUR_OWN_API_KEY") # Add your API_KEY 
model = genai.GenerativeModel('gemini-1.5-flash', safety_settings=safety_settings)

```

Aşağıda veri artırımı işlemini gerçekleştirmek için yazılan promptu, çoğaltmak istediğiniz sınıfa göre ayarlayınız.

```python

response = model.generate_content("""

    Sen bir yapay zeka modelisin ve verilen Türkçe memeografi raporunu genel anlam ve kelime anlamı açısından analiz eden bir sentetik veri üreticisisin.

    Bu rapor, BIRADS 1 kategorisine aittir ve normal, herhangi bir malignite belirtisi olmayan bulgular içermektedir.

    Aşağıda verilen raporu analiz et ve anla. Anlamını anladıktan sonra, bu raporu hazırlayan bir doktorun aynı BIRADS 1 kategorisinde benzer raporları nasıl yazmış olabileceğini düşün. Bu doğrultuda, orijinal raporun uzunluğuna yakın ve tıbbi doğruluğa sahip 4 adet yeni rapor üret.

    Sadece aşağıdaki Python liste formatında yaz. Başka hiçbir şey yazma:
    Çıktı formatı: ["<rapor1>", "<rapor2>", "<rapor3>", "<rapor4>"]

    \n\n
    Orijinal Rapor: """ + sentence)

```


### 5.4 World Level Augmentation 


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



### 5.5 MODELS


Model çıktıları:
| Model Adı                 | Doğruluk (Acc) | macro avg |
|---------------------------|----------|----------------|
| [Logistic Regression](https://github.com/Aakayy7/SYZ-2024/blob/main/TEXT-CLASSFICATION/CRF_NER.ipynb)| %91 | %79 |
| [Naive Bayes ](https://github.com/Aakayy7/SYZ-2024/blob/main/TEXT-CLASSFICATION/NER_SPACY.ipynb)| %80 | %64 |
| [XGboost](https://github.com/Aakayy7/SYZ-2024/blob/main/TEXT-CLASSFICATION/ltsm.ipynb)| %96 | %86|
| [LSTM](https://github.com/Aakayy7/SYZ-2024/blob/main/TEXT-CLASSFICATION/ltsm.ipynb)| %95 | %90|
| [BERT](https://github.com/Aakayy7/SYZ-2024/blob/main/TEXT-CLASSFICATION/ltsm.ipynb)| %99 | %99|


[BERT Modeli Linki](https://huggingface.co/AAkay/bert_classfication_model_for_syz_2024)

### 5.5 Transformers-İnterpret 


[tranformers-interpret](https://github.com/Aakayy7/SYZ-2024/blob/main/TEXT-CLASSFICATION/transformers_interpret.ipynb) kütüphanesi kullanılmıştır . Bu sayede çıktıdaki kelimelerin hedef etiket ile nasıl bir korelasyona sahip olduğu görülmektedir.


BIRADS1 

![Alt Text](https://github.com/Aakayy7/SYZ-2024/raw/main/images/inter_pre.png)
