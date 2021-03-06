# 12/2 BERT QA測試
## 模型使用 BERT -Base Chinese
![](https://i.imgur.com/Tbh5nBC.jpg)

## input length 的限制
### max_seq_length >> 調大
![](https://i.imgur.com/f3CXumS.jpg)
### 網上論壇
![](https://i.imgur.com/M7a15jB.jpg)
### BERT EXAMPLE 裡的註解
![](https://i.imgur.com/vaUjaQ6.jpg)

## 實際測試  (答對/答錯/答[CLS]) 
第一題測試 (一篇文章 六小題)
* **max_seq_length=384 ，取前300字放入 >>>>>>>>>>>>(5/1/0)**
* max_seq_length=384 ，取後300字放入 >>>>>>>>>>>>(0/0/6)
* max_seq_length=384 ，取前384字放入 >>>>>>>>>>>>(1/0/5)
* max_seq_length=384 ，取後384字放入 >>>>>>>>>>>>(0/0/6)
* **max_seq_length=512 ，取前450字放入 >>>>>>>>>>>>(5/1/0)**
* max_seq_length=512 ，取後450字放入 >>>>>>>>>>>>(1/0/5)
* max_seq_length=512 ，取前512字放入 >>>>>>>>>>>>(0/0/6)
* max_seq_length=512 ，取後512字放入 >>>>>>>>>>>>(0/0/6)
* max_seq_length=512 ，全535字放入   >>>>>>>>>>>>(1/0/5)

### 題目
蘇軾（1037年1月8日－1101年8月24日），眉州眉山（今四川省眉山市）人，北宋時著名的文學家、政治家、藝術家、醫學家。字子瞻，一字和仲，號東坡居士、鐵冠道人。嘉佑二年進士，累官至端明殿學士兼翰林學士，禮部尚書。南宋理學方熾時，加賜諡號文忠，複追贈太師。有《東坡先生大全集》及《東坡樂府》詞集傳世，宋人王宗稷收其作品，編有《蘇文忠公全集》。\n其散文、詩、詞、賦均有成就，且善書法和繪畫，是文學藝術史上的通才，也是公認韻文散文造詣皆比較傑出的大家。蘇軾的散文為唐宋四家（韓愈、柳宗元、歐蘇）之末，與唐代的古文運動發起者韓愈並稱為「韓潮蘇海」，也與歐陽修並稱「歐蘇」；更與父親蘇洵、弟蘇轍合稱「三蘇」，父子三人，同列唐宋八大家。蘇軾之詩與黃庭堅並稱「蘇黃」，又與陸游並稱「蘇陸」；其詞「以詩入詞」，首開詞壇「豪放」一派，振作了晚唐、五代以來綺靡的西崑體餘風。後世與南宋辛棄疾並稱「蘇辛」，惟蘇軾故作豪放，其實清朗；其賦亦頗有名氣，最知名者為貶謫期間借題發揮寫的前後《赤壁賦》。宋代每逢科考常出現其文命題之考試，故當時學者曰：「蘇文熟，喫羊肉、蘇文生，嚼菜羹」。藝術方面，書法名列「蘇、黃、米、蔡」北宋四大書法家（宋四家）之首；其畫則開創了湖州畫派；並在題畫文學史上佔有舉足輕重的地位。


## :heavy_check_mark:1. 蘇東坡在中國歷史上，是哪一個朝代的人？
*正確答案:北宋* 
*機器答案:北宋*
## :heavy_check_mark:2. 蘇東坡是中國哪個省份的人
*正確答案:四川省* 
*機器答案:四川省*
## :heavy_multiplication_x:3. 蘇東坡的爸爸叫什麼名字?
*正確答案:蘇洵* 
*機器答案:蘇軾*
## :heavy_check_mark:4. 蘇文忠公指的是誰?
*正確答案:蘇軾* 
*機器答案:蘇軾*
## :heavy_check_mark:5. 《蘇文忠公全集》是由何人編纂？
*正確答案:王宗稷* 
*機器答案:王宗稷*
## :heavy_check_mark:6. 韓愈在中國歷史上，是哪一個朝代的人？
*正確答案:唐代* 
*機器答案:唐代


* # 特別注意 若 max_seq_length調超過512 CUDA會爆，機器要重開才行，不然之後程式全爆
* # doc_stride調大 至512 並無任何變化
