# Speech Recognize  
[TOC]  

Colab：  
1.NLP feature  
https://colab.research.google.com/drive/1IXyOCb6f4-mykDjKvlW1rL_kOPINhks9  
2.Multiple_choice  
https://pse.is/JFYEY  
3.Question answering  
https://pse.is/LUC26

## Wordnet  
Synset: Synonym set  
Hypernym：fruit is apple's hypernym   
Hyponym：apple is fruit's hyponym  
antonymy:反義  
meronymy:part to whole relation  
![](https://i.imgur.com/CatSe49.png)  
### Synonym and antonyms
```
good = wn.synset('good.a.01')
print("good 的定義： {}".format(good.definition()))
print("good 的例子： {}".format(good.examples()))
print("good 的同義詞： {}".format(good.lemmas()))
print("good 的反義詞： {}".format(good.lemmas()[0].antonyms()))

#good 的定義： having desirable or positive qualities especially those suitable for a thing specified
#good 的例子： ['good news from the hospital', 'a good report card', 'when she was good she was very very good', 'a good knife is one good for cutting', 'this stump will make a good picnic table', 'a good check', 'a good joke', 'a good exterior paint', 'a good secretary', 'a good dress for the office']
#good 的同義詞： [Lemma('good.a.01.good')]
#good 的反義詞： [Lemma('bad.a.01.bad')]
```  
### Hypernym and Hyponym  
```
school = wn.synset('school.n.01')
print("school的上位詞: {}".format(school.hypernyms()))
print("school的下位詞: {}".format(school.hyponyms()))
```  
### More relation   
```
tree = wn.synset('tree.n.01') # 樹
print("很多樹我們稱作什麼？ {}".format(tree.member_holonyms()))
body = wn.synset('body.n.01') # 樹
print("身體有哪些部位？ {}".format(body.part_meronyms()))

#很多樹我們稱作什麼？ [Synset('forest.n.01')]
#身體有哪些部位？ [Synset('arm.n.01'), Synset('articulatory_system.n.01'), Synset('body_substance.n.01'), Synset('cavity.n.04'), Synset('circulatory_system.n.01'), Synset('crotch.n.02'), Synset('digestive_system.n.01'), Synset('endocrine_system.n.01'), Synset('head.n.01'), Synset('leg.n.01'), Synset('lymphatic_system.n.01'), Synset('musculoskeletal_system.n.01'), Synset('neck.n.01'), Synset('nervous_system.n.01'), Synset('pressure_point.n.01'), Synset('respiratory_system.n.01'), Synset('sensory_system.n.02'), Synset('torso.n.01'), Synset('vascular_system.n.01')]
```
## CKIP Tagger  
### Standard
```
ws = WS("./data")
sentence_list = [
    "唯一支持韓國瑜",
    "唯一支持韓國瑜選總統",
    "唯一支持韓國輸",
    "讓韓國愉快不起來",
    "讓韓國瑜快不起來",
    "請把手放下",
    "全臺大停電",
    "全台大停電",
word_sentence_list = ws(
    sentence_list,    
)    
print("一般模式：")
for word_sentence in word_sentence_list:
  print(word_sentence)
  
#一般模式：
#['唯一', '支持', '韓國', '瑜']
#['唯一', '支持', '韓國', '瑜選', '總統']
#['唯一', '支持', '韓國', '輸']
#['讓', '韓國', '愉快', '不', '起來']
#['讓', '韓國瑜', '快', '不', '起來']
#['請', '把', '手', '放下']
#['全', '臺', '大', '停電']
#['全', '台', '大', '停電']

```
### With weight   
```
word_to_weight = {
    "韓國瑜": 10000,
    "臺大": 100,
}
dictionary = construct_dictionary(word_to_weight)
word_sentence_list = ws(
    sentence_list,
    coerce_dictionary=dictionary
)
print("\n自訂詞典：{}".format(word_to_weight))
for word_sentence in word_sentence_list:
  print(word_sentence)
 
#自訂詞典：{'韓國瑜': 10000, '臺大': 100}
#['唯一', '支持', '韓國瑜']
#['唯一', '支持', '韓國瑜', '選', '總統']
#['唯一', '支持', '韓國', '輸']
#['讓', '韓國', '愉快', '不', '起來']
#['讓', '韓國瑜', '快', '不', '起來']
#['請', '把', '手', '放下']
#['全', '臺大', '停電']
#['全', '台', '大', '停電']
```
## Word2vec  
```
query = "高雄"
print('相似詞前十排序')
res = word2vec_model.most_similar(query, topn=10)

for item in res:
    print(item[0] +','+ str(item[1]))

'''
相似詞前十排序
臺南,0.6982719302177429
臺中,0.6878253221511841
高雄市,0.6727442741394043
臺北,0.6553664207458496
屏東,0.6454926133155823
花蓮,0.6125885248184204
左營,0.6014320850372314
新竹,0.5794859528541565
基隆,0.5742310881614685
桃園,0.5702739357948303
'''
```  
## Bert  
```
first_sent = "醒醒吧，" # 第一句話
second_sent = "發大財。" # 第二句話
tokens_tensor, segments_tensors = preprocess_two_sents(first_sent, second_sent)

print("句子裡每個字的編號組成的向量： {}".format(tokens_tensor))
print("句子裡每個字屬於哪句話的向量： {}".format(segments_tensors))

句子裡每個字的編號組成的向量： tensor([[ 101, 7008, 7008, 1416, 8024,  102, 6243, 5050, 5440, 2719,  677,  678,
         3152, 4638, 2099, 1403, 7030,  511,  102]])
句子裡每個字屬於哪句話的向量： tensor([[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
```
### Leak word  
```
sent = "蔡英文是總統。"
masked_indices = [5] # 要遮住第幾個字？
# masked_token: 被遮住的字，tokens_tensor: 句子裡每個字的編號組成的向量，segments_tensor: 句子裡每個字屬於哪句話的向量
masked_tokens, tokens_tensor, segments_tensors = preprocess(sent, masked_indices)

'''
tokenized_text before masking: ['[CLS]', '蔡', '英', '文', '是', '總', '統', '。', '[SEP]']
tokenized_text after masking: ['[CLS]', '蔡', '英', '文', '是', '[MASK]', '統', '。', '[SEP]']
原字：['總']; 模型預測：['總']
'''
```
### Relating  
```
# 蘋果作爲一個水果
apple_1_sents = ["台灣蘋果有蠻多種類的  大都靠外銷  但是我覺得好吃的是新鮮的",
                 "他說，期盼日後有機會能親訪青森縣，體驗蘋果、美麗的風景、和富含歷史的鐵道文化。",
                 "蘋果跟橘子的價格又跌了",
                 "今天買了蘋果來吃",
                 "這些日子無聊的以蘋果餐為代餐",
                 "一芳的蘋果茶真D難喝"
                 "其中進口蘋果(富士)平均每公斤下跌12.3%，奇異果下跌3.8%，甜桃下跌14.2%。",
                 "每年接近5月，很多老饕都知道智利的水蜜桃蘋果季節即將到來。",
                 "進口蘋果因防止水分流失、延長保存期故添加人工果蠟。",
                 "你可能不知道，蘋果富含對人體有益的細菌，整顆蘋果若連皮吃掉，不僅會吃進更多的纖維，也會比別人吃進多十倍對腸胃道有益的細菌。",
                 "一芳水果茶主打水果茶，不過店員製作時，只有看到蘋果和柳丁，原來他們是搭配使用釀鳳梨醬罐頭調飲料，但就被質疑不是天然的水果茶，"
                  "代工廠發聲明強調，他們都是用天然水果製成罐頭，一芳也強調原物料都合法合規。"]

# 蘋果作爲一家公司
apple_2_sents = ["蘋果即將於下月發表新款 iPhone，許多外媒、爆料者陸續揭露相關規格與情報，唯獨命名沒有定論。",
                 "蘋果獲新 Face ID 專利，就連 MacBook 都能臉部辨識",
                 "今天買了蘋果手機",
                 "蘋果公司的股價又跌了",
                 "因為沒有達到當初銷售目標，蘋果向三星電子提供 8,000 億韓圜 （約 6.84 億美元），彌補三星供應蘋果 iPhone OLED 面板的投資成本。",
                 "2013年，蘋果首度押寶指紋辨識技術，引爆全球各大智慧手機廠急起跟進。",
                 "摩根士丹利認為，由於 iPhone 在中國的銷售情形好轉，蘋果 (AAPL-US) 股價可望上漲。",
                 "展望本季，隨著新iPhone即將於9月亮相，蘋果對相關代工廠展開拉貨，和碩本季營運將逐月走高。",
                 "蘋果 2019 年第 3 季財報優於華爾街預期，盤後股價大漲逾 4%",
                 "韓國三星公司一直使用嘲諷策略來宣傳自家的產品，自從美國Apple公司在 iPhone 7開始移除了3.5mm耳機孔後，"
                 "三星除了在發表會上嘲諷iPhone沒有耳機孔外，還發布了多支廣告影片來嘲諷蘋果用戶和iPhone沒有耳機孔。"]          
```
![](https://i.imgur.com/nrUjeyf.png)  
![](https://i.imgur.com/DdwX8cy.png)  
![](https://i.imgur.com/7x4zi4x.png)  
### Multiple_choice   
利用Bert_chinese   
```
context = '關於芒果品種的小故事，土芒果是最早來台灣的芒果，是荷蘭人在台期間引進，同時也是產期最早的芒果，早生品種在4月即可採收，盛產期5至7月，果皮黃綠色，纖維較粗，酸甜有味，是許多人兒時記憶中的味道；愛文則是台灣栽培最多的主力品種，1954年由美國佛羅里達引進，產期5至7月，外皮鮮紅討喜，甜酸適中符合現代大眾的喜好；凱特同樣與愛文同時引進，但較晚生成，產期可延遲至9至10月，故有「九月檨」之稱號，果實外觀黃色，為較大之卵圓型，果肉多汁而帶微酸。'
question = '請問台灣的主力芒果品種為何？'
choices = ['小芒果', '土芒果', '愛文芒果', '凱特芒果']
label = 3
data, tokens = convert_examples_to_features(context, question, choices, tokenizer, 2)
pred = predict(data, model, tokenizer, device=device)

print('\n\n', context)
print(question)
print('Predicted answer: ', choices[pred])
print('Ground truth: ', choices[label-1])

#Predicted answer:  愛文芒果
#Ground truth:  愛文芒果
```  
```
question = '土芒果是哪國人引進的？'#'請問何者與愛文同時引進?'
choices = ['荷蘭', '台灣', '美國', '日本']
label = 1
data, tokens = convert_examples_to_features(context, question, choices, tokenizer, 4)
pred = predict(data, model, tokenizer, device=device)

print('\n\n', context)
print(question)
print('Predicted answer: ', choices[pred])
print('Ground truth: ', choices[label-1])
  
#Predicted answer:  荷蘭
#Ground truth:  荷蘭
```
### Question answering  
```
context = '國立臺灣大學，簡稱臺大、NTU，是臺灣第一所現代綜合大學，為臺灣學生人數最多的高等教育機構。其始於1928年日治時代中期創校的「臺北帝國大學」[注 1]，1945年中華民國接收臺灣後經改制與兩次易名始用現名。現設有11個學院、3個專業學院，下分54個學系、109個研究所；另設有30餘個各學術領域之國家級或校級研究中心，以及進修推廣部、臺大醫院等附屬機構，是全臺唯一學生人數超過三萬的高等教育學校[11][14]。目前亦為高教深耕計畫中參與全球鏈結全校型計畫的4所學校之一[15][16]。2020年QS世界大學排名位居第69名。此外，臺大擁有臺北市境內的3大校區、以及多處散布於全臺的分支校區與校地，總面積約3萬4千公頃，佔臺灣土地總面積的百分之一。'
question = '什麼大學是台灣學生人數最多的教育機構'
data, tokens = convert_examples_to_features(tokenizer=tokenizer, question_text=question, doc_tokens=context)
start, end = evaluate(data, model, tokenizer)
"".join(tokens[start[0]: end[0]+1])

#'國立臺灣大學'
```