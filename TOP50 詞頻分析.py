import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import jieba  # 用於中文分詞

# 下載必要的 NLTK 資料
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# 讀取 CSV 檔案
df = pd.read_csv('merged_records.csv', encoding='utf-8')

# 獲取英文停用詞列表
lst_stopwords = set(nltk.corpus.stopwords.words("english"))
# 加入一些常見的標點符號和特殊字符
lst_stopwords.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', 
                     '-', '_', '+', '=', '@', '#', '$', '%', '^', '&', '*', '/', '\\',
                     '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
# 加入新的停用詞
lst_stopwords.update(['The', 'In', 'This', 'We', 'study', 
                     'the', 'in', 'this', 'we', 'Study'])  # 加入大小寫版本

def analyze_word_frequency(texts, is_chinese=True):
    all_words = []
    
    for text in texts:
        if isinstance(text, str):
            # 移除標點符號和數字
            words = jieba.cut(text)
            # 只保留不在停用詞列表中的詞
            filtered_words = [word.strip() for word in words 
                            if word.strip() 
                            and word.strip() not in lst_stopwords 
                            and word.strip().lower() not in lst_stopwords  # 檢查小寫版本
                            and len(word.strip()) > 1]  # 過濾掉單字符
            all_words.extend(filtered_words)
    
    # 計算詞頻
    word_freq = Counter(all_words)
    
    # 取得前20個最常出現的詞
    most_common = word_freq.most_common(50)
    
    return most_common

# 分析 AB 欄位
print("\n=== AB 欄位詞頻分析（已剔除停用詞）===")
ab_frequencies = analyze_word_frequency(df['AB'])
print("\nAB 欄位詞頻分析結果（前50個最常出現的詞）：")
for word, freq in ab_frequencies:
    print(f"{word}: {freq}")

# 將結果存成 DataFrame
ab_freq_df = pd.DataFrame(ab_frequencies, columns=['詞語', '出現次數'])
print("\nAB 欄位詞頻統計表：")
print(ab_freq_df)

# 將結果儲存到 CSV 檔案
ab_freq_df.to_csv('AB_column_frequency_filtered.csv', encoding='utf-8-sig', index=False)

# # 顯示使用的停用詞列表
# print("\n使用的停用詞列表（這些詞已被剔除）：")
# print(sorted(list(lst_stopwords)))
