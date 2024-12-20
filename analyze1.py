## for data
import pandas as pd
import collections
import json
## for plotting
import matplotlib.pyplot as plt
import seaborn as sns
## for text processing
import re
import nltk
## for language detection
import langdetect 
## for sentiment
from textblob import TextBlob
## for ner
import spacy
## for vectorizer
from sklearn import feature_extraction, manifold
## for word embedding
import gensim.downloader as gensim_api
## for topic modeling
import gensim

def load_and_process_data(file_path):
    print("="*50)
    print("開始讀取數據...")
    
    # 讀取JSON數據
    lst_dics = []
    with open(file_path, mode='r', errors='ignore') as json_file:
        for dic in json_file:
            lst_dics.append(json.loads(dic))
    
    print(f"已讀取 {len(lst_dics)} 條記錄")
    
    # 創建DataFrame並過濾類別
    print("\n開始處理數據...")
    dtf = pd.DataFrame(lst_dics)
    original_size = len(dtf)
    dtf = dtf[dtf["category"].isin(['ENTERTAINMENT','POLITICS','TECH'])][["category","headline"]]
    filtered_size = len(dtf)
    
    print(f"原始數據量: {original_size}")
    print(f"過濾後數據量: {filtered_size}")
    print(f"過濾比例: {(filtered_size/original_size*100):.2f}%")
    
    dtf = dtf.rename(columns={"category":"y", "headline":"text"})
    
    # 添加語言檢測功能
    print("\n開始進行語言檢測...")
    def detect_language(text):
        try:
            return langdetect.detect(text) if text.strip() != "" else ""
        except:
            return "unknown"
    
    total = len(dtf)
    for i, (idx, row) in enumerate(dtf.iterrows(), 1):
        if i % 100 == 0:  # 每處理100條顯示一次進度
            print(f"處理進度: {(i/total*100):.2f}% ({i}/{total})", end='\r')
    
    dtf['lang'] = dtf["text"].apply(detect_language)
    print("\n語言檢測完成!")
    print("="*50)
    
    return dtf

def plot_category_distribution(dtf):
    # 設置中文字體
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 計算每個類別的數量和百分比
    category_counts = dtf['y'].value_counts()
    total = len(dtf)
    category_percentages = (category_counts / total * 100).round(2)
    
    # 創建類別分布圖
    plt.figure(figsize=(12, 6))
    ax = sns.countplot(data=dtf, x='y')
    plt.title('文獻類別分布')
    plt.xlabel('類別')
    plt.ylabel('數量')
    
    # 在每個柱子上添加數量和百分比標籤
    for i, (count, percentage) in enumerate(zip(category_counts, category_percentages)):
        ax.text(i, count, f'{count}\n({percentage}%)', 
                ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def clean_and_tokenize(text):
    print("--- 原始文本 ---")
    print(text)
    
    print("--- 清理後文本 ---")
    # 轉小寫，移除標點符號和特殊字符
    cleaned_text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    print(cleaned_text)
    
    print("--- 分詞結果 ---")
    # 分詞
    tokens = cleaned_text.split()
    print(tokens)
    
    return tokens

def analyze_data(dtf):
    print("\n開始數據分析...")
    print("="*50)
    
    # 基本統計信息
    total_records = len(dtf)
    
    # 計算語言分布及百分比
    print("分析語言分布...")
    lang_dist = dtf['lang'].value_counts()
    lang_percent = (lang_dist / total_records * 100).round(2)
    
    # 計算類別分布及百分比
    print("分析類別分布...")
    category_dist = dtf['y'].value_counts()
    category_percent = (category_dist / total_records * 100).round(2)
    
    # 顯示詳細統計結果
    print("\n數據集統計結果：")
    print("="*50)
    print(f"總記錄數：{total_records}")
    
    print("\n語言分布：")
    print("-"*30)
    for lang, (count, percent) in enumerate(zip(lang_dist, lang_percent)):
        print(f"{lang_dist.index[lang]:<10}: {count:>6} 筆 ({percent:>6.2f}%)")
    
    print("\n類別分布：")
    print("-"*30)
    for cat, (count, percent) in enumerate(zip(category_dist, category_percent)):
        print(f"{category_dist.index[cat]:<15}: {count:>6} 筆 ({percent:>6.2f}%)")
    
    # 文本處理示例
    print("\n文本處理示例：")
    print("="*50)
    print("選擇第一條文獻進行處理...")
    sample_text = dtf['text'].iloc[0]
    clean_and_tokenize(sample_text)

def main():
    print("="*50)
    print("開始執行文獻分類分析程序")
    print("="*50)
    
    # 讀取和處理數據
    file_path = 'News_Category_Dataset_v3.json'
    dtf = load_and_process_data(file_path)
    
    # 顯示數據樣本
    print("\n數據預覽（前5筆）：")
    print(dtf.head())
    
    # 分析數據
    analyze_data(dtf)
    
    # 繪製類別分布圖
    print("\n生成視覺化圖表...")
    plot_category_distribution(dtf)
    
    print("\n分析完成!")
    print("="*50)

if __name__ == "__main__":
    main()
