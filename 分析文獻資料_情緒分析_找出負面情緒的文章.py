import pandas as pd
from textblob import TextBlob
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 設定中文字型
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

def analyze_sentiment(text):
    """進行情緒分析的函數"""
    try:
        text = str(text)
        analysis = TextBlob(text)
        polarity = analysis.sentiment.polarity
        subjectivity = analysis.sentiment.subjectivity
        return polarity, subjectivity
    except:
        return np.nan, np.nan

def process_sentiment_analysis(input_file):
    """處理情緒分析的主要函數"""
    encodings = ['big5', 'cp950', 'gbk', 'gb2312', 'gb18030', 'shift-jis', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            print(f"嘗試使用 {encoding} 編碼...")
            df = pd.read_csv(input_file, encoding=encoding, usecols=['serial', 'AB'])
            print(f"成功使用 {encoding} 編碼讀取檔案")
            
            # 進行情緒分析
            print("\n進行情緒分析...")
            df[['sentiment_polarity', 'sentiment_subjectivity']] = pd.DataFrame(
                df['AB'].apply(analyze_sentiment).tolist(), 
                index=df.index
            )
            
            # 將結果分類
            df['sentiment_category'] = pd.cut(
                df['sentiment_polarity'],
                bins=[-1, -0.1, 0.1, 1],
                labels=['負面', '中性', '正面']
            )
            
            # 生成輸出檔名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f'sentiment_analysis_{timestamp}.csv'
            
            # 儲存結果
            df.to_csv(output_filename, encoding='utf-8', index=False)
            print(f"\n分析結果已儲存至 {output_filename}")
            
            return df, output_filename
            
        except UnicodeDecodeError:
            print(f"{encoding} 編碼無法讀取檔案")
            continue
    
    return None, None

def analyze_negative_sentiment(analysis_file):
    """分析負面情緒資料的函數"""
    try:
        # 讀取情緒分析結果
        df = pd.read_csv(analysis_file, encoding='utf-8')
        
        # 篩選負面情緒資料
        negative_df = df[df['sentiment_category'] == '負面']
        
        # 顯示資訊
        print("\n負面情緒資料統計：")
        print(f"總筆數：{len(negative_df)}")
        print("\n情緒極性統計：")
        print(negative_df['sentiment_polarity'].describe())
        
        # 儲存負面情緒資料
        output_filename = f'negative_sentiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        negative_df.to_csv(output_filename, encoding='utf-8', index=False)
        print(f"\n負面情緒資料已儲存至 {output_filename}")
        
        return negative_df, output_filename
        
    except Exception as e:
        print(f"分析負面情緒資料時發生錯誤：{str(e)}")
        return None, None

def generate_sentiment_statistics(df):
    """生成情緒分析的詳細統計表"""
    try:
        print("\n開始生成情緒分析統計表...")
        
        # 計算基本統計數據
        total_count = len(df)
        positive_df = df[df['sentiment_category'] == '正面']
        negative_df = df[df['sentiment_category'] == '負面']
        neutral_df = df[df['sentiment_category'] == '中性']
        
        # 創建統計表
        stats_dict = {
            '類別': ['總計', '正面', '負面', '中性'],
            '數量': [
                total_count,
                len(positive_df),
                len(negative_df),
                len(neutral_df)
            ],
            '百分比': [
                '100%',
                f'{(len(positive_df)/total_count*100):.2f}%',
                f'{(len(negative_df)/total_count*100):.2f}%',
                f'{(len(neutral_df)/total_count*100):.2f}%'
            ],
            '平均情緒極性': [
                f'{df["sentiment_polarity"].mean():.4f}',
                f'{positive_df["sentiment_polarity"].mean():.4f}',
                f'{negative_df["sentiment_polarity"].mean():.4f}',
                f'{neutral_df["sentiment_polarity"].mean():.4f}'
            ],
            '最大極性值': [
                f'{df["sentiment_polarity"].max():.4f}',
                f'{positive_df["sentiment_polarity"].max():.4f}',
                f'{negative_df["sentiment_polarity"].min():.4f}',
                f'{neutral_df["sentiment_polarity"].mean():.4f}'
            ],
            '最小極性值': [
                f'{df["sentiment_polarity"].min():.4f}',
                f'{positive_df["sentiment_polarity"].min():.4f}',
                f'{negative_df["sentiment_polarity"].max():.4f}',
                f'{neutral_df["sentiment_polarity"].mean():.4f}'
            ]
        }
        
        # 創建DataFrame
        stats_df = pd.DataFrame(stats_dict)
        
        # 儲存為CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stats_filename = f'AI_sentiment_statistics_{timestamp}.csv'
        stats_df.to_csv(stats_filename, encoding='utf-8', index=False)
        
        # 儲存為Excel（更易讀的格式）
        excel_filename = f'AI_sentiment_statistics_{timestamp}.xlsx'
        stats_df.to_excel(excel_filename, index=False, sheet_name='情緒分析統計')
        
        # 打印統計表
        print("\nAI 代理文獻情緒分析統計表：")
        print("=" * 80)
        print(stats_df.to_string(index=False))
        print("=" * 80)
        print(f"\n統計表已儲存為：\nCSV: {stats_filename}\nExcel: {excel_filename}")
        
        return stats_df, stats_filename
        
    except Exception as e:
        print(f"生成統計表時發生錯誤：{str(e)}")
        return None, None

def analyze_sentiment_comparison(analysis_file):
    """分析並比較正負面情緒資料的函數"""
    try:
        print("\n開始生成情緒分析圖表...")
        
        # 讀取情緒分析結果
        df = pd.read_csv(analysis_file, encoding='utf-8')
        
        # 生成統計表
        stats_df, stats_file = generate_sentiment_statistics(df)
        
        # 分別篩選正面和負面情緒資料
        positive_df = df[df['sentiment_category'] == '正面']
        negative_df = df[df['sentiment_category'] == '負面']
        neutral_df = df[df['sentiment_category'] == '中性']
        
        # 1. 整體情緒分布圓餅圖
        print("繪製整體情緒分布圓餅圖...")
        plt.figure(figsize=(10, 8))
        sizes = [len(positive_df), len(negative_df), len(neutral_df)]
        labels = ['正面', '負面', '中性']
        colors = ['#2ecc71', '#e74c3c', '#95a5a6']
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
        plt.title('AI 代理文獻之整體情緒分布', pad=20, fontsize=14)
        
        # 儲存圓餅圖
        pie_chart_filename = f'AI_sentiment_distribution_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(pie_chart_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"圓餅圖已儲存：{pie_chart_filename}")
        
        # 2. 情緒極性整體分布
        print("繪製情緒極性整體分布圖...")
        plt.figure(figsize=(12, 8))
        sns.histplot(data=df, x='sentiment_polarity', hue='sentiment_category', 
                    multiple="stack", 
                    palette={'正面': '#2ecc71', '負面': '#e74c3c', '中性': '#95a5a6'})
        plt.title('AI 代理文獻之情緒極性整體分布', pad=20, fontsize=14)
        plt.xlabel('情緒極性值', fontsize=12)
        plt.ylabel('文章數量', fontsize=12)
        
        # 儲存直方圖
        hist_chart_filename = f'AI_sentiment_polarity_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(hist_chart_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"直方圖已儲存：{hist_chart_filename}")
        
        return positive_df, negative_df, (pie_chart_filename, hist_chart_filename, stats_file)
        
    except Exception as e:
        print(f"分析過程發生錯誤：{str(e)}")
        return None, None, None

def main():
    """主程式"""
    # 進行情緒分析
    df, analysis_file = process_sentiment_analysis('merged_records.csv')
    if df is not None:
        # 進行情緒比較分析
        positive_df, negative_df, chart_files = analyze_sentiment_comparison(analysis_file)
        if positive_df is not None:
            print("\n分析完成！")
            print(f"圓餅圖：{chart_files[0]}")
            print(f"直方圖：{chart_files[1]}")
            return True
    return False

if __name__ == "__main__":
    main()
