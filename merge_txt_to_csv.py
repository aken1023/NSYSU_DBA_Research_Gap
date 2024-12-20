import pandas as pd
import json
import os
from tqdm import tqdm  # 用於顯示進度條
import time  # 用於計算處理時間

def load_and_merge_data(input_path, output_path):
    start_time = time.time()  # 記錄開始時間
    print("="*50)
    print(f"開始時間: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("開始讀取和合併數據...")
    
    try:
        # 讀取JSON數據
        lst_dics = []
        print("\n第一階段: 計算文件大小...")
        total_files = sum(1 for line in open(input_path, 'r', encoding='utf-8'))
        
        print(f"開始讀取文件: {input_path}")
        print(f"總行數: {total_files:,}")
        
        print("\n第二階段: 讀取JSON數據...")
        with open(input_path, mode='r', encoding='utf-8', errors='ignore') as json_file:
            for line in tqdm(json_file, total=total_files, desc="讀取JSON"):
                try:
                    lst_dics.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"\n警告: 跳過無效的JSON行: {e}")
                    continue
        
        json_time = time.time()
        print(f"\n成功讀取 {len(lst_dics):,} 條記錄")
        print(f"JSON讀取耗時: {(json_time - start_time):.2f} 秒")
        
        # 創建DataFrame
        print("\n第三階段: 處理數據...")
        dtf = pd.DataFrame(lst_dics)
        
        # 顯示所有可用類別
        print("\n所有新聞類別:")
        print("-"*30)
        category_counts = dtf['category'].value_counts()
        for category, count in category_counts.items():
            percentage = (count/len(dtf)*100)
            print(f"{category:<20}: {count:>6,} 筆 ({percentage:>6.2f}%)")
        
        # 詢問用戶是否繼續處理
        print("\n是否要繼續處理? (y/n)")
        response = input().lower()
        if response != 'y':
            print("程序已終止")
            return False
            
        original_size = len(dtf)
        
        # 過濾類別
        print("\n過濾指定類別中...")
        dtf = dtf[dtf["category"].isin(['ENTERTAINMENT','POLITICS','TECH'])][["category","headline"]]
        filtered_size = len(dtf)
        
        # 顯示處理統計
        print(f"\n數據統計:")
        print(f"原始數據量: {original_size:,} 條")
        print(f"過濾後數據量: {filtered_size:,} 條")
        print(f"保留比例: {(filtered_size/original_size*100):.2f}%")
        
        # 重命名列
        print("\n重命名欄位...")
        dtf = dtf.rename(columns={"category":"label", "headline":"text"})
        
        # 顯示類別分布
        print("\n類別分布:")
        category_counts = dtf['label'].value_counts()
        for category, count in category_counts.items():
            percentage = (count/filtered_size*100)
            print(f"{category:<15}: {count:>6,} 筆 ({percentage:>6.2f}%)")
        
        # 保存為CSV
        print(f"\n第四階段: 保存數據到 {output_path}")
        dtf.to_csv(output_path, index=False, encoding='utf-8')
        print(f"已成功保存 {len(dtf):,} 條記錄")
        
        # 顯示數據預覽
        print("\n數據預覽（前5筆）：")
        print(dtf.head())
        
        # 計算總耗時
        end_time = time.time()
        total_time = end_time - start_time
        print("\n處理完成!")
        print(f"總耗時: {total_time:.2f} 秒")
        print(f"平均處理速度: {filtered_size/total_time:.2f} 條/秒")
        
        return True
        
    except Exception as e:
        print(f"\n錯誤: {str(e)}")
        print(f"錯誤發生時間: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        return False

def main():
    print("="*50)
    print("新聞數據合併程序")
    print("="*50)
    
    # 設定輸入輸出路徑
    input_path = 'News_Category_Dataset_v3.json'
    output_path = 'news_data.csv'
    
    # 執行合併
    success = load_and_merge_data(input_path, output_path)
    
    if success:
        print("\n程序執行成功!")
    else:
        print("\n程序執行失敗!")
    print("="*50)
    print(f"結束時間: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50)

if __name__ == "__main__":
    main() 