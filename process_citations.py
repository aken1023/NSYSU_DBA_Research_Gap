import os
import csv
import sys
import codecs
from collections import defaultdict

# 設定系統編碼
if sys.stdout.encoding != 'utf-8':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

def process_file(file_path):
    """處理單個檔案並返回文獻資訊"""
    try:
        citations = []
        print(f"正在處理檔案: {file_path}", flush=True)
        
        # 嘗試不同的編碼
        encodings = ['utf-8', 'utf-8-sig', 'big5', 'gb18030', 'latin1']
        content = None
        used_encoding = None
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.readlines()
                used_encoding = encoding
                print(f"  成功使用 {encoding} 編碼讀取檔案", flush=True)
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"  使用 {encoding} 編碼時發生錯誤: {str(e)}", flush=True)
                continue
        
        if content is None:
            print(f"  無法使用任何編碼讀取檔案", flush=True)
            return []
            
        print(f"  讀取到 {len(content)} 行資料", flush=True)
        
        if len(content) < 2:  # 確保檔案至少有標題行和一行資料
            print("  檔案行數不足，跳過", flush=True)
            return []
            
        # 獲取欄位名稱
        headers = content[0].strip().split('\t')
        print(f"  找到 {len(headers)} 個欄位: {headers}", flush=True)
        
        # 處理每一行資料
        for line_num, line in enumerate(content[1:], 1):
            try:
                fields = line.strip().split('\t')
                if len(fields) >= len(headers):  # 確保有足夠的欄位
                    citation = {}
                    for i, header in enumerate(headers):
                        if i < len(fields):  # 確保不會超出索引範圍
                            citation[header] = fields[i]
                    
                    # 使用一些關鍵欄位作為唯一標識
                    key = f"{citation.get('TI', '')}-{citation.get('AU', '')}-{citation.get('PY', '')}"
                    if key not in [c.get('key') for c in citations]:
                        citation['key'] = key
                        citations.append(citation)
                else:
                    print(f"  警告：第 {line_num} 行的欄位數量不足 (預期 {len(headers)}, 實際 {len(fields)})", flush=True)
            except Exception as e:
                print(f"  處理第 {line_num} 行時發生錯誤: {str(e)}", flush=True)
                continue
        
        print(f"  從檔案中提取到 {len(citations)} 筆引用資料", flush=True)
        return citations
        
    except Exception as e:
        print(f"處理檔案 {file_path} 時發生錯誤: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        return []

def main():
    try:
        print("開始執行程式...", flush=True)
        
        # 設定輸入和輸出路徑
        input_dir = 'txt'
        output_file = 'citations.csv'
        
        # 確保輸入目錄存在
        if not os.path.exists(input_dir):
            print(f"錯誤：找不到輸入目錄 {input_dir}", flush=True)
            return
            
        # 獲取所有 txt 檔案
        txt_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
        total_files = len(txt_files)
        
        if total_files == 0:
            print(f"錯誤：在 {input_dir} 目錄中找不到任何 .txt 檔案", flush=True)
            return
            
        print(f"找到 {total_files} 個 .txt 檔案待處理", flush=True)
        processed_count = 0
        
        # 用於存儲所有不重複的引用
        all_citations = []
        unique_keys = set()
        
        # 處理每個檔案
        for filename in txt_files:
            file_path = os.path.join(input_dir, filename)
            citations = process_file(file_path)
            
            # 添加不重複的引用
            for citation in citations:
                key = citation['key']
                if key not in unique_keys:
                    unique_keys.add(key)
                    all_citations.append(citation)
            
            processed_count += 1
            progress = (processed_count / total_files) * 100
            print(f"處理進度: {progress:.2f}% ({processed_count}/{total_files})", flush=True)
        
        # 如果有引用資料，則寫入CSV檔案
        if all_citations:
            # 獲取所有可能的欄位
            all_fields = set()
            for citation in all_citations:
                all_fields.update(citation.keys())
            
            print(f"正在寫入 {len(all_citations)} 筆資料到 {output_file}", flush=True)
            
            # 寫入CSV檔案
            with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=sorted(list(all_fields)))
                writer.writeheader()
                writer.writerows(all_citations)
            
            print(f"\n處理完成！", flush=True)
            print(f"總共處理了 {total_files} 個檔案", flush=True)
            print(f"找到 {len(all_citations)} 筆不重複的引用資料", flush=True)
            print(f"結果已儲存至 {output_file}", flush=True)
        else:
            print("沒有找到任何引用資料！", flush=True)
            
    except Exception as e:
        print(f"執行過程中發生錯誤: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 