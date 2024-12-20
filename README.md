 #   N S Y S U _ D B A _ R e s e a r c h _ G a p 
 
 # AI 代理文獻研究缺口分析系統
這個專案是一個針對 AI 代理相關文獻進行情緒分析的系統，用於識別和分析研究文獻中的情緒傾向，特別關注負面情緒部分，以協助發現潛在的研究缺口。
## 功能特點
### 1. 文獻情緒分析
 自動識別文獻中的情緒傾向
 計算情緒極性值（範圍：-1 到 1）
 將文獻分類為正面、負面和中性
 特別關注負面情緒文獻，協助發現研究缺口
### 2. 視覺化分析
 **整體情緒分布圓餅圖**
 - 顯示正面、負面、中性文獻的比例
 - 直觀呈現整體情緒分布
 **情緒極性分布直方圖**
 - 展示情緒極性的詳細分布
 - 協助識別極端情緒的文獻
### 3. 統計分析報告
 生成詳細的統計數據表
 包含各類情緒的數量與比例
 提供情緒極性的描述性統計
 輸出 CSV 和 Excel 格式的報告
## 安裝需求

pip install -r requirements.txt

必要套件：
- pandas：數據處理
- textblob：文本情緒分析
- numpy：數值計算
- matplotlib：資料視覺化
- seaborn：統計資料視覺化
- openpyxl：Excel 檔案處理

## 使用方法

1. **準備資料**
   - 確保有 `merged_records.csv` 檔案
   - 檔案必須包含以下欄位：
     - `serial`：文獻編號
     - `AB`：文獻摘要

2. **執行分析**
3. 
3. **查看結果**
   - 檢視生成的情緒分布圖表
   - 查看統計分析報告
   - 分析負面情緒文獻以發現研究缺口

## 輸出檔案

1. **圖表檔案**
   - `AI_sentiment_distribution_[timestamp].png`：整體情緒分布圓餅圖
   - `AI_sentiment_polarity_[timestamp].png`：情緒極性分布直方圖

2. **統計報告**
   - `AI_sentiment_statistics_[timestamp].csv`：CSV 格式統計表
   - `AI_sentiment_statistics_[timestamp].xlsx`：Excel 格式統計表

## 情緒分析標準

- **正面情緒**：極性值 > 0.1
- **中性情緒**：-0.1 ≤ 極性值 ≤ 0.1
- **負面情緒**：極性值 < -0.1

## 應用場景

1. **研究缺口識別**
   - 通過分析負面情緒文獻發現潛在研究問題
   - 識別學術界尚未解決的挑戰

2. **文獻綜述支援**
   - 協助研究者快速了解文獻情緒傾向
   - 提供客觀的情緒分析數據

3. **研究趨勢分析**
   - 追蹤特定領域的情緒變化
   - 發現研究熱點和問題

## 注意事項

1. 文本分析主要支援英文內容
2. 每次執行都會生成新的時間戳記檔案
3. 建議定期備份重要的分析結果

## 更新日誌

### v1.0.0 (2024-03-15)
- 初始版本發布
- 實現基本的情緒分析功能
- 加入視覺化圖表
- 新增統計報表功能

## 授權

MIT License

## 作者

中山大學 DBA AKEN
