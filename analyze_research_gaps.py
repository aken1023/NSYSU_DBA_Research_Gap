import pandas as pd
import re
from collections import Counter
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 使用微軟雅黑字體
plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題

# 定義英文關鍵詞的中文翻譯字典
keyword_translations = {
    'artificial intelligence': '人工智能',
    'machine learning': '機器學習',
    'reinforcement learning': '強化學習',
    'deep learning': '深度學習',
    'neural networks': '神經網絡',
    'natural language processing': '自然語言處理',
    'computer vision': '計算機視覺',
    'robotics': '機器人技術',
    'autonomous systems': '自主系統',
    'expert systems': '專家系統',
    'knowledge representation': '知識表示',
    'data mining': '數據挖掘',
    'pattern recognition': '模式識別',
    'decision making': '決策制定',
    'optimization': '優化',
    'algorithms': '算法',
    'human-computer interaction': '人機交互',
    'multi-agent systems': '多智能體系統',
    'computer science': '計算機科學',
    'artificial neural networks': '人工神經網絡',
    'ai': '人工智能',
    'agents': '智能體',
    'agent': '智能體',
    'based': '基於',
    'human': '人類',
    'learning': '學習',
    'research': '研究',
    'artificial': '人工',
    'intelligence': '智能',
    'system': '系統',
    'systems': '系統',
    'data': '數據',
    'model': '模型',
    'models': '模型',
    'analysis': '分析',
    'design': '設計',
    'development': '開發',
    'framework': '框架',
    'method': '方法',
    'methods': '方法',
    'approach': '方法',
    'approaches': '方法',
    'performance': '性能',
    'evaluation': '評估',
    'implementation': '實現',
    'application': '應用',
    'applications': '應用',
    'technology': '技術',
    'technologies': '技術',
    'network': '網絡',
    'networks': '網絡',
    'processing': '處理',
    'control': '控制',
    'management': '管理',
    'security': '安全',
    'knowledge': '知識',
    'information': '信息',
    'communication': '通信',
    'computing': '計算',
    'software': '軟件',
    'hardware': '硬件',
    'architecture': '架構',
    'platform': '平台',
    'platforms': '平台',
    'tool': '工具',
    'tools': '工具',
    'interface': '接口',
    'interfaces': '接口',
    'environment': '環境',
    'environments': '環境',
    'solution': '解決方案',
    'solutions': '解決方案',
    'chatbot': '聊天機器人',
    'ethics': '倫理',
    'autonomous': '自主的',
    'intelligent': '智能的'
}

def load_data():
    """載入並預處理數據"""
    print("載入數據...")
    df = pd.read_csv('merged_records.csv')
    df['PY'] = pd.to_numeric(df['PY'], errors='coerce')
    df['TC'] = pd.to_numeric(df['TC'], errors='coerce')
    
    # 篩選引用次數超過20的論文
    df_filtered = df[df['TC'] >= 20].copy()
    print(f"總論文數: {len(df)}")
    print(f"引用次數>=20的論文數: {len(df_filtered)}")
    
    return df_filtered

def find_research_gaps(text):
    """在文本中尋找研究缺口相關的句子"""
    if not isinstance(text, str):
        return []
    
    # 定義表示研究缺口的關鍵詞
    gap_keywords = [
        'future research', 'further research', 'limitation', 'limited',
        'gap', 'gaps', 'need', 'needs', 'lacking', 'lack', 'missing',
        'unexplored', 'unanswered', 'challenge', 'challenges', 'future direction',
        'future work', 'further investigation', 'remain', 'remains'
    ]
    
    # 使用簡單的句子分割
    sentences = [s.strip() for s in re.split('[.!?]', text) if s.strip()]
    gap_sentences = []
    
    for sentence in sentences:
        if any(keyword in sentence.lower() for keyword in gap_keywords):
            gap_sentences.append(sentence)
    
    return gap_sentences

def analyze_keywords(df):
    """分析關鍵字趨勢"""
    print("\n分析關鍵字趨勢...")
    all_keywords = []
    
    # 收集所有關鍵字
    for keywords in df['DE'].dropna():
        if isinstance(keywords, str):
            # 分割關鍵字（假設以分號分隔）
            keywords_list = [k.strip() for k in keywords.split(';')]
            all_keywords.extend(keywords_list)
    
    # 計算關鍵字頻率
    keyword_freq = Counter(all_keywords)
    
    # 獲取前20個最常見的關鍵字
    top_keywords = dict(sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)[:20])
    
    return top_keywords

def create_heatmap(df):
    """創建研究主題熱力圖"""
    print("\n創建研究主題熱力圖...")
    
    # 提取關鍵字
    keywords = []
    for kw in df['DE'].dropna():
        if isinstance(kw, str):
            keywords.append(kw.lower())
    
    # 使用 CountVectorizer 提取最常見的主題詞
    vectorizer = CountVectorizer(max_features=15, stop_words='english')
    X = vectorizer.fit_transform(keywords)
    
    # 獲取主題詞
    terms = vectorizer.get_feature_names_out()
    
    # 創建只有中文的標籤
    translated_terms = [keyword_translations.get(term.lower(), term) for term in terms]
    
    # 創建共現矩陣
    cooc_matrix = np.zeros((len(terms), len(terms)))
    
    # 計算共現次數
    for doc in keywords:
        for i, term1 in enumerate(terms):
            if term1 in doc:
                for j, term2 in enumerate(terms):
                    if term2 in doc:
                        cooc_matrix[i][j] += 1
    
    # 創建熱力圖
    plt.figure(figsize=(15, 12))  # 增加圖片大小以適應更多文字
    sns.heatmap(cooc_matrix, 
                xticklabels=translated_terms, 
                yticklabels=translated_terms,
                cmap='YlOrRd',
                annot=True,
                fmt='.0f')
    plt.title('研究主題關聯熱力圖', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('topic_heatmap.png', dpi=300, bbox_inches='tight')
    print("熱力圖已保存為 'topic_heatmap.png'")

def analyze_topic_gaps(df):
    """分析主題研究缺口"""
    print("\n分析主題研究缺口...")
    
    # 提取包含研究缺口的摘要
    gap_abstracts = []
    for _, paper in df.iterrows():
        if isinstance(paper['AB'], str) and find_research_gaps(paper['AB']):
            gap_abstracts.append(paper['AB'].lower())
    
    # 使用 CountVectorizer 提取主題詞
    vectorizer = CountVectorizer(max_features=20, stop_words='english')
    X = vectorizer.fit_transform(gap_abstracts)
    
    # 計算每個主題詞在研究缺口中的出現頻率
    topic_gaps = dict(zip(vectorizer.get_feature_names_out(), X.sum(axis=0).A1))
    
    return dict(sorted(topic_gaps.items(), key=lambda x: x[1], reverse=True))

def analyze_recent_papers(df):
    """分析最近的論文中提到的研究缺口"""
    print("\n分析最近的論文...")
    # 獲取最近兩年的論文
    current_year = datetime.now().year
    recent_papers = df[df['PY'] >= current_year - 2].copy()
    
    gaps_in_recent = []
    for _, paper in recent_papers.iterrows():
        if isinstance(paper['AB'], str):
            gaps = find_research_gaps(paper['AB'])
            if gaps:
                gaps_in_recent.extend(gaps)
    
    return gaps_in_recent

def analyze_highly_cited(df):
    """分析高引用論文中提到的研究缺口"""
    print("\n分析高引用論文...")
    # 選擇引用次數前10%的論文
    threshold = df['TC'].quantile(0.9)
    highly_cited = df[df['TC'] >= threshold].copy()
    
    gaps_in_cited = []
    for _, paper in highly_cited.iterrows():
        if isinstance(paper['AB'], str):
            gaps = find_research_gaps(paper['AB'])
            if gaps:
                gaps_in_cited.extend(gaps)
    
    return gaps_in_cited

def analyze_key_factors_rf(df):
    """使用隨機森林分析AI Agent主題的關鍵因子"""
    print("\n使用隨機森林分析AI Agent主題的關鍵因子...")
    
    # 篩選出AI Agent相關的論文
    ai_agent_keywords = ['agent', 'agents', 'ai agent', 'intelligent agent', 'multi-agent',
                        'autonomous agent', 'software agent', 'conversational agent']
    
    def is_ai_agent_paper(row):
        if not isinstance(row['DE'], str):
            return False
        keywords = row['DE'].lower()
        return any(kw in keywords for kw in ai_agent_keywords)
    
    df['is_ai_agent'] = df.apply(is_ai_agent_paper, axis=1)
    
    # 準備文本數據
    print("處理文本數據...")
    # 合併摘要和關鍵詞
    df['text_features'] = df['AB'].fillna('') + ' ' + df['DE'].fillna('')
    
    # 使用TF-IDF提取文本特徵
    tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
    X = tfidf.fit_transform(df['text_features'])
    y = df['is_ai_agent']
    
    # 分割訓練集和測試集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 訓練隨機��林模型
    print("訓練隨機森林模型...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # 獲取特徵重要性
    feature_importance = pd.DataFrame({
        'feature': tfidf.get_feature_names_out(),
        'importance': rf.feature_importances_
    })
    
    # 排序並獲取前20個最重要的特徵
    top_features = feature_importance.sort_values('importance', ascending=False).head(20)
    
    # 繪製特徵重要性圖
    plt.figure(figsize=(15, 10))
    plt.bar(range(len(top_features)), top_features['importance'])
    
    # 只顯示中文翻譯
    feature_labels = [f"{keyword_translations.get(f.lower(), f)}" 
                     for f in top_features['feature']]
    
    plt.xticks(range(len(top_features)), feature_labels, rotation=45, ha='right')
    plt.title('AI Agent研究中的關鍵因子重要性（基於隨機森林分析）', fontsize=16)
    plt.xlabel('關鍵因子', fontsize=14)
    plt.ylabel('重要性得分', fontsize=14)
    plt.tight_layout()
    plt.savefig('key_factors_importance.png', dpi=300, bbox_inches='tight')
    print("關鍵因子重要性圖已保存為 'key_factors_importance.png'")
    
    # 輸出模型性能
    train_score = rf.score(X_train, y_train)
    test_score = rf.score(X_test, y_test)
    print(f"\n模型性能:")
    print(f"訓練集準確率: {train_score:.4f}")
    print(f"測試集準確率: {test_score:.4f}")
    
    # 輸出前20個最重要的特徵及其重要性分數
    print("\n前20個最重要的關鍵因子:")
    for _, row in top_features.iterrows():
        feature = row['feature']
        translation = keyword_translations.get(feature.lower(), feature)
        print(f"   {translation}: {row['importance']:.4f}")
    
    return top_features

def main():
    # 載入數據
    df = load_data()
    
    if len(df) == 0:
        print("沒有找到引用次數>=20的論文")
        return
    
    # 執行隨機森林分析
    analyze_key_factors_rf(df)
    
    # 1. 分析關鍵字趨勢
    top_keywords = analyze_keywords(df)
    
    # 2. 創建研究主題熱力圖
    create_heatmap(df)
    
    # 3. 分析主題研究缺口
    topic_gaps = analyze_topic_gaps(df)
    
    # 4. 分析最近論文中的研究缺口
    recent_gaps = analyze_recent_papers(df)
    
    # 5. 分析高引用論文中的研究缺口
    cited_gaps = analyze_highly_cited(df)
    
    # 輸出結果
    print("\n=== 引用次數>=20的論文研究缺口分析結果 ===")
    print("\n1. 熱門研究關鍵字 (前20個):")
    for keyword, count in top_keywords.items():
        translation = keyword_translations.get(keyword.lower(), keyword)
        print(f"   {translation}: {count}")
    
    print("\n2. 研究缺口主題詞 (前20個):")
    for topic, count in topic_gaps.items():
        translation = keyword_translations.get(topic.lower(), topic)
        print(f"   {translation}: {count}")
    
    print("\n3. 最近論文中提到的研究缺口 (部分示例):")
    for gap in recent_gaps[:5]:
        print(f"   - {gap}")
    
    print("\n4. 高引用論文中的研究缺口 (部分示例):")
    for gap in cited_gaps[:5]:
        print(f"   - {gap}")
    
    # 繪製關鍵字趨勢圖
    plt.figure(figsize=(16, 10))  # 增加圖片大小以適應更多文字
    keywords = list(top_keywords.keys())[:15]
    values = list(top_keywords.values())[:15]
    translated_keywords = [keyword_translations.get(k.lower(), k) for k in keywords]
    
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(values)), translated_keywords, rotation=45, ha='right')
    plt.title('引用次數>=20論文的主要研究關鍵字分布', fontsize=16)
    plt.xlabel('關鍵字', fontsize=14)
    plt.ylabel('出現次數', fontsize=14)
    plt.tight_layout()
    plt.savefig('keyword_trends.png', dpi=300, bbox_inches='tight')
    print("\n關鍵字趨勢圖已保存為 'keyword_trends.png'")

if __name__ == '__main__':
    main() 