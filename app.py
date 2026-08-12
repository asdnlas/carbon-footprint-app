# -*- coding: utf-8 -*-
# ============================================================
# 智能碳足迹核算与逆向优化平台（Streamlit Web应用）
# 双碳竞赛 - 机器学习 × 产品碳足迹核算
# 兼容本地(Windows)和云端(Linux Streamlit Community Cloud)
# ============================================================
# 运行方式：
#   pip install -r requirements.txt
#   streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_squared_error
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

# -------- 可选依赖 --------
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LGBM_AVAILABLE = True
except Exception:
    LGBM_AVAILABLE = False

# -------- 字体检测 --------
try:
    from matplotlib import font_manager
    _cn_fonts = [f.name for f in font_manager.fontManager.ttflist
                 if any(k in f.name for k in ['SimHei', 'Microsoft YaHei', 'Heiti',
                                                'Noto Sans CJK', 'WenQuanYi', 'Source Han',
                                                'PingFang', 'STHeiti', 'AR PL'])]
    if _cn_fonts:
        plt.rcParams['font.sans-serif'] = _cn_fonts[:3] + ['DejaVu Sans']
        CHINESE_FONT_OK = True
    else:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        CHINESE_FONT_OK = False
except Exception:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    CHINESE_FONT_OK = False

plt.rcParams['axes.unicode_minus'] = False

def T(cn, en):
    return cn if CHINESE_FONT_OK else en

FEATURE_CN_CORE = ['原材料消耗(kg)', '电力消耗(kWh)', '热力/燃料消耗', '运输(tkm)',
                   '核算边界类型', '分配方法类型', '行业来源', '产品类型', '产品单位']
FEATURE_EN_CORE = ['Material(kg)', 'Electricity(kWh)', 'Heat/Fuel', 'Transport(tkm)',
                   'Boundary Type', 'Allocation Method', 'Industry Source', 'Product Type', 'Unit']

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="智能碳足迹核算与逆向优化平台",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="auto"
)

# ============================================================
# 自定义CSS样式
# ============================================================
st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #f5f7fa 0%, #e4edf5 100%); }
    .hero-card { background: linear-gradient(135deg, #1a237e 0%, #283593 50%, #3949ab 100%);
        color: white; padding: 30px; border-radius: 16px; margin-bottom: 20px;
        box-shadow: 0 8px 32px rgba(26, 35, 126, 0.3); }
    .hero-card h1 { color: white !important; margin: 0 0 10px 0; font-size: 2.2rem !important; }
    .hero-card p { color: #c5cae9 !important; font-size: 1.1rem; margin: 0; }
    .feature-card { background: white; padding: 20px; border-radius: 12px;
        border-left: 4px solid #1a237e; box-shadow: 0 2px 12px rgba(0,0,0,0.08); height: 100%; }
    .metric-card { background: white; padding: 16px 20px; border-radius: 12px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08); text-align: center; }
    .metric-value { font-size: 2rem; font-weight: 700; color: #1a237e; }
    .metric-label { font-size: 0.9rem; color: #666; margin-top: 4px; }
    .nav-box { background: white; padding: 16px 20px; border-radius: 12px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08); margin-bottom: 20px; }
    .section-title { color: #1a237e; font-size: 1.4rem !important; font-weight: 700; margin-top: 20px !important; }
    div[data-testid="stSidebar"] { background: linear-gradient(180deg, #1a237e 0%, #283593 100%); }
    div[data-testid="stSidebar"] * { color: white !important; }
    .sidebar-section { background: rgba(255,255,255,0.1); padding: 14px; border-radius: 10px; margin-bottom: 12px; }
    .badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; }
    .badge-green { background: #e8f5e9; color: #2e7d32; }
    .badge-yellow { background: #fff8e1; color: #f57f17; }
    .badge-red { background: #ffebee; color: #c62828; }
    .info-box { background: linear-gradient(135deg, #e3f2fd, #e8eaf6); padding: 16px 20px;
        border-radius: 10px; border-left: 5px solid #1a237e; color: #1a237e; }
    .context-card { background: white; padding: 24px; border-radius: 14px;
        border-top: 4px solid #1a237e; box-shadow: 0 2px 12px rgba(0,0,0,0.08); margin-bottom: 16px; }
    .context-card h3 { color: #1a237e !important; margin-top: 0 !important; }
    .highlight-box { background: linear-gradient(135deg, #fff3e0, #ffe0b2); padding: 16px 20px;
        border-radius: 10px; border-left: 5px solid #e65100; color: #bf360c; }
    .success-box { background: linear-gradient(135deg, #e8f5e9, #c8e6c9); padding: 16px 20px;
        border-radius: 10px; border-left: 5px solid #2e7d32; color: #1b5e20; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 数据加载 + 模型训练
# ============================================================
@st.cache_resource(show_spinner=False)
def load_data_and_train_model():
    data = [
        {"source": "电解铝_马梦霞", "product": "氧化铝", "unit": "1 t", "boundary": "摇篮到大门", "allocation": "物理法", "material": None, "electricity": None, "heat": None, "transport": None, "carbon_footprint": 2030},
        {"source": "电解铝_马梦霞", "product": "预焙阳极", "unit": "1 t", "boundary": "摇篮到大门", "allocation": "物理法", "material": None, "electricity": None, "heat": None, "transport": None, "carbon_footprint": 1367},
        {"source": "电解铝_马梦霞", "product": "电解铝", "unit": "1 t", "boundary": "摇篮到大门", "allocation": "物理法", "material": 1920, "electricity": 15500, "heat": None, "transport": None, "carbon_footprint": 14302},
        {"source": "草甘膦_彭子豪", "product": "草甘膦原药", "unit": "1 t", "boundary": "摇篮到大门", "allocation": "经济价值法", "material": 320, "electricity": 9127, "heat": 18550, "transport": None, "carbon_footprint": 12520},
        {"source": "草甘膦_彭子豪", "product": "草甘膦原药", "unit": "1 t", "boundary": "摇篮到大门", "allocation": "质量法", "material": 320, "electricity": 9127, "heat": 18550, "transport": None, "carbon_footprint": 3120},
        {"source": "草甘膦_彭子豪", "product": "草甘膦原药", "unit": "1 t", "boundary": "摇篮到大门", "allocation": "化学计量法", "material": 320, "electricity": 9127, "heat": 18550, "transport": None, "carbon_footprint": 4880},
        {"source": "玻璃瓶罐_四川天马", "product": "玻璃瓶罐", "unit": "1 t", "boundary": "摇篮到大门", "allocation": "物理法", "material": 620, "electricity": 203, "heat": 185.8, "transport": None, "carbon_footprint": 926},
        {"source": "炼厂_张梦研", "product": "常顶油", "unit": "1 t", "boundary": "摇篮到大门", "allocation": "质量法", "material": None, "electricity": None, "heat": None, "transport": None, "carbon_footprint": 10.32},
        {"source": "炼厂_张梦研", "product": "常顶油", "unit": "1 t", "boundary": "摇篮到大门", "allocation": "热值法", "material": None, "electricity": None, "heat": None, "transport": None, "carbon_footprint": 11.09},
        {"source": "炼厂_张梦研", "product": "减顶气(未换热)", "unit": "1 t", "boundary": "摇篮到大门", "allocation": "烟值法", "material": None, "electricity": None, "heat": None, "transport": None, "carbon_footprint": 29.22},
        {"source": "炼厂_张梦研", "product": "渣油(大量换热)", "unit": "1 t", "boundary": "摇篮到大门", "allocation": "烟值法", "material": None, "electricity": None, "heat": None, "transport": None, "carbon_footprint": 10.05},
        {"source": "南孚电池", "product": "碱性锌锰电池", "unit": "1 万只", "boundary": "摇篮到大门", "allocation": "物理法", "material": 39023, "electricity": 36639562, "heat": 379963, "transport": None, "carbon_footprint": 125.85},
        {"source": "肉鸡屠宰_樊庆锌", "product": "屠宰场(建设期)", "unit": "1 座", "boundary": "摇篮到大门", "allocation": "物理法", "material": None, "electricity": 906577, "heat": 55, "transport": None, "carbon_footprint": 7455000},
        {"source": "肉鸡屠宰_樊庆锌", "product": "屠宰场(运营期)", "unit": "1 年产量", "boundary": "摇篮到大门", "allocation": "物理法", "material": None, "electricity": 906577, "heat": 55, "transport": None, "carbon_footprint": 944810},
        {"source": "船舶_韩子诺", "product": "散货船(运输阶段)", "unit": "1 艘", "boundary": "摇篮到大门", "allocation": "物理法", "material": None, "electricity": None, "heat": 1836, "transport": None, "carbon_footprint": 178094000},
        {"source": "船舶_韩子诺", "product": "散货船(原材料阶段)", "unit": "1 艘", "boundary": "摇篮到大门", "allocation": "物理法", "material": 6399, "electricity": None, "heat": None, "transport": None, "carbon_footprint": 22416000},
    ]
    df = pd.DataFrame(data)

    categorical_cols = ['boundary', 'allocation', 'source', 'product', 'unit']
    numeric_cols = ['material', 'electricity', 'heat', 'transport']
    for col in categorical_cols:
        df[col] = df[col].fillna('未知')
    for col in numeric_cols:
        md = df[col].median()
        df[col] = df[col].fillna(0 if pd.isna(md) else md)

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # 核心9维特征
    feature_cols_core = ['material', 'electricity', 'heat', 'transport',
                         'boundary_encoded', 'allocation_encoded',
                         'source_encoded', 'product_encoded', 'unit_encoded']
    X_core = df[feature_cols_core].copy()
    X_core.columns = FEATURE_CN_CORE

    # 多项式特征
    num_cols_cn = ['原材料消耗(kg)', '电力消耗(kWh)', '热力/燃料消耗', '运输(tkm)']
    X_num = X_core[num_cols_cn].copy()
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    X_poly = poly.fit_transform(X_num)

    # 合并
    X_full = np.hstack([X_core.values, X_poly])

    y = df['carbon_footprint'].copy()
    y_log = np.log1p(y)

    # ========== 多模型LOOCV评估 ==========
    loo = LeaveOneOut()
    model_results = {}

    candidates = {}
    candidates['RF'] = RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_split=2, min_samples_leaf=1, random_state=42)
    candidates['GBR'] = GradientBoostingRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, min_samples_split=2, random_state=42)
    candidates['Ridge'] = Ridge(alpha=1.0, random_state=42)
    if XGB_AVAILABLE:
        candidates['XGB'] = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, verbosity=0)
    if LGBM_AVAILABLE:
        candidates['LGBM'] = LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, verbosity=-1)

    for name, model in candidates.items():
        ytl, ypl = [], []
        for tr, te in loo.split(X_full):
            params = model.get_params()
            m = type(model)(**{k: v for k, v in params.items() if k in ['n_estimators','max_depth','min_samples_split','min_samples_leaf','learning_rate','alpha','random_state','verbosity']})
            m.fit(X_full[tr], y_log.iloc[tr].values)
            p = m.predict(X_full[te])
            ypl.extend(p)
            ytl.extend(y_log.iloc[te])
        model_results[name] = {'r2': r2_score(ytl, ypl), 'model': model}

    best_name = max(model_results, key=lambda k: model_results[k]['r2'])
    best_model = model_results[best_name]['model']
    best_r2_log = model_results[best_name]['r2']

    # 训练最终模型
    final_params = {k: v for k, v in best_model.get_params().items()
                    if k in ['n_estimators','max_depth','min_samples_split','min_samples_leaf','learning_rate','alpha','random_state','verbosity']}
    final_model = type(best_model)(**final_params)
    final_model.fit(X_full, y_log.values)

    # 最终LOOCV
    ytl, ypl, yto, ypo = [], [], [], []
    for tr, te in loo.split(X_full):
        m = type(final_model)(**final_params)
        m.fit(X_full[tr], y_log.iloc[tr].values)
        p = m.predict(X_full[te])
        ypl.extend(p); ytl.extend(y_log.iloc[te])
        ypo.extend(np.expm1(p)); yto.extend(y.iloc[te])

    r2_log = r2_score(ytl, ypl)
    rmse_log = float(np.sqrt(mean_squared_error(ytl, ypl)))
    r2_ori = r2_score(yto, ypo)
    rmse_ori = float(np.sqrt(mean_squared_error(yto, ypo)))

    comp_data = [{'Model': n, 'LOOCV R²': round(info['r2'], 4)} for n, info in model_results.items()]
    comp_df = pd.DataFrame(comp_data).sort_values('LOOCV R²', ascending=False).reset_index(drop=True)

    # SHAP / 特征重要性
    shap_values = None
    mean_abs_shap = None
    use_shap = False
    if SHAP_AVAILABLE and best_name in ['RF', 'GBR']:
        try:
            exp = shap.TreeExplainer(final_model)
            sv = exp.shap_values(X_full)
            if isinstance(sv, list):
                sv = sv[0]
            shap_values = sv
            mean_abs_shap = np.abs(sv).mean(axis=0)
            use_shap = True
        except Exception:
            pass

    feat_importance = None
    if not use_shap:
        if hasattr(final_model, 'feature_importances_'):
            feat_importance = final_model.feature_importances_
        elif hasattr(final_model, 'coef_'):
            coefs = np.abs(final_model.coef_)
            feat_importance = coefs / coefs.sum()

    core_imp = None
    if feat_importance is not None:
        core_imp = feat_importance[:len(FEATURE_CN_CORE)]
    elif mean_abs_shap is not None:
        core_imp = mean_abs_shap[:len(FEATURE_CN_CORE)]

    return dict(
        model=final_model, df=df, X_core=X_core, X_full=X_full,
        y=y, y_log=y_log,
        feature_names_cn=FEATURE_CN_CORE,
        label_encoders=label_encoders,
        best_model_name=best_name, model_comparison=comp_df,
        r2_log=r2_log, rmse_log=rmse_log, r2_ori=r2_ori, rmse_ori=rmse_ori,
        shap_values=shap_values, mean_abs_shap=mean_abs_shap,
        use_shap=use_shap, feat_importance=feat_importance, core_imp=core_imp,
        poly=poly, num_cols_cn=num_cols_cn
    )


# ============================================================
# 启动
# ============================================================
with st.spinner('加载模型中... Loading model...'):
    try:
        model_data = load_data_and_train_model()
    except Exception as e:
        st.error(f"模型加载失败 / Model load error: {e}")
        st.stop()

rf_model = model_data['model']
df = model_data['df']
X_core = model_data['X_core']
y = model_data['y']
y_log = model_data['y_log']
feature_names_cn = model_data['feature_names_cn']
label_encoders = model_data['label_encoders']
best_model_name = model_data['best_model_name']
model_comparison = model_data['model_comparison']
r2_log = model_data['r2_log']
rmse_log = model_data['rmse_log']
r2_ori = model_data['r2_ori']
rmse_ori = model_data['rmse_ori']
shap_values = model_data['shap_values']
mean_abs_shap = model_data['mean_abs_shap']
use_shap = model_data['use_shap']
feat_importance = model_data['feat_importance']
core_imp = model_data['core_imp']
poly = model_data['poly']
num_cols_cn = model_data['num_cols_cn']

boundary_options = label_encoders['boundary'].classes_.tolist()
allocation_options = label_encoders['allocation'].classes_.tolist()
source_options = label_encoders['source'].classes_.tolist()
product_options = label_encoders['product'].classes_.tolist()
unit_options = label_encoders['unit'].classes_.tolist()
N_CORE = len(feature_names_cn)

# ============================================================
# 侧边栏
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:10px 0 20px 0;">
        <div style="font-size:2rem;">🌿</div>
        <div style="font-size:1.1rem;font-weight:700;margin-top:5px;">智能碳足迹核算平台</div>
        <div style="font-size:0.8rem;opacity:0.8;margin-top:3px;">ML · Inverse Optimization</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("### 📊 模型性能")
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.metric("LOOCV R²", f"{r2_log:.4f}")
    st.metric("LOOCV RMSE", f"{rmse_log:.4f}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### 🔧 模型配置")
    st.markdown(f"- 最优算法: **{best_model_name}**")
    st.markdown(f"- 核心特征: {N_CORE}维 + 多项式")
    st.markdown(f"- 评估: LOOCV ({len(df)}样本)")

    st.markdown("---")
    st.markdown("### 🎯 功能菜单")
    page = st.radio(
        "",
        ["🏠 首页", "🏭 双碳背景", "🔍 碳足迹预测", "🔄 逆向优化", "📊 SHAP分析", "📂 批量预测"],
        index=0, label_visibility="collapsed"
    )

# ============================================================
# 页面1 - 首页
# ============================================================
if page == "🏠 首页":
    st.markdown("""
    <div class="hero-card">
        <h1>🌿 智能碳足迹核算与逆向优化平台</h1>
        <p>基于机器学习的产品碳足迹智能预测与优化系统 · 助力双碳目标</p>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{r2_log:.3f}</div><div class="metric-label">📈 模型 R²</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{rmse_log:.2f}</div><div class="metric-label">🎯 RMSE</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{N_CORE}</div><div class="metric-label">⚙️ 核心特征</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{best_model_name}</div><div class="metric-label">🏆 最优算法</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="section-title">🎯 系统功能</p>', unsafe_allow_html=True)

    f1, f2 = st.columns(2)
    with f1:
        st.markdown("""<div class="feature-card"><h4>🏭 双碳背景</h4>
            <p style="color:#666;">系统阐述碳足迹、双碳目标及LCA方法论，展示项目与竞赛主题的契合度。</p></div>""", unsafe_allow_html=True)
    with f2:
        st.markdown("""<div class="feature-card"><h4>🔍 碳足迹预测</h4>
            <p style="color:#666;">输入产品参数，实时预测碳足迹，通过SHAP解释特征贡献。</p></div>""", unsafe_allow_html=True)
    f3, f4 = st.columns(2)
    with f3:
        st.markdown("""<div class="feature-card"><h4>🔄 逆向优化</h4>
            <p style="color:#666;">设定碳足迹目标，反推最优参数组合，为低碳工艺设计提供决策支持。</p></div>""", unsafe_allow_html=True)
    with f4:
        st.markdown("""<div class="feature-card"><h4>📊 SHAP分析</h4>
            <p style="color:#666;">基于SHAP值的可解释性分析，揭示碳足迹关键影响因子。</p></div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="section-title">🏆 模型对比</p>', unsafe_allow_html=True)
    st.dataframe(model_comparison, use_container_width=True, hide_index=True)
    st.markdown(f"*最优模型: **{best_model_name}** (LOOCV R²={r2_log:.4f})*")

    with st.expander("📋 训练数据详情"):
        st.dataframe(df[['source','product','unit','boundary','allocation','material','electricity','heat','carbon_footprint']], use_container_width=True, height=260)

# ============================================================
# 页面2 - 双碳背景
# ============================================================
elif page == "🏭 双碳背景":
    st.markdown("""
    <div class="hero-card">
        <h1>🏭 双碳背景与项目契合度</h1>
        <p>碳足迹核算 · 机器学习 · 双碳目标 —— 项目价值与竞赛主题的深度契合</p>
    </div>""", unsafe_allow_html=True)

    st.markdown('<p class="section-title">🌍 一、双碳目标背景</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="context-card">
    <h3>🇨🇳 中国"双碳"目标</h3>
    <p><b>2030年前碳达峰</b>、<b>2060年前碳中和</b>，是以习近平同志为核心的党中央作出的重大战略决策，
    也是实现中华民族永续发展的必然选择。</p>
    <p>工业领域碳排放占全社会总排放的 <b>70%以上</b>，是实现双碳目标的关键主战场。
    产品层面的碳足迹核算是工业减排的基础和前提。</p>
    </div>""", unsafe_allow_html=True)

    st.markdown('<p class="section-title">📖 二、碳足迹与LCA方法论</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""<div class="context-card"><h3>🔬 什么是碳足迹？</h3>
            <p>碳足迹（Carbon Footprint）指一个产品在其生命周期阶段产生的温室气体排放总量，以 <b>kgCO₂e</b> 表示。</p>
            <p>碳足迹核算是识别减排机会、评估减排成效、推进绿色低碳发展的重要工具。</p></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="context-card"><h3>📊 LCA生命周期评价</h3>
            <p>生命周期评价（LCA）是ISO 14040确立的碳足迹核算标准方法：</p>
            <ul><li><b>目标与范围定义</b> — 确定核算边界</li>
            <li><b>清单分析</b> — 收集资源能源消耗数据</li>
            <li><b>影响评价</b> — 计算碳足迹</li>
            <li><b>结果解释</b> — 识别减排热点</li></ul></div>""", unsafe_allow_html=True)

    st.markdown('<p class="section-title">⚠️ 三、行业痛点与挑战</p>', unsafe_allow_html=True)
    st.markdown("""<div class="context-card"><h3>💡 传统LCA方法的局限</h3>
        <ul><li><b>数据获取困难</b>：企业生产数据不完整，很多环节缺乏实测数据</li>
        <li><b>核算成本高昂</b>：专业LCA咨询费用高，中小企业难以承担</li>
        <li><b>周期长</b>：一个产品的LCA核算通常需要数月时间</li>
        <li><b>无法实时优化</b>：传统方法只能事后核算，无法指导工艺调整</li></ul></div>""", unsafe_allow_html=True)

    st.markdown('<p class="section-title">🚀 四、项目创新与价值</p>', unsafe_allow_html=True)
    st.markdown("""<div class="success-box"><h3>✨ 本项目如何解决上述痛点？</h3>
        <p>本项目创新性地将<b>机器学习</b>与<b>碳足迹核算</b>相结合，提供<b>低成本、快速、可解释</b>的碳足迹预测方法：</p>
        <ul><li>🎯 <b>智能预测</b>：仅需常规生产参数即可预测碳足迹</li>
        <li>🔍 <b>可解释性分析</b>：SHAP特征重要性揭示关键驱动因子</li>
        <li>🔄 <b>逆向优化</b>：设定低碳目标，自动反推最优工艺参数</li>
        <li>📚 <b>多行业覆盖</b>：电解铝、农药、玻璃、炼油、电池、屠宰、船舶等</li></ul></div>""", unsafe_allow_html=True)

    st.markdown('<p class="section-title">🎯 五、与竞赛主题的契合</p>', unsafe_allow_html=True)
    st.markdown("""<div class="highlight-box"><h3>✅ 紧扣"双碳"主题</h3>
        <p><b>1. 政策契合</b>：响应国家双碳战略，聚焦工业领域碳减排核心需求</p>
        <p><b>2. 方法创新</b>：将机器学习引入传统LCA领域，实现方法学突破</p>
        <p><b>3. 实用价值</b>：为企业提供低成本碳足迹核算工具，助力绿色转型</p>
        <p><b>4. 技术深度</b>：集成LOOCV、SHAP可解释性、逆向优化等先进方法</p></div>""", unsafe_allow_html=True)

    st.markdown('<p class="section-title">📊 六、研究数据概览</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{len(df)}</div><div class="metric-label">样本量</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{len(source_options)}</div><div class="metric-label">行业数</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{r2_log:.3f}</div><div class="metric-label">LOOCV R²</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{best_model_name}</div><div class="metric-label">最优模型</div></div>', unsafe_allow_html=True)

    industry_counts = df['source'].value_counts().reset_index()
    industry_counts.columns = ['行业来源', '样本数']
    st.dataframe(industry_counts, use_container_width=True, hide_index=True)

# ============================================================
# 页面3 - 碳足迹预测
# ============================================================
elif page == "🔍 碳足迹预测":
    st.markdown('<p class="section-title">🔍 碳足迹智能预测</p>', unsafe_allow_html=True)
    st.markdown("输入产品生产参数，基于机器学习模型预测碳足迹（kgCO₂e/单位）")
    st.markdown("---")

    cl, cr = st.columns([1, 1])
    with cl:
        st.markdown("### 📝 参数输入")
        st.markdown('<div class="nav-box">', unsafe_allow_html=True)
        material = st.number_input("原材料消耗 (kg)", min_value=0.0, value=1000.0, step=100.0)
        electricity = st.number_input("电力消耗 (kWh)", min_value=0.0, value=10000.0, step=1000.0)
        heat = st.number_input("热力/燃料消耗", min_value=0.0, value=5000.0, step=500.0)
        transport = st.number_input("运输 (tkm)", min_value=0.0, value=50.0, step=10.0)
        c1, c2 = st.columns(2)
        with c1:
            boundary_sel = st.selectbox("核算边界", boundary_options)
        with c2:
            allocation_sel = st.selectbox("分配方法", allocation_options)
        c3, c4 = st.columns(2)
        with c3:
            source_sel = st.selectbox("行业来源", source_options)
        with c4:
            product_sel = st.selectbox("产品类型", product_options)
        unit_sel = st.selectbox("产品单位", unit_options)
        st.markdown('</div>', unsafe_allow_html=True)
        predict_btn = st.button("🚀 开始预测", type="primary", use_container_width=True)

    with cr:
        if predict_btn:
            try:
                b_code = label_encoders['boundary'].transform([boundary_sel])[0]
                a_code = label_encoders['allocation'].transform([allocation_sel])[0]
                s_code = label_encoders['source'].transform([source_sel])[0]
                p_code = label_encoders['product'].transform([product_sel])[0]
                u_code = label_encoders['unit'].transform([unit_sel])[0]

                input_core = pd.DataFrame([[material, electricity, heat, transport, b_code, a_code, s_code, p_code, u_code]], columns=feature_names_cn)
                input_num = input_core[num_cols_cn].copy()
                input_poly = poly.transform(input_num)
                input_full = np.hstack([input_core.values, input_poly])

                pred_log = rf_model.predict(input_full)[0]
                pred_ori = float(np.expm1(pred_log))

                st.markdown("### 📊 预测结果")
                st.markdown(f"""
                <div style="background:linear-gradient(135deg,#e8f5e9,#c8e6c9);padding:24px;border-radius:12px;border-left:5px solid #2e7d32;">
                    <div style="font-size:0.9rem;color:#2e7d32;">预测碳足迹</div>
                    <div style="font-size:2.5rem;font-weight:700;color:#1b5e20;margin:8px 0;">
                        {pred_ori:.2f} <span style="font-size:1rem;">kgCO₂e / unit</span></div></div>""", unsafe_allow_html=True)

                q25, q75 = y.quantile(0.25), y.quantile(0.75)
                if pred_ori < q25:
                    lv = '<span class="badge badge-green">🟢 低碳</span>'
                    ds = f"低于25%分位（{q25:.1f}），表现优异"
                elif pred_ori < q75:
                    lv = '<span class="badge badge-yellow">🟡 中等</span>'
                    ds = "处于25%-75%分位区间"
                else:
                    lv = '<span class="badge badge-red">🔴 高碳</span>'
                    ds = f"高于75%分位（{q75:.1f}），减排潜力大"
                st.markdown(f"**碳足迹等级**: {lv}", unsafe_allow_html=True)
                st.markdown(f"*{ds}*")
                st.markdown(f"样本范围: **{y.min():.2f} ~ {y.max():.2f}** kgCO₂e")
                st.markdown("---")

                st.markdown("### 🔬 特征贡献分析")
                try:
                    if use_shap:
                        exp = shap.TreeExplainer(rf_model)
                        sv = exp.shap_values(input_full)
                        if isinstance(sv, list): sv = sv[0]
                        sv_s = sv[0]
                        fig, ax = plt.subplots(figsize=(10, 4.5))
                        colors = ['#e53935' if v > 0 else '#43a047' for v in sv_s[:N_CORE]]
                        order = np.argsort(np.abs(sv_s[:N_CORE]))
                        ax.barh(range(len(order)), sv_s[:N_CORE][order], color=[colors[i] for i in order], edgecolor='white')
                        ax.set_yticks(range(len(order)))
                        ax.set_yticklabels([FEATURE_EN_CORE[i] if not CHINESE_FONT_OK else feature_names_cn[i] for i in order])
                        ax.set_xlabel(T('SHAP贡献值','SHAP value'))
                        ax.set_title(T('核心特征贡献','Core Feature Contributions'), fontsize=12, fontweight='bold')
                        ax.axvline(0, color='gray', ls='--', lw=0.8)
                        plt.tight_layout(); st.pyplot(fig); plt.close()

                        contrib = pd.DataFrame({T('特征','Feature'): feature_names_cn, T('SHAP值','SHAP'): np.round(sv_s[:N_CORE], 5),
                            T('方向','Direction'): ['↑ +C' if v > 0 else '↓ -C' for v in sv_s[:N_CORE]]}).sort_values(T('SHAP值','SHAP'), key=abs, ascending=False)
                        st.dataframe(contrib, use_container_width=True, hide_index=True)
                    elif core_imp is not None:
                        fig, ax = plt.subplots(figsize=(10, 4.5))
                        cmap = plt.cm.Blues(np.linspace(0.4, 0.95, N_CORE))
                        order = np.argsort(core_imp)
                        ax.barh(range(len(order)), core_imp[order], color=[cmap[i] for i in order], edgecolor='white')
                        ax.set_yticks(range(len(order)))
                        ax.set_yticklabels([FEATURE_EN_CORE[i] if not CHINESE_FONT_OK else feature_names_cn[i] for i in order])
                        ax.set_xlabel(T('特征重要性','Feature Importance'))
                        ax.set_title(T('核心特征重要性','Core Feature Importance'), fontsize=12, fontweight='bold')
                        plt.tight_layout(); st.pyplot(fig); plt.close()

                        imp_df = pd.DataFrame({T('特征','Feature'): feature_names_cn, T('重要性','Importance'): np.round(core_imp, 5)}).sort_values(T('重要性','Importance'), ascending=False)
                        st.dataframe(imp_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("暂无特征重要性数据")
                except Exception as e:
                    st.info(f"特征分析跳过: {e}")
            except Exception as e:
                st.error(f"预测出错: {e}")
        else:
            st.markdown("""<div class="info-box"><b>👈 请在左侧输入参数，点击「🚀 开始预测」</b>
                <br><small>提示：模型基于LCA样本训练，预测结果仅供参考</small></div>""", unsafe_allow_html=True)

# ============================================================
# 页面4 - 逆向优化
# ============================================================
elif page == "🔄 逆向优化":
    st.markdown('<p class="section-title">🔄 逆向参数优化</p>', unsafe_allow_html=True)
    st.markdown("设定目标碳足迹，系统反推最优参数组合，助力低碳工艺设计")
    st.markdown("---")

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("### ⚙️ 优化配置")
        st.markdown('<div class="nav-box">', unsafe_allow_html=True)
        target_cf = st.number_input("目标碳足迹 (kgCO₂e)", min_value=0.1, value=500.0, step=50.0)
        n_search = st.slider("搜索次数", min_value=5000, max_value=50000, value=20000, step=2500)
        st.markdown('</div>', unsafe_allow_html=True)
        optimize_btn = st.button("🔄 开始优化搜索", type="primary", use_container_width=True)

    with c2:
        if optimize_btn:
            with st.spinner(f"搜索中 {n_search} 次..."):
                try:
                    rng_bounds = []
                    for i in range(N_CORE):
                        if i == 0: rng_bounds.append((0, max(X_core[feature_names_cn[i]].max() * 1.2, 5000)))
                        elif i == 1: rng_bounds.append((0, max(X_core[feature_names_cn[i]].max() * 1.2, 50000)))
                        elif i == 2: rng_bounds.append((0, max(X_core[feature_names_cn[i]].max() * 1.2, 10000)))
                        elif i == 3: rng_bounds.append((0, max(100, X_core[feature_names_cn[i]].max() * 1.2)))
                        else: rng_bounds.append((0, len(label_encoders[['boundary','allocation','source','product','unit'][i-4]].classes_) - 1e-6))

                    np.random.seed(42)
                    arr_list = []
                    for i in range(N_CORE):
                        lo, hi = rng_bounds[i]
                        if i < 4: arr_list.append(np.random.uniform(lo, hi, n_search))
                        else: arr_list.append(np.random.randint(int(lo), int(hi+1), n_search).astype(float))
                    arr = np.column_stack(arr_list)
                    arr_poly = poly.transform(arr[:, :4])
                    arr_full = np.hstack([arr, arr_poly])

                    preds = rf_model.predict(arr_full)
                    pred_ori = np.expm1(preds)

                    def sd(le, code):
                        return le.inverse_transform([int(max(0, min(len(le.classes_)-1, code)))])[0]

                    result_df = pd.DataFrame(arr, columns=feature_names_cn)
                    result_df['预测碳足迹'] = pred_ori
                    result_df['核算边界'] = result_df[feature_names_cn[4]].apply(lambda x: sd(label_encoders['boundary'], x))
                    result_df['分配方法'] = result_df[feature_names_cn[5]].apply(lambda x: sd(label_encoders['allocation'], x))
                    result_df['行业来源'] = result_df[feature_names_cn[6]].apply(lambda x: sd(label_encoders['source'], x))
                    result_df['产品类型'] = result_df[feature_names_cn[7]].apply(lambda x: sd(label_encoders['product'], x))
                    result_df['产品单位'] = result_df[feature_names_cn[8]].apply(lambda x: sd(label_encoders['unit'], x))
                    result_df['偏差'] = np.abs(result_df['预测碳足迹'] - target_cf)

                    top = result_df.nsmallest(10, '偏差').reset_index(drop=True)

                    st.markdown(f"### ✅ 最优参数组合 Top 10（目标 = {target_cf:.0f}）")
                    show = top[['原材料消耗(kg)','电力消耗(kWh)','热力/燃料消耗','运输(tkm)','核算边界','分配方法','行业来源','产品类型','产品单位','预测碳足迹','偏差']].copy()
                    show.index = range(1, 11)
                    show.index.name = 'Rank'
                    show.columns = ['Material(kg)','Elec(kWh)','Heat','Transport','Boundary','Alloc.','Source','Product','Unit','Pred.CF','Delta']
                    for c in ['Material(kg)','Heat']: show[c] = show[c].round(1)
                    show['Elec(kWh)'] = show['Elec(kWh)'].round(0).astype(int)
                    show['Transport'] = show['Transport'].round(1)
                    show['Pred.CF'] = show['Pred.CF'].round(2)
                    show['Delta'] = show['Delta'].round(2)
                    st.dataframe(show, use_container_width=True)

                    csv_bytes = show.to_csv().encode('utf-8-sig')
                    st.download_button("📥 下载 CSV", data=csv_bytes, file_name=f'InverseOpt_{target_cf:.0f}.csv', mime='text/csv')

                    ra, rb, rc = st.columns(3)
                    with ra: st.markdown(f'<div class="metric-card"><div class="metric-value">{show["Pred.CF"].iloc[0]:.2f}</div><div class="metric-label">最优预测</div></div>', unsafe_allow_html=True)
                    with rb: st.markdown(f'<div class="metric-card"><div class="metric-value">{show["Delta"].iloc[0]:.2f}</div><div class="metric-label">最小偏差</div></div>', unsafe_allow_html=True)
                    with rc: st.markdown(f'<div class="metric-card"><div class="metric-value">{pred_ori.min():.0f}~{pred_ori.max():.0f}</div><div class="metric-label">搜索范围</div></div>', unsafe_allow_html=True)

                    try:
                        fig, ax = plt.subplots(figsize=(8, 4))
                        xs = [f"Top{i}" for i in range(1, 11)]
                        ax.bar(xs, top['预测碳足迹'], color='#3949ab', alpha=0.7, label='Predicted')
                        ax.axhline(target_cf, color='#e53935', ls='--', lw=2, label=f'Target {target_cf}')
                        ax.set_ylabel(T('碳足迹','Carbon Footprint'))
                        ax.set_title(T('Top10与目标对比','Top10 vs Target'), fontsize=12, fontweight='bold')
                        ax.legend()
                        plt.tight_layout(); st.pyplot(fig); plt.close()
                    except Exception as e:
                        st.info(f"图表跳过: {e}")
                except Exception as e:
                    st.error(f"优化出错: {e}")
        else:
            st.markdown("""<div class="info-box"><b>👈 左侧设置目标碳足迹 → 点击「开始优化搜索」</b>
                <br><small>系统在参数空间随机搜索，返回最接近目标值的Top 10组合</small></div>""", unsafe_allow_html=True)

# ============================================================
# 页面5 - SHAP分析
# ============================================================
elif page == "📊 SHAP分析":
    st.markdown('<p class="section-title">📊 SHAP特征重要性</p>', unsafe_allow_html=True)
    if use_shap:
        st.markdown("基于SHAP的模型可解释性分析")
    else:
        st.markdown("基于模型内置特征重要性（SHAP不可用时自动回退）")
    st.markdown("---")

    if use_shap and shap_values is not None:
        t1, t2, t3 = st.tabs([T("🏆 重要性排序","🏆 Importance"), T("🌡️ 摘要图","🌡️ Summary"), T("📈 依赖图","📈 Dependence")])
        with t1:
            try:
                mean_abs = mean_abs_shap[:N_CORE]
                sdf = pd.DataFrame({T('特征','Feature'): feature_names_cn, '|SHAP|': mean_abs}).sort_values('|SHAP|', ascending=True)
                fig, ax = plt.subplots(figsize=(10, 5))
                cmap = plt.cm.Blues(np.linspace(0.4, 0.95, len(sdf)))
                bars = ax.barh(range(len(sdf)), sdf['|SHAP|'], color=cmap, edgecolor='white')
                ax.set_yticks(range(len(sdf)))
                ax.set_yticklabels([FEATURE_EN_CORE[feature_names_cn.index(n)] if not CHINESE_FONT_OK else n for n in sdf[T('特征','Feature')]])
                ax.set_xlabel(T('平均|SHAP|值','Mean |SHAP|'))
                ax.set_title(T('SHAP特征重要性排序','SHAP Feature Importance'), fontsize=13, fontweight='bold')
                for b, v in zip(bars, sdf['|SHAP|']):
                    ax.text(b.get_width() + sdf['|SHAP|'].max()*0.015, b.get_y() + b.get_height()/2, f'{v:.4f}', va='center', fontsize=9, fontweight='bold')
                plt.tight_layout(); st.pyplot(fig); plt.close()
            except Exception as e:
                st.info(f"绘图跳过: {e}")

            rank = pd.DataFrame({'排名 Rank': range(1, N_CORE+1), T('特征','Feature'): feature_names_cn, '|SHAP|': mean_abs,
                T('等级','Level'):['⭐⭐⭐' if v > mean_abs.mean()*1.5 else '⭐⭐' if v > mean_abs.mean() else '⭐' for v in mean_abs]
                }).sort_values('|SHAP|', ascending=False).reset_index(drop=True)
            rank.index = rank['排名 Rank']
            st.dataframe(rank.drop(columns=['排名 Rank']), use_container_width=True)

        with t2:
            st.markdown("颜色=特征值高低，位置=对预测的影响方向")
            try:
                disp_names = FEATURE_CN_CORE if CHINESE_FONT_OK else FEATURE_EN_CORE
                fig, ax = plt.subplots(figsize=(10, 6))
                sv_core = shap_values[:, :N_CORE] if len(shap_values.shape) > 1 else shap_values[:N_CORE]
                shap.summary_plot(sv_core, X_core.values, feature_names=disp_names, show=False, ax=ax)
                plt.tight_layout(); st.pyplot(fig); plt.close()
            except Exception as e:
                st.info(f"摘要图跳过: {e}")
            st.markdown(f"**{T('解读','Interpretation')}**: {T('红=高特征值，蓝=低；右=增加碳足迹，左=降低','Red=high, Blue=low; Right=+CF, Left=-CF')}")

        with t3:
            feat_sel = st.selectbox(T("选择特征","Select feature"), feature_names_cn)
            try:
                disp_names = FEATURE_CN_CORE if CHINESE_FONT_OK else FEATURE_EN_CORE
                idx = feature_names_cn.index(feat_sel)
                fig, ax = plt.subplots(figsize=(10, 5))
                sv_core = shap_values[:, :N_CORE] if len(shap_values.shape) > 1 else shap_values[:N_CORE]
                shap.dependence_plot(idx, sv_core, X_core.values, show=False, ax=ax, feature_names=disp_names)
                plt.tight_layout(); st.pyplot(fig); plt.close()
            except Exception as e:
                st.info(f"依赖图跳过: {e}")

    else:
        st.markdown("""<div class="info-box"><b>📌 当前使用模型内置特征重要性（SHAP不可用时自动回退）</b><br>
            <small>功能等价，均能展示各特征对碳足迹预测的贡献程度</small></div>""", unsafe_allow_html=True)

        if core_imp is not None:
            try:
                sdf = pd.DataFrame({T('特征','Feature'): feature_names_cn, '重要性': core_imp}).sort_values('重要性', ascending=True)
                fig, ax = plt.subplots(figsize=(10, 5))
                cmap = plt.cm.Blues(np.linspace(0.4, 0.95, len(sdf)))
                bars = ax.barh(range(len(sdf)), sdf['重要性'], color=cmap, edgecolor='white')
                ax.set_yticks(range(len(sdf)))
                ax.set_yticklabels([FEATURE_EN_CORE[feature_names_cn.index(n)] if not CHINESE_FONT_OK else n for n in sdf[T('特征','Feature')]])
                ax.set_xlabel(T('特征重要性','Feature Importance'))
                ax.set_title(T('核心特征重要性排序','Core Feature Importance'), fontsize=13, fontweight='bold')
                for b, v in zip(bars, sdf['重要性']):
                    ax.text(b.get_width() + sdf['重要性'].max()*0.015, b.get_y() + b.get_height()/2, f'{v:.4f}', va='center', fontsize=9, fontweight='bold')
                plt.tight_layout(); st.pyplot(fig); plt.close()
            except Exception as e:
                st.info(f"绘图跳过: {e}")

            rank = pd.DataFrame({'排名 Rank': range(1, N_CORE+1), T('特征','Feature'): feature_names_cn, '重要性': core_imp,
                T('等级','Level'):['⭐⭐⭐' if v > core_imp.mean()*1.5 else '⭐⭐' if v > core_imp.mean() else '⭐' for v in core_imp]
                }).sort_values('重要性', ascending=False).reset_index(drop=True)
            rank.index = rank['排名 Rank']
            st.dataframe(rank.drop(columns=['排名 Rank']), use_container_width=True)

            st.markdown("---")
            st.markdown('<p class="section-title">📊 特征贡献说明</p>', unsafe_allow_html=True)
            st.markdown("""<div class="context-card">
                <ul><li><b>⭐⭐⭐ 核心特征</b>：贡献超过平均值1.5倍，是碳足迹的关键驱动因子</li>
                <li><b>⭐⭐ 重要特征</b>：贡献超过平均值，对碳足迹有显著影响</li>
                <li><b>⭐ 辅助特征</b>：贡献低于平均值，起辅助修正作用</li></ul>
                <p>💡 <b>建议</b>：重点关注核心特征，它们是减排优化的主要着力点。</p></div>""", unsafe_allow_html=True)

# ============================================================
# 页面6 - 批量预测
# ============================================================
elif page == "📂 批量预测":
    st.markdown('<p class="section-title">📂 批量数据预测</p>', unsafe_allow_html=True)
    st.markdown("上传CSV/Excel，批量预测多条产品碳足迹")
    st.markdown("---")

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("### 📥 文件上传")
        st.markdown('<div class="nav-box">', unsafe_allow_html=True)
        st.info("必填列: `material`, `electricity`, `heat`, `transport`, `boundary`, `allocation`, `source`, `product`, `unit`")
        uploaded = st.file_uploader("选择文件", type=['csv','xlsx','xls'])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("### 📄 下载模板")
        tpl = pd.DataFrame({
            'material': [1000, 500, 2000], 'electricity': [10000,5000,20000],
            'heat': [5000, 2000, 8000], 'transport': [50, 100, 200],
            'boundary': ['摇篮到大门','摇篮到大门','摇篮到大门'],
            'allocation': ['物理法', '质量法', '经济价值法'],
            'source': ['电解铝_马梦霞','草甘膦_彭子豪','玻璃瓶罐_四川天马'],
            'product': ['电解铝','草甘膦原药','玻璃瓶罐'], 'unit': ['1 t','1 t','1 t']
        })
        st.download_button("📥 CSV模板", data=tpl.to_csv(index=False).encode('utf-8-sig'), file_name='Batch_Template.csv', mime='text/csv', use_container_width=True)

    with c2:
        if uploaded is not None:
            try:
                if uploaded.name.endswith('.csv'):
                    idf = pd.read_csv(uploaded)
                else:
                    idf = pd.read_excel(uploaded)

                st.markdown("### 📋 数据预览 (前10条)")
                st.dataframe(idf.head(10), use_container_width=True)

                req = ['material','electricity','heat','transport','boundary','allocation','source','product','unit']
                miss = [c for c in req if c not in idf.columns]
                if miss:
                    st.error(f"❌ 缺少列: {miss}")
                else:
                    if st.button("🚀 执行预测", type="primary", use_container_width=True):
                        with st.spinner("预测中..."):
                            for col in ['material','electricity','heat','transport']:
                                idf[col] = idf[col].fillna(0)
                            def senc(le, v):
                                try: return le.transform([str(v)])[0]
                                except ValueError: return 0
                            idf['boundary_encoded'] = idf['boundary'].apply(lambda x: senc(label_encoders['boundary'], x))
                            idf['allocation_encoded'] = idf['allocation'].apply(lambda x: senc(label_encoders['allocation'], x))
                            idf['source_encoded'] = idf['source'].apply(lambda x: senc(label_encoders['source'], x))
                            idf['product_encoded'] = idf['product'].apply(lambda x: senc(label_encoders['product'], x))
                            idf['unit_encoded'] = idf['unit'].apply(lambda x: senc(label_encoders['unit'], x))

                            BX = idf[['material','electricity','heat','transport','boundary_encoded','allocation_encoded','source_encoded','product_encoded','unit_encoded']].copy()
                            BX.columns = feature_names_cn
                            BX_num = BX[num_cols_cn].copy()
                            BX_poly = poly.transform(BX_num)
                            BX_full = np.hstack([BX.values, BX_poly])

                            preds_o = np.expm1(rf_model.predict(BX_full))
                            idf['预测碳足迹(kgCO₂e)'] = np.round(preds_o, 2)

                            st.markdown("### ✅ 预测结果")
                            rdf = idf[['material','electricity','heat','transport','boundary','allocation','source','product','unit','预测碳足迹(kgCO₂e)']].copy()
                            rdf.columns = ['Material','Elec','Heat','Transport','Boundary','Alloc.','Source','Product','Unit','Pred.CF']
                            st.dataframe(rdf, use_container_width=True)

                            st.download_button("📥 下载结果", data=rdf.to_csv(index=False).encode('utf-8-sig'), file_name='Batch_Prediction.csv', mime='text/csv')

                            a1, a2, a3 = st.columns(3)
                            with a1: st.markdown(f'<div class="metric-card"><div class="metric-value">{len(rdf)}</div><div class="metric-label">条数</div></div>', unsafe_allow_html=True)
                            with a2: st.markdown(f'<div class="metric-card"><div class="metric-value">{rdf["Pred.CF"].mean():.1f}</div><div class="metric-label">平均</div></div>', unsafe_allow_html=True)
                            with a3: st.markdown(f'<div class="metric-card"><div class="metric-value">{rdf["Pred.CF"].min():.0f}~{rdf["Pred.CF"].max():.0f}</div><div class="metric-label">范围</div></div>', unsafe_allow_html=True)

                            try:
                                v1, v2 = st.columns(2)
                                with v1:
                                    fig, ax = plt.subplots(figsize=(6, 4))
                                    ax.hist(rdf['Pred.CF'], bins=min(20, len(rdf)), color='#3949ab', edgecolor='white', alpha=0.85)
                                    ax.set_xlabel(T('碳足迹','Carbon Footprint')); ax.set_ylabel(T('频数','Frequency'))
                                    ax.set_title(T('碳足迹分布','CF Distribution'), fontsize=12, fontweight='bold')
                                    plt.tight_layout(); st.pyplot(fig); plt.close()
                                with v2:
                                    if len(rdf) <= 30:
                                        fig, ax = plt.subplots(figsize=(6, 4))
                                        labels = [f"#{i+1}" for i in range(len(rdf))]
                                        ax.bar(labels, rdf['Pred.CF'], color='#1a237e', alpha=0.8)
                                        ax.set_xlabel(T('样本','Sample')); ax.set_ylabel(T('碳足迹','CF'))
                                        ax.set_title(T('各样本碳足迹','Per-sample CF'), fontsize=12, fontweight='bold')
                                        plt.xticks(rotation=45); plt.tight_layout(); st.pyplot(fig); plt.close()
                                    else:
                                        st.info("样本数>30，跳过条形图")
                            except Exception as e:
                                st.info(f"图表跳过: {e}")
            except Exception as e:
                st.error(f"❌ 错误: {e}")
        else:
            st.markdown("""<div class="info-box"><b>👈 上传文件后点击「执行预测」</b>
                <br><small>支持 CSV (.csv) 与 Excel (.xlsx/.xls)</small></div>""", unsafe_allow_html=True)
