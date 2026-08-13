# -*- coding: utf-8 -*-
# ============================================================
# 智能碳足迹核算与逆向优化平台（Streamlit Web应用）
# 基于机器学习的产品碳足迹智能预测与优化系统
# 核心：生产过程特征 + LCA方法论 + 机器学习
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

# ============================================================
# 核心特征定义（生产过程导向）
# ============================================================
# 数值特征：生产消耗数据
NUMERIC_FEATURES = ['原材料消耗(kg)', '电力消耗(kWh)', '热力/燃料消耗', '运输(tkm)']
NUMERIC_COLS = ['material', 'electricity', 'heat', 'transport']

# 分类特征：核算方法 + 生产过程
CATEGORICAL_FEATURES = ['核算边界类型', '分配方法类型', '生产工艺类型', '主要碳排放源']
CATEGORICAL_COLS = ['boundary', 'allocation', 'process_type', 'emission_source']

# 全部核心特征（8维）
FEATURE_CN_CORE = NUMERIC_FEATURES + CATEGORICAL_FEATURES
FEATURE_EN_CORE = ['Material(kg)', 'Electricity(kWh)', 'Heat/Fuel', 'Transport(tkm)',
                   'Boundary', 'Allocation', 'Process Type', 'Emission Source']

N_CORE = len(FEATURE_CN_CORE)  # 8

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
    # 生产过程特征数据（重新标注）
    # process_type: 生产工艺类型
    #   - 电解工艺：通过电解还原/氧化金属的工艺
    #   - 化学合成：通过化学反应合成产品
    #   - 物理加工：物理成型、蒸馏等工艺
    #   - 电化学加工：电池制造等
    #   - 食品加工：屠宰、食品处理
    #   - 造船工艺：船舶建造
    #   - 玻璃制造：玻璃熔制与成型
    # emission_source: 主要碳排放源
    #   - 电力驱动：电力消耗为主
    #   - 燃料燃烧：热力/燃料消耗为主
    #   - 化学反应：化学反应碳排放
    #   - 原料隐含：原材料隐含碳为主
    data = [
        # 电解铝行业
        {"material": 0, "electricity": 0, "heat": 0, "transport": 0,
         "boundary": "摇篮到大门", "allocation": "物理法",
         "process_type": "电解工艺", "emission_source": "电力驱动", "carbon_footprint": 2030},
        {"material": 0, "electricity": 0, "heat": 0, "transport": 0,
         "boundary": "摇篮到大门", "allocation": "物理法",
         "process_type": "电解工艺", "emission_source": "电力驱动", "carbon_footprint": 1367},
        {"material": 1920, "electricity": 15500, "heat": 0, "transport": 0,
         "boundary": "摇篮到大门", "allocation": "物理法",
         "process_type": "电解工艺", "emission_source": "电力驱动", "carbon_footprint": 14302},

        # 草甘膦农药
        {"material": 320, "electricity": 9127, "heat": 18550, "transport": 0,
         "boundary": "摇篮到大门", "allocation": "经济价值法",
         "process_type": "化学合成", "emission_source": "燃料燃烧", "carbon_footprint": 12520},
        {"material": 320, "electricity": 9127, "heat": 18550, "transport": 0,
         "boundary": "摇篮到大门", "allocation": "质量法",
         "process_type": "化学合成", "emission_source": "燃料燃烧", "carbon_footprint": 3120},
        {"material": 320, "electricity": 9127, "heat": 18550, "transport": 0,
         "boundary": "摇篮到大门", "allocation": "化学计量法",
         "process_type": "化学合成", "emission_source": "燃料燃烧", "carbon_footprint": 4880},

        # 玻璃瓶罐
        {"material": 620, "electricity": 203, "heat": 185.8, "transport": 0,
         "boundary": "摇篮到大门", "allocation": "物理法",
         "process_type": "玻璃制造", "emission_source": "燃料燃烧", "carbon_footprint": 926},

        # 炼油
        {"material": 0, "electricity": 0, "heat": 0, "transport": 0,
         "boundary": "摇篮到大门", "allocation": "质量法",
         "process_type": "物理加工", "emission_source": "化学反应", "carbon_footprint": 10.32},
        {"material": 0, "electricity": 0, "heat": 0, "transport": 0,
         "boundary": "摇篮到大门", "allocation": "热值法",
         "process_type": "物理加工", "emission_source": "化学反应", "carbon_footprint": 11.09},
        {"material": 0, "electricity": 0, "heat": 0, "transport": 0,
         "boundary": "摇篮到大门", "allocation": "烟值法",
         "process_type": "物理加工", "emission_source": "化学反应", "carbon_footprint": 29.22},
        {"material": 0, "electricity": 0, "heat": 0, "transport": 0,
         "boundary": "摇篮到大门", "allocation": "烟值法",
         "process_type": "物理加工", "emission_source": "化学反应", "carbon_footprint": 10.05},

        # 南孚电池
        {"material": 39023, "electricity": 36639562, "heat": 379963, "transport": 0,
         "boundary": "摇篮到大门", "allocation": "物理法",
         "process_type": "电化学加工", "emission_source": "电力驱动", "carbon_footprint": 125.85},

        # 肉鸡屠宰
        {"material": 0, "electricity": 906577, "heat": 55, "transport": 0,
         "boundary": "摇篮到大门", "allocation": "物理法",
         "process_type": "食品加工", "emission_source": "电力驱动", "carbon_footprint": 7455000},
        {"material": 0, "electricity": 906577, "heat": 55, "transport": 0,
         "boundary": "摇篮到大门", "allocation": "物理法",
         "process_type": "食品加工", "emission_source": "电力驱动", "carbon_footprint": 944810},

        # 船舶
        {"material": 0, "electricity": 0, "heat": 1836, "transport": 0,
         "boundary": "摇篮到大门", "allocation": "物理法",
         "process_type": "造船工艺", "emission_source": "燃料燃烧", "carbon_footprint": 178094000},
        {"material": 6399, "electricity": 0, "heat": 0, "transport": 0,
         "boundary": "摇篮到大门", "allocation": "物理法",
         "process_type": "造船工艺", "emission_source": "原料隐含", "carbon_footprint": 22416000},
    ]

    df = pd.DataFrame(data)

    # 填充缺失值
    for col in CATEGORICAL_COLS:
        df[col] = df[col].fillna('未知')
    for col in NUMERIC_COLS:
        md = df[col].median()
        df[col] = df[col].fillna(0 if pd.isna(md) else md)

    # 标签编码
    label_encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # 构建特征矩阵
    feature_cols_core = NUMERIC_COLS + [c + '_encoded' for c in CATEGORICAL_COLS]
    X_core = df[feature_cols_core].copy()
    X_core.columns = FEATURE_CN_CORE

    # 多项式特征（仅对数值特征）
    X_num = X_core[NUMERIC_FEATURES].copy()
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    X_poly = poly.fit_transform(X_num)

    # 合并核心特征 + 多项式特征
    X_full = np.hstack([X_core.values, X_poly])

    y = df['carbon_footprint'].copy()
    y_log = np.log1p(y)

    # ========== 多模型LOOCV评估 ==========
    loo = LeaveOneOut()
    model_results = {}

    candidates = {}
    candidates['RF'] = RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_split=2, min_samples_leaf=1, random_state=42)
    candidates['GBR'] = GradientBoostingRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, min_samples_split=2, random_state=42)
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
        core_imp = feat_importance[:N_CORE]
    elif mean_abs_shap is not None:
        core_imp = mean_abs_shap[:N_CORE]

    return dict(
        model=final_model, df=df, X_core=X_core, X_full=X_full,
        y=y, y_log=y_log,
        feature_names_cn=FEATURE_CN_CORE,
        label_encoders=label_encoders,
        best_model_name=best_name, model_comparison=comp_df,
        r2_log=r2_log, rmse_log=rmse_log, r2_ori=r2_ori, rmse_ori=rmse_ori,
        shap_values=shap_values, mean_abs_shap=mean_abs_shap,
        use_shap=use_shap, feat_importance=feat_importance, core_imp=core_imp,
        poly=poly, num_cols_cn=NUMERIC_FEATURES
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

# 获取选项
boundary_options = label_encoders['boundary'].classes_.tolist()
allocation_options = label_encoders['allocation'].classes_.tolist()
process_type_options = label_encoders['process_type'].classes_.tolist()
emission_source_options = label_encoders['emission_source'].classes_.tolist()

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
    st.markdown(f"- 核心特征: {N_CORE}维(数值+生产过程) + 多项式")
    st.markdown(f"- 评估: LOOCV ({len(df)}样本)")

    st.markdown("---")
    st.markdown("### 🎯 功能菜单")
    page = st.radio(
        "",
        ["🏠 首页", "🔍 碳足迹预测", "🔄 逆向优化", "📊 SHAP分析", "📂 批量预测"],
        index=0, label_visibility="collapsed"
    )

# ============================================================
# 页面1 - 首页
# ============================================================
if page == "🏠 首页":
    st.markdown("""
    <div class="hero-card">
        <h1>🌿 智能碳足迹核算与逆向优化平台</h1>
        <p>基于机器学习的产品碳足迹智能预测与优化系统 · 生产过程建模 · 助力双碳目标</p>
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
        st.markdown("""<div class="feature-card"><h4>🔍 碳足迹预测</h4>
            <p style="color:#666;">输入生产过程参数（工艺类型、能耗、物料消耗），实时预测产品碳足迹。</p></div>""", unsafe_allow_html=True)
    with f2:
        st.markdown("""<div class="feature-card"><h4>🔄 逆向优化</h4>
            <p style="color:#666;">设定碳足迹目标，反推最优生产参数组合，为低碳工艺设计提供决策支持。</p></div>""", unsafe_allow_html=True)
    f3, f4 = st.columns(2)
    with f3:
        st.markdown("""<div class="feature-card"><h4>📊 SHAP分析</h4>
            <p style="color:#666;">基于SHAP值的可解释性分析，揭示生产过程各参数对碳足迹的影响程度。</p></div>""", unsafe_allow_html=True)
    with f4:
        st.markdown("""<div class="feature-card"><h4>📂 批量预测</h4>
            <p style="color:#666;">上传数据文件，批量预测多条生产场景的碳足迹，支持CSV/Excel格式。</p></div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="section-title">🏆 模型对比</p>', unsafe_allow_html=True)
    st.dataframe(model_comparison, use_container_width=True, hide_index=True)
    st.markdown(f"*最优模型: **{best_model_name}** (LOOCV R²={r2_log:.4f})*")

    with st.expander("📋 训练数据详情"):
        st.dataframe(df[['process_type','emission_source','boundary','allocation','material','electricity','heat','carbon_footprint']], use_container_width=True, height=260)

# ============================================================
# 页面2 - 碳足迹预测
# ============================================================
elif page == "🔍 碳足迹预测":
    st.markdown('<p class="section-title">🔍 碳足迹智能预测</p>', unsafe_allow_html=True)
    st.markdown("输入**生产过程参数**，基于机器学习模型预测碳足迹（kgCO₂e）")
    st.markdown("---")

    cl, cr = st.columns([1, 1])
    with cl:
        st.markdown("### 📝 生产过程参数输入")
        st.markdown('<div class="nav-box">', unsafe_allow_html=True)

        # 生产过程特征
        st.markdown("#### 🏭 生产过程")
        c1, c2 = st.columns(2)
        with c1:
            process_sel = st.selectbox("生产工艺类型", process_type_options,
                                       help="产品采用的主要生产工艺类型")
        with c2:
            emission_sel = st.selectbox("主要碳排放源", emission_source_options,
                                        help="该产品的主要碳排放来源")

        # 核算方法特征
        st.markdown("#### 📊 核算方法")
        c1, c2 = st.columns(2)
        with c1:
            boundary_sel = st.selectbox("核算边界", boundary_options)
        with c2:
            allocation_sel = st.selectbox("分配方法", allocation_options)

        # 数值特征
        st.markdown("#### ⚡ 生产消耗数据")
        material = st.number_input("原材料消耗 (kg)", min_value=0.0, value=1000.0, step=100.0,
                                   help="生产过程中消耗的原材料质量")
        electricity = st.number_input("电力消耗 (kWh)", min_value=0.0, value=10000.0, step=1000.0,
                                      help="生产过程中消耗的电能")
        heat = st.number_input("热力/燃料消耗", min_value=0.0, value=5000.0, step=500.0,
                               help="生产过程中消耗的热能或燃料")
        transport = st.number_input("运输 (tkm)", min_value=0.0, value=50.0, step=10.0,
                                    help="产品运输的吨公里数")

        st.markdown('</div>', unsafe_allow_html=True)
        predict_btn = st.button("🚀 开始预测", type="primary", use_container_width=True)

    with cr:
        if predict_btn:
            try:
                p_code = label_encoders['process_type'].transform([process_sel])[0]
                e_code = label_encoders['emission_source'].transform([emission_sel])[0]
                b_code = label_encoders['boundary'].transform([boundary_sel])[0]
                a_code = label_encoders['allocation'].transform([allocation_sel])[0]

                input_core = pd.DataFrame([[material, electricity, heat, transport, b_code, a_code, p_code, e_code]], columns=feature_names_cn)
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
                        {pred_ori:.2f} <span style="font-size:1rem;">kgCO₂e</span></div></div>""", unsafe_allow_html=True)

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
            st.markdown("""<div class="info-box"><b>👈 请在左侧输入生产参数，点击「🚀 开始预测」</b>
                <br><small>提示：模型基于LCA样本训练，预测结果仅供参考</small></div>""", unsafe_allow_html=True)

# ============================================================
# 页面3 - 逆向优化
# ============================================================
elif page == "🔄 逆向优化":
    st.markdown('<p class="section-title">🔄 逆向参数优化</p>', unsafe_allow_html=True)
    st.markdown("设定目标碳足迹，系统反推最优生产参数组合，助力低碳工艺设计")
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
                    # 随机搜索参数范围
                    rng_bounds = []
                    for i in range(N_CORE):
                        if i == 0: rng_bounds.append((0, max(X_core[feature_names_cn[i]].max() * 1.2, 5000)))
                        elif i == 1: rng_bounds.append((0, max(X_core[feature_names_cn[i]].max() * 1.2, 50000)))
                        elif i == 2: rng_bounds.append((0, max(X_core[feature_names_cn[i]].max() * 1.2, 10000)))
                        elif i == 3: rng_bounds.append((0, max(100, X_core[feature_names_cn[i]].max() * 1.2)))
                        else: rng_bounds.append((0, len(label_encoders[CATEGORICAL_COLS[i-4]].classes_) - 1e-6))

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

                    # 反解编码
                    result_df['核算边界'] = result_df[feature_names_cn[4]].apply(lambda x: sd(label_encoders['boundary'], x))
                    result_df['分配方法'] = result_df[feature_names_cn[5]].apply(lambda x: sd(label_encoders['allocation'], x))
                    result_df['生产工艺'] = result_df[feature_names_cn[6]].apply(lambda x: sd(label_encoders['process_type'], x))
                    result_df['碳排放源'] = result_df[feature_names_cn[7]].apply(lambda x: sd(label_encoders['emission_source'], x))
                    result_df['偏差'] = np.abs(result_df['预测碳足迹'] - target_cf)

                    top = result_df.nsmallest(10, '偏差').reset_index(drop=True)

                    st.markdown(f"### ✅ 最优参数组合 Top 10（目标 = {target_cf:.0f} kgCO₂e）")
                    show = top[['原材料消耗(kg)','电力消耗(kWh)','热力/燃料消耗','运输(tkm)','核算边界','分配方法','生产工艺','碳排放源','预测碳足迹','偏差']].copy()
                    show.index = range(1, 11)
                    show.index.name = 'Rank'
                    show.columns = ['Material(kg)','Elec(kWh)','Heat','Transport','Boundary','Alloc.','Process','Emission','Pred.CF','Delta']
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
                <br><small>系统在参数空间随机搜索，返回最接近目标值的Top 10生产参数组合</small></div>""", unsafe_allow_html=True)

# ============================================================
# 页面4 - SHAP分析
# ============================================================
elif page == "📊 SHAP分析":
    st.markdown('<p class="section-title">📊 SHAP特征重要性分析</p>', unsafe_allow_html=True)
    st.markdown("揭示**生产过程各参数**对碳足迹预测的贡献程度")
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
            <small>功能等价，均能展示生产过程各参数对碳足迹预测的贡献程度</small></div>""", unsafe_allow_html=True)

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
# 页面5 - 批量预测
# ============================================================
elif page == "📂 批量预测":
    st.markdown('<p class="section-title">📂 批量数据预测</p>', unsafe_allow_html=True)
    st.markdown("上传CSV/Excel，批量预测多条生产场景的碳足迹")
    st.markdown("---")

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("### 📥 文件上传")
        st.markdown('<div class="nav-box">', unsafe_allow_html=True)
        st.info("必填列: `material`, `electricity`, `heat`, `transport`, `boundary`, `allocation`, `process_type`, `emission_source`")
        uploaded = st.file_uploader("选择文件", type=['csv','xlsx','xls'])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("### 📄 下载模板")
        tpl = pd.DataFrame({
            'material': [1000, 500, 2000], 'electricity': [10000,5000,20000],
            'heat': [5000, 2000, 8000], 'transport': [50, 100, 200],
            'boundary': ['摇篮到大门','摇篮到大门','摇篮到大门'],
            'allocation': ['物理法', '质量法', '经济价值法'],
            'process_type': ['电解工艺','化学合成','玻璃制造'],
            'emission_source': ['电力驱动','燃料燃烧','燃料燃烧']
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

                req = ['material','electricity','heat','transport','boundary','allocation','process_type','emission_source']
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
                            idf['process_type_encoded'] = idf['process_type'].apply(lambda x: senc(label_encoders['process_type'], x))
                            idf['emission_source_encoded'] = idf['emission_source'].apply(lambda x: senc(label_encoders['emission_source'], x))

                            BX = idf[['material','electricity','heat','transport','boundary_encoded','allocation_encoded','process_type_encoded','emission_source_encoded']].copy()
                            BX.columns = feature_names_cn
                            BX_num = BX[num_cols_cn].copy()
                            BX_poly = poly.transform(BX_num)
                            BX_full = np.hstack([BX.values, BX_poly])

                            preds_o = np.expm1(rf_model.predict(BX_full))
                            idf['预测碳足迹(kgCO₂e)'] = np.round(preds_o, 2)

                            st.markdown("### ✅ 预测结果")
                            rdf = idf[['material','electricity','heat','transport','boundary','allocation','process_type','emission_source','预测碳足迹(kgCO₂e)']].copy()
                            rdf.columns = ['Material','Elec','Heat','Transport','Boundary','Alloc.','Process','Emission','Pred.CF']
                            st.dataframe(rdf, use_container_width=True)

                            st.download_button("📥 下载结果", data=rdf.to_csv(index=False).encode('utf-8-sig'), file_name='Batch_Prediction.csv', mime='text/csv')

                            a1, a2, a3 = st.columns(3)
                            with a1: st.markdown(f'<div class="metric-card"><div class="metric-value">{len(rdf)}</div><div class="metric-label">条数</div></div>', unsafe_allow_html=True)
                            with a2: st.markdown(f'<div class="metric-card"><div class="metric-value">{rdf["Pred.CF"].mean():.1f}</div><div class="metric-label">平均</div></div>', unsafe_allow_html=True)
                            with a3: st.markdown(f'<div class="metric-card"><div class="metric-value">{rdf["Pred.CF"].min():.0f}~{rdf["Pred.CF"].max():.0f}</div><div class="metric-label">范围</div></div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"文件处理出错: {e}")
        else:
            st.markdown("""<div class="info-box"><b>👈 在左侧上传文件，点击「执行预测」</b>
                <br><small>支持 CSV / Excel 格式，需包含必要的生产过程参数列</small></div>""", unsafe_allow_html=True)
