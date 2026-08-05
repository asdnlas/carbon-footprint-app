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
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_squared_error
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

# -------- 可选依赖 --------
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

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

# 云端无中文字体时，图表使用英文标签
def T(cn_text, en_text):
    return cn_text if CHINESE_FONT_OK else en_text

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
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4edf5 100%);
    }
    .hero-card {
        background: linear-gradient(135deg, #1a237e 0%, #283593 50%, #3949ab 100%);
        color: white;
        padding: 30px;
        border-radius: 16px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px rgba(26, 35, 126, 0.3);
    }
    .hero-card h1 {
        color: white !important;
        margin: 0 0 10px 0;
        font-size: 2.2rem !important;
    }
    .hero-card p {
        color: #c5cae9 !important;
        font-size: 1.1rem;
        margin: 0;
    }
    .feature-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        border-left: 4px solid #1a237e;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        height: 100%;
    }
    .metric-card {
        background: white;
        padding: 16px 20px;
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1a237e;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        margin-top: 4px;
    }
    .nav-box {
        background: white;
        padding: 16px 20px;
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }
    .section-title {
        color: #1a237e;
        font-size: 1.4rem !important;
        font-weight: 700;
        margin-top: 20px !important;
    }
    .stMetric {
        background: white;
        padding: 12px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a237e 0%, #283593 100%);
    }
    div[data-testid="stSidebar"] * {
        color: white !important;
    }
    .sidebar-section {
        background: rgba(255,255,255,0.1);
        padding: 14px;
        border-radius: 10px;
        margin-bottom: 12px;
    }
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-green { background: #e8f5e9; color: #2e7d32; }
    .badge-yellow { background: #fff8e1; color: #f57f17; }
    .badge-red { background: #ffebee; color: #c62828; }
    .info-box {
        background: linear-gradient(135deg, #e3f2fd, #e8eaf6);
        padding: 16px 20px;
        border-radius: 10px;
        border-left: 5px solid #1a237e;
        color: #1a237e;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 特征名中英文映射
# ============================================================
FEATURE_CN = ['原材料消耗(kg)', '电力消耗(kWh)', '热力/燃料消耗', '运输(tkm)',
              '核算边界类型', '分配方法类型']
FEATURE_EN = ['Material (kg)', 'Electricity (kWh)', 'Heat/Fuel', 'Transport (tkm)',
              'Boundary Type', 'Allocation Method']
def F(i):
    return FEATURE_CN[i] if CHINESE_FONT_OK else FEATURE_EN[i]

# ============================================================
# 数据加载 + 模型训练（带缓存，云端仅首次加载慢）
# ============================================================
@st.cache_resource(show_spinner=False)
def load_data_and_train_model():
    data = [
        {"source": "电解铝_马梦霞", "product": "氧化铝", "unit": "1 t", "boundary": "摇篮到大门",
         "allocation": "物理法", "material": None, "electricity": None,
         "heat": None, "transport": None, "carbon_footprint": 2030},
        {"source": "电解铝_马梦霞", "product": "预焙阳极", "unit": "1 t", "boundary": "摇篮到大门",
         "allocation": "物理法", "material": None, "electricity": None,
         "heat": None, "transport": None, "carbon_footprint": 1367},
        {"source": "电解铝_马梦霞", "product": "电解铝", "unit": "1 t", "boundary": "摇篮到大门",
         "allocation": "物理法", "material": 1920, "electricity": 15500,
         "heat": None, "transport": None, "carbon_footprint": 14302},
        {"source": "草甘膦_彭子豪", "product": "草甘膦原药", "unit": "1 t", "boundary": "摇篮到大门",
         "allocation": "经济价值法", "material": 320, "electricity": 9127,
         "heat": 18550, "transport": None, "carbon_footprint": 12520},
        {"source": "草甘膦_彭子豪", "product": "草甘膦原药", "unit": "1 t", "boundary": "摇篮到大门",
         "allocation": "质量法", "material": 320, "electricity": 9127,
         "heat": 18550, "transport": None, "carbon_footprint": 3120},
        {"source": "草甘膦_彭子豪", "product": "草甘膦原药", "unit": "1 t", "boundary": "摇篮到大门",
         "allocation": "化学计量法", "material": 320, "electricity": 9127,
         "heat": 18550, "transport": None, "carbon_footprint": 4880},
        {"source": "玻璃瓶罐_四川天马", "product": "玻璃瓶罐", "unit": "1 t", "boundary": "摇篮到大门",
         "allocation": "物理法", "material": 620, "electricity": 203,
         "heat": 185.8, "transport": None, "carbon_footprint": 926},
        {"source": "炼厂_张梦研", "product": "常顶油", "unit": "1 t", "boundary": "摇篮到大门",
         "allocation": "质量法", "material": None, "electricity": None,
         "heat": None, "transport": None, "carbon_footprint": 10.32},
        {"source": "炼厂_张梦研", "product": "常顶油", "unit": "1 t", "boundary": "摇篮到大门",
         "allocation": "热值法", "material": None, "electricity": None,
         "heat": None, "transport": None, "carbon_footprint": 11.09},
        {"source": "炼厂_张梦研", "product": "减顶气(未换热)", "unit": "1 t", "boundary": "摇篮到大门",
         "allocation": "烟值法", "material": None, "electricity": None,
         "heat": None, "transport": None, "carbon_footprint": 29.22},
        {"source": "炼厂_张梦研", "product": "渣油(大量换热)", "unit": "1 t", "boundary": "摇篮到大门",
         "allocation": "烟值法", "material": None, "electricity": None,
         "heat": None, "transport": None, "carbon_footprint": 10.05},
        {"source": "南孚电池", "product": "碱性锌锰电池", "unit": "1 万只", "boundary": "摇篮到大门",
         "allocation": "物理法", "material": 39023, "electricity": 36639562,
         "heat": 379963, "transport": None, "carbon_footprint": 125.85},
        {"source": "肉鸡屠宰_樊庆锌", "product": "屠宰场(建设期)", "unit": "1 座", "boundary": "摇篮到大门",
         "allocation": "物理法", "material": None, "electricity": 906577,
         "heat": 55, "transport": None, "carbon_footprint": 7455000},
        {"source": "肉鸡屠宰_樊庆锌", "product": "屠宰场(运营期)", "unit": "1 年产量", "boundary": "摇篮到大门",
         "allocation": "物理法", "material": None, "electricity": 906577,
         "heat": 55, "transport": None, "carbon_footprint": 944810},
        {"source": "船舶_韩子诺", "product": "散货船(运输阶段)", "unit": "1 艘", "boundary": "摇篮到大门",
         "allocation": "物理法", "material": None, "electricity": None,
         "heat": 1836, "transport": None, "carbon_footprint": 178094000},
        {"source": "船舶_韩子诺", "product": "散货船(原材料阶段)", "unit": "1 艘", "boundary": "摇篮到大门",
         "allocation": "物理法", "material": 6399, "electricity": None,
         "heat": None, "transport": None, "carbon_footprint": 22416000},
    ]
    df = pd.DataFrame(data)

    categorical_cols = ['boundary', 'allocation']
    numeric_cols = ['material', 'electricity', 'heat', 'transport']
    target_col = 'carbon_footprint'

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

    feature_cols = ['material', 'electricity', 'heat', 'transport',
                    'boundary_encoded', 'allocation_encoded']
    X = df[feature_cols].copy()
    X.columns = FEATURE_CN
    y = df[target_col].copy()
    y_log = np.log1p(y)

    best_params = dict(n_estimators=100, max_depth=10,
                       min_samples_split=2, min_samples_leaf=1,
                       random_state=42)
    rf_model = RandomForestRegressor(**best_params)
    rf_model.fit(X, y_log)

    # LOOCV
    loo = LeaveOneOut()
    ytl, ypl, yto, ypo = [], [], [], []
    for tr, te in loo.split(X):
        m = RandomForestRegressor(**best_params)
        m.fit(X.iloc[tr], y_log.iloc[tr])
        p = m.predict(X.iloc[te])
        ypl.extend(p); ytl.extend(y_log.iloc[te])
        ypo.extend(np.expm1(p)); yto.extend(y.iloc[te])

    r2_log = r2_score(ytl, ypl)
    rmse_log = float(np.sqrt(mean_squared_error(ytl, ypl)))
    r2_ori = r2_score(yto, ypo)
    rmse_ori = float(np.sqrt(mean_squared_error(yto, ypo)))

    shap_values, mean_abs_shap = None, None
    if SHAP_AVAILABLE:
        try:
            exp = shap.TreeExplainer(rf_model)
            shap_values = exp.shap_values(X)
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
        except Exception:
            pass

    return dict(model=rf_model, df=df, X=X, y=y, y_log=y_log,
                feature_names_cn=FEATURE_CN, label_encoders=label_encoders,
                best_params=best_params,
                r2_log=r2_log, rmse_log=rmse_log,
                r2_ori=r2_ori, rmse_ori=rmse_ori,
                shap_values=shap_values, mean_abs_shap=mean_abs_shap)


# ============================================================
# 启动
# ============================================================
with st.spinner('加载模型中... Loading model...'):
    try:
        model_data = load_data_and_train_model()
    except Exception as e:
        st.error(f"模型加载失败 / Model load error: {e}")
        st.stop()

rf_model       = model_data['model']
df             = model_data['df']
X              = model_data['X']
y              = model_data['y']
y_log          = model_data['y_log']
feature_names_cn = model_data['feature_names_cn']
label_encoders = model_data['label_encoders']
best_params    = model_data['best_params']
r2_log         = model_data['r2_log']
rmse_log       = model_data['rmse_log']
r2_ori         = model_data['r2_ori']
rmse_ori       = model_data['rmse_ori']
shap_values    = model_data['shap_values']
mean_abs_shap  = model_data['mean_abs_shap']

boundary_options   = label_encoders['boundary'].classes_.tolist()
allocation_options = label_encoders['allocation'].classes_.tolist()

# ============================================================
# 侧边栏
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:10px 0 20px 0;">
        <div style="font-size:2rem;">🌿</div>
        <div style="font-size:1.1rem;font-weight:700;margin-top:5px;">智能碳足迹核算平台</div>
        <div style="font-size:0.8rem;opacity:0.8;margin-top:3px;">ML · Inverse Optimization</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📊 模型性能")
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.metric("LOOCV R²",   f"{r2_log:.4f}")
    st.metric("LOOCV RMSE", f"{rmse_log:.4f}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### 🔧 模型配置")
    st.markdown(f"- 算法: Random Forest")
    st.markdown(f"- 特征: {len(feature_names_cn)}")
    st.markdown(f"- 评估: LOOCV")

    st.markdown("---")
    st.markdown("### 🎯 功能菜单")
    page = st.radio(
        "",
        ["🏠 首页 Home", "🔍 碳足迹预测", "🔄 逆向优化", "📊 SHAP分析", "📂 批量预测"],
        index=0, label_visibility="collapsed"
    )

# ============================================================
# 页面1 - 首页
# ============================================================
if page == "🏠 首页 Home":
    st.markdown("""
    <div class="hero-card">
        <h1>🌿 智能碳足迹核算与逆向优化平台</h1>
        <p>基于机器学习的产品碳足迹智能预测与优化系统 · 助力双碳目标</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{r2_log:.3f}</div><div class="metric-label">📈 模型 R²</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{rmse_log:.2f}</div><div class="metric-label">🎯 RMSE</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{len(feature_names_cn)}</div><div class="metric-label">⚙️ 特征数</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-card"><div class="metric-value">6+</div><div class="metric-label">🏭 行业覆盖</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="section-title">🎯 系统功能 System Features</p>', unsafe_allow_html=True)

    f1, f2 = st.columns(2)
    with f1:
        st.markdown("""
        <div class="feature-card">
            <h4>🔍 碳足迹智能预测</h4>
            <p style="color:#666;">输入产品原材料、电力、热力等参数，实时预测碳足迹排放量，
            并通过SHAP解释各特征的贡献方向与程度。</p>
        </div>""", unsafe_allow_html=True)
    with f2:
        st.markdown("""
        <div class="feature-card">
            <h4>🔄 逆向参数优化</h4>
            <p style="color:#666;">设定目标碳足迹值，系统通过随机搜索反推最优参数组合，
            为低碳工艺设计提供决策支持。</p>
        </div>""", unsafe_allow_html=True)
    f3, f4 = st.columns(2)
    with f3:
        st.markdown("""
        <div class="feature-card">
            <h4>📊 SHAP特征分析</h4>
            <p style="color:#666;">基于SHAP值的可解释性分析，可视化展示各特征对预测结果的
            边际贡献，揭示碳足迹关键影响因子。</p>
        </div>""", unsafe_allow_html=True)
    with f4:
        st.markdown("""
        <div class="feature-card">
            <h4>📂 批量数据预测</h4>
            <p style="color:#666;">支持CSV/Excel文件批量导入，一键预测多条产品碳足迹，
            并生成可视化统计报告，结果可下载。</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="section-title">🧭 快速开始 Quick Start</p>', unsafe_allow_html=True)
    n1, n2 = st.columns(2)
    with n1:
        st.markdown('<div class="info-box">🔍 <b>单次预测</b>：进入「碳足迹预测」页面，输入参数即可预测</div>', unsafe_allow_html=True)
    with n2:
        st.markdown('<div class="info-box">📂 <b>批量预测</b>：进入「批量预测」页面，上传CSV/Excel文件</div>', unsafe_allow_html=True)

    with st.expander("📋 训练数据详情 Training Dataset (click to expand)"):
        st.dataframe(
            df[['source', 'product', 'unit', 'boundary', 'allocation',
                'material', 'electricity', 'heat', 'carbon_footprint']],
            use_container_width=True, height=260
        )
    with st.expander("🔧 模型超参数 Hyperparameters"):
        st.json(best_params)

# ============================================================
# 页面2 - 碳足迹预测
# ============================================================
elif page == "🔍 碳足迹预测":
    st.markdown('<p class="section-title">🔍 碳足迹智能预测 Carbon Footprint Prediction</p>', unsafe_allow_html=True)
    st.markdown("输入产品生产参数，基于随机森林模型预测碳足迹（kgCO₂e/单位）")
    st.markdown("---")

    cl, cr = st.columns([1, 1])
    with cl:
        st.markdown("### 📝 参数输入 Parameters")
        st.markdown('<div class="nav-box">', unsafe_allow_html=True)
        material    = st.number_input("原材料消耗 (kg)",      min_value=0.0, value=1000.0,  step=100.0)
        electricity = st.number_input("电力消耗 (kWh)",       min_value=0.0, value=10000.0, step=1000.0)
        heat        = st.number_input("热力/燃料消耗",        min_value=0.0, value=5000.0,  step=500.0)
        transport   = st.number_input("运输 (tkm)",           min_value=0.0, value=50.0,    step=10.0)
        boundary_sel   = st.selectbox("核算边界类型",   boundary_options)
        allocation_sel = st.selectbox("分配方法类型", allocation_options)
        st.markdown('</div>', unsafe_allow_html=True)
        predict_btn = st.button("🚀 开始预测 Predict", type="primary", use_container_width=True)

    with cr:
        if predict_btn:
            try:
                b_code = label_encoders['boundary'].transform([boundary_sel])[0]
                a_code = label_encoders['allocation'].transform([allocation_sel])[0]
                input_df = pd.DataFrame(
                    [[material, electricity, heat, transport, b_code, a_code]],
                    columns=feature_names_cn
                )
                pred_log = rf_model.predict(input_df)[0]
                pred_ori = float(np.expm1(pred_log))

                st.markdown("### 📊 预测结果 Result")
                st.markdown(f"""
                <div style="background:linear-gradient(135deg,#e8f5e9,#c8e6c9);
                            padding:24px;border-radius:12px;border-left:5px solid #2e7d32;">
                    <div style="font-size:0.9rem;color:#2e7d32;">预测碳足迹 / Predicted</div>
                    <div style="font-size:2.5rem;font-weight:700;color:#1b5e20;margin:8px 0;">
                        {pred_ori:.2f} <span style="font-size:1rem;">kgCO₂e / unit</span>
                    </div>
                </div>""", unsafe_allow_html=True)

                q25, q75 = y.quantile(0.25), y.quantile(0.75)
                if pred_ori < q25:
                    lv = '<span class="badge badge-green">🟢 低碳 Low</span>'
                    ds = f"低于25%分位（{q25:.1f}），表现优异"
                elif pred_ori < q75:
                    lv = '<span class="badge badge-yellow">🟡 中等 Medium</span>'
                    ds = "处于25%-75%分位区间"
                else:
                    lv = '<span class="badge badge-red">🔴 高碳 High</span>'
                    ds = f"高于75%分位（{q75:.1f}），减排潜力大"

                st.markdown(f"**碳足迹等级 Level**: {lv}", unsafe_allow_html=True)
                st.markdown(f"*{ds}*")
                st.markdown(f"现有样本范围 Dataset range: **{y.min():.2f} ~ {y.max():.2f}** kgCO₂e")
                st.markdown("---")

                # SHAP 瀑布图
                if SHAP_AVAILABLE and shap_values is not None:
                    st.markdown("### 🔬 SHAP特征贡献 Feature Contributions")
                    try:
                        exp = shap.TreeExplainer(rf_model)
                        sv  = exp.shap_values(input_df)[0]

                        fig, ax = plt.subplots(figsize=(10, 4.5))
                        colors = ['#e53935' if v > 0 else '#43a047' for v in sv]
                        order  = np.argsort(np.abs(sv))
                        ax.barh(range(len(order)), sv[order],
                                color=[colors[i] for i in order], edgecolor='white')
                        ax.set_yticks(range(len(order)))
                        ax.set_yticklabels([F(i) for i in order])
                        ax.set_xlabel(T('SHAP贡献值', 'SHAP value'))
                        ax.set_title(T('特征贡献瀑布图', 'Feature Contribution Waterfall'),
                                     fontsize=12, fontweight='bold')
                        ax.axvline(0, color='gray', ls='--', lw=0.8)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()

                        contrib = pd.DataFrame({
                            T('特征','Feature'): feature_names_cn,
                            T('SHAP值','SHAP'):  np.round(sv, 5),
                            T('方向','Direction'): ['↑ +Carbon' if v > 0 else '↓ -Carbon' for v in sv]
                        }).sort_values(T('SHAP值','SHAP'), key=abs, ascending=False)
                        st.dataframe(contrib, use_container_width=True, hide_index=True)
                    except Exception as e:
                        st.info(f"SHAP绘图跳过 / SHAP plot skipped: {e}")
            except Exception as e:
                st.error(f"预测出错 / Prediction error: {e}")
        else:
            st.markdown("""
            <div class="info-box">
                <b>👈 请在左侧输入参数，点击「🚀 开始预测」</b>
                <br><small>提示：模型基于LCA样本训练，预测结果仅供参考</small>
            </div>""", unsafe_allow_html=True)

# ============================================================
# 页面3 - 逆向优化
# ============================================================
elif page == "🔄 逆向优化":
    st.markdown('<p class="section-title">🔄 逆向参数优化 Inverse Optimization</p>', unsafe_allow_html=True)
    st.markdown("设定目标碳足迹，系统反推最优参数组合，助力低碳工艺设计")
    st.markdown("---")

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("### ⚙️ 优化配置 Config")
        st.markdown('<div class="nav-box">', unsafe_allow_html=True)
        target_cf = st.number_input("目标碳足迹 Target (kgCO₂e)", min_value=0.1, value=500.0, step=50.0)
        n_search  = st.slider("搜索次数 Search iterations", min_value=5000, max_value=50000, value=20000, step=2500)
        st.markdown('</div>', unsafe_allow_html=True)
        optimize_btn = st.button("🔄 开始优化搜索 Optimize", type="primary", use_container_width=True)

    with c2:
        if optimize_btn:
            with st.spinner(f"搜索中 Searching {n_search} 次..."):
                try:
                    rng = {
                        FEATURE_CN[0]: (0, max(X[FEATURE_CN[0]].max() * 1.2, 5000)),
                        FEATURE_CN[1]: (0, max(X[FEATURE_CN[1]].max() * 1.2, 50000)),
                        FEATURE_CN[2]: (0, max(X[FEATURE_CN[2]].max() * 1.2, 10000)),
                        FEATURE_CN[3]: (0, max(100, X[FEATURE_CN[3]].max() * 1.2)),
                        FEATURE_CN[4]: (0, len(label_encoders['boundary'].classes_)   - 1e-6),
                        FEATURE_CN[5]: (0, len(label_encoders['allocation'].classes_) - 1e-6),
                    }
                    np.random.seed(42)
                    arr = np.column_stack([np.random.uniform(*rng[f], n_search) for f in FEATURE_CN])
                    cdf = pd.DataFrame(arr, columns=FEATURE_CN)
                    cdf[FEATURE_CN[4]] = cdf[FEATURE_CN[4]].round().astype(int).clip(0, len(boundary_options)-1)
                    cdf[FEATURE_CN[5]] = cdf[FEATURE_CN[5]].round().astype(int).clip(0, len(allocation_options)-1)

                    preds = rf_model.predict(cdf)
                    cdf['预测碳足迹'] = np.expm1(preds)

                    def sd(le, code):
                        return le.inverse_transform([int(max(0, min(len(le.classes_)-1, code)))])[0]

                    cdf['偏差'] = np.abs(cdf['预测碳足迹'] - target_cf)
                    top = cdf.nsmallest(10, '偏差').reset_index(drop=True)
                    top['核算边界'] = top[FEATURE_CN[4]].apply(lambda x: sd(label_encoders['boundary'],   x))
                    top['分配方法'] = top[FEATURE_CN[5]].apply(lambda x: sd(label_encoders['allocation'], x))

                    st.markdown(f"### ✅ 最优参数组合 Top 10（目标 Target = {target_cf:.0f}）")
                    show = top[[FEATURE_CN[0], FEATURE_CN[1], FEATURE_CN[2], FEATURE_CN[3],
                                '核算边界', '分配方法', '预测碳足迹', '偏差']].copy()
                    show.index = range(1, 11)
                    show.index.name = 'Rank'
                    show.columns = ['Material(kg)', 'Elec(kWh)', 'Heat', 'Transport(tkm)',
                                    'Boundary', 'Alloc.', 'Pred.CF', 'Delta']
                    show['Material(kg)'] = show['Material(kg)'].round(1)
                    show['Heat']         = show['Heat'].round(1)
                    show['Elec(kWh)']    = show['Elec(kWh)'].round(0).astype(int)
                    show['Transport(tkm)'] = show['Transport(tkm)'].round(1)
                    show['Pred.CF']      = show['Pred.CF'].round(2)
                    show['Delta']        = show['Delta'].round(2)
                    st.dataframe(show, use_container_width=True)

                    csv_bytes = show.to_csv().encode('utf-8-sig')
                    st.download_button("📥 下载 CSV", data=csv_bytes,
                                       file_name=f'InverseOpt_{target_cf:.0f}.csv', mime='text/csv')

                    st.markdown("---")
                    ra, rb, rc = st.columns(3)
                    with ra:
                        st.markdown(f'<div class="metric-card"><div class="metric-value">{show["Pred.CF"].iloc[0]:.2f}</div><div class="metric-label">最优预测 Best</div></div>', unsafe_allow_html=True)
                    with rb:
                        st.markdown(f'<div class="metric-card"><div class="metric-value">{show["Delta"].iloc[0]:.2f}</div><div class="metric-label">最小偏差 Error</div></div>', unsafe_allow_html=True)
                    with rc:
                        st.markdown(f'<div class="metric-card"><div class="metric-value">{cdf["预测碳足迹"].min():.0f}~{cdf["预测碳足迹"].max():.0f}</div><div class="metric-label">搜索范围 Range</div></div>', unsafe_allow_html=True)

                    st.markdown("---")
                    st.markdown("### 📊 Top10 与目标对比 Comparison")
                    try:
                        fig, ax = plt.subplots(figsize=(8, 4))
                        xs = [f"Top{i}" for i in range(1, 11)]
                        ax.bar(xs, top['预测碳足迹'], color='#3949ab', alpha=0.7, label='Predicted CF')
                        ax.axhline(target_cf, color='#e53935', ls='--', lw=2, label=f'Target {target_cf}')
                        ax.set_ylabel(T('碳足迹','Carbon Footprint'))
                        ax.set_title(T('最优组合预测与目标对比','Top10 Predicted vs Target'),
                                     fontsize=12, fontweight='bold')
                        ax.legend()
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                    except Exception as e:
                        st.info(f"图表跳过 / Plot skipped: {e}")
                except Exception as e:
                    st.error(f"优化出错 / Optimizer error: {e}")
        else:
            st.markdown("""
            <div class="info-box">
                <b>👈 左侧设置目标碳足迹 → 点击「开始优化搜索」</b>
                <br><small>系统在参数空间随机搜索，返回最接近目标值的Top 10组合</small>
            </div>""", unsafe_allow_html=True)

# ============================================================
# 页面4 - SHAP分析
# ============================================================
elif page == "📊 SHAP分析":
    st.markdown('<p class="section-title">📊 SHAP特征重要性 SHAP Analysis</p>', unsafe_allow_html=True)
    st.markdown("基于SHAP的模型可解释性分析")
    st.markdown("---")

    if not SHAP_AVAILABLE or shap_values is None:
        st.error("⚠️ SHAP不可用 / SHAP not available. 请安装: `pip install shap`")
    else:
        t1, t2, t3 = st.tabs([T("🏆 重要性排序", "🏆 Importance"),
                              T("🌡️ 摘要图",   "🌡️ Summary"),
                              T("📈 依赖图",   "📈 Dependence")])
        with t1:
            st.markdown("### 特征重要性（平均|SHAP|值）Feature Importance")
            try:
                sdf = pd.DataFrame({T('特征','Feature'): feature_names_cn,
                                    '|SHAP|':  mean_abs_shap}).sort_values('|SHAP|', ascending=True)

                fig, ax = plt.subplots(figsize=(10, 5))
                cmap = plt.cm.Blues(np.linspace(0.4, 0.95, len(sdf)))
                bars = ax.barh(range(len(sdf)), sdf['|SHAP|'], color=cmap, edgecolor='white')
                ax.set_yticks(range(len(sdf)))
                ax.set_yticklabels([FEATURE_EN[feature_names_cn.index(n)]
                                    if not CHINESE_FONT_OK else n for n in sdf[T('特征','Feature')]])
                ax.set_xlabel(T('平均|SHAP|值（重要性）','Mean |SHAP| (Importance)'))
                ax.set_title(T('SHAP特征重要性排序','SHAP Feature Importance'),
                             fontsize=13, fontweight='bold')
                for b, v in zip(bars, sdf['|SHAP|']):
                    ax.text(b.get_width() + sdf['|SHAP|'].max()*0.015,
                            b.get_y() + b.get_height()/2, f'{v:.4f}',
                            va='center', fontsize=9, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig); plt.close()
            except Exception as e:
                st.info(f"绘图跳过 / Plot skipped: {e}")

            rank = pd.DataFrame({
                '排名 Rank':     range(1, 7),
                T('特征','Feature'): feature_names_cn,
                '|SHAP|':        mean_abs_shap,
                T('等级','Level'):['⭐⭐⭐' if v > mean_abs_shap.mean()*1.5
                                   else '⭐⭐' if v > mean_abs_shap.mean()
                                   else '⭐' for v in mean_abs_shap]
            }).sort_values('|SHAP|', ascending=False).reset_index(drop=True)
            rank.index = rank['排名 Rank']
            st.dataframe(rank.drop(columns=['排名 Rank']), use_container_width=True)

        with t2:
            st.markdown("### 摘要图 Summary Plot")
            st.markdown("颜色=特征值高低，位置=对预测的影响方向")
            try:
                # 摘要图不依赖rcParams中文字体，直接传feature_names
                disp_names = FEATURE_CN if CHINESE_FONT_OK else FEATURE_EN
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values, X.values,
                                  feature_names=disp_names, show=False, ax=ax)
                plt.tight_layout()
                st.pyplot(fig); plt.close()
            except Exception as e:
                st.info(f"摘要图跳过 / Summary skipped: {e}")
            st.markdown(f"**{T('解读','Interpretation')}**: {T('红=高特征值，蓝=低特征值；右=增加碳足迹，左=降低碳足迹','Red=high feature value, Blue=low; Right=+CF, Left=-CF')}")

        with t3:
            st.markdown("### 依赖图 Dependence Plot")
            feat_sel = st.selectbox(T("选择特征 Select feature", "Select feature"), feature_names_cn)
            try:
                disp_names = FEATURE_CN if CHINESE_FONT_OK else FEATURE_EN
                idx = feature_names_cn.index(feat_sel)
                fig, ax = plt.subplots(figsize=(10, 5))
                shap.dependence_plot(idx, shap_values, X.values, show=False, ax=ax,
                                     feature_names=disp_names)
                plt.tight_layout()
                st.pyplot(fig); plt.close()
            except Exception as e:
                st.info(f"依赖图跳过 / Dependence skipped: {e}")

# ============================================================
# 页面5 - 批量预测
# ============================================================
elif page == "📂 批量预测":
    st.markdown('<p class="section-title">📂 批量数据预测 Batch Prediction</p>', unsafe_allow_html=True)
    st.markdown("上传CSV/Excel，批量预测多条产品碳足迹")
    st.markdown("---")

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("### 📥 文件上传 Upload")
        st.markdown('<div class="nav-box">', unsafe_allow_html=True)
        st.info("必填列 Required columns:\n`material`, `electricity`, `heat`, `transport`, `boundary`, `allocation`")
        uploaded = st.file_uploader("选择文件 File", type=['csv','xlsx','xls'])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("### 📄 下载模板 Template")
        tpl = pd.DataFrame({
            'material':   [1000, 500, 2000],
            'electricity':[10000,5000,20000],
            'heat':       [5000, 2000, 8000],
            'transport':  [50,   100,  200],
            'boundary':   ['摇篮到大门','摇篮到大门','摇篮到大门'],
            'allocation': ['物理法', '质量法', '经济价值法']
        })
        st.download_button("📥 CSV模板", data=tpl.to_csv(index=False).encode('utf-8-sig'),
                           file_name='Batch_Template.csv', mime='text/csv', use_container_width=True)

    with c2:
        if uploaded is not None:
            try:
                if uploaded.name.endswith('.csv'):
                    idf = pd.read_csv(uploaded)
                else:
                    idf = pd.read_excel(uploaded)

                st.markdown("### 📋 数据预览 Preview (前10条)")
                st.dataframe(idf.head(10), use_container_width=True)

                req = ['material','electricity','heat','transport','boundary','allocation']
                miss = [c for c in req if c not in idf.columns]
                if miss:
                    st.error(f"❌ 缺少列 Missing columns: {miss}")
                else:
                    if st.button("🚀 执行预测 Run Prediction", type="primary", use_container_width=True):
                        with st.spinner("预测中 Predicting..."):
                            for col in ['material','electricity','heat','transport']:
                                idf[col] = idf[col].fillna(0)

                            def senc(le, v):
                                try: return le.transform([str(v)])[0]
                                except ValueError: return 0

                            idf['boundary_encoded']   = idf['boundary'].apply(
                                lambda x: senc(label_encoders['boundary'],   x))
                            idf['allocation_encoded'] = idf['allocation'].apply(
                                lambda x: senc(label_encoders['allocation'], x))

                            BX = idf[['material','electricity','heat','transport',
                                      'boundary_encoded','allocation_encoded']].copy()
                            BX.columns = FEATURE_CN
                            preds_o = np.expm1(rf_model.predict(BX))
                            idf['预测碳足迹(kgCO₂e)'] = np.round(preds_o, 2)

                            st.markdown("---")
                            st.markdown("### ✅ 预测结果 Results")
                            rdf = idf[['material','electricity','heat','transport',
                                       'boundary','allocation','预测碳足迹(kgCO₂e)']].copy()
                            rdf.columns = ['Material(kg)','Elec(kWh)','Heat','Transport(tkm)',
                                           'Boundary','Alloc.','Pred.CF']
                            st.dataframe(rdf, use_container_width=True)

                            st.download_button("📥 下载结果 Download CSV",
                                               data=rdf.to_csv(index=False).encode('utf-8-sig'),
                                               file_name='Batch_Prediction.csv', mime='text/csv')

                            st.markdown("---")
                            st.markdown("### 📈 结果统计 Stats")
                            a1, a2, a3 = st.columns(3)
                            with a1:
                                st.markdown(f'<div class="metric-card"><div class="metric-value">{len(rdf)}</div><div class="metric-label">条数 Count</div></div>', unsafe_allow_html=True)
                            with a2:
                                st.markdown(f'<div class="metric-card"><div class="metric-value">{rdf["Pred.CF"].mean():.1f}</div><div class="metric-label">平均 Mean</div></div>', unsafe_allow_html=True)
                            with a3:
                                st.markdown(f'<div class="metric-card"><div class="metric-value">{rdf["Pred.CF"].min():.0f}~{rdf["Pred.CF"].max():.0f}</div><div class="metric-label">范围 Range</div></div>', unsafe_allow_html=True)

                            st.markdown("---")
                            st.markdown("### 📊 可视化 Visualization")
                            try:
                                v1, v2 = st.columns(2)
                                with v1:
                                    fig, ax = plt.subplots(figsize=(6, 4))
                                    ax.hist(rdf['Pred.CF'], bins=min(20, len(rdf)),
                                            color='#3949ab', edgecolor='white', alpha=0.85)
                                    ax.set_xlabel(T('碳足迹','Carbon Footprint'))
                                    ax.set_ylabel(T('频数','Frequency'))
                                    ax.set_title(T('碳足迹分布','CF Distribution'), fontsize=12, fontweight='bold')
                                    plt.tight_layout()
                                    st.pyplot(fig); plt.close()
                                with v2:
                                    if len(rdf) <= 30:
                                        fig, ax = plt.subplots(figsize=(6, 4))
                                        labels = [f"#{i+1}" for i in range(len(rdf))]
                                        ax.bar(labels, rdf['Pred.CF'], color='#1a237e', alpha=0.8)
                                        ax.set_xlabel(T('样本序号','Sample'))
                                        ax.set_ylabel(T('碳足迹','Carbon Footprint'))
                                        ax.set_title(T('各样本碳足迹','Per-sample CF'), fontsize=12, fontweight='bold')
                                        plt.xticks(rotation=45)
                                        plt.tight_layout()
                                        st.pyplot(fig); plt.close()
                                    else:
                                        st.info("样本数>30，跳过条形图 / Skipped bar chart (n>30)")
                            except Exception as e:
                                st.info(f"图表跳过 / Plot skipped: {e}")
            except Exception as e:
                st.error(f"❌ 错误 Error: {e}")
        else:
            st.markdown("""
            <div class="info-box">
                <b>👈 上传文件后点击「执行预测」</b>
                <br><small>支持 CSV (.csv) 与 Excel (.xlsx/.xls)</small>
            </div>""", unsafe_allow_html=True)
