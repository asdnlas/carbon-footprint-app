# -*- coding: utf-8 -*-
# ============================================================
# 智能碳足迹核算与逆向优化系统（Streamlit Web应用）
# 双碳竞赛 - 机器学习 × 产品碳足迹核算
# ============================================================
# 运行方式：
#   pip install -r requirements.txt
#   streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_squared_error
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

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
    .glow-btn {
        background: linear-gradient(135deg, #1a237e, #3949ab);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 12px 28px;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(26, 35, 126, 0.4);
    }
    .glow-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(26, 35, 126, 0.5);
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
# 数据加载与预处理
# ============================================================
@st.cache_resource
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
        median_val = df[col].median()
        if pd.isna(median_val):
            median_val = 0
        df[col] = df[col].fillna(median_val)

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    feature_cols = ['material', 'electricity', 'heat', 'transport',
                    'boundary_encoded', 'allocation_encoded']
    feature_names_cn = ['原材料消耗(kg)', '电力消耗(kWh)', '热力/燃料消耗', '运输(tkm)',
                        '核算边界类型', '分配方法类型']

    X = df[feature_cols].copy()
    X.columns = feature_names_cn
    y = df[target_col].copy()
    y_log = np.log1p(y)

    best_params = {
        'n_estimators': 100, 'max_depth': 10,
        'min_samples_split': 2, 'min_samples_leaf': 1,
        'random_state': 42
    }
    rf_model = RandomForestRegressor(**best_params)
    rf_model.fit(X, y_log)

    loo = LeaveOneOut()
    y_true_log, y_pred_log, y_true_ori, y_pred_ori = [], [], [], []
    for train_idx, test_idx in loo.split(X):
        rf_loo = RandomForestRegressor(**best_params)
        rf_loo.fit(X.iloc[train_idx], y_log.iloc[train_idx])
        pred_l = rf_loo.predict(X.iloc[test_idx])
        y_pred_log.extend(pred_l)
        y_true_log.extend(y_log.iloc[test_idx])
        y_pred_ori.extend(np.expm1(pred_l))
        y_true_ori.extend(y.iloc[test_idx])

    r2_log = r2_score(y_true_log, y_pred_log)
    rmse_log = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
    r2_ori = r2_score(y_true_ori, y_pred_ori)
    rmse_ori = np.sqrt(mean_squared_error(y_true_ori, y_pred_ori))

    shap_values = None
    mean_abs_shap = None
    if SHAP_AVAILABLE:
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(X)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

    return {
        'model': rf_model, 'df': df, 'X': X, 'y': y, 'y_log': y_log,
        'feature_names_cn': feature_names_cn, 'label_encoders': label_encoders,
        'best_params': best_params, 'r2_log': r2_log, 'rmse_log': rmse_log,
        'r2_ori': r2_ori, 'rmse_ori': rmse_ori,
        'shap_values': shap_values, 'mean_abs_shap': mean_abs_shap,
    }


with st.spinner('正在加载模型和数据...'):
    model_data = load_data_and_train_model()

rf_model = model_data['model']
df = model_data['df']
X = model_data['X']
y = model_data['y']
y_log = model_data['y_log']
feature_names_cn = model_data['feature_names_cn']
label_encoders = model_data['label_encoders']
best_params = model_data['best_params']
r2_log = model_data['r2_log']
rmse_log = model_data['rmse_log']
r2_ori = model_data['r2_ori']
rmse_ori = model_data['rmse_ori']
shap_values = model_data['shap_values']
mean_abs_shap = model_data['mean_abs_shap']

boundary_options = label_encoders['boundary'].classes_.tolist()
allocation_options = label_encoders['allocation'].classes_.tolist()


# ============================================================
# 侧边栏（深色主题）
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:10px 0 20px 0;">
        <div style="font-size:2rem;">🌿</div>
        <div style="font-size:1.1rem;font-weight:700;margin-top:5px;">智能碳足迹核算平台</div>
        <div style="font-size:0.8rem;opacity:0.8;margin-top:3px;">机器学习 · 逆向优化</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📊 模型性能指标")
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.metric("LOOCV R²", f"{r2_log:.4f}")
    st.metric("LOOCV RMSE", f"{rmse_log:.4f}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### 🔧 模型参数")
    st.markdown(f"- **算法**: Random Forest")
    st.markdown(f"- **特征维度**: {len(feature_names_cn)}")
    st.markdown(f"- **评估方法**: LOOCV")

    st.markdown("---")
    st.markdown("### 🎯 快速导航")
    page = st.radio(
        "",
        ["🏠 首页概览", "🔍 碳足迹预测", "🔄 逆向优化", "📊 SHAP分析", "📂 批量预测"],
        index=0,
        label_visibility="collapsed"
    )


# ============================================================
# 页面1：首页
# ============================================================
if page == "🏠 首页概览":
    st.markdown("""
    <div class="hero-card">
        <h1>🌿 智能碳足迹核算与逆向优化平台</h1>
        <p>基于机器学习的产品碳足迹智能预测与优化系统 · 助力双碳目标</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-value">{}</div><div class="metric-label">📈 模型R²</div></div>'.format(f"{r2_log:.3f}"), unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><div class="metric-value">{}</div><div class="metric-label">🎯 预测精度</div></div>'.format(f"RMSE={rmse_log:.2f}"), unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><div class="metric-value">{}</div><div class="metric-label">⚙️ 特征维度</div></div>'.format(str(len(feature_names_cn))), unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><div class="metric-value">{}</div><div class="metric-label">🏭 行业覆盖</div></div>'.format("多领域"), unsafe_allow_html=True)

    st.markdown("---")

    st.markdown('<p class="section-title">🎯 系统功能</p>', unsafe_allow_html=True)

    f1, f2 = st.columns(2)
    with f1:
        st.markdown("""
        <div class="feature-card">
            <h4>🔍 碳足迹智能预测</h4>
            <p style="color:#666;">输入产品原材料、电力、热力等参数，实时预测碳足迹排放量，
            并通过SHAP解释各特征的贡献方向与程度。</p>
        </div>
        """, unsafe_allow_html=True)
    with f2:
        st.markdown("""
        <div class="feature-card">
            <h4>🔄 逆向参数优化</h4>
            <p style="color:#666;">设定目标碳足迹值，系统通过随机搜索反推最优参数组合，
            为低碳工艺设计提供决策支持。</p>
        </div>
        """, unsafe_allow_html=True)

    f3, f4 = st.columns(2)
    with f3:
        st.markdown("""
        <div class="feature-card">
            <h4>📊 SHAP特征分析</h4>
            <p style="color:#666;">基于SHAP值的可解释性分析，可视化展示各特征对预测结果的
            边际贡献，揭示碳足迹的关键影响因子。</p>
        </div>
        """, unsafe_allow_html=True)
    with f4:
        st.markdown("""
        <div class="feature-card">
            <h4>📂 批量数据预测</h4>
            <p style="color:#666;">支持CSV/Excel文件批量导入，一键预测多条产品碳足迹，
            并生成可视化统计报告，结果可下载。</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown('<p class="section-title">🧭 快速开始</p>', unsafe_allow_html=True)

    nav_col1, nav_col2 = st.columns(2)
    with nav_col1:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("**🔍 单次预测** - 进入「碳足迹预测」页面，输入参数即可预测")
        st.markdown("</div>", unsafe_allow_html=True)
    with nav_col2:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("**📂 批量预测** - 进入「批量预测」页面，上传CSV/Excel文件")
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("📋 查看样本数据详情（点击展开）"):
        st.dataframe(
            df[['source', 'product', 'unit', 'boundary', 'allocation',
                'material', 'electricity', 'heat', 'carbon_footprint']],
            use_container_width=True,
            height=250
        )

    with st.expander("🔧 查看模型配置详情"):
        st.json(best_params)


# ============================================================
# 页面2：碳足迹预测
# ============================================================
elif page == "🔍 碳足迹预测":
    st.markdown('<p class="section-title">🔍 碳足迹智能预测</p>', unsafe_allow_html=True)
    st.markdown("输入产品生产参数，基于训练好的随机森林模型预测碳足迹（kgCO₂e/单位）")
    st.markdown("---")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("### 📝 参数输入")
        st.markdown('<div class="nav-box">', unsafe_allow_html=True)
        material = st.number_input("原材料消耗 (kg)", min_value=0.0, value=1000.0, step=100.0)
        electricity = st.number_input("电力消耗 (kWh)", min_value=0.0, value=10000.0, step=1000.0)
        heat = st.number_input("热力/燃料消耗", min_value=0.0, value=5000.0, step=500.0)
        transport = st.number_input("运输 (tkm)", min_value=0.0, value=50.0, step=10.0)
        boundary_sel = st.selectbox("核算边界类型", boundary_options)
        allocation_sel = st.selectbox("分配方法类型", allocation_options)
        st.markdown('</div>', unsafe_allow_html=True)

        predict_btn = st.button("🚀 开始预测", type="primary", use_container_width=True)

    with col_right:
        if predict_btn:
            b_code = label_encoders['boundary'].transform([boundary_sel])[0]
            a_code = label_encoders['allocation'].transform([allocation_sel])[0]

            input_df = pd.DataFrame(
                [[material, electricity, heat, transport, b_code, a_code]],
                columns=feature_names_cn
            )
            pred_log = rf_model.predict(input_df)[0]
            pred_ori = np.expm1(pred_log)

            st.markdown("### 📊 预测结果")

            st.markdown("""
            <div style="background:linear-gradient(135deg,#e8f5e9,#c8e6c9);
                        padding:24px;border-radius:12px;border-left:5px solid #2e7d32;">
                <div style="font-size:0.9rem;color:#2e7d32;">预测碳足迹</div>
                <div style="font-size:2.5rem;font-weight:700;color:#1b5e20;margin:8px 0;">
                    {:.2f} <span style="font-size:1rem;">kgCO₂e/单位</span>
                </div>
            </div>
            """.format(pred_ori), unsafe_allow_html=True)

            q25, q75 = y.quantile(0.25), y.quantile(0.75)
            if pred_ori < q25:
                level_html = '<span class="badge badge-green">🟢 低碳水平</span>'
                desc = f"低于25%分位值（{q25:.1f}），表现优异"
            elif pred_ori < q75:
                level_html = '<span class="badge badge-yellow">🟡 中等水平</span>'
                desc = f"处于25%-75%分位区间"
            else:
                level_html = '<span class="badge badge-red">🔴 高碳水平</span>'
                desc = f"高于75%分位值（{q75:.1f}），减排潜力大"

            st.markdown(f"**碳足迹等级**: {level_html}", unsafe_allow_html=True)
            st.markdown(f"*{desc}*")

            st.markdown("---")

            if SHAP_AVAILABLE:
                st.markdown("### 🔬 SHAP特征贡献解释")
                st.markdown("各特征对本次预测的贡献方向与幅度：")

                explainer = shap.TreeExplainer(rf_model)
                sv = explainer.shap_values(input_df)

                fig, ax = plt.subplots(figsize=(10, 4))
                sv_vals = sv[0]
                colors = ['#e53935' if v > 0 else '#43a047' for v in sv_vals]
                sorted_idx = np.argsort(np.abs(sv_vals))
                ax.barh(range(len(sorted_idx)), sv_vals[sorted_idx],
                        color=[colors[i] for i in sorted_idx], edgecolor='white')
                ax.set_yticks(range(len(sorted_idx)))
                ax.set_yticklabels([feature_names_cn[i] for i in sorted_idx])
                ax.set_xlabel('SHAP贡献值')
                ax.set_title('特征贡献瀑布图', fontsize=12, fontweight='bold')
                ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                contrib_df = pd.DataFrame({
                    '特征': feature_names_cn,
                    'SHAP贡献值': sv[0],
                    '贡献方向': ['↑ 增加碳足迹' if v > 0 else '↓ 降低碳足迹' for v in sv[0]]
                }).sort_values('SHAP贡献值', key=abs, ascending=False)
                st.dataframe(contrib_df, use_container_width=True, hide_index=True)
        else:
            st.markdown("""
            <div class="info-box">
                <b>👈 请在左侧输入产品参数</b>，然后点击「🚀 开始预测」按钮
                <br><br>
                <small>提示：模型基于历史LCA数据训练，预测结果仅供参考</small>
            </div>
            """, unsafe_allow_html=True)


# ============================================================
# 页面3：逆向优化
# ============================================================
elif page == "🔄 逆向优化":
    st.markdown('<p class="section-title">🔄 逆向参数优化</p>', unsafe_allow_html=True)
    st.markdown("设定目标碳足迹值，系统反推最优参数组合，助力低碳工艺设计")
    st.markdown("---")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### ⚙️ 优化配置")
        st.markdown('<div class="nav-box">', unsafe_allow_html=True)
        target_cf = st.number_input(
            "目标碳足迹 (kgCO₂e/单位)",
            min_value=0.1, value=500.0, step=50.0
        )
        n_search = st.slider("搜索次数", min_value=5000, max_value=100000, value=30000, step=5000)
        st.markdown('</div>', unsafe_allow_html=True)

        optimize_btn = st.button("🔄 开始优化搜索", type="primary", use_container_width=True)

    with col2:
        if optimize_btn:
            with st.spinner(f'正在进行 {n_search} 次随机搜索...'):
                search_ranges = {
                    '原材料消耗(kg)': (0, max(X['原材料消耗(kg)'].max() * 1.2, 5000)),
                    '电力消耗(kWh)': (0, max(X['电力消耗(kWh)'].max() * 1.2, 50000)),
                    '热力/燃料消耗': (0, max(X['热力/燃料消耗'].max() * 1.2, 10000)),
                    '运输(tkm)': (0, max(100, X['运输(tkm)'].max() * 1.2)),
                    '核算边界类型': (0, len(label_encoders['boundary'].classes_) - 1e-6),
                    '分配方法类型': (0, len(label_encoders['allocation'].classes_) - 1e-6),
                }

                np.random.seed(42)
                candidates = np.column_stack([
                    np.random.uniform(*search_ranges[f], n_search)
                    for f in feature_names_cn
                ])
                candidates_df = pd.DataFrame(candidates, columns=feature_names_cn)

                candidates_df['核算边界类型'] = candidates_df['核算边界类型'].round().astype(int).clip(
                    0, len(label_encoders['boundary'].classes_) - 1)
                candidates_df['分配方法类型'] = candidates_df['分配方法类型'].round().astype(int).clip(
                    0, len(label_encoders['allocation'].classes_) - 1)

                preds_log = rf_model.predict(candidates_df)
                preds_ori = np.expm1(preds_log)
                candidates_df['预测碳足迹'] = preds_ori

                def safe_decode(le, code):
                    max_code = len(le.classes_) - 1
                    return le.inverse_transform([int(max(0, min(max_code, code)))])[0]

                candidates_df['与目标偏差'] = np.abs(candidates_df['预测碳足迹'] - target_cf)
                top10 = candidates_df.nsmallest(10, '与目标偏差').reset_index(drop=True)

                top10['核算边界'] = top10['核算边界类型'].apply(lambda x: safe_decode(label_encoders['boundary'], x))
                top10['分配方法'] = top10['分配方法类型'].apply(lambda x: safe_decode(label_encoders['allocation'], x))

                st.markdown(f"### ✅ 最优参数组合（Top 10，目标={target_cf:.0f}）")

                display_cols = ['原材料消耗(kg)', '电力消耗(kWh)', '热力/燃料消耗',
                                '运输(tkm)', '核算边界', '分配方法', '预测碳足迹', '与目标偏差']
                display_df = top10[display_cols].copy()
                display_df.index = range(1, 11)
                display_df.index.name = '序号'

                for col in ['原材料消耗(kg)', '热力/燃料消耗']:
                    display_df[col] = display_df[col].round(1)
                display_df['电力消耗(kWh)'] = display_df['电力消耗(kWh)'].round(0).astype(int)
                display_df['运输(tkm)'] = display_df['运输(tkm)'].round(1)
                display_df['预测碳足迹'] = display_df['预测碳足迹'].round(2)
                display_df['与目标偏差'] = display_df['与目标偏差'].round(2)

                st.dataframe(display_df, use_container_width=True)

                csv = display_df.to_csv().encode('utf-8-sig')
                st.download_button(
                    label="📥 下载优化结果 (CSV)",
                    data=csv,
                    file_name=f'逆向优化_目标{target_cf:.0f}.csv',
                    mime='text/csv'
                )

                st.markdown("---")
                st.markdown("### 📈 优化结果统计")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.markdown('<div class="metric-card"><div class="metric-value">{}</div><div class="metric-label">最优预测值</div></div>'.format(f"{top10['预测碳足迹'].iloc[0]:.2f}"), unsafe_allow_html=True)
                with col_b:
                    st.markdown('<div class="metric-card"><div class="metric-value">{}</div><div class="metric-label">与目标偏差</div></div>'.format(f"{top10['与目标偏差'].iloc[0]:.2f}"), unsafe_allow_html=True)
                with col_c:
                    st.markdown('<div class="metric-card"><div class="metric-value">{}</div><div class="metric-label">搜索范围</div></div>'.format(f"{preds_ori.min():.1f}~{preds_ori.max():.1f}"), unsafe_allow_html=True)

                st.markdown("---")
                st.markdown("### 📊 Top10参数与目标值对比")
                fig_cmp, ax_cmp = plt.subplots(figsize=(8, 4))
                x_labels = [f"Top{i}" for i in range(1, 11)]
                ax_cmp.bar(x_labels, top10['预测碳足迹'], color='#3949ab', alpha=0.7, label='预测碳足迹')
                ax_cmp.axhline(y=target_cf, color='#e53935', linestyle='--', linewidth=2, label=f'目标值 ({target_cf})')
                ax_cmp.set_ylabel('碳足迹 (kgCO₂e)')
                ax_cmp.set_title('最优参数组合预测值与目标对比', fontsize=12, fontweight='bold')
                ax_cmp.legend()
                plt.tight_layout()
                st.pyplot(fig_cmp)
                plt.close()

        else:
            st.markdown("""
            <div class="info-box">
                <b>👈 在左侧设置目标碳足迹值</b>，点击「开始优化搜索」
                <br><br>
                <small>系统将基于历史数据范围进行随机搜索，找到最接近目标值的参数组合</small>
            </div>
            """, unsafe_allow_html=True)


# ============================================================
# 页面4：SHAP分析
# ============================================================
elif page == "📊 SHAP分析":
    st.markdown('<p class="section-title">📊 SHAP特征重要性分析</p>', unsafe_allow_html=True)
    st.markdown("基于SHAP（Shapley Additive Explanations）的模型可解释性分析")
    st.markdown("---")

    if not SHAP_AVAILABLE:
        st.error("⚠️ SHAP库未安装。请运行: `pip install shap`")
    else:
        tab1, tab2, tab3 = st.tabs(["🏆 特征重要性排序", "🌡️ 摘要图", "📈 依赖图分析"])

        with tab1:
            st.markdown("### 特征重要性排序（平均|SHAP|值）")
            shap_df = pd.DataFrame({
                '特征': feature_names_cn,
                '平均|SHAP|值': mean_abs_shap
            }).sort_values('平均|SHAP|值', ascending=True)

            fig, ax = plt.subplots(figsize=(10, 5))
            cmap = plt.cm.Blues(np.linspace(0.4, 0.9, len(shap_df)))
            bars = ax.barh(range(len(shap_df)), shap_df['平均|SHAP|值'], color=cmap, edgecolor='white')
            ax.set_yticks(range(len(shap_df)))
            ax.set_yticklabels(shap_df['特征'])
            ax.set_xlabel('平均|SHAP|值（重要性）')
            ax.set_title('SHAP特征重要性排序', fontsize=13, fontweight='bold')
            for bar, val in zip(bars, shap_df['平均|SHAP|值']):
                ax.text(bar.get_width() + max(shap_df['平均|SHAP|值']) * 0.015,
                        bar.get_y() + bar.get_height() / 2,
                        f'{val:.4f}', va='center', fontsize=9, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            rank_df = pd.DataFrame({
                '排名': range(1, len(feature_names_cn) + 1),
                '特征': feature_names_cn,
                '平均|SHAP|值': mean_abs_shap,
                '重要性等级': ['⭐⭐⭐' if v > np.mean(mean_abs_shap) * 1.5
                              else '⭐⭐' if v > np.mean(mean_abs_shap)
                              else '⭐' for v in mean_abs_shap]
            }).sort_values('平均|SHAP|值', ascending=False).reset_index(drop=True)
            rank_df.index = rank_df['排名']
            st.dataframe(rank_df.drop(columns=['排名']), use_container_width=True)

        with tab2:
            st.markdown("### SHAP摘要图（Summary Plot）")
            st.markdown("每个点代表一个样本，颜色表示特征值高低，位置表示对预测的影响方向")

            fig2, ax2 = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values, X, show=False, ax=ax2)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

            st.markdown("""
            **解读说明：**
            - **水平位置**：SHAP值为正→增加碳足迹；为负→降低碳足迹
            - **颜色**：蓝色=特征值低，红色=特征值高
            - **纵向分布**：点越密集，说明该特征值对预测影响越稳定
            """)

        with tab3:
            st.markdown("### SHAP特征依赖图")
            st.markdown("展示各特征值与SHAP贡献之间的关系")

            dep_feature = st.selectbox("选择要分析的特征", feature_names_cn)

            fig3, ax3 = plt.subplots(figsize=(10, 5))
            feat_idx = feature_names_cn.index(dep_feature)
            shap.dependence_plot(feat_idx, shap_values, X, show=False, ax=ax3,
                                 feature_names=feature_names_cn)
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close()

            st.markdown("""
            **解读说明：**
            - 横轴：该特征的取值
            - 纵轴：对应的SHAP贡献值（正=增加碳足迹，负=降低碳足迹）
            - 颜色：第二个特征的取值（交互效应）
            """)


# ============================================================
# 页面5：批量预测
# ============================================================
elif page == "📂 批量预测":
    st.markdown('<p class="section-title">📂 批量数据预测</p>', unsafe_allow_html=True)
    st.markdown("上传CSV/Excel文件，批量预测多条产品的碳足迹")
    st.markdown("---")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 📥 文件上传")
        st.markdown('<div class="nav-box">', unsafe_allow_html=True)
        st.info("文件需包含以下列：\n`material`, `electricity`, `heat`, `transport`, `boundary`, `allocation`")
        uploaded_file = st.file_uploader(
            "选择文件",
            type=['csv', 'xlsx', 'xls'],
            help="支持CSV和Excel格式"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("### 📄 下载模板")
        template_df = pd.DataFrame({
            'material': [1000, 500, 2000],
            'electricity': [10000, 5000, 20000],
            'heat': [5000, 2000, 8000],
            'transport': [50, 100, 200],
            'boundary': ['摇篮到大门', '摇篮到大门', '摇篮到大门'],
            'allocation': ['物理法', '质量法', '经济价值法']
        })
        template_csv = template_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 下载CSV模板",
            data=template_csv,
            file_name='批量预测模板.csv',
            mime='text/csv',
            use_container_width=True
        )

    with col2:
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    input_df = pd.read_csv(uploaded_file)
                else:
                    input_df = pd.read_excel(uploaded_file)

                st.markdown("### 📋 数据预览")
                st.dataframe(input_df.head(10), use_container_width=True)

                required_cols = ['material', 'electricity', 'heat', 'transport', 'boundary', 'allocation']
                missing_cols = [c for c in required_cols if c not in input_df.columns]

                if missing_cols:
                    st.error(f"❌ 缺少必要列: {missing_cols}")
                    st.markdown("请确保文件包含：`material`, `electricity`, `heat`, `transport`, `boundary`, `allocation`")
                else:
                    if st.button("🚀 执行批量预测", type="primary", use_container_width=True):
                        with st.spinner('正在批量预测...'):
                            for col in ['material', 'electricity', 'heat', 'transport']:
                                input_df[col] = input_df[col].fillna(0)

                            def safe_encode(le, val):
                                try:
                                    return le.transform([str(val)])[0]
                                except ValueError:
                                    return 0

                            input_df['boundary_encoded'] = input_df['boundary'].apply(
                                lambda x: safe_encode(label_encoders['boundary'], x))
                            input_df['allocation_encoded'] = input_df['allocation'].apply(
                                lambda x: safe_encode(label_encoders['allocation'], x))

                            batch_X = input_df[['material', 'electricity', 'heat', 'transport',
                                                'boundary_encoded', 'allocation_encoded']].copy()
                            batch_X.columns = feature_names_cn

                            preds_log = rf_model.predict(batch_X)
                            preds_ori = np.expm1(preds_log)
                            input_df['预测碳足迹(kgCO₂e)'] = preds_ori.round(2)

                            st.markdown("---")
                            st.markdown("### ✅ 预测结果")
                            result_df = input_df[['material', 'electricity', 'heat', 'transport',
                                                  'boundary', 'allocation', '预测碳足迹(kgCO₂e)']].copy()
                            result_df.columns = ['原材料(kg)', '电力(kWh)', '热力', '运输(tkm)',
                                                 '核算边界', '分配方法', '预测碳足迹(kgCO₂e)']
                            st.dataframe(result_df, use_container_width=True)

                            csv = result_df.to_csv(index=False).encode('utf-8-sig')
                            st.download_button(
                                label="📥 下载全部结果 (CSV)",
                                data=csv,
                                file_name='批量预测结果.csv',
                                mime='text/csv'
                            )

                            st.markdown("---")
                            st.markdown("### 📈 结果统计")

                            r1, r2, r3 = st.columns(3)
                            with r1:
                                st.markdown('<div class="metric-card"><div class="metric-value">{}</div><div class="metric-label">预测条数</div></div>'.format(len(result_df)), unsafe_allow_html=True)
                            with r2:
                                st.markdown('<div class="metric-card"><div class="metric-value">{}</div><div class="metric-label">平均碳足迹</div></div>'.format(f"{result_df['预测碳足迹(kgCO₂e)'].mean():.1f}"), unsafe_allow_html=True)
                            with r3:
                                st.markdown('<div class="metric-card"><div class="metric-value">{}</div><div class="metric-label">碳足迹范围</div></div>'.format(f"{result_df['预测碳足迹(kgCO₂e)'].min():.1f}~{result_df['预测碳足迹(kgCO₂e)'].max():.1f}"), unsafe_allow_html=True)

                            st.markdown("---")
                            st.markdown("### 📊 结果可视化")
                            vis_col1, vis_col2 = st.columns(2)
                            with vis_col1:
                                fig_hist, ax_hist = plt.subplots(figsize=(6, 4))
                                ax_hist.hist(result_df['预测碳足迹(kgCO₂e)'], bins=min(20, len(result_df)),
                                             color='#3949ab', edgecolor='white', alpha=0.8)
                                ax_hist.set_xlabel('碳足迹 (kgCO₂e)')
                                ax_hist.set_ylabel('频数')
                                ax_hist.set_title('碳足迹分布直方图', fontsize=12, fontweight='bold')
                                plt.tight_layout()
                                st.pyplot(fig_hist)
                                plt.close()

                            with vis_col2:
                                if len(result_df) <= 30:
                                    fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
                                    labels = [f"#{i+1}" for i in range(len(result_df))]
                                    ax_bar.bar(labels, result_df['预测碳足迹(kgCO₂e)'], color='#1a237e', alpha=0.8)
                                    ax_bar.set_xlabel('样本序号')
                                    ax_bar.set_ylabel('碳足迹 (kgCO₂e)')
                                    ax_bar.set_title('各样本碳足迹对比', fontsize=12, fontweight='bold')
                                    plt.xticks(rotation=45)
                                    plt.tight_layout()
                                    st.pyplot(fig_bar)
                                    plt.close()
                                else:
                                    st.info("样本数>30，跳过条形图展示")

            except Exception as e:
                st.error(f"❌ 文件读取错误: {e}")
        else:
            st.markdown("""
            <div class="info-box">
                <b>👈 请在左侧上传文件</b>，然后点击「执行批量预测」
                <br><br>
                <small>支持CSV（.csv）和Excel（.xlsx/.xls）格式</small>
            </div>
            """, unsafe_allow_html=True)
