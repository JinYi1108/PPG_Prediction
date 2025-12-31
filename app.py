import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb  
import shap         
import warnings



warnings.filterwarnings("ignore")

if 'lang' not in st.session_state:
    st.session_state.lang = 'CH'




translations = {
    'CH': {
        'page_title': "PPG 风险预测模型",
        'main_title': "PPG >= 20 初筛",
        'description_1': "本模型基于XGBoost算法构建,用于预测患者PPG是否高于等于或小于20 (mmHg)。",
        'description_2': "请在左侧边栏输入以下 **5** 个指标，然后点击“预测”。",
        'sidebar_header': "患者指标输入",
        
        'sidebar_subheader_clinical': "临床表现",
        

        'varices_label': "食管胃底静脉曲张 (Esophageal gastric varices)",
        'varices_0': "0 - 直径<5mm, 无红色征象或糜烂, 表面光滑,无纤维蛋白覆盖",
        'varices_1': "1 - 直径5-10mm,局部红色征象或孤立性糜烂",
        'varices_2': "2 - 直径>10mm,弥漫性红色征象,纤维蛋白帽或血疱样斑点,渗血或喷射性出血",
        
        
        'splenomegaly_label': "脾肿大 (Splenomegaly)",
        'splenomegaly_0': "0 - 无 (≤ 12 cm)",
        'splenomegaly_1': "1 - 有 (> 12 cm)",
        'splenomegaly_2': "2 - 脾切除术后 (status post splenectomy)",

        'ascites_label': "腹水 (Ascites)",
        'ascites_0': "0 - 无/轻度",
        'ascites_1': "1 - 中-重度",
        
        'sidebar_subheader_lab': "检查指标", 

        'rpvf_label': "门静脉右支流量 (RPVF, mL/min)",
        'hb_label': "血红蛋白 (Hb, g/L)",
        
        'predict_button': "开始预测",
        

        'results_header': "预测结果与解释",
        'prob_header': "预测概率",
        'prob_metric_label': "PPG >= 20 的风险概率",
        'high_risk': "**预测结果：高风险 (>= 50%)**",
        'low_risk': "**预测结果：低风险 (< 50%)**",
        'disclaimer': "本预测基于您的输入数据，结果仅供参考，不能替代专业医疗诊断。",
        

        'shap_header': "个体预测解释",
        'shap_desc_1': "下图显示了各项指标对本次预测结果的影响：",
        'shap_desc_red': "🔴 **红色特征** 将风险推高。",
        'shap_desc_blue': "🔵 **蓝色特征** 将风险拉低。",
        'global_shap_header': "模型全局特征重要性",
        'global_shap_desc': "下图显示了模型在所有数据上学到的总体特征重要性，作为参考。",
        'global_shap_caption': "全局特征重要性 (Mean Absolute SHAP Value)",
        'image_not_found_warn': "未找到 'SHAP_Summary_Bar.png'。"
    },
    'EN': {
        'page_title': "PPG Risk Prediction Model",
        'main_title': "PPG >= 20 Preliminary Screening",
        'description_1': "This model is built based on XGBoost and is used to predict whether the patient's PPG is higher than, equal to or less than 20 (mmHg).",
        'description_2': "Please input the following **5** indicators in the sidebar and click 'Start Prediction'.",
        'sidebar_header': "Patient Indicators Input",

        'sidebar_subheader_clinical': "Clinical Manifestations",

        'varices_label': "Esophageal gastric varices",
        'varices_0': "0 - variceal diameter <5mm, absence of red color signs or erosions, smooth surface without fibrin caps",
        'varices_1': "1 - variceal diameter 5-10mm, presence of localized red color signs (RC+) or isolated erosions",
        'varices_2': "2 - variceal diameter >10mm, presence of diffuse red color signs, fibrin caps/hematocystic spots or active oozing or spurting bleeding",
        
        'splenomegaly_label': "Splenomegaly",
        'splenomegaly_0': "0 - Absent (≤ 12 cm)",
        'splenomegaly_1': "1 - Present (> 12 cm)",
        'splenomegaly_2': "2 - Status post splenectomy",

        'ascites_label': "Ascites",
        'ascites_0': "0 - Absent/Mild",
        'ascites_1': "1 - Moderate-Severe",
        'sidebar_subheader_lab': "Laboratory Indicators",


        'sidebar_subheader_lab': "Laboratory Indicators",

        'rpvf_label': "Right Portal Vein Flow (RPVF, mL/min)",
        'hb_label': "Hemoglobin (Hb, g/L)",
        
        'predict_button': "Start Prediction",
        
        'results_header': "Prediction Results and Interpretation",
        'prob_header': "Prediction Probability",
        'prob_metric_label': "Risk Probability of PPG >= 20",
        'high_risk': "**Prediction: High Risk (>= 50%)**",
        'low_risk': "**Prediction: Low Risk (< 50%)**",
        'disclaimer': "This prediction is based on your input data and is for reference only. It cannot replace professional medical diagnosis.",
        
        'shap_header': "Individual Prediction Interpretation",
        'shap_desc_1': "The plot below shows the impact of each indicator on this prediction:",
        'shap_desc_red': "🔴 **Red features** push the risk higher.",
        'shap_desc_blue': "🔵 **Blue features** pull the risk lower.",
        'global_shap_header': "Global Model Feature Importance",
        'global_shap_desc': "The plot below shows the model's overall feature importance learned from all data, for reference.",
        'global_shap_caption': "Global Feature Importance (Mean Absolute SHAP Value)",
        'image_not_found_warn': "Warning: 'SHAP_Summary_Bar.png' not found."
    }
}
lang = st.session_state.lang 
T = translations[lang]


st.set_page_config(layout="wide", page_title=T['page_title'])
XGB_FEATURES = [
    "Esophageal gastric varices",
    "Splenomegaly",
    "RPVF",
    "Ascites",
    "Hb"
]


@st.cache_resource
def load_models():

    try:

        model = joblib.load('xgb_calibrated_model.pkl')
    except FileNotFoundError:
        st.error("找不到 'xgb_calibrated_model.pkl'。")
        st.stop()
        
    try:

        base_model_shap = joblib.load('xgb_base_model_for_shap.pkl')
        explainer = shap.TreeExplainer(base_model_shap)
    except FileNotFoundError:
        st.error("找不到 'xgb_base_model_for_shap.pkl'。")
        st.stop()
    except Exception as e:
        st.error(f"加载 SHAP Explainer 失败: {e}。")
        st.stop()
        
    return model, explainer


calibrated_model, shap_explainer = load_models()



st.title(T['main_title'])
st.write(T['description_1'])
st.write(T['description_2'])

st.sidebar.radio(
    "语言 / Language",
    options=['CH', 'EN'],
    format_func=lambda x: "中文" if x == 'CH' else "English",
    key='lang',
    horizontal=True
)

st.sidebar.header(T['sidebar_header'])
inputs = {}


st.sidebar.subheader(T['sidebar_subheader_clinical'])

varices_options_map = {
    0: T['varices_0'],
    1: T['varices_1'],
    2: T['varices_2']
}

inputs['Esophageal gastric varices'] = st.sidebar.selectbox(
    T['varices_label'],
    options=[0, 1, 2], 
    index=0,           
    format_func=lambda x: varices_options_map[x] 
)

splenomegaly_options_map = {
    0: T['splenomegaly_0'],
    1: T['splenomegaly_1'],
    2: T['splenomegaly_2']
}
inputs['Splenomegaly'] = st.sidebar.selectbox(
    T['splenomegaly_label'],
    options=[0, 1, 2],  
    index=0,            
    format_func=lambda x: splenomegaly_options_map[x] 
)
ascites_options_map = {
    0: T['ascites_0'],
    1: T['ascites_1']
}
inputs['Ascites'] = st.sidebar.selectbox(
    T['ascites_label'], 
    options=[0, 1], 
    index=0, 
    format_func=lambda x: ascites_options_map[x] 
)


st.sidebar.subheader(T['sidebar_subheader_lab'])
inputs['RPVF'] = st.sidebar.number_input(
    T['rpvf_label'], 
    min_value=-1000.0000,  
    max_value=2000.0000, 
    value=0.0000,        
    step=0.01,
    format="%.4f"
)

inputs['Hb'] = st.sidebar.number_input(
    T['hb_label'], 
    min_value=0.0,  
    max_value=1000.0, 
    value=0.0,
    step=0.1,
    format="%.1f"
)


input_df = pd.DataFrame([inputs])
input_df = input_df[XGB_FEATURES]


if st.sidebar.button(T['predict_button'], type="primary"):
    

    try:
        prediction_proba = calibrated_model.predict_proba(input_df)[0][1] 
    except Exception as e:
        st.error(f"模型预测失败: {e}")
        st.stop()
        

    try:
        shap_values_obj = shap_explainer(input_df)
        

        if shap_values_obj.values.ndim == 3:
             shap_values_class1 = shap_values_obj.values[0, :, 1]
        else:
             shap_values_class1 = shap_values_obj.values[0]
        

        expected_value = shap_explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = expected_value[1] 

    except Exception as e:
        st.error(f"SHAP 值计算失败: {e}")
        st.stop()


    st.header(T['results_header'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(T['prob_header'])
        st.metric(label=T['prob_metric_label'], value=f"{prediction_proba:.2%}")
        
        if prediction_proba > 0.5:
            st.warning(T['high_risk'])
        else:
            st.success(T['low_risk'])
        
        st.write("---")
        st.write(T['disclaimer'])

    with col2:
        st.subheader(T['shap_header'])
        st.write(T['shap_desc_1'])
        st.write(T['shap_desc_red'])
        st.write(T['shap_desc_blue'])

        try:

            fig_html = shap.force_plot(expected_value, 
                                  shap_values_class1, 
                                  input_df,
                                  matplotlib=False
                                ) 
            shap_js = shap.getjs()
            full_html = f"<head><meta charset='utf-8'>{shap_js}</head><body>{fig_html.html()}</body>"
            

            st.components.v1.html(full_html, height=350, scrolling=True)
            
        except Exception as e:
            st.error(f"绘制 SHAP Force Plot 失败: {e}")

    st.header(T['global_shap_header'])
    st.write(T['global_shap_desc'])
    try:
        st.image('SHAP_Summary_Bar.png', caption=T['global_shap_caption'])
    except FileNotFoundError:
        st.warning(T['image_not_found_warn'])
