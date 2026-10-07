import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import skew
from scipy.linalg import eigh
import plotly.express as px
import re
import io

# Importação de suporte para KMO e Bartlett se disponíveis
try:
    from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
    FA_AVAILABLE = True
except ImportError:
    FA_AVAILABLE = False

# Importação do ReportLab para geração de relatórios em PDF
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Configuração da Página
st.set_page_config(page_title="Plataforma Estatística Avançada", layout="wide", page_icon="📊")

# --- CSS PREMIUM C-LEVEL ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main .block-container { 
        padding-top: 2rem; 
        background-color: #FAFAFB;
    }
    
    h1, h2, h3 { 
        color: #1A252F; 
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    
    /* Cards de Métricas */
    div[data-testid="metric-container"] {
        background-color: #FFFFFF;
        border-left: 5px solid #2980B9;
        padding: 15px 20px;
        border-radius: 8px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.04);
        transition: transform 0.2s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(0,0,0,0.08);
    }
    
    /* Botões Premium */
    div.stButton > button {
        background: linear-gradient(135deg, #1A252F 0%, #2980B9 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 6px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(41, 128, 185, 0.3);
        border: none;
        color: white;
    }
    
    /* Abas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        border-bottom-color: #2980B9 !important;
        color: #2980B9 !important;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# --- CABEÇALHO DO APP ---
st.title("📊 Plataforma de Inteligência Estatística")
st.markdown("Faça o upload da sua base de dados corporativa, configure os parâmetros e gere relatórios científicos avançados.")

# --- FUNÇÕES MATEMÁTICAS E AUXILIARES ---
def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=200)
    buf.seek(0)
    return buf

def calcular_descritiva(df, cols):
    stats = []
    for c in cols:
        s = df[c].dropna()
        if len(s) == 0: continue
        mean = s.mean()
        std = s.std()
        stats.append({
            'Variável': c,
            'Média': mean,
            'Mediana': s.median(),
            'Moda': s.mode().iloc[0] if not s.mode().empty else np.nan,
            'Desv Padrão': std if not pd.isna(std) else 0.0,
            'Variância': s.var() if not pd.isna(s.var()) else 0.0,
            'CV (%)': (std / mean * 100) if mean != 0 and not pd.isna(std) else np.nan,
            'Mínimo': s.min(),
            'Máximo': s.max(),
            'Amplitude': s.max() - s.min(),
            'Q1': s.quantile(0.25),
            'Q3': s.quantile(0.75)
        })
    return pd.DataFrame(stats).set_index('Variável')

def analisar_assimetria(s):
    if s.dropna().nunique() <= 1: return "Constante"
    sk = skew(s.dropna())
    if sk > 0.5: return "Assimétrica Positiva"
    elif sk < -0.5: return "Assimétrica Negativa"
    return "Relativamente Simétrica"

def format_p_value(p):
    return "< 0.001" if p < 0.001 else f"{p:.4f}"

def formatar_texto_latex(texto):
    txt = str(texto).replace('%', '\\%').replace('$', '\\$').replace('_', '\\_')
    return f"\\text{{{txt}}}"

def recuperar_nota_corrompida(val):
    val_str = str(val).strip()
    match = re.match(r'^2026[-/](\d{2})[-/](\d{2})', val_str)
    if match:
        mes = int(match.group(1))
        dia = int(match.group(2))
        return float(f"{dia}.{mes}") if dia <= 5 else float(f"{mes}.{dia}")
    
    match_br = re.match(r'^(\d{2})[-/](\d{2})[-/](2026|\d{2})', val_str)
    if match_br:
        d = int(match_br.group(1))
        m = int(match_br.group(2))
        if d <= 5: return float(f"{d}.{m}")
        if m <= 5: return float(f"{m}.{d}")
        
    limpo = val_str.replace(',', '.')
    limpo = re.sub(r'[^\d\.\-]+', '', limpo)
    try: 
        return float(limpo) if limpo else np.nan
    except: 
        return np.nan

def varimax_rotation(Phi, gamma=1.0, max_iter=500, tol=1e-6):
    p, k = Phi.shape
    R = np.eye(k)
    d = 0
    for i in range(max_iter):
        d_old = d
        Lambda = np.dot(Phi, R)
        u, s, vh = np.linalg.svd(np.dot(Phi.T, Lambda**3 - (gamma / p) * np.dot(Lambda, np.diag(np.sum(Lambda**2, axis=0)))))
        R = np.dot(u, vh)
        d = np.sum(s)
        if d_old != 0 and (d - d_old) / d < tol: 
            break
    return np.dot(Phi, R), R

def promax_rotation(Phi, m=4):
    L_varimax, R_varimax = varimax_rotation(Phi)
    P = np.abs(L_varimax)**m / L_varimax
    coef = np.linalg.lstsq(L_varimax, P, rcond=None)[0]
    u, s, vh = np.linalg.svd(coef)
    T = np.dot(u, vh)
    return np.dot(L_varimax, T)

def calcular_cronbach(df_vars):
    if df_vars.shape[1] < 2: return np.nan
    df_clean = df_vars.dropna()
    k = df_clean.shape[1]
    variancias_itens = df_clean.var(ddof=1).sum()
    variancia_total = df_clean.sum(axis=1).var(ddof=1)
    if variancia_total == 0: return 0.0
    return (k / (k - 1)) * (1 - (variancias_itens / variancia_total))

def gerar_pdf_relatorio(titulo, secoes):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontSize=18, textColor=colors.HexColor("#2C3E50"), spaceAfter=12)
    heading_style = ParagraphStyle('SectionHeading', parent=styles['Heading2'], fontSize=13, textColor=colors.HexColor("#2980B9"), spaceBefore=10, spaceAfter=6)
    body_style = ParagraphStyle('BodyTextCustom', parent=styles['Normal'], fontSize=9, leading=12, spaceAfter=6)
    
    story = [Paragraph(titulo, title_style), HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2C3E50"), spaceAfter=15)]
    
    for sec_title, content in secoes:
        story.append(Paragraph(sec_title, heading_style))
        if isinstance(content, str):
            story.append(Paragraph(content.replace('\n', '<br/>'), body_style))
        elif isinstance(content, pd.DataFrame):
            df_fmt = content.reset_index() if content.index.name else content.copy()
            data = [[Paragraph(str(col), ParagraphStyle('TH', parent=body_style, fontName='Helvetica-Bold', textColor=colors.white)) for col in df_fmt.columns]]
            for _, row in df_fmt.iterrows():
                row_cells = []
                for val in row:
                    val_str = f"{val:.3f}" if isinstance(val, (float, np.floating)) else str(val)
                    row_cells.append(Paragraph(val_str, body_style))
                data.append(row_cells)
            t = Table(data)
            t.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2C3E50")),
                ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('BOTTOMPADDING', (0,0), (-1,-1), 4),
                ('TOPPADDING', (0,0), (-1,-1), 4),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#BDC3C7")),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8F9F9")])
            ]))
            story.append(t)
        elif isinstance(content, io.BytesIO):
            story.append(RLImage(content, width=400, height=220))
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, io.BytesIO):
                    story.append(RLImage(item, width=380, height=200))
                    story.append(Spacer(1, 6))
        story.append(Spacer(1, 10))
        
    doc.build(story)
    buffer.seek(0)
    return buffer

# --- FUNÇÃO DO PARECER AUTOMÁTICO (SEM API) ---
def gerar_parecer_estatistico(modelo_regressao, target_col, independent_cols):
    """Gera um relatório executivo automatizado baseado puramente nas regras estatísticas do modelo."""
    r2_adj = modelo_regressao.rsquared_adj
    f_pvalue = modelo_regressao.f_pvalue
    
    parecer = []
    parecer.append(f"### 📋 Parecer Executivo Automatizado para **{target_col}**\n")
    
    # 1. Avaliação da Validade Geral
    parecer.append("#### 1. Validade e Confiabilidade do Modelo")
    if f_pvalue < 0.05:
        parecer.append(f"- O modelo é **estatisticamente significativo** ($p$-valor da Estatística F = `{format_p_value(f_pvalue)}`, menor que o limite de 0.05). Isto indica que o conjunto de variáveis independentes escolhidas explica de forma consistente as variações em `{target_col}`.")
    else:
        parecer.append(f"- ⚠️ **Alerta de Validade:** O modelo não apresentou significância estatística global ($p$-valor = `{format_p_value(f_pvalue)}` $\\ge 0.05$). Recomenda-se rever as variáveis selecionadas.")
        
    # Explicação do R² Ajustado
    qualidade_r2 = "forte" if r2_adj > 0.7 else ("moderada" if r2_adj > 0.4 else "baixa")
    parecer.append(f"- O **Poder Explicativo ($R^2$ Ajustado)** foi de **`{(r2_adj * 100):.1f}%`**, o que denota uma capacidade de predição **{qualidade_r2}** face aos dados analisados.\n")
    
    # 2. Análise dos Impactos Individuais (Drivers)
    parecer.append("#### 2. Principais Impulsionadores (Coeficientes)")
    
    coefs = modelo_regressao.params.drop('const', errors='ignore')
    pvals = modelo_regressao.pvalues.drop('const', errors='ignore')
    
    for col in independent_cols:
        if col in coefs:
            c = coefs[col]
            p = pvals[col]
            sig_txt = "significativa ($p < 0.05$)" if p < 0.05 else "não significativa estatisticamente"
            direcao = "positiva" if c > 0 else "negativa"
            
            parecer.append(f"- **{col}**: Apresenta uma relação **{direcao}** ({c:.4f}) com a variável dependente. Esta influência é considerável e {sig_txt} ($p$-valor = `{format_p_value(p)}`).")
            
    # 3. Conclusão Prática
    parecer.append("\n#### 3. Recomendações de Gestão")
    if f_pvalue < 0.05:
        parecer.append(f"- Como o modelo é válido, a direção pode utilizar a equação estimada na aba anterior para simular cenários futuros e planear ações voltadas para otimizar os resultados de `{target_col}` com base nos principais impulsionadores identificados.")
    else:
        parecer.append("- Evite tomar decisões estratégicas com base neste modelo atual até que sejam incluídas novas variáveis explicativas mais adequadas ao problema.")
        
    return "\n".join(parecer)

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Painel de Controlo")
    uploaded_file = st.file_uploader("1. Carregar Base de Dados", type=["csv", "xlsx", "xls"])
    
    df = None
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'): 
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')): 
                df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Erro ao carregar os dados: {e}")
            st.stop()

    if df is not None:
        for col in df.columns:
            if str(col).lower() not in ['obs', 'obs.', 'id', 'identificação', 'unidade', 'região']:
                df[col] = df[col].apply(recuperar_nota_corrompida)
                df[col] = pd.to_numeric(df[col], errors='coerce')

        all_numeric_cols = [c for c in df.select_dtypes(include=np.number).columns.tolist() if str(c).lower() not in ['obs', 'obs.', 'id']]
        all_numeric_cols = [c for c in all_numeric_cols if df[c].notna().sum() > 0]

        if len(all_numeric_cols) < 2:
            st.error("A base de dados precisa conter ao menos 2 colunas numéricas válidas.")
            st.stop()
        
        st.markdown("---")
        tipo_analise = st.radio("2. Tipo de Análise Técnica", ["📈 Regressão Linear Múltipla", "🧬 Análise Fatorial Exploratória (AFE)"])
        
        st.markdown("---")
        if "Regressão" in tipo_analise:
            valid_targets = [c for c in all_numeric_cols if df[c].nunique() > 1]
            target_col = st.selectbox("3. Variável Dependente (Y)", valid_targets)
            independent_cols = [c for c in all_numeric_cols if c != target_col]
        else:
            opcoes_fa = [c for c in all_numeric_cols if df[c].nunique() > 1]
            independent_cols = st.multiselect("3. Selecionar Itens para Fatoração", opcoes_fa, default=opcoes_fa)
            
            st.markdown("**Configurações da AFE:**")
            metodo_fatores = st.radio("Critério de Extração", ["Automático (Kaiser - Autovalor > 1)", "Manual (Forçar número fixo)"])
            n_fixo_fatores = 2
            if "Manual" in metodo_fatores:
                n_fixo_fatores = st.number_input("Número de Fatores Desejados", min_value=1, max_value=len(independent_cols), value=2)
            metodo_rotacao = st.selectbox("Rotação dos Fatores", ["Varimax (Fatores Independentes)", "Promax (Fatores Correlacionados)"])
            
        run_btn = st.button("🚀 Processar Análise", use_container_width=True)

# --- TELA DE BOAS VINDAS (Aparece apenas se nenhum arquivo foi enviado) ---
if uploaded_file is None:
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("### 📈 Regressão Múltipla\nIdentifique o grau de impacto das suas variáveis de negócio e gere equações matemáticas preditivas com alto nível de confiabilidade.")
        
    with col2:
        st.success("### 🧬 Análise Fatorial\nReduza a complexidade dos seus dados descobrindo fatores ocultos de comportamento, validados por testes de KMO e Bartlett.")
        
    with col3:
        st.warning("### 📕 Relatórios em PDF\nExporte as suas descobertas com um único clique em relatórios diagramados, prontos para serem apresentados à direção.")
        
    st.markdown("---")
    st.markdown("<h4 style='text-align: center; color: #7F8C8D;'>👈 Comece por enviar a sua base de dados (.csv ou .xls) na barra lateral.</h4>", unsafe_allow_html=True)


# --- EXECUÇÃO DAS ANÁLISES ---
if uploaded_file is not None and df is not None and 'run_btn' in locals() and run_btn:
    reg_independent_cols = [c for c in independent_cols if df[c].dropna().nunique() > 1]
    graficos_pdf = {}

    # ------------------ PIPELINE 1: REGRESSÃO LINEAR ------------------
    if "Regressão" in tipo_analise:
        colunas_reg = [target_col] + reg_independent_cols
        df_reg = df[colunas_reg].dropna().astype(float)
        
        if len(df_reg) < 2:
            st.error("❌ **Dados insuficientes:** Após remover os valores nulos, não sobraram linhas suficientes para processar a regressão.")
            st.stop()
        
        X_multi = sm.add_constant(df_reg[reg_independent_cols])
        Y = df_reg[target_col]
        modelo_multi = sm.OLS(Y, X_multi).fit()

        # --- KPIs Executivos Premium ---
        st.markdown("### 🎯 Performance do Modelo Executivo")
        kpi1, kpi2, kpi3 = st.columns(3)
        with kpi1:
            st.metric(label="Poder Explicativo (R² Ajustado)", value=f"{(modelo_multi.rsquared_adj * 100):.1f}%", delta="Confiança Alta" if modelo_multi.rsquared_adj > 0.7 else "Confiança Moderada/Baixa")
        with kpi2:
            st.metric(label="Estatística F (Significância)", value=f"{modelo_multi.fvalue:.2f}", delta="Modelo Válido (p<0.05)" if modelo_multi.f_pvalue < 0.05 else "Alerta de Validade", delta_color="normal" if modelo_multi.f_pvalue < 0.05 else "inverse")
        with kpi3:
            st.metric(label="Volume de Amostra (n)", value=f"{int(modelo_multi.nobs)} registos")
        st.markdown("<br>", unsafe_allow_html=True)

        df_desc = calcular_descritiva(df_reg, colunas_reg)

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Descritiva", "📊 Distribuições", "🔗 Correlação (Heatmap)", "🧮 Equação & Insights", "📋 Diagnóstico Avançado"])

        with tab1:
            st.dataframe(df_desc.style.format("{:.2f}"), use_container_width=True)
            st.subheader("🔎 Análise de Assimetria")
            for var in colunas_reg:
                st.markdown(f"- **{var}:** {analisar_assimetria(df_reg[var])}")

        with tab2:
            cols_ui = st.columns(2)
            graficos_pdf['histograms'] = []
            for i, col in enumerate(colunas_reg):
                with cols_ui[i % 2]:
                    fig, ax = plt.subplots(figsize=(5, 3))
                    sns.histplot(df_reg[col], kde=True, ax=ax, color='#3498DB')
                    ax.set_title(f"Distribuição: {col}")
                    st.pyplot(fig)
                    graficos_pdf['histograms'].append(fig_to_bytes(fig))
                    plt.close()

        with tab3:
            matriz_corr = df_reg.corr()
            
            # Gráfico Interativo na Tela (Plotly)
            fig_plotly = px.imshow(matriz_corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r")
            fig_plotly.update_layout(margin=dict(t=10, l=10, r=10, b=10))
            st.plotly_chart(fig_plotly, use_container_width=True)
            
            # Gráfico Oculto para o PDF (Matplotlib)
            fig_pdf, ax_pdf = plt.subplots(figsize=(6, 4))
            sns.heatmap(matriz_corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax_pdf)
            graficos_pdf['corr_heatmap'] = fig_to_bytes(fig_pdf)
            plt.close(fig_pdf)

        with tab4:
            intercepto = modelo_multi.params['const']
            partes_equacao = [f"{intercepto:.4f}"]
            for col in reg_independent_cols:
                coef = modelo_multi.params[col]
                partes_equacao.append(f"{'+' if coef >= 0 else '-'} ({abs(coef):.4f} \\cdot {formatar_texto_latex(col)})")
            st.info("### Equação Estimada:")
            st.write(f"$$\\widehat{{{formatar_texto_latex(target_col)}}} = {' '.join(partes_equacao)}$$")
            
            # --- INTEGRAÇÃO DA NOVA FERRAMENTA DE INSIGHTS ---
            st.markdown("---")
            st.subheader("💡 Insights Automáticos de Negócio")
            st.markdown("Clique abaixo para gerar um relatório analítico estruturado com base estritamente nos cálculos estatísticos do modelo.")
            
            if st.button("📊 Gerar Relatório Executivo por Regras", use_container_width=True):
                with st.spinner("A processar indicadores estatísticos..."):
                    relatorio_gerado = gerar_parecer_estatistico(modelo_multi, target_col, reg_independent_cols)
                    st.markdown(relatorio_gerado)

        with tab5:
            st.text(modelo_multi.summary().tables[0].as_text())
            st.text(modelo_multi.summary().tables[1].as_text())

        # --- EXPORTAÇÃO REGRESSÃO ---
        st.markdown("---")
        st.subheader("📥 Central de Exportação de Relatórios")
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            coef_df = pd.DataFrame({'Coeficiente': modelo_multi.params, 'P-Valor': modelo_multi.pvalues, 'Erro Padrão': modelo_multi.bse})
            st.download_button(label="📄 Extrair Coeficientes (CSV)", data=coef_df.to_csv().encode('utf-8'), file_name="coeficientes_regressao.csv", mime="text/csv", use_container_width=True)
        with col_exp2:
            if PDF_AVAILABLE:
                resumo_texto = f"R²: {modelo_multi.rsquared:.4f} | R² Ajustado: {modelo_multi.rsquared_adj:.4f}\nF-statistic: {modelo_multi.fvalue:.4f} (p-val: {modelo_multi.f_pvalue:.4f})"
                secoes_pdf = [
                    ("1. Estatísticas Descritivas", df_desc),
                    ("2. Resumo do Modelo OLS", resumo_texto),
                    ("3. Coeficientes do Modelo", coef_df),
                    ("4. Matriz de Correlação", graficos_pdf.get('corr_heatmap')),
                    ("5. Gráficos de Distribuição (Histogramas)", graficos_pdf.get('histograms'))
                ]
                st.download_button(label="📕 Baixar Relatório Executivo Completo (PDF)", data=gerar_pdf_relatorio("Relatório Científico: Regressão Linear Múltipla", secoes_pdf), file_name="relatorio_regressao.pdf", mime="application/pdf", use_container_width=True)
            else:
                st.info("Para gerar o PDF, instale a biblioteca reportlab (`pip install reportlab`).")

    # ------------------ PIPELINE 2: ANÁLISE FATORIAL COMPLETA ------------------
    else:
        st.header("🧬 Módulo Analítico: Fatoração Exploratória")
        
        if len(reg_independent_cols) < 3:
            st.warning("Selecione pelo menos 3 variáveis numéricas para processar a análise fatorial.")
        else:
            df_fa = df[reg_independent_cols].dropna().astype(float)
            if len(df_fa) < 3:
                st.error("❌ **Dados insuficientes:** Tente remover variáveis com muitos dados nulos na aba lateral.")
                st.stop()
                
            corr_matrix_df = df_fa.corr()
            corr_matrix_np = corr_matrix_df.values
            ev, _ = eigh(corr_matrix_np)
            ev = sorted(ev, reverse=True)
            n_fatores = max(1, sum(1 for x in ev if x >= 1.0)) if "Automático" in metodo_fatores else min(n_fixo_fatores, len(reg_independent_cols))

            tab_fa1, tab_fa2, tab_fa3, tab_fa4, tab_fa5 = st.tabs(["📋 Validação Amostral", "📐 Variância Explicada", "📊 Cargas & Heatmap", "🧬 Comunalidades", "📥 Extrair Escores"])

            with tab_fa1:
                if not FA_AVAILABLE:
                    st.warning("Biblioteca 'factor_analyzer' ausente na nuvem.")
                else:
                    try:
                        chi_square, p_value_bartlett = calculate_bartlett_sphericity(df_fa)
                        kmo_all, kmo_model = calculate_kmo(df_fa)
                        
                        c1, c2 = st.columns(2)
                        with c1:
                            st.metric(label="KMO Geral (Kaiser-Meyer-Olkin)", value=f"{kmo_model:.3f}", delta="Amostra Adequada" if kmo_model >= 0.6 else "Amostra Fraca", delta_color="normal" if kmo_model >= 0.6 else "inverse")
                        with c2:
                            st.metric(label="Bartlett Sphericity (p-valor)", value=format_p_value(p_value_bartlett), delta="Matriz Elegível" if p_value_bartlett < 0.05 else "Alerta", delta_color="normal" if p_value_bartlett < 0.05 else "inverse")
                        
                        st.markdown("<br><b>Medida de Adequabilidade Amostral Individual (MSA)</b>", unsafe_allow_html=True)
                        df_msa = pd.DataFrame({'Variável': reg_independent_cols, 'MSA Individual': kmo_all})
                        st.dataframe(df_msa.style.map(lambda x: 'background-color: #ffcccc' if x < 0.5 else '', subset=['MSA Individual']).format({'MSA Individual': '{:.3f}'}), use_container_width=True)
                    except Exception as e:
                        st.error(f"Erro ao computar KMO/Bartlett: {e}")

            with tab_fa2:
                var_explicada = [(x / sum(ev)) * 100 for x in ev]
                var_acumulada = np.cumsum(var_explicada)
                df_var_exp = pd.DataFrame([{"Fator": f"Fator {i+1}", "Autovalor": ev[i], "% da Variância": var_explicada[i], "% Acumulada": var_acumulada[i]} for i in range(len(ev))]).set_index("Fator")
                st.dataframe(df_var_exp.style.format("{:.3f}"), use_container_width=True)
                
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.scatter(range(1, len(ev) + 1), ev, color='#E74C3C', zorder=3)
                ax.plot(range(1, len(ev) + 1), ev, color='#34495E', linestyle='--')
                ax.axhline(y=1, color='gray', linestyle=':')
                ax.set_title("Scree Plot")
                st.pyplot(fig)
                graficos_pdf['scree_plot'] = fig_to_bytes(fig)
                plt.close()

            with tab_fa3:
                try:
                    A = df_fa.values - np.mean(df_fa.values, axis=0)
                    _, _, Vht = np.linalg.svd(A, full_matrices=False)
                    cargas_iniciais = Vht[:n_fatores].T * np.sqrt(ev[:n_fatores])
                    cargas_rotacionadas = promax_rotation(cargas_iniciais) if "Promax" in metodo_rotacao and n_fatores > 1 else (varimax_rotation(cargas_iniciais)[0] if n_fatores > 1 else cargas_iniciais)
                        
                    colunas_fatores = [f"Fator {i+1}" for i in range(n_fatores)]
                    df_cargas = pd.DataFrame(cargas_rotacionadas, columns=colunas_fatores, index=reg_independent_cols)
                    
                    c1, c2 = st.columns([1, 1.2])
                    with c1:
                        st.dataframe(df_cargas.style.format("{:.3f}").background_gradient(cmap="bwr", vmin=-1, vmax=1), use_container_width=True)
                    with c2:
                        # Gráfico Interativo Plotly
                        fig_plotly_fa = px.imshow(df_cargas, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
                        fig_plotly_fa.update_layout(margin=dict(t=10, l=10, r=10, b=10))
                        st.plotly_chart(fig_plotly_fa, use_container_width=True)
                        
                        # Gráfico para o PDF
                        fig_fa, ax_fa = plt.subplots(figsize=(5, 4))
                        sns.heatmap(df_cargas, annot=True, cmap="bwr", center=0, fmt=".2f", vmin=-1, vmax=1, ax=ax_fa)
                        graficos_pdf['cargas_heatmap'] = fig_to_bytes(fig_fa)
                        plt.close(fig_fa)
                except Exception as e:
                    st.error(f"Erro ao calcular cargas: {e}")

            with tab_fa4:
                try:
                    comunalidades = np.sum(cargas_rotacionadas**2, axis=1)
                    df_comun = pd.DataFrame({'Variável': reg_independent_cols, 'Comunalidade': comunalidades}).set_index('Variável')
                    st.dataframe(df_comun.style.map(lambda x: 'background-color: #ffcccc' if x < 0.5 else '', subset=['Comunalidade']).format({'Comunalidade': '{:.3f}'}), use_container_width=True)
                    
                    st.markdown("---")
                    st.subheader("Confiabilidade da Escala (Alfa de Cronbach)")
                    for fat in colunas_fatores:
                        dominantes = df_cargas.index[df_cargas[fat].abs() >= 0.40].tolist()
                        if len(dominantes) >= 2:
                            st.markdown(f"- **{fat}:** $\\alpha$ = **{calcular_cronbach(df_fa[dominantes]):.3f}** (Itens: {', '.join(dominantes)})")
                        else:
                            st.markdown(f"- **{fat}:** Itens insuficientes para confiabilidade.")
                except Exception as e:
                    st.error(f"Erro no processamento das comunalidades: {e}")

            with tab_fa5:
                try:
                    R_inv = np.linalg.pinv(corr_matrix_np)
                    escores_np = np.dot(((df_fa - df_fa.mean()) / df_fa.std()).values, np.dot(R_inv, cargas_rotacionadas))
                    df_escores = pd.DataFrame(escores_np, columns=colunas_fatores, index=df_fa.index)
                    
                    df_completo_com_fatores = df.copy()
                    for fat in colunas_fatores: df_completo_com_fatores[fat] = df_escores[fat]
                        
                    st.dataframe(df_escores.head(10).style.format("{:.4f}"), use_container_width=True)
                    st.download_button(label="📥 Download Base Atualizada c/ Fatores (CSV)", data=df_completo_com_fatores.to_csv(index=False).encode('utf-8'), file_name="base_com_escores_fatoriais.csv", mime="text/csv", use_container_width=True)
                except Exception as e:
                    st.error(f"Erro ao computar os escores: {e}")

            # --- EXPORTAÇÃO AFE ---
            st.markdown("---")
            col_exp1, col_exp2 = st.columns(2)
            with col_exp1:
                if 'df_cargas' in locals(): st.download_button("📄 Download Cargas Fatoriais (CSV)", df_cargas.to_csv().encode('utf-8'), "cargas_fatoriais.csv", "text/csv", use_container_width=True)
            with col_exp2:
                if PDF_AVAILABLE and 'df_cargas' in locals() and 'df_var_exp' in locals():
                    kmo_str = f"KMO: {kmo_model:.3f} | Bartlett p-val: {format_p_value(p_value_bartlett)}" if FA_AVAILABLE else "Não disponível"
                    secoes_pdf_fa = [("1. Testes de Adequabilidade Amostral", kmo_str), ("2. Variância Total Explicada", df_var_exp), ("3. Gráfico de Sedimentação (Scree Plot)", graficos_pdf.get('scree_plot')), ("4. Matriz de Cargas Fatoriais Rotacionadas", df_cargas), ("5. Heatmap das Cargas Fatoriais", graficos_pdf.get('cargas_heatmap')), ("6. Comunalidades das Variáveis", df_comun)]
                    st.download_button("📕 Baixar Relatório Completo AFE (PDF)", gerar_pdf_relatorio("Relatório Analítico: Análise Fatorial", secoes_pdf_fa), "relatorio_afe.pdf", "application/pdf", use_container_width=True)
