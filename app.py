import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import skew
from scipy.linalg import eigh
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
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Configuração da Página
st.set_page_config(page_title="Plataforma Estatística Avançada", layout="wide", page_icon="📊")

st.markdown("""
    <style>
    .main .block-container { padding-top: 2rem; }
    h1, h2, h3 { color: #2C3E50; }
    .stAlert { margin-top: 1rem; }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Plataforma de Análise Estatística Avançada")
st.markdown("Faça o upload da sua base de dados, configure os parâmetros e gere relatórios científicos completos.")

# --- FUNÇÕES MATEMÁTICAS E DE GERADORAS ---

def calcular_descritiva(df, cols):
    stats = []
    for c in cols:
        s = df[c].dropna()
        if len(s) == 0:
            continue
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
    if s.dropna().nunique() <= 1: 
        return "Constante"
    sk = skew(s.dropna())
    if sk > 0.5: 
        return "Assimétrica Positiva"
    elif sk < -0.5: 
        return "Assimétrica Negativa"
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

# Rotações de Fatores
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
    if df_vars.shape[1] < 2:
        return np.nan
    df_clean = df_vars.dropna()
    k = df_clean.shape[1]
    variancias_itens = df_clean.var(ddof=1).sum()
    variancia_total = df_clean.sum(axis=1).var(ddof=1)
    if variancia_total == 0:
        return 0.0
    alfa = (k / (k - 1)) * (1 - (variancias_itens / variancia_total))
    return alfa

# Gerador de PDF Generico
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
            # Formatar Tabela para ReportLab
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
        story.append(Spacer(1, 10))
        
    doc.build(story)
    buffer.seek(0)
    return buffer

# Interface Lateral (Sidebar)
with st.sidebar:
    st.header("⚙️ Painel de Controle")
    uploaded_file = st.file_uploader("1. Carregar Base de Dados", type=["csv", "xlsx"])
    
    df = None
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'): 
                df = pd.read_csv(uploaded_file)
            else: 
                df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Erro ao carregar os dados: {e}")
            st.stop()

    if df is not None:
        for col in df.columns:
            if str(col).lower() not in ['obs', 'obs.', 'id', 'identificação', 'unidade', 'região']:
                df[col] = df[col].apply(recuperar_nota_corrompida)

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

# --- EXECUÇÃO DAS ANÁLISES ---
if uploaded_file is not None and df is not None and 'run_btn' in locals() and run_btn:
    reg_independent_cols = [c for c in independent_cols if df[c].dropna().nunique() > 1]
    
    # ------------------ PIPELINE 1: REGRESSÃO LINEAR ------------------
    if "Regressão" in tipo_analise:
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Descritiva", "📊 Distribuições", "🔗 Correlação", "🧮 Equação Estimada", "📋 Diagnóstico"])
        
        colunas_reg = [target_col] + reg_independent_cols
        df_reg = df[colunas_reg].dropna()
        
        X_multi = sm.add_constant(df_reg[reg_independent_cols])
        Y = df_reg[target_col]
        modelo_multi = sm.OLS(Y, X_multi).fit()

        df_desc = calcular_descritiva(df_reg, colunas_reg)

        with tab1:
            st.header("Módulo 1: Descritiva das Variáveis")
            st.dataframe(df_desc.style.format("{:.2f}"), use_container_width=True)
            
            st.subheader("🔎 Análise de Assimetria")
            for var in colunas_reg:
                estado_assimetria = analisar_assimetria(df_reg[var])
                st.markdown(f"- **{var}:** {estado_assimetria}")

        with tab2:
            st.header("Módulo 2: Distribuições")
            cols_ui = st.columns(2)
            for i, col in enumerate(colunas_reg):
                with cols_ui[i % 2]:
                    fig, ax = plt.subplots(figsize=(5, 3))
                    sns.histplot(df_reg[col], kde=True, ax=ax, color='#3498DB')
                    st.pyplot(fig)
                    plt.close()

        with tab3:
            st.header("Módulo 3: Correlações Lineares")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(df_reg.corr(), annot=True, cmap="coolwarm", fmt=".2f")
            st.pyplot(fig)
            plt.close()

        with tab4:
            st.header("Módulo 4: Equação de Regressão Múltipla")
            intercepto = modelo_multi.params['const']
            partes_equacao = [f"{intercepto:.4f}"]
            for col in reg_independent_cols:
                coef = modelo_multi.params[col]
                partes_equacao.append(f"{'+' if coef >= 0 else '-'} ({abs(coef):.4f} \\cdot {formatar_texto_latex(col)})")
            st.info("### Equação Estimada:")
            st.write(f"$$ \\widehat{{{formatar_texto_latex(target_col)}}} = {' '.join(partes_equacao)} $$")

        with tab5:
            st.header("Módulo 5: Diagnóstico Completo")
            st.text(modelo_multi.summary().tables[0].as_text())
            st.text(modelo_multi.summary().tables[1].as_text())

        # --- EXPORTAÇÃO DA REGRESSÃO ---
        st.markdown("---")
        st.subheader("📥 Exportar Relatório de Regressão")
        col_exp1, col_exp2 = st.columns(2)
        
        with col_exp1:
            # Exportar CSV das Estatísticas Descritivas e Coeficientes
            coef_df = pd.DataFrame({'Coeficiente': modelo_multi.params, 'P-Valor': modelo_multi.pvalues, 'Erro Padrão': modelo_multi.bse})
            csv_data = coef_df.to_csv().encode('utf-8')
            st.download_button(
                label="📄 Baixar Coeficientes (CSV)",
                data=csv_data,
                file_name="coeficientes_regressao.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        with col_exp2:
            if PDF_AVAILABLE:
                resumo_texto = f"R²: {modelo_multi.rsquared:.4f} | R² Ajustado: {modelo_multi.rsquared_adj:.4f}\nF-statistic: {modelo_multi.fvalue:.4f} (p-val: {modelo_multi.f_pvalue:.4f})"
                secoes_pdf = [
                    ("1. Estatísticas Descritivas", df_desc),
                    ("2. Resumo do Modelo OLS", resumo_texto),
                    ("3. Coeficientes do Modelo", coef_df)
                ]
                pdf_bytes = gerar_pdf_relatorio("Relatório Científico: Regressão Linear Múltipla", secoes_pdf)
                st.download_button(
                    label="📕 Baixar Relatório Completo (PDF)",
                    data=pdf_bytes,
                    file_name="relatorio_regressao.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            else:
                st.info("Para gerar o PDF, instale a biblioteca reportlab (`pip install reportlab`).")

    # ------------------ PIPELINE 2: ANÁLISE FATORIAL COMPLETA ------------------
    else:
        st.header("🧬 Relatório Científico: Análise Fatorial Exploratória")
        
        if len(reg_independent_cols) < 3:
            st.warning("Selecione pelo menos 3 variáveis numéricas para processar a análise fatorial.")
        else:
            df_fa = df[reg_independent_cols].dropna().astype(float)
            
            tab_fa1, tab_fa2, tab_fa3, tab_fa4, tab_fa5 = st.tabs([
                "📋 1. Validação & Adequabilidade", 
                "📐 2. Variância Explicada (Kaiser)", 
                "📊 3. Matriz de Cargas & Heatmap", 
                "🧬 4. Comunalidades & Confiabilidade",
                "📥 5. Escores Fatoriais (Download)"
            ])
            
            corr_matrix_df = df_fa.corr()
            corr_matrix_np = corr_matrix_df.values
            ev, _ = eigh(corr_matrix_np)
            ev = sorted(ev, reverse=True)
            
            if "Automático" in metodo_fatores:
                n_fatores = max(1, sum(1 for x in ev if x >= 1.0))
            else:
                n_fatores = min(n_fixo_fatores, len(reg_independent_cols))

            # --- TAB 1: ADEQUABILIDADE ---
            with tab_fa1:
                st.subheader("Testes de Validação Amostral (SPSS Style)")
                if not FA_AVAILABLE:
                    st.warning("Biblioteca 'factor_analyzer' ausente na nuvem. Exibindo Matriz de Correlação.")
                else:
                    try:
                        chi_square, p_value_bartlett = calculate_bartlett_sphericity(df_fa)
                        kmo_all, kmo_model = calculate_kmo(df_fa)
                        
                        c1, c2 = st.columns(2)
                        with c1:
                            st.metric(label="KMO Geral (Kaiser-Meyer-Olkin)", value=f"{kmo_model:.3f}")
                            if kmo_model >= 0.6: 
                                st.success("✓ Amostra Adequada (KMO > 0.60)")
                            else: 
                                st.warning("⚠️ Amostra Fraca (KMO < 0.60)")
                        with c2:
                            st.metric(label="Bartlett Sphericity (p-valor)", value=format_p_value(p_value_bartlett))
                            if p_value_bartlett < 0.05: 
                                st.success("✓ Matriz elegível para agrupamento fatorial.")
                        
                        st.markdown("#### Medida de Adequabilidade Amostral Individual (MSA)")
                        df_msa = pd.DataFrame({'Variável': reg_independent_cols, 'MSA Individual': kmo_all})
                        
                        styler_msa = df_msa.style
                        try:
                            styler_msa = styler_msa.map(lambda x: 'background-color: #ffcccc' if x < 0.5 else '', subset=['MSA Individual'])
                        except AttributeError:
                            styler_msa = styler_msa.applymap(lambda x: 'background-color: #ffcccc' if x < 0.5 else '', subset=['MSA Individual'])
                            
                        st.dataframe(styler_msa.format({'MSA Individual': '{:.3f}'}), use_container_width=True)
                        st.caption("Valores de MSA abaixo de 0.50 sugerem que a variável correspondente deve ser removida do modelo.")
                    except Exception as e:
                        st.error(f"Erro ao computar KMO/Bartlett: {e}")

            # --- TAB 2: VARIÂNCIA EXPLICADA ---
            with tab_fa2:
                st.subheader("Tabela de Variância Total Explicada")
                total_var = sum(ev)
                var_explicada = [(x / total_var) * 100 for x in ev]
                var_acumulada = np.cumsum(var_explicada)
                
                rows_variancia = []
                for i in range(len(ev)):
                    rows_variancia.append({
                        "Fator": f"Fator {i+1}",
                        "Autovalor (Eigenvalue)": ev[i],
                        "% da Variância": var_explicada[i],
                        "% Acumulada": var_acumulada[i]
                    })
                df_var_exp = pd.DataFrame(rows_variancia).set_index("Fator")
                st.dataframe(df_var_exp.style.format("{:.3f}"), use_container_width=True)
                
                st.markdown("---")
                st.subheader("Gráfico de Sedimentação (Scree Plot)")
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.scatter(range(1, len(ev) + 1), ev, color='#E74C3C', zorder=3)
                ax.plot(range(1, len(ev) + 1), ev, color='#34495E', linestyle='--')
                ax.axhline(y=1, color='gray', linestyle=':')
                ax.set_title("Scree Plot")
                ax.set_xlabel("Fatores")
                ax.set_ylabel("Autovalores")
                st.pyplot(fig)
                plt.close()

            # --- TAB 3: MATRIZ DE CARGAS & HEATMAP ---
            with tab_fa3:
                st.subheader(f"Cargas Fatoriais Rotacionadas ({metodo_rotacao})")
                try:
                    A = df_fa.values - np.mean(df_fa.values, axis=0)
                    _, _, Vht = np.linalg.svd(A, full_matrices=False)
                    cargas_iniciais = Vht[:n_fatores].T * np.sqrt(ev[:n_fatores])
                    
                    if n_fatores > 1:
                        if "Promax" in metodo_rotacao:
                            cargas_rotacionadas = promax_rotation(cargas_iniciais)
                        else:
                            cargas_rotacionadas, _ = varimax_rotation(cargas_iniciais)
                    else:
                        cargas_rotacionadas = cargas_iniciais
                        
                    colunas_fatores = [f"Fator {i+1}" for i in range(n_fatores)]
                    df_cargas = pd.DataFrame(cargas_rotacionadas, columns=colunas_fatores, index=reg_independent_cols)
                    
                    c1, c2 = st.columns([1, 1])
                    with c1:
                        st.dataframe(df_cargas.style.format("{:.3f}").background_gradient(cmap="bwr", vmin=-1, vmax=1), use_container_width=True)
                    with c2:
                        fig, ax = plt.subplots(figsize=(5, 4))
                        sns.heatmap(df_cargas, annot=True, cmap="bwr", center=0, fmt=".2f", vmin=-1, vmax=1, ax=ax)
                        ax.set_title("Heatmap das Cargas Fatoriais")
                        st.pyplot(fig)
                        plt.close()
                        
                    st.markdown("#### 💡 Sugestão de Nomeação Conceitual dos Fatores")
                    for fat in colunas_fatores:
                        top_vars = df_cargas[fat].abs().nlargest(3).index.tolist()
                        st.markdown(f"- **{fat}:** Poderia ser rotulado como reflexo de **{', '.join(top_vars)}** (maiores forças de carga).")
                        
                except Exception as e:
                    st.error(f"Erro ao calcular cargas fatoriais: {e}")

            # --- TAB 4: COMUNALIDADES & CONFIABILIDADE ---
            with tab_fa4:
                st.subheader("Comunalidades ($h^2$) das Variáveis")
                try:
                    comunalidades = np.sum(cargas_rotacionadas**2, axis=1)
                    df_comun = pd.DataFrame({'Variável': reg_independent_cols, 'Comunalidade': comunalidades}).set_index('Variável')
                    
                    styler_comun = df_comun.style
                    try:
                        styler_comun = styler_comun.map(lambda x: 'background-color: #ffcccc' if x < 0.5 else '', subset=['Comunalidade'])
                    except AttributeError:
                        styler_comun = styler_comun.applymap(lambda x: 'background-color: #ffcccc' if x < 0.5 else '', subset=['Comunalidade'])
                    
                    st.dataframe(styler_comun.format({'Comunalidade': '{:.3f}'}), use_container_width=True)
                    st.caption("Valores destacados em vermelho (< 0.50) explicam pouca variância e deveriam ser removidos.")
                    
                    st.markdown("---")
                    st.subheader("Confiabilidade da Escala (Alfa de Cronbach por Fator)")
                    for i, fat in enumerate(colunas_fatores):
                        dominantes = df_cargas.index[df_cargas[fat].abs() >= 0.40].tolist()
                        if len(dominantes) >= 2:
                            alfa_fator = calcular_cronbach(df_fa[dominantes])
                            st.markdown(f"- **{fat}:** $\\alpha$ = **{alfa_fator:.3f}** (Baseado em: {', '.join(dominantes)})")
                        else:
                            st.markdown(f"- **{fat}:** Não há itens suficientes carregando fortemente para calcular a consistência interna.")
                except Exception as e:
                    st.error(f"Erro no processamento das comunalidades/Cronbach: {e}")

            # --- TAB 5: ESCORES FATORIAIS ---
            with tab_fa5:
                st.subheader("Geração e Download dos Escores Fatoriais")
                st.markdown("Use esta aba para converter os dados originais das colunas em novos fatores resumidos (Escores) salvos em uma nova planilha.")
                
                try:
                    R_inv = np.linalg.pinv(corr_matrix_np)
                    pesos_escores = np.dot(R_inv, cargas_rotacionadas)
                    
                    df_fa_std = (df_fa - df_fa.mean()) / df_fa.std()
                    escores_np = np.dot(df_fa_std.values, pesos_escores)
                    
                    df_escores = pd.DataFrame(escores_np, columns=colunas_fatores, index=df_fa.index)
                    df_completo_com_fatores = df.copy()
                    for fat in colunas_fatores:
                        df_completo_com_fatores[fat] = df_escores[fat]
                        
                    st.markdown("#### Pré-visualização dos Novos Fatores Gerados por Registro:")
                    st.dataframe(df_escores.head(10).style.format("{:.4f}"), use_container_width=True)
                    
                    csv_export = df_completo_com_fatores.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Baixar Base de Dados Atualizada com os Fatores (CSV)",
                        data=csv_export,
                        file_name="base_com_escores_fatoriais.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Erro ao computar os escores fatoriais: {e}")

            # --- EXPORTAÇÃO DA ANÁLISE FATORIAL ---
            st.markdown("---")
            st.subheader("📥 Exportar Relatório da Análise Fatorial")
            col_exp1, col_exp2 = st.columns(2)
            
            with col_exp1:
                if 'df_cargas' in locals():
                    csv_cargas = df_cargas.to_csv().encode('utf-8')
                    st.download_button(
                        label="📄 Baixar Cargas Fatoriais (CSV)",
                        data=csv_cargas,
                        file_name="cargas_fatoriais.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
            with col_exp2:
                if PDF_AVAILABLE and 'df_cargas' in locals() and 'df_var_exp' in locals():
                    kmo_str = f"KMO: {kmo_model:.3f} | Bartlett p-val: {format_p_value(p_value_bartlett)}" if FA_AVAILABLE else "Não disponível"
                    secoes_pdf_fa = [
                        ("1. Testes de Adequabilidade Amostral", kmo_str),
                        ("2. Variância Total Explicada", df_var_exp),
                        ("3. Matriz de Cargas Fatoriais Rotacionadas", df_cargas),
                        ("4. Comunalidades das Variáveis", df_comun)
                    ]
                    pdf_bytes_fa = gerar_pdf_relatorio("Relatório Científico: Análise Fatorial Exploratória", secoes_pdf_fa)
                    st.download_button(
                        label="📕 Baixar Relatório Completo (PDF)",
                        data=pdf_bytes_fa,
                        file_name="relatorio_analise_fatorial.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                else:
                    st.info("Para gerar o PDF, instale a biblioteca reportlab (`pip install reportlab`).")
