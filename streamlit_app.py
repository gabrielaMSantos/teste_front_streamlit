"""
Streamlit App - Frontend para API de Machine Learning.

Aplicativo interativo que fornece uma interface gráfica para interagir com a API FastAPI,
permitindo treinar modelos LSTM, consultar status de treinamento e fazer previsões de
preços de ações.

Classes:
    Nenhuma

Funções:
    consultar_status_treinamento: Consulta o status de um job de treinamento.
    iniciar_treinamento: Inicia um novo treinamento de modelo via API.
    fazer_previsao: Realiza uma previsão usando o modelo treinado.
    validar_precos_manuais: Valida e converte string de preços em lista de floats.
    renderizar_status_treinamento: Renderiza a interface de status de treinamento.
    renderizar_secao_treinamento: Renderiza a interface de treinamento.
    renderizar_secao_previsao: Renderiza a interface de previsão.
    main: Função principal que orquestra a aplicação.

Módulos dependentes:
    streamlit: Framework para aplicações web interativas.
    requests: Biblioteca para requisições HTTP.
"""

# =============================
# IMPORTS
# =============================
from typing import Dict, Any, List, Optional

import requests
import streamlit as st

# =============================
# CONSTANTES
# =============================
import os

# Detectar ambiente: desenvolvimento (localhost) ou produção (via Nginx proxy)
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# Permite sobrescrever o destino da API via variável (ex.: apontar para API hospedada)
API_BASE_URL = os.getenv("API_BASE_URL") or "https://fase-1-hkv8.onrender.com/api" 

TRAIN_ENDPOINT = f"{API_BASE_URL}/train"
STATUS_ENDPOINT = f"{API_BASE_URL}/train/status"
PREDICT_ENDPOINT = f"{API_BASE_URL}/predict"

# Configuração padrão da página
PAGE_TITLE = "ML Stock Prediction App"
PAGE_LAYOUT = "centered"

# Mapeamento de status com tipos
STATUS_MAP = {
    "pending": ("", "Treinamento em fila", "warning"),
    "running": ("", "Treinamento em andamento", "info"),
    "completed": ("", "Treinamento finalizado", "success"),
    "failed": ("", "Treinamento falhou", "error")
}

# Configurações da interface
NUM_PRECOS_OBRIGATORIOS = 60


# =============================
# FUNÇÕES AUXILIARES
# =============================

def consultar_status_treinamento(job_id: str) -> Optional[Dict[str, Any]]:
    """
    Consulta o status de um job de treinamento na API.

    Realiza uma requisição GET ao endpoint de status da API para obter
    informações sobre um treinamento em andamento.

    Args:
        job_id (str): ID único do job de treinamento.

    Returns:
        Optional[Dict[str, Any]]: Dicionário com dados do status ou None se houver erro.

    Raises:
        Exibe mensagem de erro via Streamlit se a conexão falhar.
    """
    try:
        url = f"{STATUS_ENDPOINT}/{job_id}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
        st.warning(f"Status HTTP {response.status_code} ao acessar {url}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Erro ao conectar com a API: {e}\nURL: {STATUS_ENDPOINT}/{job_id}")
        return None


def iniciar_treinamento(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Inicia um novo treinamento de modelo via API.

    Envia uma requisição POST com os parâmetros de treinamento para a API
    e retorna a resposta contendo o Job ID do treinamento.

    Args:
        payload (Dict[str, Any]): Dicionário com parâmetros de treinamento:
            - symbol (str): Símbolo da ação
            - start_date (str): Data inicial (formato YYYY-MM-DD)
            - end_date (str): Data final (formato YYYY-MM-DD)
            - epochs (int): Número de épocas
            - batch_size (int): Tamanho do lote
            - learning_rate (float): Taxa de aprendizado

    Returns:
        Optional[Dict[str, Any]]: Resposta da API ou None se houver erro.

    Raises:
        Exibe mensagem de erro via Streamlit se a conexão falhar.
    """
    try:
        response = requests.post(TRAIN_ENDPOINT, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()
        st.warning(f"Status HTTP {response.status_code} ao conectar com {TRAIN_ENDPOINT}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Erro ao conectar com a API: {e}\nURL: {TRAIN_ENDPOINT}")
        return None


def fazer_previsao(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Realiza uma previsão usando o modelo treinado.

    Envia uma requisição POST com dados históricos de preços ou símbolo
    para obter a previsão do próximo preço de fechamento.

    Args:
        payload (Dict[str, Any]): Dicionário com dados de previsão:
            - symbol (str): Símbolo da ação (opcional)
            - start_date (str): Data inicial (opcional, formato YYYY-MM-DD)
            - end_date (str): Data final (opcional, formato YYYY-MM-DD)
            - last_60_days_prices (List[float]): Lista de 60 preços (alternativa)

    Returns:
        Optional[Dict[str, Any]]: Resultado da previsão ou None se houver erro.

    Raises:
        Exibe mensagem de erro via Streamlit se a conexão falhar.
    """
    try:
        response = requests.post(PREDICT_ENDPOINT, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()
        st.warning(f"Status HTTP {response.status_code} ao conectar com {PREDICT_ENDPOINT}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Erro ao conectar com a API: {e}\nURL: {PREDICT_ENDPOINT}")
        return None


def validar_precos_manuais(prices_text: str) -> Optional[List[float]]:
    """
    Valida e converte string de preços em lista de floats.

    Verifica se a quantidade de preços é exatamente 60 (obrigatório para LSTM)
    e se todos os valores são numéricos válidos.

    Args:
        prices_text (str): String com preços separados por vírgula.

    Returns:
        Optional[List[float]]: Lista de preços validados ou None se inválido.

    Raises:
        Exibe mensagens de erro via Streamlit se a validação falhar.
    """
    try:
        prices = [float(p.strip()) for p in prices_text.split(",")]
        if len(prices) != NUM_PRECOS_OBRIGATORIOS:
            st.error(
                f"É necessário informar exatamente {NUM_PRECOS_OBRIGATORIOS} valores. "
                f"Você forneceu {len(prices)}."
            )
            return None
        return prices
    except ValueError:
        st.error("Todos os valores devem ser numéricos.")
        return None


def renderizar_status_treinamento() -> None:
    """
    Renderiza a seção de consulta de status de treinamento.

    Exibe um formulário para o usuário informar um Job ID e consultar
    o status atual do treinamento associado, mostrando resultado visual
    com emojis e cores.
    """
    st.divider()
    st.header("Status do Treinamento")

    job_id_input = st.text_input(
        "Job ID do treinamento",
        value=st.session_state.get("job_id", ""),
        help="Informe o Job ID recebido ao iniciar o treinamento"
    )

    if st.button("Consultar status", key="btn_status"):
        if not job_id_input:
            st.warning("Informe um Job ID para consultar o status.")
        else:
            with st.spinner("Consultando status..."):
                data = consultar_status_treinamento(job_id_input)

            if data:
                status_value = data.get("status")

                # Renderiza status com feedback visual
                if status_value in STATUS_MAP:
                    _, msg, tipo = STATUS_MAP[status_value]
                    if tipo == "warning":
                        st.warning(f"{msg}")
                    elif tipo == "info":
                        st.info(f"{msg}")
                    elif tipo == "success":
                        st.success(f"{msg}")
                    elif tipo == "error":
                        st.error(f"{msg}")
                else:
                    st.write("Status desconhecido")

                st.json(data)
            else:
                st.error("Erro ao consultar status")


def renderizar_secao_treinamento() -> None:
    """
    Renderiza a seção de treinamento de modelo.

    Exibe formulário com campos para configuração de treinamento:
    - Símbolo da ação
    - Datas de início e fim
    - Hiperparâmetros (epochs, batch_size, learning_rate)

    Ao clicar em treinar, envia requisição à API e salva o Job ID
    na sessão para consultas posteriores.
    """
    st.divider()
    st.header("Treinamento de Modelo")
    st.subheader("Parâmetros do Treinamento")

    symbol = st.text_input(
        "Símbolo do ativo",
        value="AAPL",
        help="Ex: AAPL, GOOGL, MSFT, TSLA"
    )

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Data inicial", help="Data de início para coleta de dados históricos")
    with col2:
        end_date = st.date_input("Data final", help="Data final para coleta de dados históricos")

    col3, col4 = st.columns(2)
    with col3:
        epochs = st.number_input(
            "Epochs",
            min_value=1,
            value=50,
            help="Número de iterações de treinamento"
        )
    with col4:
        batch_size = st.number_input(
            "Batch size",
            min_value=1,
            value=64,
            help="Tamanho do lote de treinamento"
        )

    learning_rate = st.number_input(
        "Learning rate",
        min_value=0.000001,
        value=0.001,
        format="%.6f",
        help="Taxa de aprendizado do otimizador"
    )

    if st.button("Treinar modelo", key="train_model"):
        payload = {
            "symbol": symbol,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate
        }

        with st.expander("Payload enviado"):
            st.json(payload)

        with st.spinner("Iniciando treinamento..."):
            data = iniciar_treinamento(payload)

        if data:
            st.success("Treinamento iniciado em background!")
            st.json(data)

            if "job_id" in data:
                st.session_state["job_id"] = data["job_id"]
                st.info(f"Job ID salvo: `{data['job_id']}`")
        else:
            st.error("Erro ao iniciar treinamento")


def renderizar_secao_previsao() -> None:
    """
    Renderiza a seção de previsão de preços.

    Oferece dois modos de previsão:
    1. Automática: usuario fornece símbolo, datas (opcionais)
    2. Manual: usuário fornece exatamente 60 preços históricos

    Realiza validação dos dados e exibe resultado com destaque
    para o preço previsto e informações adicionais.
    """
    st.divider()
    st.header("Fazer Previsão")
    st.markdown("Escolha como deseja fornecer os dados para a previsão:")

    opcao = st.radio(
        "Modo de previsão",
        [
            "Busca automática (símbolo)",
            "Dados manuais (últimos 60 preços)"
        ],
        help="Busca automática usa a API do Yahoo Finance. "
             "Dados manuais requerem exatamente 60 valores."
    )

    payload = None

    # Modo 1: Busca automática
    if opcao == "Busca automática (símbolo)":
        symbol_pred = st.text_input(
            "Símbolo da ação",
            value="AAPL",
            key="symbol_predict"
        )

        col_start, col_end = st.columns(2)
        with col_start:
            start_date_pred = st.date_input(
                "Data inicial (opcional)",
                value=None,
                key="start_pred"
            )
        with col_end:
            end_date_pred = st.date_input(
                "Data final (opcional)",
                value=None,
                key="end_pred"
            )

        payload = {"symbol": symbol_pred}

        if start_date_pred:
            payload["start_date"] = start_date_pred.strftime("%Y-%m-%d")
        if end_date_pred:
            payload["end_date"] = end_date_pred.strftime("%Y-%m-%d")

    # Modo 2: Dados manuais
    else:
        st.markdown(
            f"Informe **exatamente {NUM_PRECOS_OBRIGATORIOS} valores**, "
            "separados por vírgula."
        )

        prices_text = st.text_area(
            "Últimos 60 preços de fechamento",
            placeholder="150.1, 151.0, 152.3, 153.5, ...",
            help="Cole os últimos 60 preços de fechamento separados por vírgula"
        )

        if prices_text:
            prices = validar_precos_manuais(prices_text)
            if prices:
                payload = {"last_60_days_prices": prices}
                st.success(f"{len(prices)} preços validados com sucesso!")

    # Botão de previsão
    if st.button("Fazer previsão", key="btn_predict"):
        if payload is None:
            st.warning("Preencha corretamente os dados para previsão.")
        else:
            with st.spinner("Calculando previsão..."):
                result = fazer_previsao(payload)

            if result:
                st.success("Previsão realizada com sucesso!")

                # Exibe métricas de forma destacada
                if "predicted_price" in result:
                    col_metric1, col_metric2 = st.columns(2)
                    with col_metric1:
                        st.metric(
                            "Preço previsto",
                            f"${result['predicted_price']:.2f}"
                        )
                    with col_metric2:
                        if "symbol" in result:
                            st.metric("Ativo", result["symbol"])

                # Exibe JSON completo em expander
                with st.expander("Resposta completa da API"):
                    st.json(result)
            else:
                st.error("Erro ao realizar previsão")


def renderizar_rodape() -> None:
    """
    Renderiza o rodapé da aplicação com informações de crédito.
    """
    st.divider()
    st.markdown(
        """
        <div style='text-align: center; color: gray; font-size: 0.9em;'>
            <p>Desenvolvido para o Tech Challenge Fase 4 - FIAP</p>
            <p>Backend: FastAPI | Frontend: Streamlit | ML: PyTorch LSTM</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# =============================
# FUNÇÃO PRINCIPAL
# =============================

def main() -> None:
    """
    Função principal da aplicação Streamlit.

    Configura a página e orquestra a renderização de todas as seções:
    - Interface principal
    - Seção de status de treinamento
    - Seção de treinamento
    - Seção de previsão
    - Rodapé
    """
    # Configuração da página (deve ser a primeira chamada Streamlit)
    st.set_page_config(
        page_title=PAGE_TITLE,
        layout=PAGE_LAYOUT,
        initial_sidebar_state="auto"
    )

    # Interface principal
    st.title("Aplicação de Machine Learning")
    
    # Info sobre ambiente e API (Streamlit roda local; API pode estar remota)
    info_api = f"**API Backend:** `{API_BASE_URL}`"

    st.markdown(
        f"""
        Este app permite:
        - Treinar modelos LSTM para previsão de ações
        - Consultar status de treinamentos em andamento
        - Realizar previsões com modelos treinados

        {info_api}
        **Ambiente:** {ENVIRONMENT.upper()}
        """
    )

    # Seções da aplicação
    renderizar_status_treinamento()
    renderizar_secao_treinamento()
    renderizar_secao_previsao()
    renderizar_rodape()


# =============================
# EXECUÇÃO
# =============================

if __name__ == "__main__":
    main()