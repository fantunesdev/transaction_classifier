# 🧠 Transaction Classifier

<p align="left">
  <img src="https://img.shields.io/badge/Python-ED8B00?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/River-0A9396?style=for-the-badge" alt="River">
  <img src="https://img.shields.io/badge/Poetry-60A5FA?style=for-the-badge&logo=python&logoColor=white" alt="Poetry">
  <img src="https://img.shields.io/badge/Uvicorn-090909?style=for-the-badge&logo=uvicorn&logoColor=white" alt="Uvicorn">
  <img src="https://img.shields.io/badge/Requests-20232A?style=for-the-badge&logo=python&logoColor=white" alt="Requests">
</p>

Um microserviço de machine learning criado com **FastAPI** e **River** que aprende com o histórico de lançamentos financeiros dos usuários e prevê **categoria** e **subcategoria** com base na descrição de um lançamento.

Este serviço é parte do ecossistema da aplicação [MyFinance](https://github.com/fantunesdev/myfinance), sendo responsável por **classificar automaticamente os lançamentos financeiros** de forma personalizada para cada usuário.

---

## 🚀 Funcionalidades

- 🔍 **Previsão de categoria e subcategoria** com base na descrição (e opcionalmente na categoria).
- 📈 **Treinamento do modelo** com base nas subcategorias e no histórico real de lançamentos.
- 🔁 Preparado para receber **feedback** e permitir aprendizado contínuo (em breve).
- 🔐 Integração com autenticação JWT da aplicação Django.
- 🔌 API leve e rápida com FastAPI.

---

## 🧪 Exemplo de uso

### 🔧 Treinar o modelo

#### Envio:
```http
POST /train/{user_id}
Authorization: Bearer <jwt_token>
```

#### Resposta:
```http
{
  "success": true,
  "message": "Modelo do usuário 1 treinado com sucesso! Número de subcategorias: 57. Número de lançamentos: 3423."
}
```

### 🧠 Fazer uma previsão

#### Envio:
```http
POST /predict/{user_id}
Authorization: Bearer <jwt_token>
Content-Type: application/json

{
  "description": "Supermercado Condor"
}
```

#### Resposta:
```http
{
  "category_id": 2,
  "subcategory_id": 31
}

```

## Instalação local

#### Clone o repositório
> git clone https://github.com/seu-usuario/transaction-classifier.git
> cd transaction-classifier


#### Crie o ambiente
> poetry install --no-root


#### Inicie o servidor
> uvicorn api.main:app --reload --host 0.0.0.0 --port 8001

## 🧰 Tecnologias utilizadas

<p align="left">
  <img src="https://img.shields.io/badge/Python-ED8B00?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/River-0A9396?style=for-the-badge" alt="River">
  <img src="https://img.shields.io/badge/Poetry-60A5FA?style=for-the-badge&logo=python&logoColor=white" alt="Poetry">
  <img src="https://img.shields.io/badge/Uvicorn-090909?style=for-the-badge&logo=uvicorn&logoColor=white" alt="Uvicorn">
  <img src="https://img.shields.io/badge/Requests-20232A?style=for-the-badge&logo=python&logoColor=white" alt="Requests">
</p>

## 🧠 Próximos passos
🚧 Implementação de feedback contínuo com correções do usuário  
🔄 Normalização automática de descrições  
📊 Métricas de acurácia personalizadas por usuário

## Autor
Fernando Antunes