# Setup e Execução do Projeto

Este guia explica como criar e ativar um ambiente virtual Python, instalar dependências e executar o script Python responsável por 
informar melhor local de compra de produtos de acordo com histórico.

## 1. Criar um Ambiente Virtual (venv)

Antes de instalar as dependências, é recomendado criar um ambiente virtual para isolar os pacotes do projeto.

Execute o seguinte comando no terminal:

```sh
python -m venv venv
```

Isso criará um diretório chamado `venv` no seu projeto, contendo o ambiente virtual.

---

## 2. Ativar o Ambiente Virtual

### **Windows (CMD ou PowerShell)**
```sh
venv\Scripts\activate
```

### **Mac/Linux (Terminal)**
```sh
source venv/bin/activate
```

Após a ativação, você verá o nome do ambiente virtual no início do prompt do terminal, indicando que está ativo.

---

## 3. Instalar Dependências

Com o ambiente virtual ativado, instale as dependências do projeto com:

```sh
pip install -r requirements.txt
```

Isso instalará todas as bibliotecas listadas no arquivo `requirements.txt`.

---

## 4. Executar o Script Python

Para executar o script Python, use:

```sh
python best-purchase-location.py
```

---

5. Dataset de Entrada e Saída

O script utiliza o dataset **product_history.csv** para gerar a tabela de saída **purchase_recommendations.xlsx**, que contém as recomendações geradas.

Certifique-se de que o arquivo **product_history.csv** está presente no diretório do projeto antes de executar o script.


