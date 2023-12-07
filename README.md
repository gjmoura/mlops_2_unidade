# 🤖 Machine Learning Operations
Projeto da disciplina Projeto de Sistemas Baseados em Aprendizado de Máquina da UFRN, que consiste em desenvolver o projeto guiado da plataforma do Dataquest, Classifying Disaster-Related Tweets as Real or Fake, aplicando conhecimentos de boas práticas de programaçãoe em projetos de aprendizado de máquina, como debug, testes, logg, clean code, formatação de código e desenvolver uma pipeline para execução do projeto utilizando Airflow.

## 📒 Projeto
# Classifying Disaster-Related Tweets as Real or Fake
Esse é projeto se chama <i>Classifying Disaster-Related Tweets as Real or Fake</i> e constrói um modelo de classificação de texto de aprendizagem profunda para prever se tweets estão relacionados a desastres reais. Utilizando como base o projeto de protifólio disponível na plataforma [Dataquest](https://app.dataquest.io/) e um conjunto de dados obtido do Kaggle,o algoritmo passa por etapas que incluem a exploração e visualização dos dados, o pré-processamento textual, a segregação dos dados em conjuntos de treino e teste, o treinamento do modelo com TensorFlow e Hugging Face Transformers. Além disso, foi feita uma integração com o Weights and Biases para monitoramento e registro de experimentos. 

## Dependências
- wandb
- Python 3.8+
- numpy
- pandas
- pytest
- requests
- tensorflow
- seaborn
- transformers
- matplotlib
- scikit-learn
- nltk

Instale as dependência do projeto:
```
pip install -r requirements.txt
```


## Como executar
Primeiramente, você precisará criar uma conta no [Weight & Biases](https://wandb.ai/site). Em seguida, encontre sua ``API_KEY`` e coloque-a dentro do arquivo ``config.json``.

Garanta que você tem o python 3.10.x instalado. Ele será necessário para instalar o [airflow 2.7.1](https://airflow.apache.org/docs/apache-airflow/stable/release_notes.html#airflow-2-7-1-2023-09-07), que é a versão mais recente até a data de implementação desta pipeline.

<strong>Para criar um ambiente virtual, siga os passos abaixo.</strong>
Obs.: Os comando a seguir devem ser executados a partir da pasta ``Project_2``.

Criar um ambiente virtual chamado airflow
```
python3.10 -m venv airflow
```

Ativar o ambiente:
```
source ./airflow/bin/activate
```

Para instalar o airflow:
```
AIRFLOW_VERSION=2.7.1
PYTHON_VERSION="$(python --version | cut -d " " -f 2 | cut -d "." -f 1-2)"
CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"
```

Depois que estiver instalado, você pode executar a aplicação:
```
airflow standalone
```

Após isso, será criado automaticamente a pasta ``~/airflow`` e a aplicação ficará disponível na porta ``http://localhost:8080``. No terminal será disponibilizado seu <strong>login e senha</strong>, e é de extrema importância que você os encontre para que possa entrar na aplicação.

Agora será preciso fazer uma auteração nas configurações do airflow.

Entre na pasta ``dags/`` e obtenha e copie para área de transferência o caminho até este diretótio:
```
cd dags/
pwd
```

Depois de copiar o caminho da pasta, entre no arquivo de configurações do airflow. Para obter o caminho até o esse arquivo:
```
find ~/ -name "airflow.cfg"
```

Entre no arquivo com um editor de texto do próprio terminal:
```
nano {path_of_the_airflow.cfg_file}
```

Edite a variável ``dags_folder`` para que aponte para a sua pasta dags, cujo caminho você já possui na área de transferência. No meu caso ficou assim:
```
dags_folder = /home/augusto/Downloads/mlops2023/dags/tweets_classifying.py
```

Agora, pare a execução do airflow e a execute novamente para aplicar as mudanças feitas:
```
airflow standalone
```

Na página inicial, em que há uma lista com as DAGs, procure pela opção "tweets_classifying" e despause essa DAG clicando no toggle ao lado de seu título. Em seguida, clique no título e você obterá mais informações sobre ela, como grafos e logs sobre cada task da pipeline. Para executar a pipeline, basta clicar no ícone de "play" na parte superior direita.


## ℹ Mais informações

Alunos:
- Adson Emanuel Santos Amaral
- Gustavo Jerônimo Moura de França
- José Augusto Agripino de Oliveira
