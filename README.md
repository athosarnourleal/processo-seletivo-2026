# RAG MULTI AGENT PROJECT

um sistema multi agente em python criado com base em persistent client Chromadb, streamlit e langchain para o processo seletivo do grupo LAPES da faculdade CESUPA

- autor: Athor Arnour Leal
- trilha escolhida: dados/IA
- email: athosarnourleal@gmail.com
- telefone: (91)98512-4864

## RETRIEVAL PIPELINE
```
app.py          ./interface/app.py
    |
    | 
    v
runRetrievalPipeline()          ./run/retrievalPipeline.py
    |
    |
    v
QueryValidationAgent.validateAndReformulateQuery()          ./agents/QueryValidationAgent.py
    Returns:  "DENIED" -> FINISH with explanation for denial
              "DONE" -> FINISH returning basic message responding query
              "PROCEED" -> reformulate query 
    |
    | ("PROCEED")
    v
VectorStoreManager.searchQuery()          ./database/VectorStoreManagement.py
    |
    |
    v
MainAgent.answerQuery()          ./agents/MainAgent.py
    |
    |
    v
AnswerValidationAgent.validateAnswer()          ./agents/QueryValidationAgent.py
    Returns:  True -> finish return explanation
              "invalid" -> finish returning reason for denial
```

### DIAGRAMA GRÁFICO

<img src="/retrieval_pipeline.png" width="75%">

## AGENTES BASEADOS EM LLM
### #QueryValidationAgent:
recebe uma query principal e retorna um de três tipos de resposta:

- **PROCEED**: continua a pipeline e reformula a query(correção de erros, clarificar ambiguidades e adaptar formato para maior facilidade de pesquisa)
- **DENIED**: termina a execução e retorna uma explicação do por que foi rejeitado(query contém linguagem ofensiva, query não é legivel ou query está fora do escopo)
- **DONE**: finaliza a execução em casos onde a query contem apenas mensagens basicas como saudações ou desculpas, retornando uma resposta sem precisar acionar o resto da pipeline

### #MainAgent:
recebe o **contexto**(chunks resultados da busca) e a **query reformulada** e, baseado neles, executa as funçoes de:
- acionar a pesquisa web como forma de fallback
- gerar a resposta(baseado no contexto ou na pesquisa web)

### #AnswerValidationAgent:
recebe a **query** e a **resposta** gerada pelo Main Agent e então classifica a saida como "valid" ou "invalid", retornando ou True ou o a explicação do por que foi invalidado

requisitos para uma resposta válida:
- a resposta responde completamente a pergunta
- a resposta apresenta as fontes usadas na sua construção

## FERRAMENTAS

### • TavilySearch(langchain tool)
realiza a pesquisa na web com base na query e na classificação

### • VectorStoreManager
Funções principais:
- conectar com o Chroma Client
- cria/conectar a coleção
- fazer embedding/indexação dos chunks e adicioná-los ao database
- fazer embedding da query e realizar pesquisa via cosine similarity

## REQUISITOS BÁSICOS
- conta no google ai studio
- conta na openai
- conta no tavily

## SETUP DO AMBIENTE
- preencher o **.env** no mesmo formato do **.env_example**
``` env
# ai agents
GOOGLE_API_KEY='insira sua key aqui'

# vectorstore embeddings
OPENAI_API_KEY='insira sua key aqui'

# web search
TAVILY_API_KEY='insira sua key aqui'
```
- crie o venv e baixar os requirements nele
``` bash
# criar e acessar venv
python3 -m venv venv
source venv/bin/activate

# baixar requirements
pip install -r ./requirements.txt
```
## rodando codigos
``` bash
# run ingestion pipeline
python3 ./database/ingestion_pipeline.py

# run streamlit interface
streamlit run ./interface/app.py
```

## DECISOES TECNICAS E SEUS TRADE-OFFS

### Agente Próprio ao inves de agentes langchain
durante o desenvolvimento diversos erros foram causados pelos novos funcionamentos introduzidos no langchain v1.0, isso, esse fator, aliado com a escasses de materiais de ensino atualizados, tornando a opção de criar a classe BasicGoogleGenAIAgent a mais viável, principalmente devido a maior liberdade e controle na pipeline de execução da IA(ainda assim usei o langchain.tools e langchain.messages principalmente por conveniencia)

Trade-offs: ter que trabalhar diretamente com o google acabou exigindo um maior estudo sobre o funcionamento das IA's, o que acabou trazendo atrito principalmente na integração de tools(especialmente o Tavily Search) e, assim, atrasando o desenvolvimento

### Ingestão de pdfs que não inclui as imagens
durante o projeto implementei a leitura de pdf que incluía imagens, entretanto o alto custo em tokens utilizados por todos os métodos que achei acabou dificultando muito os testes, logo optei por uma extração simples de pdf do pypdf e removi a função de leitura de imagens

Trade-offs: acabei limitando a quantidade de pdfs que funcionariam devidamente na ingestão

## justificativa para o Corpus

### Por que esse corpus e o que ele representa como problema real?
escolhi principalmente por entender a dificuldade que muitos alunos têm em relação às matérias do ENEM, principalmente nas áreas de ciencias exatas, situação essa que não se limita apenas à um caso isolado, mas sim é um fenômeno frequente na sociedade atual

### Como a combinação RAG e busca web resolve algo que nenhum dos dois sozinho resolve bem nesse domínio?
o sistem RAG e a pesquisa web são um contunto que traz uma ótima complementaridade, o sistema RAG(quando apresentado com documentações certificadas) reduz os problemas relacionados à fontes não confiaveis que surgem quando se depende apenas da pesquisa web e a busca web também complementa o conjunto ao eliminar o principal problema de sistemas RAG de baixa escala como o do projeto: a quantidade limitada de dados, a qual pode levar em muitos casos à lacunas nas respostas.

### Onde o sistema vai falhar e por que isso é esperado dado o corpus escolhido?
em alguns casos algumas perguntas que poderiam ser resolvidas com apenas o documento acabam ativando uma pesquisa web mesmo assim, isso ocorre pois muitas vezes subtópicos de uma matéria acabam por ser desconectados do título dela durante o processo de chunking, isso é um problema que poderia ser corridigo pelo aumento do tamanho dos chunks ou dos overlaps entre chunks, entretanto decidi não aplicar esses métodos de correção para não aumentar o número de tokens de cada run do sistema.

### Por que esse corpus permite exercitar todas as partes obrigatórias do desafio, incluindo o fallback?
a imensa disponibilidade de documentos de livre uso relacionados aos assuntos do ENEM facilita muito a seleção dos documentos do corpus, além disso essa diponibilidade também aumenta as chances da busca web achar resultados melhores

## CUSTO POR EXECUÇÃO
**ingestion pipeline**
- custo em tokens: ???? tokens
- custo monetário: US$ 0.03
**retrieval pipeline(custo estimado)**
- média de tokens gastos ~= 36304.4 tokens
- tokens do tavily(quando ativado) ~= 2 créditos
