# RAG MULTI AGENT PROJECT

um sistema multi-agente em python criado com base em persistent client Chromadb(com embeddings OpenAi), streamlit e googleGenAi para o processo seletivo do grupo LAPES da faculdade CESUPA

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
AnswerValidationAgent.validateAnswer()          ./agents/AnswerValidationAgent.py
    Returns:  True -> finish return explanation
              "invalid" -> finish returning reason for denial

    |
    |  
    v
app.py          ./interface/app.py
```

### DIAGRAMA GRÁFICO

<img src="/retrieval_pipeline.png" width="100%">

## AGENTES BASEADOS EM LLM
### # QueryValidationAgent:
recebe uma query principal e retorna um de três tipos de resposta:

- **PROCEED**: continua a pipeline e reformula a query(correção de erros, clarificar ambiguidades e adaptar formato para maior facilidade de pesquisa)
- **DENIED**: termina a execução e retorna uma explicação do por que foi rejeitado(query contém linguagem ofensiva, query não é legivel ou query está fora do escopo)
- **DONE**: termina a execução em casos onde a query contem apenas mensagens basicas como saudações ou desculpas, retornando uma resposta sem precisar acionar o resto da pipeline

a resposta é concedida no formato de um **json** com os campos: status(`DONE, DENIED ou PROCEED`) e message(query reformulada quando `"PROCEED"`, mensagem respondendo a query quando `"DONE"` ou justificativa do por quê foi a query foi barrada quando `"DENIED"`), segue abaixo um exemplo:
``` json
{
    "status": "DONE",
    "message": "bom dia!"
}
```

### # MainAgent:
recebe o **contexto**(chunks resultados da busca) e a **query reformulada** e, baseado neles, executa as funçoes de:
- acionar a pesquisa web como forma de fallback
- gerar a resposta(baseado no contexto ou na pesquisa web)
- citar as fontes usadas na geração da resposta

### # AnswerValidationAgent:
recebe a **query** e a **resposta** gerada pelo Main Agent e então classifica a saida como `"valid"` ou `"invalid"`, retornando:
- `True` (quando `"valid"`)
- pedido de desculpas para o usuario explicando do por que foi invalidado(quando `"invalid"`)

requisitos para uma resposta `"valid"`:
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
- conta no `google ai studio`
- conta na `openai`
- conta no `tavily`

## SETUP DO AMBIENTE
- realizar o git clone do projeto e entrar na pasta principal
``` bash
git clone https://github.com/athosarnourleal/processo-seletivo-2026.git
cd processo-seletivo-2026
```
- renomear o **.env_example** para **.env** e preencher com as chaves
``` env
# ai agents
GOOGLE_API_KEY='insira sua key aqui'

# vectorstore embeddings
OPENAI_API_KEY='insira sua key aqui'

# web search
TAVILY_API_KEY='insira sua key aqui'
```
- criar o virtual environment e baixar importar packages do `requirements.txt` nele
``` bash
# criar e acessar venv
python3 -m venv venv
source venv/bin/activate
```
```bash
# baixar requirements
pip install -r ./requirements.txt
```
## RODANDO CÓDIGO
``` bash
# rodar ingestion pipeline
python3 ./database/ingestion_pipeline.py
```
```bash
# rodar interface streamlit
streamlit run ./interface/app.py
```
## BENCHMARKS E AVALIAÇÃO
as métrica de avaliação que usei para validar(manualmente) as respostas do sistema foram:
- query deve ser respondida **completamente** pela resposta gerada
- resposta deve ser fiel à documentação retornada da pesquisa dos documentos ou da pesquisa web
- resposta deve citar as fontes usadas na sua construção(fontes essas que devem estar presentes nos documentos ou na pesquisa web)
- documentos/web search resgatados devem, quando possível, ser correlacionados à query

como o sistema RAG que produzi possui uma pipeline mais simplificada, essas métricas, que focam diretamente nas partes mais cruciais do processo, foram as que fizeram mais sentido

diversos exemplos de perguntas(tanto de validação, quanto de teste) e suas respostas estão presentes em: `./docs/question_examples.txt`

## DECISÕES TÉCNICAS E SEUS TRADE-OFFS

### Agente Próprio ao inves de agentes baseados em frameworks
durante o desenvolvimento, decidi me voltar ao framework do langchain pois ele aparentava ser mais acessível, entretanto diversos erros foram causados pelos novos funcionamentos introduzidos no framework langchain v1.0, esse fator, aliado com a ausência de materiais de ensino atualizados, tornou a opção de criar a classe BasicGoogleGenAIAgent(com apenas algumas ferramentas de langchain) mais viável do que reaprender outro framework do zero. Além disso, criar um agente sem framework proporcionou maior liberdade e aprendizado acerca da execução das LLMs.

Trade-offs: ter que trabalhar diretamente com o google genai acabou exigindo um maior estudo sobre o funcionamento das IA's, o que acabou trazendo atrito principalmente na integração de tools(principalmente Tavily Search) e isso levou a atrasos no desnenvolvimento.

### Ingestão de pdfs que não inclui as imagens
durante o projeto tentei implementar a leitura de pdf que incluía imagens, entretanto o alto custo em tokens utilizados por todos os métodos que tentei acabou dificultando muito os testes, logo optei por uma extração somente de texto baseada em pypdf.

Trade-offs: essa situação limitando a quantidade de pdfs que funcionariam devidamente na ingestão.

## JUSTIFICATIVA PARA O CORPUS
tema do corpus: química, física e biologia par o Enem

### Por que esse corpus e o que ele representa como problema real?
escolhi principalmente por entender a dificuldade que muitos alunos têm em relação às matérias do ENEM, principalmente nas áreas de ciencias exatas. Situação essa que, além de frequente, é um obstaculo muito grande nos estudos do aluno, o que faz com que ferramentas que auxiliem esse processo sejam extremamente valirizadas.

### Como a combinação RAG e busca web resolve algo que nenhum dos dois sozinho resolve bem nesse domínio?
o sistem RAG e a pesquisa web são um contunto que traz uma ótima complementaridade, o sistema RAG(quando apresentado com documentações certificadas) reduz os problemas relacionados à fontes não confiaveis que surgem quando se depende apenas da pesquisa web e a busca web também complementa o conjunto ao eliminar o principal problema de sistemas RAG de baixa escala como o do projeto: a quantidade limitada de dados, a qual pode levar em muitos casos à lacunas nas respostas.

### Onde o sistema vai falhar e por que isso é esperado dado o corpus escolhido?
em alguns casos algumas perguntas que poderiam ser resolvidas com apenas o documento acabam ativando uma pesquisa web mesmo assim, isso ocorre pois muitas vezes subtópicos de uma matéria acabam por ser desconectados do título dela durante o processo de chunking, isso é um problema que poderia ser corridigo pelo aumento do tamanho dos chunks ou dos overlaps entre chunks, entretanto decidi não aplicar esses métodos de correção para não aumentar o número de tokens de cada run do sistema.

### Por que esse corpus permite exercitar todas as partes obrigatórias do desafio, incluindo o fallback?
a imensa disponibilidade de documentos de livre uso relacionados aos assuntos do ENEM facilita muito a seleção dos documentos do corpus, além disso essa diponibilidade e acessibilidade também aumenta as chances da busca web achar resultados melhores.

## CUSTO POR EXECUÇÃO
**ingestion pipeline**
- custo em tokens(OpenAI): 262178 tokens / ingestion
  
**retrieval pipeline(custo estimado)**
- média de tokens gastos ~= 36304,4 tokens / retrieval pipeline
- tokens do tavily(quando ativado) = 2 créditos / ativação
