# POC: Análisis de Experiencia del Cliente mediante PLN

**Procesamiento de Lenguaje Natural — Universidad de La Sabana**

Este proyecto implementa una Prueba de Concepto (POC) que analiza automáticamente reseñas de clientes de Amazon usando técnicas de Procesamiento de Lenguaje Natural. El sistema clasifica sentimientos, descubre temas y explica las razones lingüísticas detrás de cada predicción.

---

## Dataset

- **Fuente**: Amazon Review Data (Web Scrapping) — ~60,000 reseñas reales
- **Columnas principales**:
  - `Review_text` — el texto del comentario (entrada)
  - `Own_Rating` — etiqueta de sentimiento: Positive, Negative o Neutral (salida)
- **Categorías de producto**: smartTv, mobile, books, refrigerator, mobile accessories
- **Distribución**: ~78% positivas, ~15% negativas, ~7% neutras

---

## ¿Cómo funciona el notebook paso a paso?

### Paso 1: Importar librerías y cargar datos

Se importan las herramientas necesarias y se carga el archivo CSV con las reseñas.

```python
import pandas as pd
import nltk, spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from gensim.models import LdaModel

df = pd.read_csv("Amazon Review Data Web Scrapping - Amazon Review Data Web Scrapping.csv")
```

Las librerías principales son:
- **pandas/numpy** — para manejar tablas de datos
- **NLTK** — para tokenizar, eliminar stopwords y lematizar
- **spaCy** — para análisis gramatical (POS tagging, NER, dependencias)
- **scikit-learn** — para convertir texto en números (TF-IDF) y entrenar clasificadores
- **gensim** — para descubrir temas automáticamente (LDA)

Se toma una **muestra estratificada de 10,000 reseñas** para que el POC corra eficientemente sin perder representatividad.

---

### Paso 2: Preprocesamiento de texto

Los comentarios vienen "sucios" — con mayúsculas, signos, links, etc. La clase `TextPreprocessor` los limpia paso a paso:

```python
class TextPreprocessor:
    def preprocess(self, text):
        text = self.basic_cleaning(text)    # Quitar HTML, URLs, emails
        text = self.normalize_case(text)     # Todo a minúsculas
        text = self.remove_punctuation(text) # Quitar signos: !?.,;
        text = self.remove_numbers(text)     # Quitar números
        tokens = self.tokenize(text)         # Cortar en palabras
        tokens = self.remove_stopwords(tokens)   # Quitar "the", "is", "a"...
        tokens = self.remove_short_words(tokens) # Quitar palabras de 1 letra
        tokens = self.lemmatize_words(tokens)    # "running" → "run"
        return ' '.join(tokens)
```

**Ejemplo**:
- Original: `"This phone is ABSOLUTELY amazing!!! Best purchase ever 😍"`
- Procesado: `"phone absolutely amazing best purchase ever"`

¿Por qué hacemos esto? Porque a la computadora no le importan las mayúsculas ni los signos — solo las **palabras con significado**. Limpiar el texto mejora mucho la precisión del modelo.

---

### Paso 3: Módulo 1 — Análisis de Sentimientos

#### 3a. Convertir palabras en números (TF-IDF)

La computadora no entiende palabras, solo números. **TF-IDF** (Term Frequency-Inverse Document Frequency) convierte cada comentario en un vector numérico donde las palabras más "especiales" de un comentario (que no aparecen en todos lados) tienen un valor más alto.

```python
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,    # Máximo 5000 palabras en el vocabulario
    ngram_range=(1, 2),   # Palabras sueltas Y pares de palabras ("poor quality")
    min_df=2,             # La palabra debe aparecer al menos en 2 documentos
    max_df=0.8            # Ignorar palabras que aparecen en más del 80% de docs
)
X = tfidf_vectorizer.fit_transform(df_sample['processed_text'])
```

#### 3b. Entrenar clasificadores

Se divide el dataset en 70% entrenamiento y 30% prueba, y se entrenan 4 modelos:

| Modelo | Cómo funciona (en simple) |
|--------|--------------------------|
| **Naive Bayes** | Usa probabilidades — si un comentario tiene "terrible" y "waste", probablemente es negativo |
| **Logistic Regression** | Pesa cada palabra — "excellent" suma puntos a positivo, "broken" a negativo |
| **SVM** | Dibuja una línea (o plano) que separa los grupos lo mejor posible |
| **Random Forest** | 100 "árboles de decisión" que votan — la mayoría gana |

```python
classifiers = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'SVM (Linear)': SVC(kernel='linear', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}
```

#### 3c. Evaluar

Se mide qué tan bien predice cada modelo con métricas estándar:
- **Accuracy** — porcentaje de aciertos totales
- **Precision** — de lo que dijo "negativo", cuánto realmente lo era
- **Recall** — de lo que realmente era negativo, cuánto detectó
- **F1-score** — balance entre precision y recall
- **Cross-Validation (5-fold)** — se repite la prueba 5 veces para verificar estabilidad

El mejor modelo (SVM/Logistic Regression) logra ~**85% de accuracy**.

#### 3d. Palabras más relevantes

Usando los coeficientes de Logistic Regression podemos ver qué palabras "pesan" más para cada sentimiento:
- **Positivo**: awesome, best, excellent, nice, love, amazing
- **Negativo**: worst, poor, return, waste, hang, defective
- **Neutro**: average, product ok, sometimes

---

### Paso 4: Módulo 2 — Identificación de Temas (LDA)

**LDA** (Latent Dirichlet Allocation) es un algoritmo que lee todos los comentarios y descubre automáticamente de qué **temas** hablan, sin que nosotros le digamos cuáles son.

```python
# Crear diccionario y corpus
dictionary = corpora.Dictionary(texts_tokenized)
corpus = [dictionary.doc2bow(text) for text in texts_tokenized]

# Entrenar LDA con 5 temas
lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=5,
                      random_state=42, passes=10, alpha='auto')
```

El modelo descubrió temas como:
1. **Calidad general de producto** (good, product, quality, excellent)
2. **Teléfonos y cámaras** (phone, camera, battery, performance)
3. **Uso y batería** (battery, charge, screen, samsung)
4. **Compras y opiniones** (dont, buy, amazon, bought)
5. **Libros y valor** (nice, book, money, value, awesome)

Luego se asigna el **tema dominante** a cada reseña y se analiza cómo se distribuyen los temas por sentimiento (ej. los comentarios negativos se concentran más en el tema 4 de "compras/quejas").

---

### Paso 5: Módulo 3 — Análisis Sintáctico-Semántico

Usando **spaCy**, analizamos la gramática de cada frase para extraer información lingüística:

```python
class SyntacticSemanticAnalyzer:
    def analyze(self, text):
        doc = self.nlp(text)

        # POS Tagging — marca cada palabra como sustantivo, adjetivo, verbo...
        token_info = [{'text': token.text, 'pos': token.pos_, 'dep': token.dep_}
                      for token in doc]

        # NER — detecta nombres propios: "Samsung" → ORG, "Amazon" → ORG
        entities = [(ent.text, ent.label_) for ent in doc.ents]

        # Patrones ADJ+NOUN — "poor quality", "excellent service"
        adj_noun = [f"{doc[i].text} {doc[i+1].text}"
                    for i in range(len(doc)-1)
                    if doc[i].pos_ == 'ADJ' and doc[i+1].pos_ == 'NOUN']

        # Patrones VERB+NOUN — "love phone", "recommend product"
        verb_noun = [f"{token.lemma_} {child.text}"
                     for token in doc if token.pos_ == 'VERB'
                     for child in token.children if child.pos_ == 'NOUN']
```

**¿Por qué importa?** Estos patrones son la **evidencia lingüística** que explica el sentimiento:
- Reseña positiva → patrones como "good phone", "excellent quality", "best product"
- Reseña negativa → patrones como "poor quality", "worst product", "bad experience"

---

### Paso 6: Pipeline Integrado

La clase `ReviewAnalysisPipeline` junta los 3 módulos. Le das un texto y te devuelve todo junto:

```python
pipeline = ReviewAnalysisPipeline(
    preprocessor=preprocessor,
    vectorizer=tfidf_vectorizer,
    classifier=lr_model,
    lda_model=lda_model,
    dictionary=dictionary,
    syntactic_analyzer=analyzer
)

result = pipeline.analyze("Terrible product. Screen broke after two days.")
# → Sentimiento: NEGATIVO (87% confianza)
# → Tema: 4 (compras/quejas)
# → Patrones: "Terrible product"
# → Evidencia TF-IDF: "terrible", "broke", "screen"
```

---

### Paso 7: Capa de Explicabilidad

El sistema no es una "caja negra". Para cada predicción muestra:

1. **Sentimiento predicho** + nivel de confianza (probabilidades por clase)
2. **Tema identificado** + sus palabras clave
3. **Patrones lingüísticos** encontrados en el texto (ADJ+NOUN, VERB+NOUN)
4. **Términos TF-IDF más relevantes** del texto que influyeron en la decisión
5. **Entidades nombradas** detectadas (marcas, productos, organizaciones)

Esto permite generar conclusiones interpretables como:

> *"Los comentarios negativos se concentran en problemas de calidad del producto y compras, evidenciados por patrones como 'poor quality' y 'worst product', con términos clave como 'waste', 'return' y 'defective'."*

---

## Estructura del proyecto

```
proyecto_pln/
├── POC_Analisis_Reviews.ipynb    ← Notebook principal del POC
├── Amazon Review Data Web Scrapping - Amazon Review Data Web Scrapping.csv  ← Dataset
├── Seleccion_de_Modulo_Critico.md  ← Documento de selección de módulos
└── README.md                      ← Este archivo
```

## Tecnologías y librerías

| Librería | Versión | Uso |
|----------|---------|-----|
| pandas | 3.0+ | Manejo de datos tabulares |
| numpy | 1.26+ | Operaciones numéricas |
| nltk | 3.9+ | Tokenización, stopwords, lematización |
| spacy | 3.7+ | POS tagging, NER, dependency parsing |
| scikit-learn | 1.8+ | TF-IDF, clasificadores, métricas |
| gensim | 4.0+ | Topic modeling (LDA) |
| matplotlib / seaborn | — | Visualizaciones |

## Resultados principales

| Módulo | Métrica principal | Resultado |
|--------|-------------------|-----------|
| Análisis de Sentimientos | Accuracy / F1 weighted | ~85% / ~0.81 |
| Identificación de Temas | Temas descubiertos | 5 temas coherentes |
| Análisis Sintáctico | Patrones extraídos | ADJ+NOUN, VERB+NOUN, NER |