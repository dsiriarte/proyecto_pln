# Análisis de Experiencia del Cliente mediante PLN

**Procesamiento de Lenguaje Natural — Universidad de La Sabana**

Sistema que analiza reseñas de Amazon usando técnicas de PLN para clasificar sentimientos, descubrir temas y extraer patrones lingüísticos.

---

## Dataset

- **Fuente**: Amazon Review Data (~60,000 reseñas)
- **Categorías**: smartTv, mobile, books, refrigerator, mobile accessories
- **Etiquetas**: Positive (~78%), Negative (~15%), Neutral (~7%)

---

## Notebook principal: `Analisis_Reviews final.ipynb`

El notebook está organizado en las siguientes secciones:

### 1. Carga y exploración de datos
Carga del CSV, limpieza de nulos y muestra estratificada de 10,000 reseñas.

### 2. Preprocesamiento de texto
Clase `TextPreprocessor` que limpia HTML, normaliza, tokeniza, elimina stopwords y lematiza con NLTK.

### 3. Módulo de Sentimientos
- Representación con **TF-IDF** (5,000 features, unigramas + bigramas)
- Comparación de 4 clasificadores: Naive Bayes, Logistic Regression, SVM, Random Forest
- Cross-validation 5-fold y reportes de clasificación

### 4. Módulo de Temas (LDA)
- Topic modeling con **Gensim LDA**
- Visualización de temas y heatmap tema-sentimiento

### 5. Módulo Sintáctico
- **spaCy**: POS Tagging, NER, patrones ADJ+NOUN y VERB+NOUN

### 6. Pipeline integrado y explicabilidad
- Clase `ReviewAnalysisPipeline` que une los 3 módulos
- Capa de explicabilidad: confianza, términos TF-IDF relevantes, patrones lingüísticos

### 7. Iteraciones de mejora
- **Iteración 1**: Balanceo de clases (`class_weight=balanced`)
- **Iteración 2**: Hypertuning con GridSearchCV
- **Iteración 3**: Optimización de K en LDA usando coherence score (K óptimo = 4)

### 8. Análisis complementarios
- Rendimiento por categoría de producto
- Análisis de errores (matrices de confusión, longitud de texto)
- Registro de iteraciones y decisiones de diseño
- Comparativa final POC vs sistema optimizado

---

## Resultados

| Métrica | POC | Sistema Final |
|---------|-----|---------------|
| Accuracy | 0.85 | 0.79 |
| Macro F1 | 0.53 | 0.57 |
| Recall Neutral | 0.01 | 0.21 |
| Recall Negative | 0.53 | 0.70 |
| LDA Temas | K=5 (manual) | K=4 (coherence=0.57) |

El sistema final prioriza **balance entre clases** sobre accuracy pura, mejorando significativamente la detección de reseñas neutrales y negativas.

---

## Estructura del proyecto

```
proyecto_pln/
├── Analisis_Reviews final.ipynb   ← Notebook principal (entregable)
├── POC_Analisis_Reviews.ipynb     ← POC inicial (referencia)
├── Amazon Review Data Web Scrapping - Amazon Review Data Web Scrapping.csv
├── Seleccion_de_Modulo_Critico.md
└── README.md
```

## Tecnologías

| Librería | Uso |
|----------|-----|
| pandas / numpy | Datos tabulares |
| nltk | Tokenización, stopwords, lematización |
| spacy | POS tagging, NER |
| scikit-learn | TF-IDF, clasificadores, GridSearchCV |
| gensim | Topic modeling (LDA), coherence |
| matplotlib / seaborn | Visualizaciones |