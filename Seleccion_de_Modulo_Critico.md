Módulos seleccionados
1. Módulo de Análisis de Sentimientos
Descripción
Este módulo tiene como propósito clasificar automáticamente los comentarios de los clientes en categorías de sentimiento: positivo, negativo o neutro, a partir del contenido textual de las reseñas.
Justificación
El análisis de sentimientos constituye una de las aplicaciones más relevantes del Procesamiento de Lenguaje Natural (PLN) en el ámbito de la experiencia de cliente, ya que permite capturar la percepción global del usuario frente a un producto o servicio.
Su implementación en el POC permite:
· Detectar niveles de satisfacción o insatisfacción 
· Identificar alertas tempranas en la experiencia del cliente 
· Priorizar acciones de mejora basadas en datos 
· Automatizar el análisis de grandes volúmenes de comentarios 
Funcionalidad
El módulo recibe como entrada un comentario textual (Review_text) y genera como salida:
· Una etiqueta de sentimiento (positivo, negativo o neutro) 
· Una posible puntuación asociada a la predicción 
· Elementos explicativos que justifican la clasificación 
Técnicas utilizadas
· Tokenización 
· Limpieza y normalización de texto 
· Lematización 
· Representación vectorial (TF-IDF) 
· Modelos de clasificación supervisada (Logistic Regression, Naive Bayes) 
· Evaluación mediante métricas (precision, recall, F1-score) 
Uso del dataset
Se utiliza la variable Own_Rating como etiqueta supervisada para entrenar el modelo de clasificación, permitiendo aplicar técnicas de aprendizaje automático sobre datos previamente categorizados.

2. Módulo de Identificación de Temas
Descripción
Este módulo tiene como objetivo identificar los temas principales presentes en los comentarios de los clientes, permitiendo agruparlos en categorías relevantes como calidad, servicio, precio o entrega.
Justificación
Mientras el análisis de sentimientos responde a la pregunta “cómo se siente el cliente”, este módulo permite responder “sobre qué está opinando”, lo cual resulta fundamental para la generación de insights accionables a nivel organizacional.
Su implementación permite:
· Detectar patrones recurrentes en los comentarios 
· Identificar áreas críticas del negocio 
· Complementar el análisis de sentimiento con contexto semántico 
· Facilitar la toma de decisiones basada en evidencia textual 
Funcionalidad
El módulo recibe un conjunto de textos y produce como salida:
· Temas identificados 
· Palabras clave asociadas a cada tema 
· Agrupaciones semánticas de comentarios 
Técnicas utilizadas
· Tokenización 
· Eliminación de stopwords 
· Representación de texto (TF-IDF) 
· Modelado de temas (LDA) 
· Extracción de keywords 

3. Módulo de Análisis Sintáctico-Semántico (Enfoque Avanzado de PLN)
Descripción
Este módulo complementa los anteriores mediante la aplicación de técnicas avanzadas de análisis lingüístico, permitiendo extraer estructuras gramaticales y patrones semánticos relevantes dentro de los textos.
Justificación
La incorporación de análisis sintáctico permite enriquecer la interpretación del lenguaje más allá de enfoques estadísticos tradicionales, aportando un nivel superior de comprensión del texto y aumentando la explicabilidad del modelo.
Este módulo es clave para cumplir con los requerimientos académicos del curso, al integrar explícitamente técnicas de análisis sintáctico y semántico.
Su implementación permite:
· Identificar relaciones gramaticales entre palabras 
· Detectar patrones lingüísticos relevantes (ej. adjetivo + sustantivo) 
· Mejorar la interpretabilidad de los resultados del modelo 
· Complementar la clasificación con evidencia lingüística 
Funcionalidad
Dado un comentario textual, el módulo genera como salida:
· Tokens procesados 
· Lemas 
· Etiquetas gramaticales (POS tagging) 
· Patrones lingüísticos relevantes (ej. “poor quality”, “excellent service”) 
· Palabras clave que explican el resultado del modelo 
Técnicas utilizadas
· Tokenización 
· Lematización 
· POS Tagging (Part-of-Speech Tagging) 
· Análisis de dependencias gramaticales (opcional) 
· Extracción de patrones sintácticos (ADJ + NOUN, VERB + NOUN) 

Relación entre los módulos
Los tres módulos se integran para proporcionar una visión completa de la experiencia del cliente:
· El análisis de sentimientos identifica la polaridad de la opinión 
· La identificación de temas contextualiza el contenido de la opinión 
· El análisis sintáctico-semántico explica las razones lingüísticas detrás de la clasificación 
Esto permite generar conclusiones como:
“Los comentarios negativos se concentran principalmente en problemas relacionados con la calidad del producto y el tiempo de entrega, evidenciados por la presencia de términos como ‘poor quality’ y ‘late delivery’.”

Explicabilidad del sistema
Una de las contribuciones clave del POC es la incorporación de una capa de explicabilidad, mediante la cual el sistema no solo predice una categoría, sino que también proporciona:
· Palabras clave relevantes 
· Estructuras lingüísticas identificadas 
· Evidencia textual que respalda la predicción 
Esto permite interpretar los resultados del modelo de forma transparente, lo cual es fundamental en aplicaciones de analítica de negocio.

Alcance dentro del POC
En esta fase inicial, los módulos serán implementados como prototipos funcionales en Python, capaces de:
· Procesar entradas de texto 
· Generar salidas estructuradas e interpretables 
· Demostrar la viabilidad técnica de las funcionalidades propuestas 
No se busca un sistema optimizado para producción, sino una validación del enfoque metodológico y técnico del proyecto.

Evaluación del modelo
El desempeño del sistema será evaluado mediante:
· Accuracy 
· Precision 
· Recall 
· F1-score 
· Matriz de confusión 
Se utilizará una división de los datos en conjuntos de entrenamiento y prueba, garantizando una evaluación objetiva del modelo.

Conclusión
La selección de estos módulos permite construir un POC robusto que combina técnicas de aprendizaje automático con análisis lingüístico avanzado, logrando no solo clasificar información textual, sino también interpretarla y explicarla.
Este enfoque garantiza una alineación tanto con los objetivos del proyecto como con los criterios académicos del curso, demostrando un nivel de profundidad técnica acorde a un programa de maestría en inteligencia artificial.
