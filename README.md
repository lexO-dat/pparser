# PDF to Markdown Parser System

## Características

- **Extracción de texto estructurado**: Mantiene jerarquías, títulos, párrafos, listas
- **Extracción de imágenes**: Detecta y guarda imágenes con referencias en Markdown * con bugs pero en proceso 
- **Conversión de tablas**: Convierte tablas a formato Markdown/CSV * algunos casos fallan
- **Fórmulas matemáticas**: Detecta y convierte a LaTeX * salen muy mal, pero trabajando en eso
- **Formularios y encuestas**: Detecta preguntas de selección múltiple
- **Procesamiento por páginas**: Iteración progresiva con validaciones intermedias
- **Arquitectura multiagente**: Sistema modular y extensible

## Instalación

```bash
pip install -r requirements.txt
```

## Configuración

Crea un archivo `.env` con tu API key de OpenAI:

```
OPENAI_API_KEY=tu_api_key_aqui
```

## Uso

### Procesamiento básico

```python
from pparser import PDFProcessor

processor = PDFProcessor()
result = processor.process_pdf("documento.pdf", output_dir="output/")
```

### Procesamiento en lote

```python
from pparser import BatchProcessor

batch = BatchProcessor()
batch.process_directory("pdfs/", "outputs/")
```

## Arquitectura

El sistema utiliza los siguientes agentes especializados:

- **TextExtractor**: Extrae texto estructurado
- **ImageExtractor**: Detecta y extrae imágenes  
- **TableExtractor**: Convierte tablas a Markdown/CSV
- **FormulaExtractor**: Detecta y convierte fórmulas matemáticas
- **FormDetector**: Identifica formularios y preguntas
- **StructureBuilder**: Ensambla el Markdown final
- **QualityValidator**: Verifica la calidad de la conversión

## Estructura del proyecto

```
pparser/
├── agents/           # Agentes especializados
├── extractors/       # Módulos de extracción
├── utils/           # Utilidades y helpers
├── workflows/       # Flujos de LangGraph
└── processors/      # Procesadores principales
```

## Flujo de procesamiento

1. **Análisis inicial**: Determina estructura del PDF
2. **Extracción paralela**: Cada agente procesa su especialidad
3. **Consolidación**: Ensambla todos los elementos
4. **Validación**: Verifica calidad y completitud
5. **Generación**: Crea Markdown final y assets

## Formato de salida

```
output/
├── documento.md     # Markdown estructurado
├── images/          # Imágenes extraídas
├── tables/          # Tablas en CSV (opcional)
└── metadata.json    # Información del procesamiento
```
