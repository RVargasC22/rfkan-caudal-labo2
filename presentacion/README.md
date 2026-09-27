# Presentación — 20 slides, 16:9

Tres archivos con el mismo contenido (metodología / implementación / resultados),
generados desde una única fuente: `content.py`. Los números salen de
`lab/repo/outputs/experiments/experiments_results.csv` y las figuras de
`lab/repo/outputs/analysis/`.

## Deck web (Artifact tipo Slides)

https://claude.ai/artifact/7Eh6oV3GD1UnMwT5ofg6Ss (de la cuenta anterior; le faltan los valores de
L=336 en las slides `seqlen` y `grilla`: regenerar con `build_deck.py` y publicar esas dos)

Para presentar y grabar el video desde el navegador (modo Present, flechas).
También se descarga como PPTX o PDF desde el menú del deck. Es privado: para
que otra persona lo abra hay que compartirlo desde el menú Share.

Se genera con `build_deck.py <carpeta>`, que escribe `project/deck.json` y
`project/slides/<id>.html`. Las figuras se suben como assets del artifact y
`deck_assets.json` guarda sus URLs.

## `RFKAN_Presentacion.pptx`

Generado con `python-pptx` (`build_pptx.py`): tablas y gráfico de líneas
**nativos de Office** (clic derecho → "Editar datos"), notas del orador en cada
slide. Para editar en PowerPoint, LibreOffice Impress o Google Slides.

## `SPEECH.md`

Guion del video: las notas del orador de cada slide, agrupadas por bloque.

## Regenerar con resultados nuevos

```bash
cd presentacion
../lab/.venv/bin/python build_pptx.py          # pptx + SPEECH.md
../lab/.venv/bin/python build_deck.py <dir>    # archivos del deck web; luego se publican al artifact
```
