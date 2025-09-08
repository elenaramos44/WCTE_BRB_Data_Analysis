#!/bin/bash

echo "Intentando activar entorno virtual..."
source /dipc/elena/myenv/bin/activate && echo "Entorno activado OK" || echo "Error activando entorno"
python3 -c "import awkward; print('awkward import OK')" || echo "awkward no encontrado"

