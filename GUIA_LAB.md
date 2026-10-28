# Guía para el laboratorio (Irwin)

La estación del laboratorio (Lenovo ThinkStation P5, RTX 6000 Ada 48 GB) es la **máquina de ejecución final**:
todos los resultados que se reportan en el artículo salen de aquí. La PC de Owen es solo para desarrollo.

## 0. Antes de instalar: preguntar en el laboratorio
1. ¿Puedo instalar Python, Git y Ollama? (si no hay permisos de administrador, ver nota abajo)
2. ¿Puedo dejar procesos corriendo de noche?
3. ¿Hay acceso remoto (AnyDesk, SSH)?
4. ¿La GPU la usa alguien más al mismo tiempo? (las mediciones de tiempo y VRAM solo valen con la GPU libre)
5. ¿Se puede guardar un PDF con licencia (ISO 27001) en el equipo?

Anotar el código/número de la PC usada en `maquinas/lab_p5.yaml` (campo `id_equipo_lab`).

## 1. Instalación (una sola vez)
En CMD:
```cmd
winget install Git.Git
winget install Python.Python.3.13
winget install Ollama.Ollama
```
Cerrar y volver a abrir CMD.

> Sin permisos de administrador: los instaladores de Python (python.org, opción "Install for current user")
> y de Ollama (ollama.com) funcionan a nivel de usuario. Git tiene versión portable (git-scm.com).

## 2. Traer el proyecto
```cmd
cd %USERPROFILE%
git clone https://github.com/Owen-Guerrero/tesis-rag.git
cd tesis-rag
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
git config user.name "TU-USUARIO-GITHUB"
git config user.email "tu-correo"
```
Usa **tu propia cuenta de GitHub** (Owen te invita como colaborador): así el historial muestra lo que hizo cada uno.

## 3. Cada vez que llegas al laboratorio
```cmd
cd %USERPROFILE%\tesis-rag
.venv\Scripts\activate.bat
git pull
```

## 4. Primera tarea: comparación de LLM candidatos (OE2 / OE3)
Descargar los candidatos (unos 50 GB en total; puede tardar):
```cmd
ollama pull llama3.1:8b
ollama pull qwen2.5:14b
ollama pull gpt-oss:20b
ollama pull qwen2.5:32b
```
Cerrar todo lo que use la GPU y ejecutar:
```cmd
python pruebas\prueba_llm.py --maquina lab_p5 --candidatos
```
Genera un JSON por modelo en `logs/` y la tabla `logs/resumen_llm.csv`.
Revisar en pantalla que cada respuesta esté en español y basada en el párrafo de AC-2.

Subir los resultados:
```cmd
git add logs maquinas
git commit -m "OE2/OE3: verificacion de candidatos en laboratorio"
git push
```

## Reglas
- Nunca editar `config.yaml` para una prueba puntual: los parámetros del experimento son comunes.
- Si algo falla, copiar el mensaje completo de la terminal y pasarlo al grupo.
- No subir el PDF de ISO al repositorio (ya está excluido en `.gitignore`).
