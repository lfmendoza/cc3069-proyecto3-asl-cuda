# Cierre antes de entregar

Lo hace Fernando en una rama `fernando/cierre`, con los otros dos revisando el
PR.

1. Todos los PRs integrados y CI en verde.
2. Tabla de resultados del `README.md` completa.
3. PDF final:
   ```bash
   quarto render informe/informe.qmd --to pdf
   ```
   Abrirlo completo. Revisar propiedades del PDF (autor/título) y que no haya
   secciones con comentarios `<!-- ... -->` pendientes:
   `git grep -n "<!--" docs informe`
4. Notebook con salidas de la GPU (`adrian/notebook-ejecutado` integrado).
5. Borrar lo que no es entrega:
   ```bash
   git rm -r .coordinacion
   git rm -r --cached results/*/scratch 2>/dev/null || true
   ```
   y cualquier carpeta de resultados de pruebas (`results/*-quick`, etc.).
6. Barrido final de texto (no debe devolver nada):
   ```bash
   git grep -n -i -E "TODO|FIXME|pendiente|completar con|lorem"
   git grep -n -E "C:\\\\Users|/home/" -- ':!*.lock'
   git log --format=%B | grep -i -E "co-authored|generated|signed-off" || true
   ```
7. Commit `chore: preparación de la entrega`, merge a `main`, tag:
   ```bash
   git tag -a v1.0 -m "Entrega Proyecto 3"
   git push origin main --tags
   ```
8. Para Canvas: el PDF, el notebook y el enlace al repositorio en el tag `v1.0`.
