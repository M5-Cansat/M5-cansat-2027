# Eme5 | Mundial CanSat 2027

**Repositorio de trabajo colaborativo del equipo Eme5**, dedicado al diseño, desarrollo, integración, validación y documentación de los subsistemas del CanSat para la misión 2027.

> **Estado:** desarrollo activo. La rama `main` contiene únicamente lo aprobado como estable; `develop` reúne los avances integrados y revisados. Tener código en una rama de trabajo **no** significa que esté probado físicamente ni incorporado a `main`.

**Accesos del equipo:** [Repositorio](https://github.com/M5-Cansat/M5-cansat-2027) · [Issues](https://github.com/M5-Cansat/M5-cansat-2027/issues) · [Pull Requests](https://github.com/M5-Cansat/M5-cansat-2027/pulls) · [Proyecto y cronograma](https://github.com/orgs/M5-Cansat/projects/1)

---

## 1. ¿Para qué sirve este repositorio?

Es la **fuente de referencia compartida** del proyecto. Aquí se conservan las versiones de firmware, software, diseños, resultados de pruebas y documentos, junto con la historia de quién propuso cada cambio, por qué se hizo y cómo se verificó.

Para cualquier contribución, buscamos poder responder:

1. **¿Qué requisito, problema o actividad atiende?** → un *Issue*.
2. **¿Dónde se desarrolló?** → una rama identificable.
3. **¿Qué cambió?** → uno o varios *commits* descriptivos.
4. **¿Cómo se revisó?** → un *Pull Request* (PR).
5. **¿Qué evidencia demuestra que funciona?** → pruebas, registros y documentación enlazados.

**Regla central: `Issue → rama → commits → pruebas → Pull Request → integración`.**

## 2. Estructura del repositorio

La estructura se organiza por **subsistema**, no por integrante. Los directorios pueden ampliarse mediante PR cuando surjan necesidades justificadas.

```text
M5-cansat-2027/
├── README.md                         # Esta guía
├── .gitignore                        # Exclusiones (compilados, temporales, etc.)
│
├── firmware/
│   ├── stm32f411/                     # Proyecto STM32CubeIDE de la STM32F411
│   └── stm32l432kc/                   # Proyecto STM32CubeIDE de la STM32L432KC
│
├── hardware/
│   ├── electronica/                   # Esquemáticos, PCB y documentación eléctrica
│   ├── mecanica/                      # CAD, mecanismos, estructura y materiales
│   └── rover/                         # Desarrollo mecánico/electrónico del rover
│
├── software/
│   ├── navegacion/                    # Algoritmos de navegación, control y RL
│   │   └── rl/                        # Propuesta de ubicación de trabajos de RL
│   └── estacion-terrena/              # Recepción, visualización y análisis de telemetría
│
├── simulaciones/                      # Modelos y estudios de comportamiento
├── pruebas/
│   ├── datos/                         # Datos de ensayo apropiados para versionar
│   └── resultados/                    # Informes, gráficas y evidencias
│
└── docs/
    ├── requisitos/                    # Requisitos, criterios y matriz de trazabilidad
    ├── arquitectura/                   # Diagramas, interfaces y decisiones técnicas
    ├── gestion/                        # Planificación, bitácoras y coordinación
    └── entregables/
        ├── CoDR/
        ├── PDR/
        ├── CDR/
        └── RAM/
```

**Nota sobre el estado real:** los firmwares F411 y L432KC se publicaron inicialmente en ramas separadas, mediante los PR [#10](https://github.com/M5-Cansat/M5-cansat-2027/pull/10) y [#11](https://github.com/M5-Cansat/M5-cansat-2027/pull/11). **No supongas que sus carpetas ya existen en `develop` o `main` hasta que dichos PR estén fusionados.** La carpeta `software/navegacion/rl/` es una ubicación propuesta: puede no estar creada todavía.

### ¿Dónde debe ir cada contribución?

| Tipo de trabajo | Ubicación recomendada | Evidencia mínima |
| --- | --- | --- |
| Firmware F411 | `firmware/stm32f411/` | Compilación, versión de placa, pruebas de periféricos |
| Firmware L432KC | `firmware/stm32l432kc/` | Compilación, GPS/LoRa/UART, pruebas de periféricos |
| Aprendizaje por refuerzo / navegación | `software/navegacion/` | Código reproducible, parámetros, métricas y semillas |
| Interfaz o recepción de datos | `software/estacion-terrena/` | Trama de ejemplo, capturas o prueba de recepción |
| Electrónica / mecánica / rover | `hardware/...` | Versión de diseño, planos y validación |
| Simulación | `simulaciones/` | Hipótesis, archivos fuente y resultados |
| Ensayos | `pruebas/` | Protocolo, condiciones, fechas y resultados |
| Requisitos, decisiones y entregables | `docs/` | Documento y relación con requisitos o Issues |

No subas resultados gigantes o binarios innecesarios al historial normal de Git. Coordina su almacenamiento y enlaza desde el Issue o la documentación.

## 3. Modelo de ramas

Este proyecto utiliza **una sola rama estable y una sola rama de integración para todo el repositorio**. Cada microcontrolador tiene su **carpeta**, sus Issues y sus ramas de trabajo, pero **no necesita una rama permanente `main` y otra `dev` propias**.

| Rama / prefijo | Propósito | Regla |
| --- | --- | --- |
| `main` | Versiones generales **estables**, revisadas y liberadas | No se trabaja directamente aquí |
| `develop` | Integración de cambios revisados de todos los subsistemas | Se actualiza mediante PR |
| `feature/<subsistema>-<objetivo>` | Incorporar funcionalidad o trabajo nuevo | Rama temporal, normalmente desde `develop` |
| `fix/<subsistema>-<problema>` | Corregir defectos concretos | Rama temporal |
| `research/<tema>` | Investigación, prototipos y algoritmos por validar | Hipótesis y experimentos documentados |
| `experiment/<tema>` | Experimento aislado y reproducible | No se integra sin evaluación |
| `docs/<tema>` | Documentación, procedimientos y guías | También pasa por PR |
| `test/<tema>` | Pruebas o automatización de pruebas | Evidencia y criterios de aceptación |
| `release/<version>` | Preparación opcional de una liberación conjunta | Solo cuando sea necesario |

**Ejemplos de nombres:**

```text
feature/f411-lectura-imu
feature/f411-filtro-complementario
feature/l432-telemetria-lora
fix/l432-buffer-gps
research/rl-entorno-base
experiment/rl-politica-v1
docs/guia-colaboracion
test/f411-validacion-i2c
```

**Ramas iniciales ya creadas:**

- `feature/fw-stm32f411-imu-freertos` → firmware de la STM32F411; [PR #10](https://github.com/M5-Cansat/M5-cansat-2027/pull/10).
- `feature/l432-importacion` → dos etapas históricas del firmware STM32L432KC; [PR #11](https://github.com/M5-Cansat/M5-cansat-2027/pull/11).

**Importante:** mientras los PR iniciales estén pendientes, `develop` puede no contener el firmware. Para avanzar sobre ese mismo código, coordina con la persona responsable: se puede trabajar en la rama existente o crear una rama dependiente a partir de ella y orientar temporalmente el PR hacia su rama base. Evita copiar el firmware a mano o introducirlo por segunda vez en otra rama.

## 4. Procedimiento para contribuir (PowerShell, Windows)

### 4.1. Primera vez: obtener el repositorio

Necesitas Git, acceso autorizado al repositorio privado y, opcionalmente, GitHub CLI (`gh`).

```powershell
git clone https://github.com/M5-Cansat/M5-cansat-2027.git
cd M5-cansat-2027
git branch -a
```

Configura tu identidad de Git si aún no lo has hecho:

```powershell
git config --global user.name "Tu Nombre"
git config --global user.email "tu-correo-asociado-a-github"
```

### 4.2. Antes de empezar una tarea

1. Comprueba si existe un **Issue**; si no existe, créalo antes de programar.
2. Completa en GitHub Projects los campos pertinentes: **Prioridad**, **Subsistema**, **Fase**, **Fecha de inicio** y **Fecha límite**.
3. Define el **resultado esperado** y cómo lo probarás; no basta con escribir «implementar».
4. Comprueba que no haya otra persona trabajando en el mismo archivo o interfaz.

### 4.3. Crear una rama para tu Issue

Si la base requerida ya está en `develop`:

```powershell
git status
git switch develop
git pull --ff-only origin develop
git switch -c feature/f411-lectura-imu
```

Sustituye el nombre por el subsistema y objetivo reales. **No crees de nuevo una rama que ya existe**: usa `git switch nombre-de-rama`.

### 4.4. Trabajar y registrar cambios

```powershell
# Modifica y prueba los archivos de tu subsistema.
git status --short
git diff --stat

# Agrega solamente lo relacionado con el Issue.
git add firmware/stm32f411/
git diff --cached --stat

git commit -m "feat(f411): leer acelerometro por I2C (refs #NUMERO)"
git push -u origin feature/f411-lectura-imu
```

`#NUMERO` es el número del **Issue real**; no lo dejes literalmente así al ejecutar el comando. Si tu tarea está en otra carpeta, reemplaza también la ruta de `git add`.

No hagas `git add .` sin revisar primero los archivos afectados. **Evita confirmar `.elf`, `.bin`, `Debug/`, `Release/`, archivos temporales, contraseñas o datos personales.** Mantén los archivos necesarios para reconstruir el proyecto STM32CubeIDE, como `.ioc`, `.project`, `.cproject`, código fuente, HAL/FreeRTOS y linker scripts.

### 4.5. Abrir un Pull Request

```powershell
gh pr create `
  --base develop `
  --head feature/f411-lectura-imu `
  --draft `
  --title "feat(f411): lectura de acelerometro" `
  --body "Refs #NUMERO. Cambios: lectura I2C. Pruebas: PENDIENTES. Evidencias: PENDIENTES."
```

- Usa **Draft** si falta compilar, probar, documentar o solicitar revisión.
- Actualiza la descripción con resultados verificables; cuando esté listo, marca el PR como **Ready for review**.
- Solicita revisión de otra persona del equipo cuando sea posible.
- **No fusiones** código cuyo alcance o pruebas no estén suficientemente claros.
- Los PR de funcionalidades normales se dirigen a `develop`. La integración de `develop` en `main` se gestiona mediante una revisión de liberación. **Excepción controlada:** una mejora de documentación general (por ejemplo, esta guía de incorporación) puede proponerse mediante PR directamente a `main` para que aparezca en la portada del repositorio, siempre con revisión; después se sincroniza el cambio con `develop`.

### 4.6. Después de fusionar el PR

```powershell
git switch develop
git pull --ff-only origin develop
git branch -d feature/f411-lectura-imu
```

El borrado local es **opcional** y debe hacerse solo después de confirmar que el PR se fusionó. Los cambios permanecen en el historial.

## 5. Trazabilidad obligatoria

Cada actividad técnica debe tener un hilo claro de documentación:

```text
Requisito / necesidad
        ↓
GitHub Issue + responsable + criterios de aceptación
        ↓
GitHub Project (fase, subsistema, prioridad y fechas)
        ↓
Rama de trabajo
        ↓
Commits pequeños con mensajes descriptivos
        ↓
Prueba o experimento reproducible
        ↓
Pull Request + revisión + evidencias
        ↓
Integración en develop
        ↓
Validación de sistema y liberación en main
```

### Plantilla recomendada para un Issue

```markdown
## Objetivo
¿Qué necesidad se resuelve?

## Contexto / requisito asociado
Identificador de requisito, interfaz o antecedente.

## Alcance y entregables
¿Qué archivos, módulos o documentos se espera producir?

## Criterios de aceptación
- [ ] Resultado técnico medible 1
- [ ] Resultado técnico medible 2
- [ ] Evidencia de pruebas adjunta o enlazada

## Dependencias y riesgos
¿Qué debe estar listo antes? ¿Qué podría fallar?
```

### Convención de commits

Formato sugerido: `tipo(subsistema): cambio concreto (refs #Issue)`.

| Tipo | Uso | Ejemplo |
| --- | --- | --- |
| `feat` | Nueva función | `feat(l432): enviar telemetria por LoRa (refs #42)` |
| `fix` | Corrección | `fix(l432): evitar truncamiento silencioso (refs #42)` |
| `test` | Pruebas | `test(f411): registrar validacion de IMU (refs #42)` |
| `docs` | Documentación | `docs(gestion): definir flujo de ramas (refs #42)` |
| `refactor` | Mejora interna sin cambiar el propósito | `refactor(rl): separar configuracion (refs #42)` |
| `chore` | Mantenimiento / importación | `chore(f411): actualizar configuracion CubeIDE (refs #42)` |

**Los números de Issue anteriores son ejemplos**, no referencias a actividades reales del repositorio.

### Plantilla recomendada para un Pull Request

```markdown
## Issue relacionado
Refs #NUMERO

## ¿Qué se cambió y por qué?
Descripción breve, archivos o módulos afectados.

## Validación realizada
- [ ] Se revisó el diff
- [ ] Compila / ejecuta (indicar entorno y versión)
- [ ] Prueba funcional realizada o limitación explicada
- [ ] Evidencias enlazadas (logs, capturas, resultados)
- [ ] Documentación e interfaces actualizadas

## Riesgos, limitaciones y siguientes pasos
Señalar qué NO ha sido probado y qué queda pendiente.
```

**No cierres un Issue solo porque se creó un PR.** Ciérralo cuando estén satisfechos sus criterios de aceptación. Si usas palabras como `Closes #...`, comprueba que el flujo de cierre automático corresponde a la rama de destino que usa GitHub; en PR hacia `develop` puede ser preferible `Refs #...` y cerrar el Issue de forma explícita tras la validación.

## 6. Qué cuenta como una prueba válida

No confundimos **código subido**, **compilación exitosa** y **funcionamiento en hardware**. Son niveles de evidencia diferentes.

| Nivel | Qué demuestra | Ejemplos |
| --- | --- | --- |
| Registrado | Archivos y cambios reproducibles en Git | Commit, configuración `.ioc` |
| Compilado | El proyecto construye sin errores | Log de STM32CubeIDE o CI |
| Probado en banco | Subsistema funciona en condiciones controladas | Lectura IMU, recepción LoRa, GPS |
| Integrado | Interactúa correctamente con otros módulos | Protocolo de tramas y pruebas conjuntas |
| Validado para misión | Cumple criterios definidos y documentados | Ensayo formal con condiciones, resultados y aceptación |

En pruebas físicas de firmware registra al menos: placa exacta, revisión del firmware/commit, alimentación, conexionado y pines físicos, sensores conectados, parámetros de comunicación, método de medición, resultados y problemas observados.

Para RL o simulaciones registra: versión de código, dataset o escenario, preprocesamiento, parámetros, semillas aleatorias cuando apliquen, métricas y gráficas comparables.

Guarda un resumen claro en `pruebas/resultados/` o en el documento correspondiente, y enlázalo desde el Issue/PR. Si se requieren datos grandes, almacénalos fuera de Git con una referencia verificable.

## 7. Reglas de colaboración

- **No** hacer commits directos a `main` ni a `develop`; usar PR y revisión.
- **No** mezclar en un mismo PR tareas independientes o múltiples subsistemas sin justificación.
- **No** cambiar interfaces, pines o tramas compartidas sin documentar y coordinar el impacto.
- **No** subir credenciales, claves, tokens, contraseñas, carpetas de usuario o datos personales.
- **No** afirmar «probado» si solo se compiló o simuló.
- **Sí** abrir Issues de defectos detectados y enlazarlos con su solución.
- **Sí** documentar decisiones de arquitectura y cambios de requisitos.
- **Sí** conservar el proyecto CubeIDE reproducible y revisar `.gitignore` antes de publicar.
- **Sí** incorporar pruebas y evidencias en cada PR.

> **Responsabilidad colectiva:** cada integrante actualiza el Issue y el PR de su trabajo. Quien revisa valida que el cambio sea entendible, reproducible y compatible con el resto del sistema.

## 8. Revisión y liberación de versiones

1. Se desarrolla en una rama de trabajo.
2. Se revisa y fusiona en `develop` cuando cumple los criterios acordados.
3. Se realizan pruebas de integración entre subsistemas.
4. Cuando exista una versión **estable**, se prepara un PR de `develop` hacia `main`.
5. Se registra una versión (*tag*) y su alcance, incluyendo las limitaciones conocidas.

Pueden identificarse componentes en las notas de versión como `f411-v0.1.0` o `l432kc-v0.1.0`, pero **un tag de Git apunta al estado de todo el repositorio**, no solamente a una carpeta.

## 9. Para empezar hoy

Si acabas de unirte al equipo: lee esta guía, entra al [tablero del proyecto](https://github.com/orgs/M5-Cansat/projects/1), elige un Issue asignado, comprueba en qué rama está el código que necesitas, crea tu rama de trabajo y abre un PR con evidencias. Ante dudas sobre interfaces compartidas, coordina primero con la persona responsable del subsistema.

**La trazabilidad es parte del entregable:** no basta con lograr que un módulo funcione; necesitamos poder explicar cómo se construyó, por qué cambió y cómo se validó.
