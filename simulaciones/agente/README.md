# m5-rover — Carrito autónomo de doble cara con SARSA

Proyecto de la Electiva de Aprendizaje por Refuerzo (Unicauca). Un carrito se lanza a una cancha de fútbol, cae en un punto desconocido (derecho o al revés) y debe llegar **solo** a menos de 2 m de una persona usando únicamente:

- 2 GPS NEO-6M (uno en el carrito y otro con la persona, cuya coordenada llega por LoRa)
- MPU6050 (sabe si está al revés, detecta golpes y mide el giro)
- 2 motorreductores con puente H (velocidad constante de 0.3 m/s)

Los sensores y motores se **simulan** con su error real, el agente se entrena con **SARSA** y la tabla Q se exporta a CSV para cargarla en la tarjeta.

## Resultados del modelo final

| Métrica | Valor |
|---|---|
| Llega a la persona | 92.9 % (derecho 91.8 %, al revés 93.8 %) |
| Llega en menos de 4 min | 55 % |
| Se sale de la cancha | 5.4 % |
| Distancia real al terminar | 3.9 m (por el error del GPS) |

Configuración: SARSA tabular, α = 0.1 decreciente, γ = 0.95, ε = 0.1, Q₀ = 0, recompensa por acercarse + castigo por tiempo, 30 000 episodios.

A 0.3 m/s solo llega en 4 min si arranca a ≤ 47 m; para cruzar la diagonal de la cancha haría falta ≈ 0.67 m/s.

## Estructura

```
pyproject.toml, uv.lock     dependencias (uv)
sim/
  cancha_env.py             entorno Gymnasium: cancha, carrito, GPS, MPU6050, recompensa
  sarsa.py                  SARSA / Expected SARSA, ε-greedy / UCB, semi-gradiente, evaluación, exportación
  test_cancha_env.py        pruebas del entorno
  test_sarsa.py             pruebas del agente
  crear_notebook.py         genera el notebook
  simulacion_carrito.ipynb  la entrega: MDP, experimentos, modelo final, animaciones
  resultados/
    q_table.csv             144 estados × 4 acciones: la matriz para la tarjeta
    politica.csv            la mejor acción por estado, en formato legible
    experimentos.csv        todos los experimentos (21 configuraciones × 10 semillas)
    *.gif                   animaciones para la presentación
    semigradiente/          la misma exportación para la versión semi-gradiente
```

## Cómo correrlo

Requiere [uv](https://docs.astral.sh/uv/).

```
uv sync                                   # instala Python 3.14 y las dependencias
cd sim
uv run python test_cancha_env.py          # debe imprimir OK
uv run python test_sarsa.py               # debe imprimir OK
uv run jupyter notebook simulacion_carrito.ipynb
```

Para regenerar el notebook desde cero y volver a entrenar todo (unos 25 min con el portátil conectado a la corriente):

```
cd sim
uv run python crear_notebook.py simulacion_carrito.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 simulacion_carrito.ipynb
```

## Antes de usarlo en el carrito real

Medir en el carrito armado la velocidad (`VEL`), cuánto gira en 1 s (`GIRO`) y la deriva del giroscopio (`GYRO_DERIVA`), cambiarlos en `sim/cancha_env.py` y volver a entrenar. La tarjeta solo hace inferencia: calcula el estado igual que `CanchaEnv._estado()` y busca la acción en `politica.csv` (el pseudocódigo está al final del notebook).
