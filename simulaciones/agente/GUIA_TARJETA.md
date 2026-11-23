# Guía: pasar el agente entrenado a la NUCLEO-L432KC y conectar los sensores

Supuestos: LoRa **RFM95/SX1276 (SPI)** y puente H **L298N**. Si tu LoRa es por UART (tipo E32), cambia el Paso 12.

---

## Parte 1: pasar el agente entrenado a la tarjeta

**Paso 1. Exporta la política a C.** En la PC, dentro de la carpeta `sim`:

```bash
uv run python -c "
import csv
acc = [r['accion'] for r in csv.DictReader(open('resultados/politica.csv'))]
print('#define N_ESTADOS', len(acc))
print('static const uint8_t politica[N_ESTADOS] = {' + ','.join(acc) + '};')
" > politica.h
```

Esto crea `politica.h`: 144 números, uno por estado (0 = Avanzar, 1 = Retroceder, 2 = Izquierda, 3 = Derecha).

**Paso 2. Instala STM32CubeIDE** (gratis, de st.com).

**Paso 3. Crea el proyecto:** *File → New → STM32 Project → Board Selector →* **NUCLEO-L432KC**.

**Paso 4. Activa los periféricos** en el archivo `.ioc`:

| Periférico | Configuración | Para |
|---|---|---|
| USART1 | 9600 baud, interrupción RX | GPS |
| SPI1 | Master, pines PB3/PB4/PB5 | LoRa |
| I2C1 | 400 kHz, PB6/PB7 | MPU6050 |
| TIM1 | CH1 y CH4 en PWM (~10 kHz) | Motores |
| TIM2 | Interrupción a 100 Hz | Leer el giroscopio |
| USART2 | 115200 baud | Mensajes a la PC |

Luego genera el código (*Generate Code*).

**Paso 5. Copia `politica.h`** a la carpeta `Core/Inc/` del proyecto.

**Paso 6. Escribe en `main.c` el lazo que se repite cada 1 segundo:**

1. Lee la posición del carrito (GPS) y la de la persona (LoRa).
2. Calcula las 4 variables del estado, **igual que en `sim/cancha_env.py`**:
   - `d`: distancia entre los GPS (<5, 5–15, 15–40, >40 m)
   - `a`: ángulo hacia la persona en sectores de 45°, u 8 si el rumbo aún no se conoce
   - `f`: volteado, según el acelerómetro
   - `b`: bloqueado, por un golpe o por 8 s sin avanzar
3. `estado = ((d*9 + a)*2 + f)*2 + b`
4. `accion = politica[estado]`
5. Mueve los motores durante 1 s con esa acción. Si la distancia es menor que 2 m, detente.

> No inviertas las acciones cuando el carrito está volteado: el agente ya ve `f` y lo compensa. Lo único que se corrige con `f` es el signo del giroscopio.

**Paso 7. Compila** con *Project → Build*.

**Paso 8. Graba:** conecta la Nucleo por USB y presiona **Run**.

**Paso 9. Verifica** con un monitor serie a 115200 baud que se impriman `estado`, `accion` y `dist`.

---

## Parte 2: conexiones de los sensores a la tarjeta

**Paso 10. GPS NEO-6M**

| GPS | Nucleo |
|---|---|
| VCC | 5V |
| GND | GND |
| TX | **D0** (PA10) |

**Paso 11. MPU6050**

| MPU6050 | Nucleo |
|---|---|
| VCC | 3V3 |
| GND | GND |
| SCL | **D5** (PB6) |
| SDA | **D4** (PB7) |

**Paso 12. LoRa RFM95** (⚠️ funciona solo a 3.3 V)

| LoRa | Nucleo |
|---|---|
| VCC | 3V3 |
| GND | GND |
| SCK | **D13** (PB3) |
| MISO | **D12** (PB4) |
| MOSI | **D11** (PB5) |
| NSS | **D2** (PA12) |
| DIO0 | **D6** (PB1) |
| RESET | **A3** (PA4) |

**Paso 13. Puente H L298N** (quita los jumpers de ENA y ENB)

| L298N | Nucleo / batería |
|---|---|
| ENA | **D9** (PA8) |
| IN1, IN2 | **A0, A1** |
| ENB | **D10** (PA11) |
| IN3, IN4 | **A2, D3** |
| GND | GND de la Nucleo **y** (−) de la batería de motores |
| 12V/VS | (+) de la batería de motores |
| 5V | **No conectar** |

**Paso 14. Alimentación**

- Batería de 9 V: el (+) a **VIN** y el (−) a **GND**.
- Batería de motores: solo al puente H.
- **Todas las tierras unidas.**

**Paso 15. Protecciones**

- Un capacitor de 100 nF soldado en las terminales de cada motor.
- Un capacitor de 470 µF entre VS y GND del puente H.
- Antenas del GPS y del LoRa lejos de los motores.

⚠️ No uses los pines **A4, A5** (unidos a D4/D5 por los puentes SB16/SB18), **A7** ni **PA15** (comunicación USB con la PC).

---

## Parte 3: pruebas, en este orden

16. Solo la Nucleo por USB: haz parpadear el LED.
17. El GPS a cielo abierto: verifica que imprima lat/lon.
18. El MPU6050: voltéalo y gíralo con la mano; revisa que `f` cambie y que el giro marque ~90°.
19. El LoRa: verifica que lleguen los datos de la persona.
20. Los motores con las ruedas en el aire: prueba las 4 acciones.
21. En el pasto: calibra el PWM a 0.3 m/s y verifica que cada giro sea de ~45°.
22. Prueba completa con la persona a ~20 m.

---

*Los pines vienen del manual UM1956 de la placa; confírmalos en CubeMX antes de soldar.*
