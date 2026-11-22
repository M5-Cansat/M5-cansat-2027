"""Entorno Gymnasium: carrito de doble cara que busca a una persona en una cancha de futbol.

Hardware simulado (solo esto "ve" el agente):
  - 2 GPS NEO-6M (carrito y persona; la de la persona llega por LoRa, no se simula porque
    la persona esta quieta). Error lento que deriva (~2 m) + ruido rapido, 1 lectura/s.
  - MPU6050: acelerometro -> si cayo al reves (F) y golpes (choques);
    giroscopio -> cuanto giro (con deriva). NO tiene brujula: el rumbo absoluto se estima
    con el GPS al recorrer una recta de 4 m; entre medias lo actualiza el giroscopio.
  - 2 motorreductores + puente H (traccion diferencial, 2 ruedas locas: una por cara).
    Si F=1 (al reves) avanzar/retroceder e izquierda/derecha quedan invertidos.

Todo es Python puro (sin numpy por paso) para que el entrenamiento sea rapido.
"""
import math
import random

import gymnasium as gym
from gymnasium import spaces

# ---------------- Cancha y carrito ----------------
LARGO, ANCHO = 105.0, 68.0        # m, cancha FIFA estandar
DT = 1.0                          # s por accion (= una lectura del GPS NEO-6M)
VEL = 0.3                         # m/s (velocidad constante acordada)
GIRO = math.radians(45)           # rad por accion de giro (ajustar al medir el carrito)
RADIO_CARRO = 0.15                # m
RADIO_META = 2.0                  # m: "llego" si la distancia GPS < 2 m
MAX_PASOS = 600                   # 10 min por intento en entrenamiento
LIMITE_PRUEBA = 240               # 4 min: meta de la prueba real (se reporta)
N_OBSTACULOS = 8

# ---------------- Sensores ----------------
GPS_SESGO_STD = 2.0               # m, error lento tipico del NEO-6M
GPS_SESGO_TAU = 60.0              # s, que tan rapido cambia ese error
GPS_RUIDO = 0.3                   # m, ruido rapido por lectura
GYRO_RUIDO = math.radians(3)      # rad de error por cada giro
GYRO_DERIVA = math.radians(0.1)   # rad/s de deriva tras calibrar
DIST_RUMBO = 4.0                  # m en recta para fijar el rumbo con el GPS
P_DETECTA_GOLPE = 0.9             # el acelerometro detecta el choque
P_FALSO_GOLPE = 0.005             # baches del pasto que parecen choque
N_SIN_AVANCE = 8                  # s seguidos empujando sin que el GPS muestre avance
MIN_AVANCE = 0.4                  # fraccion del avance esperado para no considerarse trabado
PASOS_EVASION = 6                 # s de movimiento recto sin golpes para salir del modo "bloqueado"

# ---------------- Recompensas ----------------
K_PROGRESO = 3.0                  # por metro que disminuye la distancia GPS
R_TIEMPO = -1.0                   # por segundo
R_CHOQUE = -5.0
R_META = 100.0
R_FUERA = -50.0
RECOMPENSAS = ("progreso_tiempo", "progreso", "dispersa")

# ---------------- Discretizacion (debe ser IDENTICA en la tarjeta) ----------------
BORDES_DIST = [5.0, 15.0, 40.0]   # -> 4 rangos de distancia GPS
N_DIST = len(BORDES_DIST) + 1
N_ANG = 8                         # sectores de 45 grados; indice 8 = rumbo desconocido
N_ESTADOS = N_DIST * (N_ANG + 1) * 2 * 2

AVANZAR, RETROCEDER, IZQUIERDA, DERECHA = range(4)
NOMBRES_ACCION = ["Avanzar", "Retroceder", "Izquierda", "Derecha"]
NOMBRES_DIST = ["<5 m", "5-15 m", "15-40 m", ">40 m"]
NOMBRES_ANG = ["al frente", "45 izq", "90 izq", "135 izq", "atras", "135 der", "90 der", "45 der", "rumbo ?"]


def envolver(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def codificar_estado(dist, ang_rel, volteado, bloqueado):
    """dist: m entre los 2 GPS. ang_rel: rumbo a la meta - rumbo del carrito (rad, + = izquierda)
    o None si aun no se conoce el rumbo. volteado/bloqueado: bool."""
    d = sum(dist > b for b in BORDES_DIST)
    if ang_rel is None:
        a = N_ANG
    else:
        a = int(((ang_rel + math.pi / N_ANG) % (2 * math.pi)) // (2 * math.pi / N_ANG)) % N_ANG
    return ((d * (N_ANG + 1) + a) * 2 + int(volteado)) * 2 + int(bloqueado)


def decodificar_estado(s):
    s, b = divmod(s, 2)
    s, f = divmod(s, 2)
    d, a = divmod(s, N_ANG + 1)
    return d, a, f, b


class GPS:
    """GPS barato: sesgo Gauss-Markov que deriva lento + ruido blanco."""

    def __init__(self, rng):
        self.rng = rng
        self.sx, self.sy = rng.gauss(0, GPS_SESGO_STD), rng.gauss(0, GPS_SESGO_STD)

    def leer(self, x, y):
        a = math.exp(-DT / GPS_SESGO_TAU)
        b = math.sqrt(1 - a * a) * GPS_SESGO_STD
        self.sx = a * self.sx + self.rng.gauss(0, b)
        self.sy = a * self.sy + self.rng.gauss(0, b)
        return x + self.sx + self.rng.gauss(0, GPS_RUIDO), y + self.sy + self.rng.gauss(0, GPS_RUIDO)


class CanchaEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, recompensa="progreso_tiempo", n_obstaculos=N_OBSTACULOS):
        assert recompensa in RECOMPENSAS
        self.recompensa = recompensa
        self.n_obstaculos = n_obstaculos
        self.observation_space = spaces.Discrete(N_ESTADOS)
        self.action_space = spaces.Discrete(4)

    # ---------------- mundo ----------------
    def _choca(self, x, y):
        return any(math.hypot(x - ox, y - oy) < r + RADIO_CARRO for ox, oy, r in self.obstaculos)

    def _libre(self, x, y, margen):
        return all(math.hypot(x - ox, y - oy) > r + margen for ox, oy, r in self.obstaculos)

    def reset(self, seed=None, options=None):
        """options (opcional, para pruebas): pos=(x,y), meta=(x,y), theta, volteado, obstaculos=[(x,y,r)]."""
        super().reset(seed=seed)
        o = options or {}
        r = self.rng = random.Random(int(self.np_random.integers(2**31)))
        self.meta = o.get("meta") or (r.uniform(1, LARGO - 1), r.uniform(1, ANCHO - 1))
        self.pos = o.get("pos")
        while self.pos is None or math.dist(self.pos, self.meta) < 5:
            self.pos = (r.uniform(3, LARGO - 3), r.uniform(3, ANCHO - 3))
        self.obstaculos = list(o.get("obstaculos", []))
        while len(self.obstaculos) < self.n_obstaculos and "obstaculos" not in o:
            ob = (r.uniform(0, LARGO), r.uniform(0, ANCHO), r.uniform(0.4, 1.5))
            if math.dist(ob[:2], self.pos) > ob[2] + 3 and math.dist(ob[:2], self.meta) > ob[2] + 3:
                self.obstaculos.append(ob)
        self.theta = o.get("theta", r.uniform(-math.pi, math.pi))  # nariz real (desconocida)
        self.volteado = o.get("volteado", r.random() < 0.5)          # el acelerometro lo sabe
        self.gps1, self.gps2 = GPS(r), GPS(r)
        self.gyro_sesgo = r.gauss(0, GYRO_DERIVA)
        self._leer_gps()
        self.rumbo_est = None
        self.tramo_ini, self.tramo_dir, self.empuje = self.gps_car, None, []
        self.bloqueado, self.libre = False, 0
        self.pasos = 0
        s, self.dist_prev = self._estado()
        return s, {}

    def _leer_gps(self):
        self.gps_car = self.gps1.leer(*self.pos)
        self.gps_meta = self.gps2.leer(*self.meta)

    # -------- sensores -> estado (esto mismo debe correr en la tarjeta) --------
    def _estado(self):
        dx, dy = self.gps_meta[0] - self.gps_car[0], self.gps_meta[1] - self.gps_car[1]
        dist = math.hypot(dx, dy)
        ang = None if self.rumbo_est is None else envolver(math.atan2(dy, dx) - self.rumbo_est)
        return codificar_estado(dist, ang, self.volteado, self.bloqueado), dist

    def step(self, a):
        r = self.rng
        signo = -1 if self.volteado else 1
        recto = a in (AVANZAR, RETROCEDER)
        choque, giro_real = False, 0.0
        # ---- fisica (el carrito real) ----
        if recto:
            self.theta += r.gauss(0, math.radians(2))  # el pasto lo desvia un poco
            v = VEL * DT * (1 if a == AVANZAR else -1) * signo * r.uniform(0.85, 1.1)
            nx = self.pos[0] + v * math.cos(self.theta)
            ny = self.pos[1] + v * math.sin(self.theta)
            if self._choca(nx, ny):
                choque = True  # las ruedas giran pero el obstaculo no lo deja avanzar
            else:
                self.pos = (nx, ny)
        else:
            giro_real = GIRO * (1 if a == IZQUIERDA else -1) * signo * r.uniform(0.8, 1.2)
            self.theta = envolver(self.theta + giro_real)
        self._leer_gps()

        # ---- estimador de rumbo: giroscopio + recta medida por GPS ----
        # (el firmware sabe F por el acelerometro y corrige el signo del giroscopio)
        gyro = giro_real + self.gyro_sesgo * DT + (r.gauss(0, GYRO_RUIDO) if giro_real else 0.0)
        if self.rumbo_est is not None:
            self.rumbo_est = envolver(self.rumbo_est + gyro)
        hacia_nariz = recto and ((a == AVANZAR) != self.volteado)
        if recto:
            if self.tramo_dir != hacia_nariz:
                self.tramo_ini, self.tramo_dir, self.empuje = self.gps_car, hacia_nariz, []
            dx, dy = self.gps_car[0] - self.tramo_ini[0], self.gps_car[1] - self.tramo_ini[1]
            if math.hypot(dx, dy) >= DIST_RUMBO:
                self.rumbo_est = envolver(math.atan2(dy, dx) + (0 if hacia_nariz else math.pi))
                self.tramo_ini = self.gps_car
        else:
            self.tramo_ini, self.tramo_dir, self.empuje = self.gps_car, None, []

        # ---- deteccion de bloqueo sin sensor de distancia ----
        # "bloqueado" = choco hace poco: sigue en 1 hasta recorrer PASOS_EVASION s en recta sin golpes,
        # asi el agente recuerda que esta rodeando un obstaculo y no vuelve a chocar de inmediato.
        if recto:
            golpe = (choque and r.random() < P_DETECTA_GOLPE) or r.random() < P_FALSO_GOLPE
            self.empuje = (self.empuje + [self.gps_car])[-N_SIN_AVANCE:]
            sin_avance = (len(self.empuje) == N_SIN_AVANCE and
                          math.dist(self.empuje[0], self.empuje[-1]) < MIN_AVANCE * (N_SIN_AVANCE - 1) * VEL * DT)
            if golpe or sin_avance:
                self.bloqueado, self.libre, self.empuje = True, 0, []
            else:
                self.libre += 1
                if self.libre >= PASOS_EVASION:
                    self.bloqueado = False

        s, dist = self._estado()
        self.pasos += 1
        fuera = not (0 <= self.pos[0] <= LARGO and 0 <= self.pos[1] <= ANCHO)
        llego = dist < RADIO_META

        # ---- recompensa ----
        rec = 0.0
        if self.recompensa != "dispersa":
            rec += K_PROGRESO * (self.dist_prev - dist)
            rec += R_CHOQUE if choque else 0.0
        if self.recompensa == "progreso_tiempo":
            rec += R_TIEMPO
        rec += R_META if llego else (R_FUERA if fuera else 0.0)
        self.dist_prev = dist

        info = {"llego": llego, "fuera": fuera, "choque": choque, "t": self.pasos * DT,
                "dist_real": math.dist(self.pos, self.meta)}
        return s, rec, llego or fuera, self.pasos >= MAX_PASOS, info

    def foto(self):
        """Todo lo necesario para dibujar un cuadro de la animacion."""
        return {"pos": self.pos, "theta": self.theta, "volteado": self.volteado, "gps_car": self.gps_car,
                "gps_meta": self.gps_meta, "rumbo_est": self.rumbo_est, "bloqueado": self.bloqueado}
