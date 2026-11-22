"""Chequeos rapidos del entorno.  Uso: cd sim && uv run python test_cancha_env.py"""
import math

import cancha_env as ce

# Codificacion del estado ida y vuelta
for s in range(ce.N_ESTADOS):
    d, a, f, b = ce.decodificar_estado(s)
    dist = [3, 10, 30, 80][d]
    ang = None if a == ce.N_ANG else a * math.pi / 4
    assert ce.codificar_estado(dist, ang, f, b) == s
assert ce.decodificar_estado(ce.codificar_estado(10, math.radians(-46), 0, 0))[1] == 7

# Al reves: "Avanzar" mueve hacia atras de la nariz e "Izquierda" gira a la derecha
env = ce.CanchaEnv(n_obstaculos=0)
for volteado, signo in [(False, 1), (True, -1)]:
    env.reset(seed=1, options={"pos": (50, 34), "meta": (90, 34), "theta": 0.0, "volteado": volteado})
    env.step(ce.AVANZAR)
    assert signo * (env.pos[0] - 50) > 0.2, (volteado, env.pos)
    t0 = env.theta
    env.step(ce.IZQUIERDA)
    assert signo * ce.envolver(env.theta - t0) > 0, volteado

# El rumbo se estima tras una recta y coincide con el real (+-25 grados), aun al reves
for volteado in (False, True):
    env.reset(seed=2, options={"pos": (50, 34), "meta": (10, 10), "theta": 1.0, "volteado": volteado})
    for _ in range(25):
        env.step(ce.AVANZAR)
    assert env.rumbo_est is not None
    assert abs(ce.envolver(env.rumbo_est - env.theta)) < math.radians(25), math.degrees(env.rumbo_est - env.theta)

# Contra un obstaculo: no avanza y se marca bloqueado
env = ce.CanchaEnv()
env.reset(seed=3, options={"pos": (50, 34), "meta": (90, 34), "theta": 0.0, "volteado": False,
                           "obstaculos": [(51.0, 34.0, 0.6)]})
for _ in range(10):
    env.step(ce.AVANZAR)
assert env.pos[0] < 50.3 and env.bloqueado
env.step(ce.IZQUIERDA)
assert env.bloqueado  # girar no basta: debe moverse en recta sin golpes
for _ in range(ce.PASOS_EVASION):
    env.step(ce.RETROCEDER)
assert not env.bloqueado
print("OK")
