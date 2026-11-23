"""SARSA tabular (Unidad 2) y SARSA semi-gradiente lineal (Unidad 3) para CanchaEnv.

Tambien: evaluacion greedy, experimentos en paralelo y exportacion a CSV.
"""
import copy
import csv
import math
import os
import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import cancha_env as ce

# algoritmo: "sarsa" | "esarsa" (Expected SARSA).  eps: numero fijo | "decreciente" | "ucb".
# alpha_decae: False | "lineal" | "clase" (alpha0 / (1 + 0.0005 ep), Tarea 9).
BASE = dict(metodo="tabular", algoritmo="sarsa", alpha=0.1, alpha_decae="lineal", gamma=0.95, eps="decreciente",
            c_ucb=2.0, q0=0.0, recompensa="progreso_tiempo", episodios=8000, semilla=0, validar_cada=0,
            n_validacion=200)


def epsilon(cfg, ep):
    if cfg["eps"] == "decreciente":  # de 1.0 a 0.05 durante el 70% de los episodios
        return max(0.05, 1.0 - 0.95 * ep / (0.7 * cfg["episodios"]))
    return cfg["eps"]


def tasa(cfg, ep):
    if cfg["alpha_decae"] == "lineal":  # baja hasta 10% al final: estabiliza Q con recompensas ruidosas (GPS)
        return cfg["alpha"] * max(0.1, 1 - ep / cfg["episodios"])
    if cfg["alpha_decae"] == "clase":
        return cfg["alpha"] / (1 + 0.0005 * ep)
    return cfg["alpha"]


def valor_esperado(q, eps):
    """Expected SARSA: sum_a pi(a|s') Q(s',a) con pi epsilon-greedy (eps/|A| a cada una, 1-eps extra a la mejor)."""
    return eps * sum(q) / len(q) + (1 - eps) * max(q)


# ---------------- Unidad 3: caracteristicas lineales (apiladas por accion) ----------------
# x(s): sesgo + one-hot de distancia + (angulo x volteado) + (bloqueado x volteado)
N_CARACT = 1 + ce.N_DIST + (ce.N_ANG + 1) * 2 + 2 * 2


def caracteristicas(s):
    d, a, f, b = ce.decodificar_estado(s)
    return (0, 1 + d, 1 + ce.N_DIST + a * 2 + f, 1 + ce.N_DIST + (ce.N_ANG + 1) * 2 + b * 2 + f)


class Agente:
    """Q tabular, o q(s,a,w) = w_a . x(s) (x binaria: se guardan solo los indices activos)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.alpha = cfg["alpha"]
        if cfg["metodo"] == "tabular":
            self.Q = [[cfg["q0"]] * 4 for _ in range(ce.N_ESTADOS)]
        else:
            self.W = [[0.0] * 4 for _ in range(N_CARACT)]
            self.W[0] = [cfg["q0"]] * 4  # el sesgo hace de inicializacion optimista
        if cfg["eps"] == "ucb":
            self.N, self.t = [[0] * 4 for _ in range(ce.N_ESTADOS)], 0

    def q(self, s):
        if self.cfg["metodo"] == "tabular":
            return self.Q[s]
        x = caracteristicas(s)
        return [sum(self.W[i][a] for i in x) for a in range(4)]

    def mejor(self, s):
        q = self.q(s)
        return q.index(max(q))

    def actualizar(self, s, a, objetivo):
        if self.cfg["metodo"] == "tabular":
            self.Q[s][a] += self.alpha * (objetivo - self.Q[s][a])
        else:
            # semi-gradiente: w <- w + alpha [objetivo - q(s,a,w)] x(s)   (solo la fila de la accion a)
            x = caracteristicas(s)
            paso = self.alpha / len(x) * (objetivo - sum(self.W[i][a] for i in x))
            for i in x:
                self.W[i][a] += paso

    def tabla(self):
        return np.array([self.q(s) for s in range(ce.N_ESTADOS)])


def entrenar(cfg):
    """SARSA on-policy (o Expected SARSA) con epsilon-greedy o UCB. Devuelve (agente, historial)."""
    cfg = {**BASE, **cfg}
    assert not (cfg["algoritmo"] == "esarsa" and cfg["eps"] == "ucb"), "Expected SARSA necesita pi(a|s): usar epsilon"
    env = ce.CanchaEnv(recompensa=cfg["recompensa"])
    rng = random.Random(cfg["semilla"])
    ag = Agente(cfg)
    g = cfg["gamma"]
    hist = {"recompensa": [], "retorno": [], "llego": [], "tiempo": []}

    def elegir(s, eps):
        if cfg["eps"] == "ucb":  # Unidad 1: argmax Q + c sqrt(ln t / N(s,a)); primero las nunca probadas
            ag.t += 1
            n, q = ag.N[s], ag.q(s)
            a = n.index(0) if 0 in n else max(range(4), key=lambda b: q[b] + cfg["c_ucb"] * math.sqrt(math.log(ag.t) / n[b]))
            n[a] += 1
            return a
        return rng.randrange(4) if rng.random() < eps else ag.mejor(s)

    for ep in range(cfg["episodios"]):
        eps = epsilon(cfg, ep)
        ag.alpha = tasa(cfg, ep)
        s, _ = env.reset(seed=cfg["semilla"] * 1_000_003 + ep)
        a = elegir(s, eps)
        total, G, desc = 0.0, 0.0, 1.0
        while True:
            s2, r, terminado, truncado, info = env.step(a)
            a2 = elegir(s2, eps)
            if terminado:
                objetivo = r
            elif cfg["algoritmo"] == "esarsa":  # R + gamma sum_a pi(a|S') Q(S',a)
                objetivo = r + g * valor_esperado(ag.q(s2), eps)
            else:  # SARSA: R + gamma Q(S',A')
                objetivo = r + g * ag.q(s2)[a2]
            ag.actualizar(s, a, objetivo)
            s, a, total = s2, a2, total + r
            G += desc * r  # G0 = sum_k gamma^k R_{k+1}
            desc *= g
            if terminado or truncado:
                break
        hist["recompensa"].append(total)
        hist["retorno"].append(G)
        hist["llego"].append(info["llego"])
        hist["tiempo"].append(info["t"])
        # con GPS ruidoso la tabla oscila: se valida cada tanto (lanzamientos distintos a los de prueba)
        # y se conserva la mejor tabla, en vez de la ultima
        if cfg["validar_cada"] and (ep + 1) % cfg["validar_cada"] == 0:
            val = evaluar(ag, n=cfg["n_validacion"], seed0=5 * 10**6)["todos"]["llega_%"]
            hist.setdefault("validacion", []).append((ep + 1, val))
            if val >= max(v for _, v in hist["validacion"]):
                mejor = copy.deepcopy(ag.__dict__)
    if cfg["validar_cada"]:
        ag.__dict__.update(mejor)
    return ag, hist


def correr(ag, seed, options=None, guardar=False):
    """Un lanzamiento con la politica greedy (epsilon=0), como correra en la tarjeta."""
    env = ce.CanchaEnv()
    s, _ = env.reset(seed=seed, options=options)
    d0 = math.dist(env.pos, env.meta)
    fotos, acciones = [env.foto()], []
    while True:
        a = ag.mejor(s)
        s, _, terminado, truncado, info = env.step(a)
        if guardar:
            fotos.append(env.foto())
            acciones.append(a)
        if terminado or truncado:
            info.update(d0=d0, volteado=env.volteado)
            return (info, env, fotos, acciones) if guardar else info


def evaluar(ag, n=500, seed0=10**7):
    return resumir([correr(ag, seed0 + i) for i in range(n)])


def resumir(filas):
    def m(f):
        sub = [x for x in filas if f(x)]
        ok = [x for x in sub if x["llego"]]
        n = max(1, len(sub))
        prom = lambda v: float(np.mean(v)) if ok else float("nan")
        return {
            "llega_%": 100 * len(ok) / n,
            "llega_en_4min_%": 100 * sum(x["t"] <= ce.LIMITE_PRUEBA for x in ok) / n,
            "se_sale_%": 100 * sum(x["fuera"] for x in sub) / n,
            "tiempo_medio_s": prom([x["t"] for x in ok]),
            "dist_real_final_m": prom([x["dist_real"] for x in ok]),
            # 100% = tardo lo mismo que ir en linea recta sin girar
            "tiempo_vs_ideal_%": prom([100 * x["t"] * ce.VEL / x["d0"] for x in ok]),
        }
    return {"todos": m(lambda x: True), "derecho": m(lambda x: not x["volteado"]),
            "al_reves": m(lambda x: x["volteado"]), "filas": filas}


def _experimento(cfg):
    ag, hist = entrenar(cfg)
    return cfg, hist, evaluar(ag, n=400)["todos"]


def experimentos(cfgs, procesos=None):
    """Entrena varias configuraciones en paralelo (una por nucleo)."""
    cfgs = [{**BASE, **c} for c in cfgs]
    with ProcessPoolExecutor(max_workers=procesos or min(len(cfgs), os.cpu_count())) as ex:
        return list(ex.map(_experimento, cfgs))


# ---------------- Exportacion ----------------
def exportar(ag, carpeta):
    os.makedirs(carpeta, exist_ok=True)
    Q = ag.tabla()
    np.savetxt(os.path.join(carpeta, "q_table.csv"), Q, delimiter=",", fmt="%.4f")
    with open(os.path.join(carpeta, "politica.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["estado", "dist_idx", "ang_idx", "volteado", "bloqueado", "accion", "accion_nombre",
                    "Q_avanzar", "Q_retroceder", "Q_izquierda", "Q_derecha"])
        for s in range(ce.N_ESTADOS):
            d, a, fl, b = ce.decodificar_estado(s)
            acc = int(np.argmax(Q[s]))
            w.writerow([s, d, a, fl, b, acc, ce.NOMBRES_ACCION[acc], *[f"{v:.4f}" for v in Q[s]]])
    if ag.cfg["metodo"] != "tabular":
        np.savetxt(os.path.join(carpeta, "pesos_semigradiente.csv"), np.array(ag.W), delimiter=",", fmt="%.6f")
