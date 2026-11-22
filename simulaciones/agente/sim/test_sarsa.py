"""Chequeos rapidos de sarsa.py.  Uso: cd sim && uv run python test_sarsa.py"""
import sarsa

# Expected SARSA: con eps=0 es el maximo; con eps es eps*promedio + (1-eps)*maximo
assert sarsa.valor_esperado([1, 2, 3, 6], 0.0) == 6
assert abs(sarsa.valor_esperado([1, 2, 3, 6], 0.2) - (0.2 * 3 + 0.8 * 6)) < 1e-12

# alpha de la Tarea 9: alpha0 / (1 + 0.0005 ep)  ->  a los 2000 episodios es la mitad
assert abs(sarsa.tasa({**sarsa.BASE, "alpha_decae": "clase"}, 2000) - 0.05) < 1e-12

# cada variante entrena sin errores y guarda el retorno descontado por episodio
for extra in (dict(algoritmo="esarsa"), dict(eps="ucb"), dict(eps=0.0, q0=50.0), dict(alpha_decae=False)):
    ag, h = sarsa.entrenar({"episodios": 30, **extra})
    assert len(h["retorno"]) == 30, extra
print("OK")
