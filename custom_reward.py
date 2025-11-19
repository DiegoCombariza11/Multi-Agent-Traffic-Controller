import numpy as np

def fairness_reward(traffic_signal):
    """
    Recompensa híbrida: Presión + Castigo por Espera Máxima.
    Si ALGUIEN espera demasiado, el agente sufre una penalización masiva.
    """
    ts_pressure = traffic_signal.get_pressure()

    max_wait = 0
    for lane in traffic_signal.lanes:
        wait = traffic_signal.env.sumo.lane.getWaitingTime(lane)
        if wait > max_wait:
            max_wait = wait
            
    penalty = 0
    if max_wait > 40:
        penalty = (max_wait - 60) ** 2  # Al cuadrado para que duela mucho
        

    return -ts_pressure - penalty