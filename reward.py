def reward_function(traffic_signal):
    """
    Combina 'Diff Waiting Time' (recompensa vaciar colas) con penalización por espera excesiva.
    Evita valores exponenciales que rompen la red neuronal.
    """
    diff_wait = traffic_signal._diff_waiting_time_reward()
    
    max_wait = 0
    for lane in traffic_signal.lanes:
        wait = traffic_signal.env.sumo.lane.getWaitingTime(lane)
        if wait > max_wait:
            max_wait = wait
    
    penalty = 0
    if max_wait > 40:
        penalty = (max_wait - 40) * 1.5 
        
    reward = (diff_wait - penalty) / 100.0
    
    return reward