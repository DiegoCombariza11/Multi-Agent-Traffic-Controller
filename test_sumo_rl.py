import sumo_rl
import supersuit as ss
from stable_baselines3 import DQN

def test():
    # Configuración idéntica pero con GUI=True
    env = sumo_rl.SumoEnvironment(
        net_file="TestLightsSogamosoNet.net.xml",
        route_file="osm.passenger.trips.xml",
        out_csv_name="test_results",
        use_gui=True, # <--- VERLO
        num_seconds=3600,
        min_green=10,
        delta_time=5,
        reward_function='max-pressure'
    )
    
    # Aplicar mismos wrappers
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, base_class="stable_baselines3")

    # Cargar modelo
    model = DQN.load("sumo_rl_final_model")

    obs = env.reset()
    done = False
    
    print("Corriendo simulación...")
    while not done:
        # El modelo predice la acción determinística (sin exploración)
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

if __name__ == "__main__":
    test()