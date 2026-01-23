import gymnasium as gym
import time

def run_simulation(env_name, max_steps=1000):
    print(f"Uruchamianie symulacji dla środowiska: {env_name}")
    env = gym.make(env_name, render_mode='human')
    observation, info = env.reset()
    total_reward = 0

    for step in range(max_steps):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        env.render()

        if terminated or truncated:
            break
        time.sleep(0.01)

    env.close()
    print(f"Zakończono symulację dla {env_name}\n")

if __name__ == "__main__":

    environments = ["InvertedPendulum-v4"]

    for env_name in environments:
        run_simulation(env_name, max_steps=500)
