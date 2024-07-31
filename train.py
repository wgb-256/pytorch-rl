# train.py
import gym
import os
from dqn_agent import DQNAgent
from utils import create_gif

def train(env_name, episodes, batch_size, gif_frequency=50):
    env = gym.make(env_name, render_mode="rgb_array")
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)

    os.makedirs("./gifs", exist_ok=True)

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        truncated = False
        frames = []  # To store frames for GIF creation

        while not (done or truncated):
            action = agent.act(state)
            next_state, reward, done, truncated, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(agent.memory) > batch_size:
                loss = agent.replay(batch_size)
                agent.writer.add_scalar('Loss', loss, episode)

            # Capture frame for GIF
            if (episode + 1) % gif_frequency == 0:
                frame = env.render()
                frames.append(frame)

        agent.writer.add_scalar('Reward', total_reward, episode)
        print(f"Episode: {episode+1}/{episodes}, Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

        # Create GIF after specified episodes
        if (episode + 1) % gif_frequency == 0:
            create_gif(frames, f"./gifs/episode_{episode+1}.gif")

    env.close()
    agent.writer.close()

if __name__ == "__main__":
    train("CartPole-v1", episodes=500, batch_size=32, gif_frequency=50)