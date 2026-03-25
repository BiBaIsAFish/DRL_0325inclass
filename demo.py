import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class BanditSimulation:
    def __init__(self, true_probs, steps=100000, initial_money=10000):
        self.true_probs = true_probs
        self.k = len(true_probs)
        self.steps = steps
        self.initial_money = initial_money

    def pull_arm(self, action):
        return 1 if np.random.rand() < self.true_probs[action] else 0

    def run_ab_testing(self):
        rewards = np.zeros(self.steps)
        for t in range(self.steps):
            action = np.random.randint(self.k)
            rewards[t] = self.pull_arm(action)
        return rewards

    def run_epsilon_greedy(self, epsilon=0.1):
        q_values = np.zeros(self.k)
        action_counts = np.zeros(self.k)
        rewards = np.zeros(self.steps)
        for t in range(self.steps):
            if np.random.rand() < epsilon:
                action = np.random.randint(self.k)
            else:
                action = np.argmax(q_values)
            reward = self.pull_arm(action)
            rewards[t] = reward
            action_counts[action] += 1
            q_values[action] += (reward - q_values[action]) / action_counts[action]
        return rewards

    def run_optimistic_initial_values(self, initial_value=5.0):
        q_values = np.full(self.k, initial_value)
        action_counts = np.zeros(self.k)
        rewards = np.zeros(self.steps)
        for t in range(self.steps):
            action = np.argmax(q_values)
            reward = self.pull_arm(action)
            rewards[t] = reward
            action_counts[action] += 1
            q_values[action] += (reward - q_values[action]) / action_counts[action]
        return rewards

    def run_softmax(self, tau=0.1):
        q_values = np.zeros(self.k)
        action_counts = np.zeros(self.k)
        rewards = np.zeros(self.steps)
        for t in range(self.steps):
            exp_q = np.exp(q_values / tau)
            probs = exp_q / np.sum(exp_q)
            action = np.random.choice(self.k, p=probs)
            reward = self.pull_arm(action)
            rewards[t] = reward
            action_counts[action] += 1
            q_values[action] += (reward - q_values[action]) / action_counts[action]
        return rewards

    def run_ucb(self, c=2.0):
        q_values = np.zeros(self.k)
        action_counts = np.zeros(self.k)
        rewards = np.zeros(self.steps)
        for t in range(self.steps):
            if t < self.k:
                action = t
            else:
                ucb_values = q_values + c * np.sqrt(np.log(t) / action_counts)
                action = np.argmax(ucb_values)
            reward = self.pull_arm(action)
            rewards[t] = reward
            action_counts[action] += 1
            q_values[action] += (reward - q_values[action]) / action_counts[action]
        return rewards

    def run_thompson_sampling(self):
        alpha = np.ones(self.k)
        beta = np.ones(self.k)
        rewards = np.zeros(self.steps)
        for t in range(self.steps):
            samples = np.random.beta(alpha, beta)
            action = np.argmax(samples)
            reward = self.pull_arm(action)
            rewards[t] = reward
            if reward == 1:
                alpha[action] += 1
            else:
                beta[action] += 1
        return rewards

def main():
    st.set_page_config(page_title="Multi-Armed Bandit Simulation", layout="wide")
    st.title("🎰 Multi-Armed Bandit Algorithms Comparison")
    
    true_probs = [0.8, 0.7, 0.5]
    steps = 100000
    num_runs = 10 # 在網頁上為了避免等待太久，我先將平均次數調降為 10 次，你可以視情況調回 50
    
    # 使用 st.spinner 讓使用者知道程式正在運算
    with st.spinner(f'Running simulations ({num_runs} runs)... Please wait.'):
        results = {
            "A/B Testing (Random)": np.zeros(steps),
            r"$\epsilon$-greedy Action": np.zeros(steps),
            "Optimistic Initial Values": np.zeros(steps),
            "Softmax (Boltzmann)": np.zeros(steps),
            "UCB Action": np.zeros(steps),
            "Thompson Sampling": np.zeros(steps)
        }
        
        for _ in range(num_runs):
            sim = BanditSimulation(true_probs, steps=steps)
            results["A/B Testing (Random)"] += sim.run_ab_testing()
            results[r"$\epsilon$-greedy Action"] += sim.run_epsilon_greedy(epsilon=0.1)
            results["Optimistic Initial Values"] += sim.run_optimistic_initial_values(initial_value=5.0)
            results["Softmax (Boltzmann)"] += sim.run_softmax(tau=0.1)
            results["UCB Action"] += sim.run_ucb(c=2.0)
            results["Thompson Sampling"] += sim.run_thompson_sampling()

    # --- 1. 繪製並顯示折線圖 ---
    st.subheader("Action Compare (Cumulative Average Reward)")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for label, reward_history in results.items():
        avg_reward_history = reward_history / num_runs
        cumulative_average = np.cumsum(avg_reward_history) / (np.arange(1, steps + 1))
        ax.plot(cumulative_average, label=label)

    ax.set_title('Action Compare')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Average Reward')
    ax.legend(loc='lower right')
    
    # 將圖表輸出到 Streamlit
    st.pyplot(fig)

    # --- 2. 製作並顯示資料表格 ---
    st.subheader("Performance Summary Table")
    optimal_reward = steps * max(true_probs)
    
    table_info = {
        "A/B Testing (Random)": {"name": "A/B Test", "style": "Static", "notes": "Simple but wasteful"},
        "Optimistic Initial Values": {"name": "Optimistic", "style": "Implicit", "notes": "Front-loaded exploration"},
        r"$\epsilon$-greedy Action": {"name": "ε-Greedy", "style": "Random", "notes": "Easy baseline"},
        "Softmax (Boltzmann)": {"name": "Softmax", "style": "Probabilistic", "notes": "Smooth control"},
        "UCB Action": {"name": "UCB", "style": "Confidence-based", "notes": "Efficient"},
        "Thompson Sampling": {"name": "Thompson", "style": "Bayesian", "notes": "Best practical"}
    }
    
    table_data = []
    for label, reward_history in results.items():
        total_reward = np.sum(reward_history) / num_runs
        regret = optimal_reward - total_reward
        info = table_info[label]
        
        table_data.append({
            "Method": info['name'],
            "Exploration Style": info['style'],
            "Total Reward": int(total_reward), # 轉成整數比較好看
            "Regret": int(regret),             # 轉成整數比較好看
            "Notes": info['notes']
        })
        
    # 轉換成 Pandas DataFrame 並顯示
    df = pd.DataFrame(table_data)
    
    # 隱藏 DataFrame 預設的 index (0, 1, 2...) 讓表格看起來更乾淨
    st.dataframe(df, hide_index=True, use_container_width=True)

if __name__ == "__main__":
    main()