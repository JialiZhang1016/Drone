# Drone Route Planning with Reinforcement Learning

This project focuses on developing a reinforcement learning (RL) agent for optimal drone route planning. The agent aims to maximize data collection from various locations while considering factors such as weather conditions, flight time constraints, and criticality levels of locations.

## Installation

### Prerequisites

- Python 3.7 or higher
- `virtualenv` or `venv` for creating a virtual environment

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/JialiZhang1016/Drone.git
   cd Drone
   ```
2. **Create a Virtual Environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use 'venv\Scripts\activate'
   ```
3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Configuration

Before running the training script, you can adjust the parameters in the configuration files located in the `config` directory. Parameters include:

- Number of locations (`num_locations`)
- Maximum flight time (`T_max`)
- Weather probabilities (`weather_prob`)
- Flight times between locations (`T_flight_good`, `T_flight_bad`)
- Data collection time bounds (`T_data_lower`, `T_data_upper`)
- Criticality levels (`criticality`)

### Training the Agent

To train the DQN agent:

```bash
python train.py
```
The results will be saved in the `runs` directory, named with the format `num_locations_num_episodes_timestamp`.

### Plotting Results

To visualize training results:

```bash
python plot_results.py runs/5_500_YYYYMMDD-HHMMSS
```

This will generate plots for:

- Total Reward per Episode
- Average Loss per Episode
- Episode Length
- Epsilon Decay
- Success Rate
- Average Reward
- Moving Average Reward

Plots will be saved in the corresponding `runs` directory.

### Evaluating the Agent

To evaluate a trained agent:

```bash
python evaluate.py
```

## Agents

### DQN Agent

Located at `agent/dqn_agent.py`, this agent uses a Deep Q-Network to learn the optimal policy for route planning.

### Random Agent

Located at `agent/random_agent.py`, this agent selects actions randomly and serves as a baseline for comparison.

## Environment

The custom drone environment is defined in `drone_env.py`, following the OpenAI Gym interface. Key components include:

- **Observation Space**: Includes current location, remaining time, visited locations, and weather conditions.
- **Action Space**: Consists of the next location to visit and the data collection time at that location.
- **Reward Function**: Calculates rewards based on data collected and costs incurred.
- **Constraints**: Enforces flight time limitations, data collection time bounds, and safety considerations.

## Results

Training results and model checkpoints are stored in the `runs` directory. Each run folder contains:

- Saved model (`policy_net.pth`)
- Numpy arrays of recorded metrics (e.g., `all_rewards.npy`, `all_losses.npy`)
- Plots of the training metrics (if generated)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

