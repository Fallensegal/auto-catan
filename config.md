# Auto-Catan Configs

## Config One - Small Pooling

```json
{
  "LR_START": 0.0001,
  "LR_END": 0.00001,
  "LR_DECAY": 200000,
  "EPS_START": 1,
  "EPS_END": 0.05,
  "EPS_DECAY": 20000,
  "GAMMA": 0.95,
  "BATCH_SIZE": 32,
  "REWARD_FUNCTION": "Incremental_All",
  "MODEL_SELECT": "Small_Pooling",
  "STOCHASTIC": true,
  "TRAINING_LOOPS": 50,
  "EPISODES_PER_LOOP": 30,
  "GAMES_PER_BENCHMARK": 100,
  "MEMORY": 10000,
  "TOTAL_ACTIONS": 960,
  "MLFLOW_ADDRESS": "http://mlflow:8250",
  "PRINT_ACTIONS": false
}
```


## Config Two - Small

```json
{
  "LR_START": 0.000003,
  "LR_END": 0.00001,
  "LR_DECAY": 200000,
  "EPS_START": 1,
  "EPS_END": 0.05,
  "EPS_DECAY": 20000,
  "GAMMA": 0.999,
  "BATCH_SIZE": 32,
  "REWARD_FUNCTION": "Incremental_All",
  "MODEL_SELECT": "Small",
  "STOCHASTIC": true,
  "TRAINING_LOOPS": 50,
  "EPISODES_PER_LOOP": 30,
  "GAMES_PER_BENCHMARK": 100,
  "MEMORY": 10000,
  "TOTAL_ACTIONS": 960,
  "MLFLOW_ADDRESS": "http://mlflow:8250",
  "PRINT_ACTIONS": false
}

```