from pydantic import BaseModel, Field

class TrainingInput(BaseModel):
    """Input ORM to Serialize Model Hyper-Parameters"""

    # Optimizer Settings

    """Optimizer Learning Rate"""
    LR_START: float
    """Terminating Optimizer Learning Rate"""
    LR_END: float
    """Optimizer Learning Decay Rate"""
    LR_DECAY: float
    
    # Greedy Algorithm Setting

    """Starting Epsilon-Greedy Value (exploration)"""
    EPS_START: float
    """Terminating Epsilon-Greedy Value (exploration)"""
    EPS_END: float
    """Epsilon-Greedy Decay Rate"""
    EPS_DECAY: float
    """Discount factor for Q-learning Reward"""
    GAMMA: float

    # Model Settings

    """Training Batch Size"""
    BATCH_SIZE: int
    """Model Reward Function"""
    REWARD_FUNCTION: str
    """Q-Function Approximating Model"""
    MODEL_SELECT: str
    """Switch betweening using stochastic or determinstic policy"""
    STOCHASTIC: bool

    # Training and Benchmark Settings

    """Number of Epochs in training loop"""
    TRAINING_LOOPS: int
    """Number of trajectory episodes per epoch"""
    EPISODES_PER_LOOP: int
    """Number of Train/Validation sections per Benchmark"""
    GAMES_PER_BENCHMARK: int

    MEMORY: int
    TOTAL_ACTIONS: int = Field(default=((21*11*4) + 36))

    MLFLOW_ADDRESS: str = Field(default='mlflow:8250')
