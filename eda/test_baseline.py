import torch
import numpy as np
from v2.critic_model_def import PredictabilityLSTM, FORMATION_LIST, SEQ_LEN, n_off_teams
from v2.config import PREDICTABILITY_MODEL_PATH

def test_baseline():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Model
    checkpoint = torch.load(PREDICTABILITY_MODEL_PATH, map_location=device)
    model = PredictabilityLSTM().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create Zero Inputs (No History)
    seq = torch.zeros((1, SEQ_LEN, 10), dtype=torch.float32).to(device)  # Zero history
    drive_context = torch.zeros((1, 25), dtype=torch.float32).to(device) # Zero drive context
    
    # Create Standard Game State (1st & 10 at 35 yard line)
    # Norms: Down (0-3 -> /3), ToGo (1-99 -> /18), StartYard (0-100 -> /100)
    gs = np.zeros(6, dtype=np.float32)
    gs[0] = (1 - 1) / 3.0    # 1st Down (-1 to 0-index)
    gs[1] = (10 - 1) / 18.0  # 10 to go
    gs[2], gs[3], gs[4] = 0.0, 1.0, 0.0 # Time/Lead placeholders
    gs[5] = 35 / 100.0       # 35 yard line
    
    t_gs = torch.from_numpy(gs).unsqueeze(0).to(device)
    t_id = torch.tensor([0], dtype=torch.long).to(device) # Team 0
    
    # Run Inference
    with torch.no_grad():
        _, formation_logits = model(seq, drive_context, t_gs, t_id)
        probs = torch.softmax(formation_logits, dim=1).cpu().numpy()[0]
        
    print("\nBaseline Probabilities (1st & 10, Zero History):")
    pairs = list(zip(FORMATION_LIST, probs))
    pairs.sort(key=lambda x: x[1], reverse=True)
    
    for fmt, p in pairs:
        print(f"{fmt}: {p:.4f}")
        
    print(f"\nMax Prob: {max(probs):.4f}")
    print(f"Mean Prob: {np.mean(probs):.4f}")

if __name__ == "__main__":
    test_baseline()
