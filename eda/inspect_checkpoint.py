import torch
from v2.config import PREDICTABILITY_MODEL_PATH

def inspect_checkpoint():
    try:
        checkpoint = torch.load(PREDICTABILITY_MODEL_PATH, map_location='cpu')
        print(f"Checkpoint Type: {type(checkpoint)}")
        if isinstance(checkpoint, dict):
            print(f"Root Keys: {list(checkpoint.keys())}")
            
            # Check Root Keys directly
            if 'off_team_to_idx' in checkpoint:
                print(f"Found 'off_team_to_idx' in Root.")
                oti = checkpoint['off_team_to_idx']
                print(f"NE Index: {oti.get('NE')}")
                print(f"NE_Space Index: {oti.get('NE ')}")
                
                # Print all keys to be sure
                print(f"All Keys: {list(oti.keys())}")
            else:
                print("'off_team_to_idx' NOT found in Root.")

            if 'formation_to_idx' in checkpoint:
                print(f"Found 'formation_to_idx' in Root.")
                print(f"Trips Right Index: {checkpoint['formation_to_idx'].get('Trips Right')}")
            else:
                print("'formation_to_idx' NOT found in Root.")
                
        else:
            print("Checkpoint is not a dict.")
            
    except Exception as e:
        print(f"Error loading checkpoint: {e}")

if __name__ == "__main__":
    inspect_checkpoint()
