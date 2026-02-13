import json
import time
from .critic import CriticSystem
from .play_generator import PlayGenerator
from .config import DATA_PATH

def main():
    print("Initializing V2 Feedback Loop...")
    
    # Initialize components
    critic = CriticSystem()
    generator = PlayGenerator()
    
    # Define Context (Patriots vs Vikings, 2-High)
    game_context = {
        "down": 1,
        "distance": 10,
        "field_position": "Own 35",
        "field_position_yards": 35,
        "off_team_code": "NE",
        "defense_personnel": "Ni (4-2-5) // 2-High Shell",
        "coverage": "Cover 2"
    }
    
    memory = []
    memory = []
    max_loops = 10
    
    print(f"Starting Loop. Max Iterations: {max_loops}")
    print("-" * 50)
    
    final_output = None
    
    for i in range(max_loops):
        print(f"\n--- Iteration {i+1} ---")
        
        # 1. Generate
        print("Generating play...")
        play = generator.generate(game_context, memory)
        if not play:
            print("Generation failed. Stopping.")
            break
            
        print(f"Proposed Play: {play.get('Play Name')} ({play.get('formation')})")
        
        # 2. Critique
        print("Running Critic Models...")
        critique = critic.evaluate(play, game_context)
        print(f"Scores -> Predictability: {critique['predictability_score']}, Exp. Yards: {critique['expected_yards']}")
        print(f"Feedback: {critique['critique']}")
        
        # 3. Update Memory (With Self-Reflection)
        print("Synthesizing Rationale...")
        rationale = generator.analyze_feedback(play, critique, game_context)
        print(f"LLM Rationale: {rationale}\n")
        
        iteration_data = {
            "iteration": i + 1,
            "play": play,
            "critique": critique,
            "rationale": rationale
        }
        memory.append(iteration_data)
        
        # 4. Check Termination Condition (SKIPPED - User requested full 3 loops)
        # Simply print status but do not break
        p_score = critique['predictability_score']
        ey = critique['expected_yards']
        
        print(f"   -> Recorded. P={p_score:.4f}, EY={ey:.2f}")

    # Select Best Play from Memory
    print("-" * 50)
    print("Loop Completed. Selecting Best Play...")
    
    # Selection Logic: Maximize Expected Yards while Penalizing High Predictability
    # Score = Expected_Yards - (Predictability_Score * 5)
    # Examples:
    #   Play A: EY=6.0, P=0.1 (Pred) -> 6.0 - 0.5 = 5.5
    #   Play B: EY=5.5, P=-0.1 (Unpred) -> 5.5 - (-0.5) = 6.0 -> Wins!
    
    def score_play(item):
        c = item['critique']
        return c['expected_yards'] - (c['predictability_score'] * 5)
        
    final_output = max(memory, key=score_play)
    best_score = score_play(final_output)
    print(f"Selected Iteration {final_output['iteration']} with Composite Score: {best_score:.4f}")

    # Create Summary Object
    summary_item = {
        "type": "summary",
        "best_iteration": final_output['iteration'],
        "score": best_score,
        "play": final_output['play'],
        "critique": final_output['critique'],
        "rationale": (
            f"This play achieved the highest Composite Score of {best_score:.4f}. "
            f"It balanced a high Expected Yards value ({final_output['critique']['expected_yards']:.2f}) "
            f"with a favorable Predictability Score ({final_output['critique']['predictability_score']:.4f}). "
            f"The combination of {final_output['play']['personnel']} personnel and "
            f"{final_output['play']['personnel_formation']} effectively optimized the trade-off between efficiency and novelty."
        )
    }
    
    # Append to memory for trace
    memory.append(summary_item)

    # Save Trace
    with open("v2_feedback_trace.json", "w") as f:
        json.dump(memory, f, indent=2)
        
    print("-" * 50)
    print("Final Selected Play:")
    print(json.dumps(final_output['play'], indent=2))
    print("Scores:", final_output['critique'])
    print(f"Trace saved to v2_feedback_trace.json")

if __name__ == "__main__":
    main()
