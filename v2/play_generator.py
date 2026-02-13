import os
import json
from openai import OpenAI
from .config import SYSTEM_PROMPT_PATH, OPENAI_MODEL, OPENAI_API_KEY

class PlayGenerator:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        with open(SYSTEM_PROMPT_PATH, 'r') as f:
            self.system_prompt = f.read()
        self.model = OPENAI_MODEL

    def generate(self, game_context, memory):
        """
        Generates a play using OpenAI API, considering memory/critic feedback.
        """
        
        # Construct User Prompt
        input_vars = f"""
        Current Game Situation:
        - Down: {game_context.get('down')}
        - Distance: {game_context.get('distance')}
        - Field Position: {game_context.get('field_position')}
        - Defense: {game_context.get('defense_personnel', 'Unknown')}
        """
        
        memory_text = ""
        if memory:
            memory_text = "\n\nCRITICAL FEEDBACK HISTORY (Previously rejected/refined plays):\n"
            for i, item in enumerate(memory):
                play = item['play']
                critique = item['critique']
                scores = f"Predictability Score: {critique.get('predictability_score')}, Expected Yards: {critique.get('expected_yards')}"
                feedback = critique.get('critique', '')
                
                memory_text += f"""
                Attempt #{i+1}:
                - Play Name: {play.get('Play Name', 'Unnamed')}
                - Formation: {play.get('formation', 'Unknown')}
                - Scores: {scores}
                - FEEDBACK: {feedback}
                """
            
            memory_text += "\n\nINSTRUCTION: You must prioritization the CRITIC FEEDBACK above all else. If the predictability score was high, CHANGE the formation or concept drastically. Do not repeat mistakes."

        user_content = input_vars + memory_text

        # Call API
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            
            # Simple JSON extraction (Naive)
            # Find first { and last }
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end != -1:
                json_str = content[start:end]
                play_json = json.loads(json_str)
                
                # --- PRUNING LOGIC START ---
                play_json = self._prune_to_11_personnel(play_json)
                # --- PRUNING LOGIC END ---
                
                return play_json
            else:
                print("Error: No JSON found in response.")
                return None
                
        except Exception as e:
            print(f"Error calling OpenAI: {e}")
            return None

    def _prune_to_11_personnel(self, play_json):
        """
        Enforce Rule of 11: Exactly 5 Skill Players.
        Prioritizes keeping RB, TE, L1, R1.
        Drops L3, R3, L2, R2 in that order if over limit.
        """
        routes = play_json.get("route_responsibilities", {})
        if len(routes) <= 5:
            return play_json

        print(f"   [Constraint Violation] Generated {len(routes)} players. Pruning to 5.")
        
        # Priority Queue (Higher score = Keep)
        priority = {
            "RB": 100,
            "TE": 90,
            "L1": 80,
            "R1": 80,
            "L2": 60,
            "R2": 60,
            "L3": 20,
            "R3": 20
        }
        
        # Sort keys by priority desc
        sorted_keys = sorted(routes.keys(), key=lambda k: priority.get(k, 0), reverse=True)
        
        # Keep top 5
        kept_keys = sorted_keys[:5]
        
        new_routes = {k: routes[k] for k in kept_keys}
        play_json["route_responsibilities"] = new_routes
        
        return play_json

    def analyze_feedback(self, play_json, critique, game_context):
        """
        Asks the LLM to explain WHY the critic gave the scores it did.
        """
        prompt = f"""
        You are an elite Offensive Coordinator analyzing the results of a play simulation.
        
        CONTEXT:
        - Scenario: {game_context}
        - Play Designed: {play_json.get('play_name')} ({play_json.get('personnel_formation')})
        - Route Concept: {play_json.get('ey_ps_analysis')}
        
        CRITIC FEEDBACK (The Truth):
        - Predictability Score: {critique.get('predictability_score')} (Target: < 0.0, Low is Unpredictable)
        - Expected Yards: {critique.get('expected_yards')} (Target: > 5.0)
        - Assessment: {critique.get('critique')}
        
        TASK:
        Explain WHY the models evaluated the play this way. 
        - If Predictability is High (> 0.05), explain what specific formation or tendency caused it.
        - If Expected Yards is Low (< 4.5), explain which routes failed to stretch the defense or were too short.
        - Synthesize a "Lesson Learned" for the next iteration.
        
        OUTPUT FORMAT:
        Return a JSON object with a single key "rationale".
        Example: {{ "rationale": "The model penalized the predictability (0.15) because Trips Right is the most common formation on 3rd & 5..." }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant evaluating football plays."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            return json.loads(content).get("rationale", "No rationale generated.")
        except Exception as e:
            print(f"Error generating analysis: {e}")
            return "Analysis failed."
