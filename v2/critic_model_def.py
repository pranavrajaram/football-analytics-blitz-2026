import torch
import torch.nn as nn
import numpy as np

# Constants from notebook analysis
SEQ_LEN = 8
SEQ_INPUT_DIM = 10
DRIVE_CONTEXT_DIM = 25
CURRENT_GAME_STATE_DIM = 6
n_off_teams = 32
n_formations = 14
n_coverage = 15  

FORMATION_LIST = [
    'Balanced', 'Slot Left', 'Twin Right', 'Twin Left', 'Trips Right',
    'Slot Right', 'Trips Left', 'Single Left', 'Single Right', 'Spread',
    'Bunch Right', 'Bunch Left', 'All Tight', 'Goal Line'
]
FORMATION_TO_IDX = {f: i for i, f in enumerate(FORMATION_LIST)}

class PredictabilityLSTM(nn.Module):
    """Past-context only: 8-play sequence + drive DNA + OffTeam (play-caller) embedding; LayerNorm; Focal-ready."""

    def __init__(self, seq_input_size=SEQ_INPUT_DIM, drive_context_size=DRIVE_CONTEXT_DIM, current_game_state_size=CURRENT_GAME_STATE_DIM,
                 n_off_teams=n_off_teams, embed_dim=8, hidden_size=64, n_formations=n_formations, dropout=0.3):
        super().__init__()
        self.n_formations = n_formations
        self.off_team_embed = nn.Embedding(n_off_teams, embed_dim)
        self.layer_norm = nn.LayerNorm(seq_input_size)
        self.lstm = nn.LSTM(seq_input_size, hidden_size, batch_first=True, dropout=dropout)
        combined_size = hidden_size + drive_context_size + current_game_state_size + embed_dim
        self.fc = nn.Sequential(
            nn.Linear(combined_size, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.pass_head = nn.Linear(64, 1)
        self.formation_head = nn.Linear(64, n_formations)
        self.log_var_pass = nn.Parameter(torch.zeros(1))
        self.log_var_form = nn.Parameter(torch.zeros(1))

    def forward(self, seq, drive_context, current_game_state, off_team_idx):
        seq = self.layer_norm(seq)
        team_emb = self.off_team_embed(off_team_idx)
        _, (h_n, _) = self.lstm(seq)
        lstm_out = h_n[-1]
        combined = torch.cat([lstm_out, drive_context, current_game_state, team_emb], dim=1)
        x = self.fc(combined)
        pass_logits = self.pass_head(x).squeeze(-1)
        formation_logits = self.formation_head(x)
        return pass_logits, formation_logits

def scale_seq_and_current(seq, drive_context, current_game_state):
    seq = seq.copy()
    dc = drive_context.copy()
    gs = current_game_state.copy()
    # Seq (10): 0-3 RB/WR/TE/OL, 4-6 outcome, 7 Personnel_Matchup, 8-9 holistic
    seq[:, 0] = np.clip(seq[:, 0], 0, 3) / 3.0
    seq[:, 1] = np.clip(seq[:, 1], 0, 5) / 5.0
    seq[:, 2] = np.clip(seq[:, 2], 0, 3) / 3.0
    seq[:, 3] = np.clip(seq[:, 3], 4, 8) / 8.0
    seq[:, 4] = np.clip(seq[:, 4], 0, 1)
    seq[:, 5] = np.clip(seq[:, 5], -20, 50) / 50.0
    seq[:, 6] = np.clip(seq[:, 6], -15, 15) / 15.0
    seq[:, 7] = np.clip(seq[:, 7], -5, 5) / 5.0
    seq[:, 8] = np.clip(seq[:, 8], -7, 7) / 7.0
    seq[:, 9] = np.clip(seq[:, 9], 0, 4) / 4.0
    # Current game state (the play we're predicting): Down, ToGo, TimeSin, TimeCos, OffLeadBefore, StartYard
    gs[0] = (gs[0] - 1) / 3.0
    gs[1] = (gs[1] - 1) / 18.0
    gs[2] = np.clip(gs[2], -1, 1)
    gs[3] = np.clip(gs[3], -1, 1)
    gs[4] = np.clip(gs[4], -40, 40) / 80.0
    gs[5] = np.clip(gs[5], 0, 100) / 100.0
    # Drive context
    dc[0] = np.clip(dc[0], -15, 15) / 15.0
    dc[1] = np.clip(dc[1], -20, 50) / 50.0
    dc[5] = np.clip(dc[5], 0, 120) / 120.0
    dc[6] = dc[6] / max(n_coverage - 1, 1)
    dc[7] = np.clip(dc[7], -7, 7) / 7.0
    dc[8] = np.clip(dc[8], 0, 4) / 4.0
    dc[9] = np.clip(dc[9], 0, 5) / 5.0
    dc[10] = np.clip(dc[10], 0, 1)
    for i in range(11, len(dc)):
        dc[i] = np.clip(dc[i], 0, 1)
    return seq, dc, gs
