# scripts/rl/agent_v3.py
# Agent PPO V3 pour ClashAI — IA réactive mid-combat.
#
# Changements vs V2 :
#   - 5 nouvelles actions : activation des capacités héros
#   - 1 nouvelle action : wait_combat (observer pendant le combat)
#   - Vecteur d'observation étendu avec combat features (15 dims)
#     + hero ability status (5 dims) + phase indicator
#   - L'agent opère en 2 phases :
#       Phase DEPLOY (comme V2) : poser troupes, sorts, wait, done
#       Phase COMBAT (nouveau)  : abilities héros, sorts restants, observer
#
# Architecture :
#   Grille (12, 40, 40) → CNN → 256
#   Vecteur (76,)         → MLP → 128 → 64
#   [village_features(20) + remaining_troops(14) + deploy_map(20) + step(1)
#    + combat_features(15) + hero_status(5) + phase(1)]
#   Fusion (320,)        → 256 shared
#                        → Actor  → 289 logits (masqués)
#                        → Critic → 1 valeur
#
# Action space (289 actions) :
#   0..279  : deploy troop_type (0-13) at position (0-19)
#   280     : wait_short  (0.5s — pause entre groupes)
#   281     : wait_long   (2.0s — funnel)
#   282     : done        (fin déploiement → passe en phase combat)
#   283     : ability_roi
#   284     : ability_reine
#   285     : ability_grand_gardien
#   286     : ability_championne
#   287     : ability_prince_gargouille
#   288     : wait_combat (2s d'observation pendant le combat)

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# =============================================================================
#                         CONFIGURATION
# =============================================================================

# Liste MAÎTRE de toutes les troupes/sorts/héros (identique V2)
TROOP_TYPES = [
    # --- Troupes ---
    {'name': 'golem',              'default_max': 2,  'role': 'tank'},
    {'name': 'sorcier',            'default_max': 6,  'role': 'ranged'},
    {'name': 'sorciere',           'default_max': 10, 'role': 'ranged'},
    {'name': 'pekka',              'default_max': 2,  'role': 'melee'},
    {'name': 'archere',            'default_max': 5,  'role': 'ranged'},
    # --- Héros ---
    {'name': 'roi',                'default_max': 1,  'role': 'hero'},
    {'name': 'reine',              'default_max': 1,  'role': 'hero'},
    {'name': 'grand_gardien',      'default_max': 1,  'role': 'hero'},
    {'name': 'championne',         'default_max': 1,  'role': 'hero'},
    {'name': 'prince_gargouille',  'default_max': 1,  'role': 'hero'},
    # --- Siège ---
    {'name': 'lance_buche',        'default_max': 1,  'role': 'siege'},
    # --- Sorts ---
    {'name': 'soin',               'default_max': 2,  'role': 'spell'},
    {'name': 'rage',               'default_max': 3,  'role': 'spell'},
    {'name': 'gel',                'default_max': 1,  'role': 'spell'},
]

NUM_TROOP_TYPES = len(TROOP_TYPES)      # 14
NUM_POSITIONS = 20

# --- Action space ---
NUM_DEPLOY_ACTIONS = NUM_TROOP_TYPES * NUM_POSITIONS  # 280
ACTION_WAIT_SHORT = NUM_DEPLOY_ACTIONS                # 280
ACTION_WAIT_LONG = NUM_DEPLOY_ACTIONS + 1             # 281
ACTION_DONE = NUM_DEPLOY_ACTIONS + 2                  # 282

# Nouvelles actions V3
# 5 héros avec ability. Au runtime, seuls les héros détectés dans la barre
# puis déployés auront leur ability démasquée. Si un héros est en amélioration
# → pas dans la barre → pas déployé → ability masquée automatiquement.
HERO_NAMES = ['roi', 'reine', 'grand_gardien', 'championne', 'prince_gargouille']
NUM_HERO_ABILITIES = len(HERO_NAMES)                  # 5
ACTION_ABILITY_START = NUM_DEPLOY_ACTIONS + 3         # 283
ACTION_ABILITY_ROI = ACTION_ABILITY_START             # 283
ACTION_ABILITY_REINE = ACTION_ABILITY_START + 1       # 284
ACTION_ABILITY_GG = ACTION_ABILITY_START + 2          # 285
ACTION_ABILITY_CHAMP = ACTION_ABILITY_START + 3       # 286
ACTION_ABILITY_PG = ACTION_ABILITY_START + 4          # 287
ACTION_WAIT_COMBAT = ACTION_ABILITY_START + 5         # 288

TOTAL_ACTIONS = ACTION_WAIT_COMBAT + 1                # 289

MAX_STEPS_PER_EPISODE = 65  # Plus de steps car phase combat ajoutée
MAX_COMBAT_STEPS = 25       # Max de steps pendant la phase combat

# Index rapide nom → idx
TROOP_NAME_TO_IDX = {t['name']: i for i, t in enumerate(TROOP_TYPES)}
HERO_NAME_TO_ABILITY = {name: ACTION_ABILITY_START + i for i, name in enumerate(HERO_NAMES)}

# --- Observation sizes ---
GRID_CHANNELS = 12     # V3 : 10 catégories + danger_sol + danger_air
GRID_SIZE = 40
VILLAGE_FEATURES = 20  # V3 : 8 base + infernos/eagle/CC/quadrants/scatter
TROOP_FEATURES = NUM_TROOP_TYPES          # 14
DEPLOY_MAP_SIZE = NUM_POSITIONS            # 20
STEP_FEATURES = 1

# Nouvelles features V3
from clashai.combat.combat_observer import COMBAT_FEATURES_SIZE  # 15
HERO_STATUS_SIZE = NUM_HERO_ABILITIES      # 5
PHASE_SIZE = 1                             # 0=deploy, 1=combat

VECTOR_SIZE = (VILLAGE_FEATURES + TROOP_FEATURES + DEPLOY_MAP_SIZE 
               + STEP_FEATURES + COMBAT_FEATURES_SIZE + HERO_STATUS_SIZE 
               + PHASE_SIZE)  # 20+14+20+1+15+5+1 = 76

# PPO Hyperparamètres
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
ENTROPY_COEF = 0.025         # Légèrement réduit (plus d'actions)
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5
LEARNING_RATE = 2e-4
PPO_EPOCHS = 4
BATCH_SIZE = 8


# =============================================================================
#                     FONCTIONS UTILITAIRES
# =============================================================================

def decode_action(action_idx):
    """
    Décode un index d'action.

    Returns:
        ('deploy', troop_idx, position_idx)
        ('wait_short', None, None)
        ('wait_long', None, None)
        ('done', None, None)
        ('ability', hero_idx, None)        # NOUVEAU V3
        ('wait_combat', None, None)        # NOUVEAU V3
    """
    if action_idx < NUM_DEPLOY_ACTIONS:
        troop_idx = action_idx // NUM_POSITIONS
        position_idx = action_idx % NUM_POSITIONS
        return ('deploy', troop_idx, position_idx)
    elif action_idx == ACTION_WAIT_SHORT:
        return ('wait_short', None, None)
    elif action_idx == ACTION_WAIT_LONG:
        return ('wait_long', None, None)
    elif action_idx == ACTION_DONE:
        return ('done', None, None)
    elif ACTION_ABILITY_START <= action_idx < ACTION_ABILITY_START + NUM_HERO_ABILITIES:
        hero_idx = action_idx - ACTION_ABILITY_START
        return ('ability', hero_idx, None)
    elif action_idx == ACTION_WAIT_COMBAT:
        return ('wait_combat', None, None)
    else:
        return ('done', None, None)  # Safety fallback


def encode_action(action_type, troop_idx=None, position_idx=None):
    """Encode une action en index."""
    if action_type == 'deploy':
        return troop_idx * NUM_POSITIONS + position_idx
    elif action_type == 'wait_short':
        return ACTION_WAIT_SHORT
    elif action_type == 'wait_long':
        return ACTION_WAIT_LONG
    elif action_type == 'done':
        return ACTION_DONE
    elif action_type == 'ability':
        return ACTION_ABILITY_START + troop_idx
    elif action_type == 'wait_combat':
        return ACTION_WAIT_COMBAT
    else:
        return ACTION_DONE


def get_initial_troop_counts():
    """Retourne le vecteur initial des troupes (tous à default_max)."""
    return np.array([t['default_max'] for t in TROOP_TYPES], dtype=np.float32)


def get_troop_counts_from_finder(troop_finder):
    """
    Construit le vecteur de troupes à partir du TroopFinder.
    """
    TEMPLATE_ALIASES = {
        'lance_buche_vide': 'lance_buche',
    }
    counts = np.zeros(NUM_TROOP_TYPES, dtype=np.float32)
    for i, troop in enumerate(TROOP_TYPES):
        if troop_finder.is_available(troop['name']):
            counts[i] = troop['default_max']
        else:
            for alias, real_name in TEMPLATE_ALIASES.items():
                if real_name == troop['name'] and troop_finder.is_available(alias):
                    counts[i] = troop['default_max']
                    break
    return counts


def compute_action_mask(remaining_troops, phase='deploy', hero_ability_mask=None):
    """
    Calcule le masque d'actions valides.

    Args:
        remaining_troops: array (14,)
        phase: 'deploy' ou 'combat'
        hero_ability_mask: array (5,) ou None — 1.0 si ability dispo

    Returns:
        mask: array (289,)
    """
    mask = np.zeros(TOTAL_ACTIONS, dtype=np.float32)

    if phase == 'deploy':
        # Phase déploiement : mêmes règles que V2
        for troop_idx in range(NUM_TROOP_TYPES):
            if remaining_troops[troop_idx] > 0:
                start = troop_idx * NUM_POSITIONS
                end = start + NUM_POSITIONS
                mask[start:end] = 1.0

        mask[ACTION_WAIT_SHORT] = 1.0
        mask[ACTION_WAIT_LONG] = 1.0
        mask[ACTION_DONE] = 1.0
        # Pas d'abilities ni wait_combat pendant le deploy

    elif phase == 'combat':
        # Phase combat : sorts restants + abilities + wait_combat
        
        # Sorts restants (on peut encore les lancer pendant le combat)
        for troop_idx in range(NUM_TROOP_TYPES):
            if TROOP_TYPES[troop_idx]['role'] == 'spell' and remaining_troops[troop_idx] > 0:
                start = troop_idx * NUM_POSITIONS
                end = start + NUM_POSITIONS
                mask[start:end] = 1.0

        # Abilities héros
        if hero_ability_mask is not None:
            for i in range(NUM_HERO_ABILITIES):
                if hero_ability_mask[i] > 0:
                    mask[ACTION_ABILITY_START + i] = 1.0

        # Wait combat (toujours disponible en phase combat)
        mask[ACTION_WAIT_COMBAT] = 1.0
        
        # Done est aussi disponible (pour mettre fin à la phase combat
        # et passer directement à l'attente des résultats)
        mask[ACTION_DONE] = 1.0

    return mask


# =============================================================================
#                     RÉSEAU ACTOR-CRITIC V3
# =============================================================================

class ActorCriticV3(nn.Module):
    """
    Réseau Actor-Critic V3.
    
    Changements vs V2 :
        - Vecteur d'entrée élargi (76 dims vs 43)
        - MLP vecteur plus large (128→64 vs 128→64)
        - Actor output : 289 actions (vs 283)
        - Reste structurellement identique (CNN + MLP → shared → actor/critic)
    """

    def __init__(self):
        super(ActorCriticV3, self).__init__()

        # CNN pour la grille du village (identique V2)
        self.grid_cnn = nn.Sequential(
            nn.Conv2d(GRID_CHANNELS, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),       # → 32×20×20

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),       # → 64×10×10

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),       # → 128×5×5
        )

        self.grid_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 5 * 5, 256),
            nn.ReLU(),
        )

        # MLP pour le vecteur étendu V3
        self.vector_fc = nn.Sequential(
            nn.Linear(VECTOR_SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # Fusion → Backbone partagé
        # 256 (grid) + 64 (vector) = 320
        self.shared = nn.Sequential(
            nn.Linear(320, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(256, 192),
            nn.ReLU(),
            nn.Linear(192, TOTAL_ACTIONS),  # 289
        )

        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def forward(self, grid, vector, action_mask=None):
        """
        Forward pass.

        Args:
            grid: (batch, 12, 40, 40)
            vector: (batch, 76)
            action_mask: (batch, 289) — 1.0 = valide, 0.0 = invalide
        """
        g = self.grid_cnn(grid)
        g = self.grid_fc(g)

        v = self.vector_fc(vector)

        combined = torch.cat([g, v], dim=1)
        shared = self.shared(combined)

        logits = self.actor(shared)
        value = self.critic(shared)

        if action_mask is not None:
            logits = logits + (action_mask - 1.0) * 1e8

        return logits, value


# =============================================================================
#                         ROLLOUT BUFFER V3
# =============================================================================

class RolloutBufferV3:
    """
    Buffer identique au V2 mais compatible avec les nouvelles dimensions.
    """

    def __init__(self):
        self.episodes = []
        self._current_episode = []

    def start_episode(self):
        self._current_episode = []

    def store_step(self, grid, vector, action, log_prob, value, action_mask):
        self._current_episode.append({
            'grid': grid.copy() if isinstance(grid, np.ndarray) else grid,
            'vector': vector.copy() if isinstance(vector, np.ndarray) else vector,
            'action': action,
            'log_prob': log_prob,
            'value': value,
            'action_mask': action_mask.copy() if isinstance(action_mask, np.ndarray) else action_mask,
        })

    def end_episode(self, final_reward, step_rewards=None):
        if not self._current_episode:
            return

        n_steps = len(self._current_episode)

        if step_rewards and len(step_rewards) >= n_steps:
            rewards = [float(step_rewards[i]) for i in range(n_steps)]
            rewards[-1] += final_reward
        else:
            rewards = [0.0] * n_steps
            rewards[-1] = final_reward

        for i, step in enumerate(self._current_episode):
            step['reward'] = rewards[i]
            step['done'] = (i == n_steps - 1)

        self.episodes.append(self._current_episode)
        self._current_episode = []

    def num_episodes(self):
        return len(self.episodes)

    def total_steps(self):
        return sum(len(ep) for ep in self.episodes)

    def clear(self):
        self.episodes.clear()
        self._current_episode = []

    def get_batch(self, device):
        all_grids = []
        all_vectors = []
        all_actions = []
        all_log_probs = []
        all_values = []
        all_masks = []
        all_advantages = []
        all_returns = []

        for episode in self.episodes:
            n = len(episode)
            rewards = [s['reward'] for s in episode]
            values = [s['value'] for s in episode]

            advantages = []
            returns = []
            gae = 0.0

            for t in reversed(range(n)):
                if t == n - 1:
                    next_value = 0.0
                else:
                    next_value = values[t + 1]

                delta = rewards[t] + GAMMA * next_value - values[t]
                gae = delta + GAMMA * GAE_LAMBDA * gae
                advantages.insert(0, gae)
                returns.insert(0, gae + values[t])

            for i, step in enumerate(episode):
                all_grids.append(step['grid'])
                all_vectors.append(step['vector'])
                all_actions.append(step['action'])
                all_log_probs.append(step['log_prob'])
                all_values.append(step['value'])
                all_masks.append(step['action_mask'])
                all_advantages.append(advantages[i])
                all_returns.append(returns[i])

        batch = {
            'grids': torch.FloatTensor(np.array(all_grids)).to(device),
            'vectors': torch.FloatTensor(np.array(all_vectors)).to(device),
            'actions': torch.LongTensor(all_actions).to(device),
            'log_probs': torch.FloatTensor(all_log_probs).to(device),
            'values': torch.FloatTensor(all_values).to(device),
            'masks': torch.FloatTensor(np.array(all_masks)).to(device),
            'advantages': torch.FloatTensor(all_advantages).to(device),
            'returns': torch.FloatTensor(all_returns).to(device),
        }

        adv = batch['advantages']
        if len(adv) > 1:
            batch['advantages'] = (adv - adv.mean()) / (adv.std() + 1e-8)

        return batch


# =============================================================================
#                          AGENT PPO V3
# =============================================================================

class PPOAgentV3:
    """
    Agent PPO V3 avec réactivité mid-combat.
    
    Changements vs PPOAgentV2 :
        - Réseau ActorCriticV3 (289 actions, 64-dim vector)
        - Buffer compatible deux phases
        - Chargement partiel de checkpoints V2 possible
    """

    def __init__(self, device=None, lr=LEARNING_RATE):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.network = ActorCriticV3().to(self.device)
        self.optimizer = optim.Adam(
            self.network.parameters(), lr=lr, eps=1e-5
        )

        self.buffer = RolloutBufferV3()

        self.update_count = 0
        self.total_episodes = 0

        n_params = sum(p.numel() for p in self.network.parameters())
        print("🤖 Agent PPO V3 initialisé")
        print(f"   Device      : {self.device}")
        print(f"   Actions     : {TOTAL_ACTIONS} "
              f"({NUM_TROOP_TYPES}×{NUM_POSITIONS} deploy + 3 ctrl "
              f"+ {NUM_HERO_ABILITIES} abilities + 1 wait_combat)")
        print(f"   Vector      : {VECTOR_SIZE} dims")
        print(f"   Paramètres  : {n_params:,}")
        print(f"   Batch size  : {BATCH_SIZE} épisodes")

    def select_action(self, grid, vector, action_mask):
        """
        Choisit une action.

        Args:
            grid: np.array (12, 40, 40)
            vector: np.array (76,)
            action_mask: np.array (289,)

        Returns:
            action: int
            log_prob: float
            value: float
        """
        grid_t = torch.FloatTensor(grid).unsqueeze(0).to(self.device)
        vector_t = torch.FloatTensor(vector).unsqueeze(0).to(self.device)
        mask_t = torch.FloatTensor(action_mask).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, value = self.network(grid_t, vector_t, mask_t)

        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.squeeze().item()

    def store_step(self, grid, vector, action, log_prob, value, action_mask):
        self.buffer.store_step(grid, vector, action, log_prob, value, action_mask)

    def buffer_ready(self):
        return self.buffer.num_episodes() >= BATCH_SIZE

    def update(self):
        """PPO update."""
        if not self.buffer_ready():
            return None

        self.update_count += 1
        batch = self.buffer.get_batch(self.device)
        total_steps = len(batch['actions'])

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0

        for epoch in range(PPO_EPOCHS):
            logits, values = self.network(
                batch['grids'], batch['vectors'], batch['masks']
            )
            dist = Categorical(logits=logits)

            new_log_probs = dist.log_prob(batch['actions'])
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - batch['log_probs'])

            surr1 = ratio * batch['advantages']
            surr2 = torch.clamp(
                ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON
            ) * batch['advantages']
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = nn.MSELoss()(values.squeeze(), batch['returns'])

            loss = (policy_loss
                    + VALUE_COEF * value_loss
                    - ENTROPY_COEF * entropy)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.network.parameters(), MAX_GRAD_NORM
            )
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()

        self.buffer.clear()

        return {
            'update': self.update_count,
            'policy_loss': total_policy_loss / PPO_EPOCHS,
            'value_loss': total_value_loss / PPO_EPOCHS,
            'entropy': total_entropy / PPO_EPOCHS,
            'total_steps': total_steps,
        }

    def save(self, path):
        """Sauvegarde atomique."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        tmp_path = path + '.tmp'
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'update_count': self.update_count,
            'total_episodes': self.total_episodes,
            'version': 'v3',
        }, tmp_path)
        if os.path.exists(path):
            os.replace(tmp_path, path)
        else:
            os.rename(tmp_path, path)
        print(f"💾 Agent V3 sauvegardé : {path}")

    def load(self, path):
        """Charge un checkpoint V3 (ou tente un chargement partiel de V2)."""
        if not os.path.exists(path):
            print(f"⚠️  Fichier introuvable : {path}")
            return False
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            version = checkpoint.get('version', 'v2')
            
            if version == 'v3':
                # Chargement normal V3
                self.network.load_state_dict(checkpoint['network'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                # Chargement partiel depuis V2 — on charge ce qu'on peut
                print("   📦 Checkpoint V2 détecté, chargement partiel...")
                v2_state = checkpoint['network']
                v3_state = self.network.state_dict()
                
                loaded = 0
                for key in v2_state:
                    if key in v3_state and v2_state[key].shape == v3_state[key].shape:
                        v3_state[key] = v2_state[key]
                        loaded += 1
                
                self.network.load_state_dict(v3_state)
                print(f"   ✅ {loaded}/{len(v2_state)} couches chargées depuis V2")
                # Ne pas charger l'optimizer (incompatible)
            
            self.update_count = checkpoint.get('update_count', 0)
            self.total_episodes = checkpoint.get('total_episodes', 0)
            print(f"📂 Agent V3 chargé : {path} "
                  f"(update #{self.update_count}, {self.total_episodes} épisodes)")
            return True
        except (EOFError, RuntimeError) as e:
            print(f"⚠️  Checkpoint corrompu ({e}), démarrage from scratch")
            return False


# =============================================================================
#                            TEST
# =============================================================================

def test_agent():
    """Test du réseau V3 avec données aléatoires."""
    print("🧪 Test Agent PPO V3\n")

    agent = PPOAgentV3()

    # Simuler un épisode avec les 2 phases
    for ep in range(BATCH_SIZE):
        remaining = get_initial_troop_counts()
        deploy_map = np.zeros(NUM_POSITIONS, dtype=np.float32)
        grid = np.random.rand(GRID_CHANNELS, GRID_SIZE, GRID_SIZE).astype(np.float32)
        features = np.random.rand(VILLAGE_FEATURES).astype(np.float32)
        combat_features = np.zeros(COMBAT_FEATURES_SIZE, dtype=np.float32)
        hero_status = np.zeros(HERO_STATUS_SIZE, dtype=np.float32)

        agent.buffer.start_episode()

        # --- Phase DEPLOY ---
        phase = 'deploy'
        for step in range(MAX_STEPS_PER_EPISODE):
            step_norm = np.array([step / MAX_STEPS_PER_EPISODE], dtype=np.float32)
            phase_indicator = np.array([0.0 if phase == 'deploy' else 1.0], dtype=np.float32)
            
            vector = np.concatenate([
                features, remaining / 10.0, deploy_map, step_norm,
                combat_features, hero_status, phase_indicator
            ])

            mask = compute_action_mask(remaining, phase=phase)
            action, log_prob, value = agent.select_action(grid, vector, mask)
            agent.store_step(grid, vector, action, log_prob, value, mask)

            action_type, troop_idx, pos_idx = decode_action(action)

            if action_type == 'deploy':
                remaining[troop_idx] = max(0, remaining[troop_idx] - 1)
                if pos_idx is not None:
                    deploy_map[pos_idx] += 0.2
                name = TROOP_TYPES[troop_idx]['name']
                print(f"   [{phase}] Step {step:2d}: {name} → pos {pos_idx}")
            elif action_type == 'done':
                print(f"   [{phase}] Step {step:2d}: DONE → passage en combat")
                phase = 'combat'
                combat_features[0] = 1.0  # Simuler phase combat
                continue
            elif action_type == 'ability':
                hero = HERO_NAMES[troop_idx]
                print(f"   [{phase}] Step {step:2d}: ⚡ ability {hero}")
            elif action_type == 'wait_combat':
                print(f"   [{phase}] Step {step:2d}: 👁️ observe (2s)")
            else:
                print(f"   [{phase}] Step {step:2d}: {action_type}")

            if phase == 'combat' and step > 40:
                break  # Fin arbitraire pour le test

        reward = float(np.random.randint(-50, 400))
        agent.buffer.end_episode(reward)
        agent.total_episodes += 1
        print(f"   → Ep {ep+1}: reward={reward:.0f}\n")

    # PPO Update
    print(f"🔄 PPO Update ({agent.buffer.total_steps()} steps)...")
    stats = agent.update()
    if stats:
        print(f"   ✅ Update #{stats['update']}")
        print(f"   Policy: {stats['policy_loss']:.4f}")
        print(f"   Value:  {stats['value_loss']:.4f}")
        print(f"   Entropy: {stats['entropy']:.4f}")

    # Test save/load
    agent.save('/tmp/test_agent_v3.pth')
    agent2 = PPOAgentV3()
    agent2.load('/tmp/test_agent_v3.pth')

    print("\n📊 Comparaison V2 vs V3 :")
    print(f"   Actions  : 283 → {TOTAL_ACTIONS}")
    print(f"   Vector   : 43  → {VECTOR_SIZE}")
    print(f"   Max steps: 50  → {MAX_STEPS_PER_EPISODE}")

    print("\n✅ Test Agent V3 terminé !")


if __name__ == "__main__":
    test_agent()