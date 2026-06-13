# clashai/combat/agent_v4/bc.py
# BehavioralCloningMixin — V4.1 imitation pretraining on heuristic demos.

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from clashai.combat.agent_v4.constants import MAX_GRAD_NORM, LEARNING_RATE


class BehavioralCloningMixin:
    """Pretrains the actor by behavioral cloning before PPO exploration."""

    def pretrain_bc(self, demonstrations, epochs=10, lr=1e-3,
                    mini_batch_size=64):
        """
        Pré-entraîne l'actor par behavioral cloning sur des
        démonstrations heuristiques.

        L'agent apprend à imiter les actions de l'heuristique avant
        de commencer l'exploration PPO. Ça donne un bien meilleur
        point de départ que de partir de zéro.

        Args:
            demonstrations: list of (grid, vector, action, mask) tuples
            epochs: nombre de passes sur le dataset
            lr: learning rate (plus élevé que PPO car supervisé)
            mini_batch_size: taille des mini-batches

        Returns:
            final_accuracy: float
        """
        n = len(demonstrations)
        if n == 0:
            print(" WARNING: Aucune démonstration, BC ignoré")
            return 0.0

        print(f"\n{'='*60}")
        print(f" Behavioral Cloning — {n} démonstrations")
        print(f" Epochs: {epochs} | LR: {lr} | Batch: {mini_batch_size}")
        print(f"{'='*60}")

        grids = torch.FloatTensor(
            np.array([d[0] for d in demonstrations])
        ).to(self.device)
        vectors = torch.FloatTensor(
            np.array([d[1] for d in demonstrations])
        ).to(self.device)
        actions = torch.LongTensor(
            [d[2] for d in demonstrations]
        ).to(self.device)
        masks = torch.FloatTensor(
            np.array([d[3] for d in demonstrations])
        ).to(self.device)

        # Separate optimizer for BC (higher lr)
        bc_optimizer = optim.Adam(
            self.network.parameters(), lr=lr, eps=1e-5
        )

        best_accuracy = 0.0

        for epoch in range(epochs):
            indices = torch.randperm(n, device=self.device)
            total_loss = 0.0
            num_batches = 0

            self.network.train()

            for start in range(0, n, mini_batch_size):
                batch_idx = indices[start:start + mini_batch_size]

                # Do not apply action mask during BC: when the heuristic
                # deploys a role after its counter hits 0, that action has
                # mask=0 in the stored demo. Passing the mask sets
                # logit[target] = -1e8, making CE loss ≈ 1e8 per sample
                # and causing the total BC loss to blow up (~284 000).
                logits, _ = self.network(
                    grids[batch_idx],
                    vectors[batch_idx],
                    None,
                )

                loss = nn.CrossEntropyLoss()(logits, actions[batch_idx])

                bc_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.parameters(), MAX_GRAD_NORM
                )
                bc_optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            # Accuracy over the full dataset
            self.network.eval()
            with torch.no_grad():
                logits, _ = self.network(grids, vectors, masks)
                preds = logits.argmax(dim=1)
                accuracy = (preds == actions).float().mean().item()

            avg_loss = total_loss / max(num_batches, 1)
            best_accuracy = max(best_accuracy, accuracy)

            print(f" Epoch {epoch+1:2d}/{epochs}: "
                  f"loss={avg_loss:.4f} accuracy={accuracy:.1%}")

        # Reset the PPO optimizer after BC
        # (Adam moments from BC are not relevant for PPO)
        self.optimizer = optim.Adam(
            self.network.parameters(), lr=LEARNING_RATE, eps=1e-5
        )

        print(f"\n BC terminé — accuracy finale: {best_accuracy:.1%}")
        return best_accuracy
