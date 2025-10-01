"""
Critic Head Pretrainer - Value Head 사전학습 컴포넌트 (v2.0.0).

RM(Reward Model)의 시퀀스 보상을 사용하여 가치함수 헤드를 학습하고,
Critic WMTP의 Stage 2에서 사용할 수 있도록 저장합니다.

[리팩토링 v2.0.0]
- CriticDeltaScorer 제거 후 GAE 로직 직접 구현
- 연구제안서 수식 직접 구현: L_VF(ϕ) = E_t[(V_ϕ(s_t) - R_t)²]
- Value Head 직접 관리 및 학습

연구 철학:
Stage 1: Value Head가 RM의 보상 신호로부터 각 토큰의 가치를 예측하도록 학습
Stage 2: 학습된 Value Head로 TD error 계산하여 토큰 중요도 가중치 생성
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from rich.console import Console
from rich.progress import track

from src.components.base import BaseComponent
from src.components.registry import pretrainer_registry
from src.utils.reward_utils import compute_sequence_rewards

console = Console()


@pretrainer_registry.register(
    "critic-head-pretrainer", category="pretrainer", version="2.0.0"
)
class CriticHeadPretrainer(BaseComponent):
    """Critic WMTP를 위한 Value Head 사전학습 트레이너.

    연구제안서 Stage 1 구현:
    1. RM이 시퀀스 레벨 보상 R 제공
    2. GAE로 토큰별 가치 목표값 V̂_t 계산
    3. Value Head V_ϕ(h_t)가 V̂_t를 예측하도록 MSE loss로 학습
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Value Head Pretrainer 초기화.

        Args:
            config: 학습 설정
                - lr: 학습률 (기본값: 1e-4)
                - num_epochs: 학습 에포크 수 (기본값: 3)
                - gamma: GAE 할인율 (기본값: 0.99)
                - gae_lambda: GAE λ (기본값: 0.95)
                - max_steps: 최대 학습 스텝 (기본값: 1000)
        """
        super().__init__(config)

        # 학습 하이퍼파라미터
        self.lr = self.config.get("lr", 1e-4)
        self.num_epochs = self.config.get("num_epochs", 3)
        self.gamma = self.config.get("gamma", 0.99)  # GAE discount
        self.gae_lambda = self.config.get("gae_lambda", 0.95)  # GAE lambda
        self.max_steps = self.config.get("max_steps", 1000)

        # Value Head는 run()에서 생성
        self.value_head: nn.Module | None = None

    def spread_reward_to_tokens(
        self, sequence_reward: float, seq_length: int
    ) -> np.ndarray:
        """시퀀스 보상을 토큰별로 균등 분배.

        간단한 균등 분배 방식. 향후 더 정교한 방법 고려 가능.

        Args:
            sequence_reward: 시퀀스 전체 보상
            seq_length: 시퀀스 길이

        Returns:
            토큰별 보상 배열 [seq_length]
        """
        # 균등 분배 (가장 간단한 방식)
        return np.full(seq_length, sequence_reward / seq_length, dtype=np.float32)

    def compute_gae_returns(
        self, rewards: np.ndarray, values: np.ndarray, next_value: float = 0.0
    ) -> np.ndarray:
        """GAE(Generalized Advantage Estimation)로 가치 목표값 계산.

        연구제안서의 토큰별 가치 분배 구현.

        Args:
            rewards: 토큰별 보상 [T]
            values: 현재 가치 예측값 [T]
            next_value: 마지막 상태의 가치 (보통 0)

        Returns:
            GAE 기반 가치 목표값 [T]
        """
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        returns = np.zeros(T, dtype=np.float32)

        # GAE 계산 (역방향)
        gae = 0
        for t in reversed(range(T)):
            next_val = next_value if t == T - 1 else values[t + 1]

            # TD error: δ_t = r_t + γV(s_{t+1}) - V(s_t)
            delta = rewards[t] + self.gamma * next_val - values[t]

            # GAE: A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages[t] = gae

            # Return = Value + Advantage
            returns[t] = values[t] + advantages[t]

        return returns

    def run(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Value Head 사전학습 실행.

        Args:
            ctx: 실행 컨텍스트
                - base_model: 베이스 언어 모델
                - rm_model: Reward Model
                - train_dataloader: 훈련 데이터 로더
                - run_name: 실행 이름 (S3 경로용)

        Returns:
            학습 결과 딕셔너리
        """
        self.validate_initialized()

        base_model = ctx["base_model"]
        rm_model = ctx.get("rm_model")
        train_loader = ctx["train_dataloader"]
        run_name = ctx.get("run_name", "default")

        if rm_model is None:
            console.print(
                "[yellow]⚠ No RM model provided for Stage 1 training[/yellow]"
            )
            return {"skipped": True, "message": "No RM model"}

        # Hidden size 추출
        hidden_size = None
        if hasattr(base_model, "config"):
            # HuggingFace 스타일 모델
            config = base_model.config
            hidden_size = getattr(
                config, "hidden_size", getattr(config, "n_embd", None)
            )

        if hidden_size is None:
            raise ValueError(
                f"Failed to extract hidden_size from base model. "
                f"Model config attributes: {dir(base_model.config) if hasattr(base_model, 'config') else 'No config'}"
            )

        # 🎯 Value Head 생성 (연구제안서: V_ϕ(h_t) → scalar)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),  # 은닉층
            nn.ReLU(),  # 활성화
            nn.Linear(hidden_size // 2, 1),  # 스칼라 가치 출력
        )

        # Device 설정
        device = next(base_model.parameters()).device
        self.value_head = self.value_head.to(device)

        # Optimizer와 Loss function
        optimizer = torch.optim.AdamW(self.value_head.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        # Hidden states 출력 활성화
        orig_flag = getattr(
            getattr(base_model, "config", object()), "output_hidden_states", False
        )
        if hasattr(base_model, "config"):
            base_model.config.output_hidden_states = True

        # 모델을 평가 모드로 (Value Head만 학습)
        base_model.eval()
        self.value_head.train()

        console.print("[cyan]Starting Stage 1: Value Head Pretraining[/cyan]")
        console.print(f"  - Hidden size: {hidden_size}")
        console.print(f"  - Learning rate: {self.lr}")
        console.print(f"  - Max steps: {self.max_steps}")

        total_loss = 0.0
        step_count = 0

        # 🔄 Training loop
        for epoch in range(self.num_epochs):
            console.print(f"\n[bold]Epoch {epoch + 1}/{self.num_epochs}[/bold]")

            for step, batch in enumerate(track(train_loader, description="Training")):
                if step >= self.max_steps:
                    break

                input_ids = batch.get("input_ids")
                attention_mask = batch.get("attention_mask")

                if input_ids is None:
                    continue

                # Device로 이동
                input_ids = input_ids.to(device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)

                # 📊 Hidden states 추출 (gradient 불필요)
                with (
                    torch.no_grad(),
                    torch.autocast(
                        device_type=("cuda" if torch.cuda.is_available() else "cpu"),
                        dtype=torch.bfloat16,
                    ),
                ):
                    outputs = base_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                    )
                    # 안전한 hidden_states 추출
                    from src.utils.model_utils import extract_hidden_states

                    hidden_states = extract_hidden_states(outputs)

                # 🎁 RM으로부터 시퀀스 보상 계산 (공통 유틸리티 사용)
                reward_tensor = compute_sequence_rewards(
                    rm_model, input_ids, attention_mask, amp_dtype=torch.bfloat16
                )
                rewards = reward_tensor.tolist()  # list[float]로 변환

                # 📐 토큰별 가치 목표값 계산 (GAE)
                B, S, H = hidden_states.shape
                value_targets = []

                for b in range(B):
                    # 시퀀스 보상을 토큰별로 분배
                    seq_reward = float(rewards[b])
                    token_rewards = self.spread_reward_to_tokens(seq_reward, S)

                    # 초기 가치 예측 (0으로 시작)
                    init_values = np.zeros(S, dtype=np.float32)

                    # GAE로 가치 목표값 계산
                    value_target = self.compute_gae_returns(
                        token_rewards, init_values, next_value=0.0
                    )
                    value_targets.append(value_target)

                # 🎯 Value Head 학습
                # Flatten: [B, S, H] → [B*S, H]
                hs_flat = hidden_states.reshape(B * S, H)

                # 가치 목표값 텐서 생성
                vt_flat = torch.tensor(
                    np.concatenate(value_targets, axis=0),
                    device=device,
                    dtype=hs_flat.dtype,
                ).view(B * S, 1)

                # Forward pass
                pred_values = self.value_head(hs_flat)

                # MSE Loss: L_VF(ϕ) = E_t[(V_ϕ(s_t) - R_t)²]
                loss = loss_fn(pred_values, vt_flat)

                # Backward pass - 연구 취지에 맞게 자연스러운 학습 허용
                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                # 연구 취지 유지: gradient clipping 대신 모니터링만
                total_norm = 0.0
                for p in self.value_head.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1.0 / 2)

                if total_norm > 50.0:  # High threshold for Stage 1
                    console.print(
                        f"[yellow]⚠ Stage 1 large gradient: {total_norm:.2f}[/yellow]"
                    )

                optimizer.step()

                # 통계
                total_loss += loss.item()
                step_count += 1

                if step % 100 == 0:
                    avg_loss = total_loss / max(step_count, 1)
                    console.print(
                        f"  Step {step}: Loss = {loss.item():.4f}, "
                        f"Avg Loss = {avg_loss:.4f}"
                    )

        # 🔚 Hidden states 설정 복원
        if hasattr(base_model, "config"):
            base_model.config.output_hidden_states = orig_flag

        # 💾 Value Head 저장
        save_location = self._save_value_head(run_name)

        avg_final_loss = total_loss / max(step_count, 1)
        console.print("\n[green]✅ Stage 1 Training Complete[/green]")
        console.print(f"  - Final avg loss: {avg_final_loss:.4f}")
        console.print(f"  - Value Head saved to: {save_location}")

        return {
            "saved": save_location,
            "final_loss": avg_final_loss,
            "total_steps": step_count,
        }

    def _save_value_head(self, run_name: str) -> str:
        """Value Head를 저장.

        Args:
            run_name: 실행 이름 (경로 생성용)

        Returns:
            저장 위치 문자열
        """
        # S3 경로 또는 로컬 경로 생성
        checkpoint_dir = Path(f"./checkpoints/critic/{run_name}")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Value Head state dict 저장
        vh_path = checkpoint_dir / "value_head_stage1.pt"
        torch.save(self.value_head.state_dict(), vh_path)

        # 메타데이터도 저장
        meta_path = checkpoint_dir / "value_head_meta.json"
        import json

        with open(meta_path, "w") as f:
            json.dump(
                {
                    "version": "2.0.0",
                    "hidden_size": self.value_head[0].in_features,
                    "intermediate_size": self.value_head[0].out_features,
                    "lr": self.lr,
                    "gamma": self.gamma,
                    "gae_lambda": self.gae_lambda,
                },
                f,
                indent=2,
            )

        return str(vh_path)
