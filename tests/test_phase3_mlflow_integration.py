"""
Phase 3: MLflow 체크포인트 통합 테스트

이 테스트는 Phase 3의 주요 구현 사항들을 검증합니다:
1. DistributedManager MLflow 통합
2. S3 체크포인트 직접 저장/로드
3. Critic Value Head MLflow 저장
4. Pipeline 체크포인트 처리 개선
"""

import io
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn


class TestPhase3MLflowIntegration:
    """Phase 3 MLflow 체크포인트 통합 테스트."""

    def test_distributed_manager_mlflow_save_local(self):
        """로컬 체크포인트 저장 및 MLflow 업로드 테스트."""
        from src.utils.dist import DistributedManager

        # Mock 설정
        dist_manager = DistributedManager()
        mock_model = MagicMock(spec=nn.Module)
        mock_optimizer = MagicMock()
        mock_mlflow = MagicMock()

        # state_dict 모킹
        mock_model.state_dict.return_value = {"layer": torch.randn(10, 10)}
        mock_optimizer.state_dict.return_value = {"param": torch.randn(10, 10)}

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = f"{tmpdir}/checkpoint.pt"

            # 로컬 저장 테스트
            with patch("torch.save") as mock_save:
                dist_manager.save_checkpoint(
                    model=mock_model,
                    optimizer=mock_optimizer,
                    checkpoint_path=checkpoint_path,
                    epoch=1,
                    step=100,
                    mlflow_manager=mock_mlflow,
                )

                # torch.save가 호출되었는지 확인
                mock_save.assert_called_once()

                # MLflow log_artifact가 호출되었는지 확인
                mock_mlflow.log_artifact.assert_called_once_with(
                    local_path=checkpoint_path, artifact_path="checkpoints"
                )

    def test_distributed_manager_s3_save(self):
        """S3 직접 체크포인트 저장 테스트."""
        from src.utils.dist import DistributedManager

        dist_manager = DistributedManager()
        mock_model = MagicMock(spec=nn.Module)
        mock_optimizer = MagicMock()
        mock_mlflow = MagicMock()

        mock_model.state_dict.return_value = {"layer": torch.randn(10, 10)}
        mock_optimizer.state_dict.return_value = {"param": torch.randn(10, 10)}

        checkpoint_path = "s3://wmtp/checkpoints/model.pt"

        with patch("src.utils.s3.S3Manager") as MockS3Manager:
            mock_s3 = MockS3Manager.return_value
            mock_s3.upload_from_bytes.return_value = checkpoint_path

            # S3 저장 테스트
            dist_manager.save_checkpoint(
                model=mock_model,
                optimizer=mock_optimizer,
                checkpoint_path=checkpoint_path,
                epoch=1,
                step=100,
                mlflow_manager=None,  # MLflow 없이 테스트
            )

            # S3 upload가 호출되었는지 확인
            mock_s3.upload_from_bytes.assert_called_once()

    def test_distributed_manager_s3_load(self):
        """S3에서 체크포인트 로드 테스트."""
        from src.utils.dist import DistributedManager

        dist_manager = DistributedManager()
        dist_manager.device = "cpu"

        mock_model = MagicMock(spec=nn.Module)
        mock_optimizer = MagicMock()

        checkpoint_path = "s3://wmtp/checkpoints/model.pt"

        # Mock 체크포인트 데이터
        checkpoint_data = {
            "model": {"layer": torch.randn(10, 10)},
            "optimizer": {"param": torch.randn(10, 10)},
            "epoch": 5,
            "step": 500,
        }

        with patch("src.utils.s3.S3Manager") as MockS3Manager:
            mock_s3 = MockS3Manager.return_value

            # S3에서 반환할 BytesIO 객체 생성
            buffer = io.BytesIO()
            torch.save(checkpoint_data, buffer)
            buffer.seek(0)
            mock_s3.stream_model.return_value = buffer

            # S3 로드 테스트
            loaded = dist_manager.load_checkpoint(
                model=mock_model, optimizer=mock_optimizer, checkpoint_path=checkpoint_path
            )

            # S3 stream_model이 호출되었는지 확인
            mock_s3.stream_model.assert_called_once_with("checkpoints/model.pt")

            # 체크포인트 데이터 확인
            assert loaded["epoch"] == 5
            assert loaded["step"] == 500

    def test_critic_value_head_mlflow_save(self):
        """Critic Value Head MLflow 저장 테스트."""
        from src.components.trainer.critic_stage1_pretrainer import (
            CriticStage1Pretrainer,
        )

        trainer = CriticStage1Pretrainer({"lr": 1e-4})
        trainer.setup({})

        # Mock 컨텍스트
        mock_base_model = MagicMock()
        mock_base_model.config.hidden_size = 768
        mock_base_model.config.output_hidden_states = False

        mock_rm_model = MagicMock()
        mock_mlflow = MagicMock()
        mock_mlflow.list_artifacts.return_value = []  # 기존 아티팩트 없음

        # Mock 데이터로더
        mock_dataloader = [
            {
                "input_ids": torch.randint(0, 1000, (2, 10)),
                "attention_mask": torch.ones(2, 10),
            }
        ]

        ctx = {
            "base_model": mock_base_model,
            "rm_model": mock_rm_model,
            "train_dataloader": mock_dataloader,
            "mlflow_manager": mock_mlflow,
        }

        with patch("torch.save"), patch(
            "src.components.reward.sequence_reward.compute_sequence_rewards"
        ) as mock_compute:
            mock_compute.return_value = torch.tensor([1.0, 2.0])

            # Hidden states 모킹
            mock_outputs = MagicMock()
            mock_outputs.hidden_states = [torch.randn(2, 10, 768)]
            mock_base_model.return_value = mock_outputs

            # 실행
            result = trainer.run(ctx)

            # MLflow log_model이 호출되었는지 확인
            mock_mlflow.log_model.assert_called_once()
            assert result["saved"] == "MLflow/S3"

    def test_pipeline_s3_checkpoint_resume(self):
        """Pipeline에서 S3 체크포인트 재개 테스트."""
        from pathlib import Path

        checkpoint_data = {"epoch": 3, "step": 300, "mlflow_run_id": "test_run_123"}

        # S3 경로로 재개 테스트
        s3_checkpoint = "s3://wmtp/checkpoints/resume.pt"

        with patch("src.utils.s3.S3Manager") as MockS3Manager:
            mock_s3 = MockS3Manager.return_value

            # S3에서 반환할 BytesIO 객체 생성
            buffer = io.BytesIO()
            torch.save(checkpoint_data, buffer)
            buffer.seek(0)
            mock_s3.stream_model.return_value = buffer

            # Pipeline 코드 일부 시뮬레이션
            if s3_checkpoint.startswith("s3://"):
                s3_key = s3_checkpoint.replace("s3://wmtp/", "")
                checkpoint_bytes = mock_s3.stream_model(s3_key)
                loaded_data = torch.load(checkpoint_bytes, map_location="cpu")

                assert loaded_data["epoch"] == 3
                assert loaded_data["step"] == 300
                assert loaded_data["mlflow_run_id"] == "test_run_123"

    def test_s3_manager_upload_from_bytes(self):
        """S3Manager upload_from_bytes 메서드 테스트."""
        from src.utils.s3 import S3Manager

        with patch("boto3.client") as mock_boto:
            mock_s3_client = MagicMock()
            mock_boto.return_value = mock_s3_client
            mock_s3_client.head_bucket.return_value = True

            s3_manager = S3Manager(bucket="test-bucket", region="us-east-1")
            s3_manager.connected = True

            # 바이트 데이터 업로드 테스트
            test_data = b"test checkpoint data"
            s3_key = "checkpoints/test.pt"

            with patch.object(s3_manager.s3_client, "put_object") as mock_put:
                s3_uri = s3_manager.upload_from_bytes(test_data, s3_key)

                # put_object가 올바른 인자로 호출되었는지 확인
                mock_put.assert_called_once()
                call_args = mock_put.call_args
                assert call_args.kwargs["Bucket"] == "test-bucket"
                assert call_args.kwargs["Key"] == s3_key
                assert call_args.kwargs["Body"] == test_data

                assert s3_uri == f"s3://test-bucket/{s3_key}"


if __name__ == "__main__":
    # 테스트 실행
    pytest.main([__file__, "-v"])