import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_toolbox_cockpit.backends.llama_cpp.config import (
    get_model_config,
    get_toolbox_defaults,
    get_vision_projector_config,
    load_models,
)
from ai_toolbox_cockpit.backends.llama_cpp.model_manager import (
    get_hf_quants,
    get_local_vision_projectors,
    scan_local_models,
)
from ai_toolbox_cockpit.backends.llama_cpp.server_runner import build_server_cmd


REPO = "ggml-org/MiMo-V2.6-Flash-RL-GGUF"
MODEL_DIR = "MiMo-V2.6-Flash-RL-GGUF"
MXFP4 = "MiMo-V2.6-Flash-RL-MXFP4-*-of-*.gguf"
Q2_K = "MiMo-V2.6-Flash-RL-Q2_K-*-of-*.gguf"
PROJECTOR = "mmproj-MiMo-V2.6-Flash-RL-Q8_0.gguf"
TOOLBOX_ID = "strix-halo-llama-rocm-10-0"


class LlamaMiMo26Tests(unittest.TestCase):
    def test_catalog_exposes_both_quants_without_advertising_broken_mtp(self) -> None:
        configs = {model["repo"]: model for model in load_models()}
        model = configs[REPO]

        self.assertEqual(
            get_vision_projector_config(model)["patterns"], [PROJECTOR]
        )
        self.assertNotIn("mtp", model)
        self.assertEqual(
            get_toolbox_defaults(model, TOOLBOX_ID),
            {
                "context_size": 262144,
                "parallel_sequences": 1,
                "gpu_layers": 999,
                "flash_attention": True,
                "load_mode": "none",
            },
        )

        files = [
            "MiMo-V2.6-Flash-RL-MXFP4-00001-of-00002.gguf",
            "MiMo-V2.6-Flash-RL-MXFP4-00002-of-00002.gguf",
            "MiMo-V2.6-Flash-RL-Q2_K-00001-of-00002.gguf",
            "MiMo-V2.6-Flash-RL-Q2_K-00002-of-00002.gguf",
            PROJECTOR,
            "mtp-MiMo-V2.6-Flash-RL-MXFP4.gguf",
        ]
        with patch(
            "ai_toolbox_cockpit.backends.llama_cpp.model_manager.HfApi.list_repo_files",
            return_value=files,
        ):
            quants = get_hf_quants(REPO)

        self.assertIn(MXFP4, quants)
        self.assertIn(Q2_K, quants)
        self.assertIn(PROJECTOR, quants)

    def test_local_quants_are_recognized_and_q8_projector_is_optional(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            models_dir = Path(temporary)
            repo_dir = models_dir / MODEL_DIR
            repo_dir.mkdir()
            mxfp4_1 = repo_dir / "MiMo-V2.6-Flash-RL-MXFP4-00001-of-00002.gguf"
            mxfp4_2 = repo_dir / "MiMo-V2.6-Flash-RL-MXFP4-00002-of-00002.gguf"
            q2_1 = repo_dir / "MiMo-V2.6-Flash-RL-Q2_K-00001-of-00002.gguf"
            q2_2 = repo_dir / "MiMo-V2.6-Flash-RL-Q2_K-00002-of-00002.gguf"
            projector = repo_dir / PROJECTOR
            mtp = repo_dir / "mtp-MiMo-V2.6-Flash-RL-MXFP4.gguf"
            for path in (mxfp4_1, mxfp4_2, q2_1, q2_2, projector, mtp):
                path.touch()

            with patch(
                "ai_toolbox_cockpit.backends.llama_cpp.model_manager.get_models_dir",
                return_value=models_dir,
            ):
                discovered = scan_local_models()
                projectors = get_local_vision_projectors(
                    str(mxfp4_1), [PROJECTOR]
                )

            self.assertEqual(
                [entry["name"] for entry in discovered],
                [f"{MODEL_DIR}/{MXFP4}", f"{MODEL_DIR}/{Q2_K}"],
            )
            self.assertEqual(projectors, [projector])
            self.assertEqual(get_model_config(str(mxfp4_1))["repo"], REPO)
            self.assertEqual(get_model_config(str(q2_1))["repo"], REPO)

    def test_strix_command_matches_validated_text_and_vision_launches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            models_dir = Path(temporary)
            repo_dir = models_dir / MODEL_DIR
            repo_dir.mkdir()
            model = repo_dir / "MiMo-V2.6-Flash-RL-MXFP4-00001-of-00002.gguf"
            second_shard = repo_dir / "MiMo-V2.6-Flash-RL-MXFP4-00002-of-00002.gguf"
            q2_model = repo_dir / "MiMo-V2.6-Flash-RL-Q2_K-00001-of-00002.gguf"
            q2_second_shard = repo_dir / "MiMo-V2.6-Flash-RL-Q2_K-00002-of-00002.gguf"
            projector = repo_dir / PROJECTOR
            for path in (
                model, second_shard, q2_model, q2_second_shard, projector,
            ):
                path.touch()

            kwargs = {
                "engine": "podman",
                "image": "docker.io/kyuz0/amd-strix-halo-toolboxes:rocm-10.0",
                "model_path": str(repo_dir / MXFP4),
                "context_size": 262144,
                "use_fa": True,
                "use_no_mmap": True,
                "custom_args": "--jinja",
                "ngl": 999,
                "parallel_sequences": 1,
                "supports_load_mode": True,
                "load_mode": "none",
                "engine_args": [],
                "platform_id": "strix-halo",
            }
            with patch(
                "ai_toolbox_cockpit.backends.llama_cpp.model_manager.get_models_dir",
                return_value=models_dir,
            ):
                text_command = build_server_cmd(**kwargs)
                q2_command = build_server_cmd(
                    **{**kwargs, "model_path": str(repo_dir / Q2_K)}
                )
                vision_command = build_server_cmd(
                    **kwargs, vision_projector_path=str(projector)
                )

        self.assertEqual(
            text_command[text_command.index("-m") + 1],
            f"/models/{MODEL_DIR}/{model.name}",
        )
        self.assertEqual(text_command[text_command.index("-c") + 1], "262144")
        self.assertEqual(text_command[text_command.index("-ngl") + 1], "999")
        self.assertEqual(text_command[text_command.index("-fa") + 1], "1")
        self.assertEqual(
            text_command[text_command.index("--load-mode") + 1], "none"
        )
        self.assertEqual(text_command[text_command.index("-np") + 1], "1")
        self.assertNotIn("--mmproj", text_command)
        self.assertEqual(
            q2_command[q2_command.index("-m") + 1],
            f"/models/{MODEL_DIR}/{q2_model.name}",
        )
        self.assertEqual(
            q2_command[q2_command.index("--load-mode") + 1], "none"
        )
        self.assertEqual(q2_command[q2_command.index("-fa") + 1], "1")
        self.assertEqual(
            vision_command[vision_command.index("--mmproj") + 1],
            f"/models/{MODEL_DIR}/{PROJECTOR}",
        )


if __name__ == "__main__":
    unittest.main()
