import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import unittest
from model_infer import test_model_inference
from light_tts.server.api_cli import make_argument_parser
from light_tts.utils.envs_utils import set_env_start_args, get_env_start_args


class TestModelInfer(unittest.TestCase):
    def test_model_infer(self):
        args = get_env_start_args()
        args.data_type = torch.float16
        test_model_inference(args)
        return


if __name__ == "__main__":
    import torch

    parser = make_argument_parser()
    parser.add_argument("--batch_size", type=int, default=None, help="batch size")
    parser.add_argument("--input_len", type=int, default=64, help="input sequence length")
    parser.add_argument("--output_len", type=int, default=128, help="output sequence length")
    parser.add_argument(
        "--torch_profile",
        action="store_true",
        help="Enable torch profiler to profile the model",
    )
    args = parser.parse_args()
    set_env_start_args(args)
    torch.multiprocessing.set_start_method("spawn")
    unittest.main(argv=["first-arg-is-ignored"])
