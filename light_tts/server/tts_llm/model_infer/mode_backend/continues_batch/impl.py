import torch
from typing import List, Tuple
from light_tts.server.core.objs.req import ReqRunStatus
from light_tts.server.tts_llm.model_infer.mode_backend.base_backend import ModeBackend
from light_tts.server.tts_llm.model_infer.infer_batch import InferReq
from light_tts.server.tts_llm.model_infer.infer_batch import g_infer_context
from light_tts.utils.log_utils import init_logger
from .pre_process import prepare_prefill_inputs, prepare_decode_inputs
from .post_process import sample

logger = init_logger(__name__)


class ContinuesBatchBackend(ModeBackend):
    def __init__(self) -> None:
        super().__init__()

    def prefill(self, reqs: List[Tuple]):
        req_ids = self._init_reqs(reqs)
        kwargs, run_reqs = prepare_prefill_inputs(req_ids)
        logits = self.model.forward(**kwargs)

        mask = ~kwargs["b_next_fill"]
        next_token_ids = sample(
            logits[mask],
            kwargs["output_token_ids"][mask],
            kwargs["ignore_eos"][mask],
            self.model.eos_token,
            # kwargs["bistream"][mask],
            # self.model.speech_token_size + 2
        )
        next_token_ids = next_token_ids.detach().cpu().numpy()

        self.post_handle(run_reqs, next_token_ids, mask)
        return

    def decode(self):
        kwargs, run_reqs = prepare_decode_inputs(g_infer_context.infer_req_ids)
        logits = self.model.forward(**kwargs)

        mask = ~kwargs["b_next_fill"]
        next_token_ids = sample(
            logits[mask],
            kwargs["output_token_ids"][mask],
            kwargs["ignore_eos"][mask],
            self.model.eos_token,
            # kwargs["bistream"][mask],
            # self.model.speech_token_size + 2
        )
        next_token_ids = next_token_ids.detach().cpu().numpy()

        self.post_handle(run_reqs, next_token_ids, mask)
        return

    def post_handle(self, run_reqs: List[InferReq], next_token_ids, mask):
        finished_req_ids = []
        append_prefill_req_ids = []

        index = 0
        # 2 for <sos> and <task_id>
        text_vocab = self.model.text_vob_size + 2
        for req_obj in run_reqs:
            req_obj: InferReq = req_obj
            req_obj.cur_kv_len = req_obj.get_cur_total_len()

            if mask[index]:
                next_token_id = next_token_ids[index]
                index += 1
            else:
                next_token_id = self.model.fill_token

            if next_token_id == self.model.fill_token:
                req_obj.next_fill_index = len(req_obj.output_token_ids) + self.mix_ratio[1] + 1
            else:
                req_obj.set_next_gen_token_id(next_token_id, next_token_id + text_vocab)  # 生成的token id 需要偏移
                req_obj.cur_output_len += 1

            req_obj.output_token_ids.append(next_token_id)

            req_obj.shm_req.shm_cur_kv_len = req_obj.cur_kv_len
            req_obj.shm_req.shm_cur_output_len = req_obj.cur_output_len

            req_obj.update_finish_status(
                self.model.eos_token, self.model.stop_token_ids, fill_token_id=self.model.fill_token
            )

            if req_obj.finish_status.is_finished() or req_obj.shm_req.router_aborted:
                finished_req_ids.append(req_obj.shm_req.request_id)
            elif req_obj.shm_req.req_status.get_status() == ReqRunStatus.WAIT_FOR_TEXT:
                append_prefill_req_ids.append(req_obj.shm_req.request_id)
            # shm_cur_kv_len shm_cur_output_len 是 router 调度进程需要读的信息
            # finish_token_index finish_status candetoken_out_len 是
            # detokenization 进程需要的信息，注意这些变量的写入顺序避免异步协同问题。

            if req_obj.finish_status.is_finished():
                req_obj.shm_req.finish_token_index = req_obj.get_cur_total_len() - 1
                req_obj.shm_req.finish_status = req_obj.finish_status

            # req_obj.shm_req.candetoken_out_len = req_obj.cur_output_len

        g_infer_context.filter(finished_req_ids, append_prefill_req_ids)
        return
