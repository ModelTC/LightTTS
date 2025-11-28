import os
import math
import ctypes
import numpy as np
from .sampling_params import SamplingParams
from .out_token_circlequeue import CircularQueue
from .shm_array import ShmArray
from light_tts.server.req_id_generator import convert_sub_id_to_group_id
from light_tts.utils.envs_utils import get_unique_server_name
from light_tts.utils.envs_utils import get_env_start_args
from typing import List, Any, Union


class FinishStatus(ctypes.Structure):
    _pack_ = 4
    _fields_ = [("status", ctypes.c_int)]

    NO_FINISH = 0
    FINISHED_STOP = 1
    FINISHED_LENGTH = 2

    def __init__(self, init_state=NO_FINISH):
        self.status = init_state

    def set_status(self, new_status):
        assert 0 <= new_status <= 2
        self.status = new_status

    def get_status(self):
        return self.status

    def is_finished(self):
        return self.FINISHED_STOP <= self.status <= self.FINISHED_LENGTH

    def get_finish_reason(self):
        if self.status == self.FINISHED_STOP:
            return "stop"
        elif self.status == self.FINISHED_LENGTH:
            return "length"
        return None

class ReqRunStatus(ctypes.Structure):
    _pack_ = 4
    _fields_ = [("status", ctypes.c_int)]

    WAIT_IN_QUEUE = 0 # 在队列中等待
    RUNNING = 1 # 运行
    WAIT_APPEND_PREFILL = 2 # 等待再次prefill
    WAIT_FOR_TEXT = 3 # 等待文本输入

    def __init__(self, init_state=WAIT_IN_QUEUE):
        self.status = init_state
    
    def set_status(self, new_status):
        assert 0 <= new_status <= 3
        self.status = new_status

    def get_status(self):
        return self.status

class PrefixTokenIdsStruct(ctypes.Structure):
    _pack_ = 4
    _fields_ = [("size", ctypes.c_int), ("data", ctypes.c_int64 * 10)]

    def __init__(self):
        self.size = 0

    def set_token_ids(self, ids: List[int]):
        self.size = len(ids)
        self.data[: len(ids)] = ids

    def get_token_ids(self):
        return list(self.data[: self.size])

class Req(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        # cosyvoice2
        ("stream", ctypes.c_bool),
        ("text_len", ctypes.c_int),
        ("semantic_len", ctypes.c_int),
        ("speech_index", ctypes.c_int),
        ("need_extract_speech", ctypes.c_bool),
        ("token_offset", ctypes.c_int),
        ("gen_finished", ctypes.c_bool),
        ("shm_gen_audios_len", ctypes.c_int),
        ("prompt_token_pad", ctypes.c_int),
        ("speed", ctypes.c_float),
        # bistream
        ("bistream", ctypes.c_bool),
        ("bistream_input_finished", ctypes.c_bool),
        ("ignore_eos", ctypes.c_bool),
        ("text_cache_start", ctypes.c_int),
        ("sos_eos_id", ctypes.c_int),
        ("task_id", ctypes.c_int),
        ("bistream_first", ctypes.c_int),
        # normal
        ("index_in_shm_mem", ctypes.c_int),
        ("ref_count", ctypes.c_int),  # 个人不要操作这个计数  # 个人不要操作这个引用计数
        ("request_id", ctypes.c_int),  # 引用计数
        ("group_req_id", ctypes.c_int),
        ("input_len", ctypes.c_int),
        ("shm_infer_released", ctypes.c_bool),  # 推理进程用于标记请求对象已经被推理进程释放，router进程得到信息后亦可释放shm req对象
        ("shm_cur_kv_len", ctypes.c_int),  # 推理进程记录自己当前占用kv 显存长度
        ("shm_cur_output_len", ctypes.c_int),  # 推理进程记录自己输出长度的计数
        # candetoken_out_len 推理进程修改这个数据，让detokenization进程知道需要detoken的长度，
        # 虽然某种程度上 cur_output_len 也有同样的功能，但是为了避免多进程访问导致的问题，添加
        # candetoken_out_len 变量单独传输这个信息。
        ("candetoken_out_len", ctypes.c_int),
        ("prompt_cache_len", ctypes.c_int),  # 用于记录prompt cache 的命中长度，用于统计
        ("is_paused", ctypes.c_bool),  # 标记一个Req因为显存资源管理的原因被临时暂停了。
        ("finish_status", FinishStatus),
        ("req_status", ReqRunStatus),
        ("is_aborted", ctypes.c_bool),
        # 这个标记变量是router进程读取到is_aborted信息后，将这个router_aborted 变量标记为True，因为推理进程
        # 直接读取 is_aborted 变量可能会存在异步问题，但是router的执行线程和推理进程之间是线性运行的，所以router
        # 进程写入的router_aborted信息，所有推理进程可以保证同时读取到的是正确信息，不会出现异步问题。
        ("router_aborted", ctypes.c_bool),
        # 当FinishStatus 是正常结束状态时，finish_token_index 用于标识结束的
        # token 的index位置
        ("finish_token_index", ctypes.c_int),
        ("out_tokens_queue", CircularQueue),
        ("sample_params", SamplingParams),
        # can_released_mark的作用是：
        # 只有整个流程中的最后一个处理模块，一般是 detokenization 进程，标记这个参数为True后，主管理进程才能真
        # 的释放请求对像。
        ("can_released_mark", ctypes.c_bool),
        ("prompt_ids", ctypes.c_int64 * 32768),
        ("output_ids", ctypes.c_int64 * 32768), # for bistream
        ("text_cache", ctypes.c_int64 * 32768), # for bistream
    ]

    def get_str(self):
        return (
            f"request_id:{self.request_id}, input_len:{self.input_len},"
            f"shm_cur_kv_len:{self.shm_cur_kv_len},"
            f"shm_cur_output_len:{self.shm_cur_output_len},"
            f"finish_status:{self.finish_status.is_finished()}"
        )
    
    def init(
        self,
        request_id: int,
        prompt_ids: List[int],
        request_dict: dict,
        sample_param: Union[dict, SamplingParams],
        sos_eos_id: int = 0,
        task_id: int = 0,
        speed: float = 1.0,
    ):
        # 只是为了有更好的编码辅助类型提示
        self.index_in_shm_mem: int = self.index_in_shm_mem
        self.ref_count: int = self.ref_count

        self.request_id = request_id
        self.group_req_id = convert_sub_id_to_group_id(request_id)
        self.is_paused = False
        self.finish_status = FinishStatus()
        self.is_aborted = False
        self.router_aborted = False
        self.shm_infer_released = False
        self.shm_cur_kv_len = 0
        self.shm_cur_output_len = 0
        self.candetoken_out_len = 0
        self.prompt_cache_len = 0
        self.prompt_token_pad = 0
        self.finish_token_index = -1
        self.can_released_mark = False
        self.reward_score = math.nan
        self.cumlogprob = 0.0
        self.speed = speed
        if isinstance(sample_param, SamplingParams):
            self.sample_params = sample_param
        else:
            self.sample_params = SamplingParams()
            self.sample_params.init(**sample_param)

        self.stream = request_dict.get("stream", False)
        self.text_len = len(prompt_ids)
        self.semantic_len = request_dict.get("semantic_len", 0)
        self.gen_finished = False
        self.bistream = request_dict.get("bistream", False)
        self.sos_eos_id = sos_eos_id
        self.task_id = task_id

        if self.bistream:
            self.assign_slice(self.text_cache, 0, prompt_ids)
            self.input_len = 1
            self.prompt_ids[0] = self.sos_eos_id
            self.text_cache_start = 0
            self.req_status = ReqRunStatus(ReqRunStatus.WAIT_FOR_TEXT)
            self.bistream_first = True
        else:
            self.assign_slice(self.prompt_ids, 0, prompt_ids)
            self.input_len = self.text_len + self.semantic_len # not exactly, need set

        self.speech_index = request_dict.get("speech_index", -1)
        self.need_extract_speech = request_dict.get("need_extract_speech", False)
        self.token_offset = 0
        self.ignore_eos = True

        self.bistream_input_finished = request_dict.get("bistream_input_finished", False)
        self.out_tokens_queue = CircularQueue()
        self.post_init()

    def try_to_fill_text(self):
        if self.bistream_first:
            self.init_prompt()
        else:
            self.fill_token()
    
    def init_prompt(self):
        while len(self.audio_ids) > 0:
            if self.text_len >= self.mix_ratio[0]:
                self.assign_slice(
                    self.prompt_ids,
                    self.input_len,
                    list(self.text_cache[self.text_cache_start:self.mix_ratio[0] + self.text_cache_start]) 
                    + self.audio_ids[:self.mix_ratio[1]]
                )
                self.text_cache_start += self.mix_ratio[0]
                self.text_len -= self.mix_ratio[0]
                self.input_len += self.mix_ratio[0] + min(self.mix_ratio[1], len(self.audio_ids))
                self.audio_ids = self.audio_ids[self.mix_ratio[1]:]
            else:
                break

        if len(self.audio_ids) > 0:
            if not self.bistream_input_finished:
                return
            else:
                self.assign_slice(
                    self.prompt_ids,
                    self.input_len,
                    list(self.text_cache[self.text_cache_start:self.text_len + self.text_cache_start])
                    + [self.task_id] + self.audio_ids
                )
                self.text_cache_start += self.text_len
                self.input_len += self.text_len + len(self.audio_ids) + 1
                self.text_len = 0
                self.audio_ids = []
                self.ignore_eos = False

        self.bistream_first = False
        self.req_status.set_status(ReqRunStatus.WAIT_IN_QUEUE)
    
    def fill_token(self):
        # 从text_cache中填充token
        if self.text_len > self.mix_ratio[0]:
            self.assign_slice(
                self.prompt_ids,
                self.input_len,
                list(self.text_cache[self.text_cache_start:self.mix_ratio[0] + self.text_cache_start])
            )
            self.text_cache_start += self.mix_ratio[0]
            self.text_len -= self.mix_ratio[0]
            self.input_len += self.mix_ratio[0]
            self.req_status.set_status(ReqRunStatus.WAIT_APPEND_PREFILL)
        else:
            if not self.bistream_input_finished:
                self.req_status.set_status(ReqRunStatus.WAIT_FOR_TEXT)
            else:
                self.assign_slice(
                    self.prompt_ids,
                    self.input_len,
                    list(self.text_cache[self.text_cache_start:self.text_len + self.text_cache_start])
                    + [self.task_id] + self.audio_ids
                )
                self.text_cache_start += self.text_len
                self.input_len += self.text_len + len(self.audio_ids) + 1
                self.text_len = 0
                self.audio_ids = []
                self.ignore_eos = False
                self.req_status.set_status(ReqRunStatus.WAIT_APPEND_PREFILL)

    def set_gen_audios(self, gen_audios: np.ndarray):
        self.create_gen_audios_shm_array(gen_audios)
        self.shm_gen_audios.arr[:] = gen_audios
        self.gen_finished = True
    
    def get_gen_audios(self) -> np.ndarray:
        self.link_gen_audios_shm_array()
        return self.shm_gen_audios.arr

    def create_gen_audios_shm_array(self, gen_audios: np.ndarray):
        service_uni_name = get_unique_server_name()
        self.shm_gen_audios_len = gen_audios.shape[0]
        name = f"{service_uni_name}_shm_gen_audios_{self.index_in_shm_mem}"
        self.shm_gen_audios = ShmArray(name, gen_audios.shape, dtype=np.float32)
        self.shm_gen_audios.create_shm()
        return

    def link_gen_audios_shm_array(self):
        service_uni_name = get_unique_server_name()
        name = f"{service_uni_name}_shm_gen_audios_{self.index_in_shm_mem}"
        self.shm_gen_audios = ShmArray(name, (self.shm_gen_audios_len,), dtype=np.float32)
        self.shm_gen_audios.link_shm()
        return

    def post_init(self):
        # 子类继承进行一些额外的初始化操作
        pass

    def get_last_gen_token(self):
        return self.output_ids[self.get_output_len() - 1]
    
    def get_output_len(self):
        return self.shm_cur_output_len

    def set_speech_token(self, speech_token: List[int]):
        self.semantic_len = len(speech_token)
        self.input_len = self.text_len + self.semantic_len
        self.assign_slice(
            self.prompt_ids, self.text_len, speech_token
        )
    
    def append_bistream(self, text_ids, min_token_text_ratio, max_token_text_ratio):
        self.assign_slice(self.text_cache, self.text_len, text_ids)
        self.text_len += len(text_ids)
        self.sample_params.min_new_tokens += len(text_ids) * min_token_text_ratio
        self.sample_params.max_new_tokens += len(text_ids) * max_token_text_ratio
    
    @staticmethod
    def assign_slice(ctypes_arr, start: int, values: Union[List[int], np.ndarray]):
        # 推断目标类型
        item_type = ctypes_arr._type_
        item_size = ctypes.sizeof(item_type)

        if isinstance(values, list):
            values = np.array(values, dtype=np.int64)

        nbytes = values.nbytes
        if start + len(values) > len(ctypes_arr):
            raise IndexError("assign_slice would exceed target ctypes array bounds")

        offset_ptr = ctypes.byref(ctypes_arr, start * item_size)
        ctypes.memmove(offset_ptr, values.ctypes.data, nbytes)


    def get_prompt_ids(self):
        return list(self.prompt_ids[: self.input_len])

    def to_router_rpc_obj(self):
        return (self.request_id, self.index_in_shm_mem)

    def can_release(self):
        # 只有管理节点有一个引用
        ref_count_ok = self.ref_count == 1
        can_released_mark = self.can_released_mark

        if self.is_aborted and can_released_mark and ref_count_ok:
            return True

        if self.finish_status.is_finished() and can_released_mark and ref_count_ok and self.out_tokens_queue.is_empty():
            return True

        return False

    def get_used_tokens(self):
        return max(0, self.shm_cur_kv_len)

    def get_tuple_tokens(self, is_busy, router_max_new_token_len):
        raise NotImplementedError("Subclasses should implement this method")

    def get_decode_need_tokens(self):
        raise NotImplementedError("Subclasses should implement this method")

    def get_first_router_need_tokens(self):
        raise NotImplementedError("Subclasses should implement this method")

    def get_all_prompt_metadata(self):
        """
        return_all_prompt_logprobs mode use to return all logprobs cacul ppl
        """
        metadata = {}
        cur_ids = list(self.prompt_ids[: self.input_len])
        all_prompts = []
        for index in range(len(cur_ids) - 1):
            tmp_dict = {int(cur_ids[index + 1]): float(self.shm_logprobs.arr[index + 1])}
            all_prompts.append([int(cur_ids[index]), tmp_dict])

        metadata["prompt_logprobs"] = all_prompts
        metadata["prompt_token_ids"] = [int(e) for e in cur_ids]
        return metadata
    
# 由于目前加入了很多异步调度的方法，为了缓解异步调度带来的很多
# 估计不准确的问题，通过加长输出的长度，进行偏向保守一些的调度
# 理论上不会多估计太多的 token 占用量, 同时得到较高的token显存
# 使用率
ADDED_OUTPUT_LEN = 6


class NormalReq(Req):
    _pack_ = 4

    def get_tuple_tokens(self, is_busy, router_max_new_token_len):
        has_out_len = self.shm_cur_output_len
        if self.sample_params.ignore_eos:
            cur_max_new_token_len = self.sample_params.max_new_tokens
        elif is_busy:
            cur_max_new_token_len = self.sample_params.max_new_tokens
        else:
            # 用当前输出长度的 1.1 倍作为预估输出长度的另一个参考量，用于更新估计的最大输出长度量
            # 后续会更新为更合理的统计条件概率估计方式 to do
            cur_max_new_token_len = min(
                self.sample_params.max_new_tokens, max(int(1.1 * has_out_len), router_max_new_token_len)
            )

        a_len = max(self.input_len + has_out_len + 1, self.shm_cur_kv_len + 1)
        b_len = max(0, cur_max_new_token_len - has_out_len - 1) + ADDED_OUTPUT_LEN

        return (a_len, b_len)

    def get_decode_need_tokens(self):

        return 1

    def get_first_router_need_tokens(self):

        return self.input_len + self.shm_cur_output_len