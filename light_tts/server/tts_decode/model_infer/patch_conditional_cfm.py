# 
import torch
from cosyvoice.flow.flow_matching import ConditionalCFM

def patched_forward_estimator(self, x, mask, mu, t, spks, cond, streaming=False):
    """
    带有torch.cuda.synchronize()的forward_estimator补丁版本
    解决TensorRT有概率结果错误的问题
    """
    if isinstance(self.estimator, torch.nn.Module):
        return self.estimator(x, mask, mu, t, spks, cond, streaming=streaming)
    else:
        [estimator, stream], trt_engine = self.estimator.acquire_estimator()
        with stream:
            estimator.set_input_shape('x', (2, 80, x.size(2)))
            estimator.set_input_shape('mask', (2, 1, x.size(2)))
            estimator.set_input_shape('mu', (2, 80, x.size(2)))
            estimator.set_input_shape('t', (2,))
            estimator.set_input_shape('spks', (2, 80))
            estimator.set_input_shape('cond', (2, 80, x.size(2)))
            data_ptrs = [x.contiguous().data_ptr(),
                            mask.contiguous().data_ptr(),
                            mu.contiguous().data_ptr(),
                            t.contiguous().data_ptr(),
                            spks.contiguous().data_ptr(),
                            cond.contiguous().data_ptr(),
                            x.data_ptr()]
            for i, j in enumerate(data_ptrs):
                estimator.set_tensor_address(trt_engine.get_tensor_name(i), j)
            # 执行前确保所有CUDA操作完成
            torch.cuda.synchronize()
            # run trt engine
            assert estimator.execute_async_v3(torch.cuda.current_stream().cuda_stream) is True
            # 执行后等待所有CUDA操作完成 - 关键修复
            torch.cuda.synchronize()
        self.estimator.release_estimator(estimator, stream)
        return x

# 应用猴子补丁
def apply_forward_estimator_patch():
    """应用forward_estimator的猴子补丁"""
    import os
    
    # 保存原始方法（可选，用于调试或回滚）
    if not hasattr(ConditionalCFM, '_original_forward_estimator'):
        ConditionalCFM._original_forward_estimator = ConditionalCFM.forward_estimator
        # 替换方法
    ConditionalCFM.forward_estimator = patched_forward_estimator
        
        # 只在verbose模式下打印
        # if os.getenv('LIGHTTTS_VERBOSE', '0') == '1':
        #     print("正在应用 forward_estimator 猴子补丁...")
        #     print("现在所有 ConditionalCFM 实例都会使用带有 torch.cuda.synchronize() 的版本")

def remove_forward_estimator_patch():
    """移除补丁，恢复原始方法（可选）"""
    if hasattr(ConditionalCFM, '_original_forward_estimator'):
        ConditionalCFM.forward_estimator = ConditionalCFM._original_forward_estimator
        delattr(ConditionalCFM, '_original_forward_estimator')
        print("forward_estimator 补丁已移除，恢复原始方法")
    else:
        print("没有找到原始方法，无法移除补丁")

# 自动应用补丁版本 - 导入即生效
# print("自动应用 ConditionalCFM.forward_estimator 补丁...")
apply_forward_estimator_patch()

# 使用示例
if __name__ == "__main__":
    # 测试补丁是否生效
    print("测试补丁:")
    print(f"当前方法: {ConditionalCFM.forward_estimator.__name__}")
    print("补丁已自动应用，可以直接使用!")