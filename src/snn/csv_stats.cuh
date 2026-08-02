#ifndef SNN_STAGE2E_CSV_STATS_H
#define SNN_STAGE2E_CSV_STATS_H

#include "types.h"

namespace stage2e {

// GPU 单次遍历归约: 求和 + 非零计数 (替代全量 D2H 拷贝 + CPU 遍历)
// 输出通过 cudaMemcpy 同步回读到 out_sum / out_nz
void device_sum_count(const float* d, int n, float* out_sum, int* out_nz);

// GPU 突触权重统计: mean / abs_mean / min / max
// 采样窗口 [offset, offset+n) 环形滚动 (每步 offset += n), 长期覆盖全数组;
// 输出通过 cudaMemcpy 同步回读
void device_synapse_weight_stats(const BioSynapse* synapses, int n, int offset,
                                 float* mean_w, float* mean_abs_w,
                                 float* min_w, float* max_w);

} // namespace stage2e

#endif
